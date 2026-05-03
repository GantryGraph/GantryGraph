"""FastAPI server for cloud deployment of GantryEngine agents.

Requires the ``[cloud]`` extra::

    pip install gantrygraph[cloud]

Usage::

    from gantrygraph import GantryEngine
    from gantrygraph.cloud import serve

    def make_agent() -> GantryEngine:
        return GantryEngine(llm=my_llm, tools=[...])

    serve(make_agent, host="0.0.0.0", port=8080)

Endpoints:
    POST /run                     → { job_id, status }          202
    GET  /status/{job_id}         → RunResponse
    GET  /stream/{job_id}         → Server-Sent Events stream    text/event-stream
    POST /resume/{job_id}         → Resume a suspended job       202
    GET  /health                  → { status: "ok" }
"""
from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import Callable
from typing import Any, Literal

# Pydantic is a core dependency — always available regardless of [cloud] extra
from pydantic import BaseModel

try:
    import uvicorn
    from fastapi import BackgroundTasks, FastAPI, HTTPException
    from fastapi.responses import StreamingResponse

    _HAS_CLOUD = True
except ImportError:
    _HAS_CLOUD = False

_INSTALL_MSG = "serve() requires the [cloud] extra: pip install 'gantrygraph[cloud]'"

# ── global job state ──────────────────────────────────────────────────────────

_jobs: dict[str, RunResponse] = {}
_job_queues: dict[str, asyncio.Queue[dict[str, Any] | None]] = {}
_suspended_engines: dict[str, Any] = {}  # job_id → suspended GantryEngine

_JOB_TTL = 3600  # seconds before completed/failed/suspended jobs are removed


# ── request / response models (always importable) ─────────────────────────────

class RunRequest(BaseModel):
    task: str


class RunResponse(BaseModel):
    job_id: str
    status: Literal["queued", "running", "completed", "failed", "suspended"]
    result: str | None = None
    error: str | None = None
    thread_id: str | None = None  # set when status == "suspended"


class ResumeRequest(BaseModel):
    approved: bool = True


# ── public entry point ────────────────────────────────────────────────────────

def serve(
    engine_factory: Callable[[], Any],
    host: str = "0.0.0.0",
    port: int = 8080,
) -> None:
    """Start a FastAPI HTTP server that exposes a GantryEngine via REST.

    Args:
        engine_factory: Called for each POST /run request to create a fresh
                        ``GantryEngine`` instance.
        host:           Bind host (default ``0.0.0.0``).
        port:           Bind port (default ``8080``).
    """
    if not _HAS_CLOUD:
        raise ImportError(_INSTALL_MSG)
    app = _build_app(engine_factory)
    uvicorn.run(app, host=host, port=port)


def _build_app(engine_factory: Callable[[], Any]) -> FastAPI:
    """Build the FastAPI app (separated from uvicorn for testability)."""
    if not _HAS_CLOUD:
        raise ImportError(_INSTALL_MSG)

    app = FastAPI(
        title="gantrygraph agent server",
        description="REST API for GantryEngine autonomous agents.",
        version="0.1.0",
    )

    # ── POST /run ─────────────────────────────────────────────────────────────

    @app.post("/run", response_model=RunResponse, status_code=202)
    async def run_task(req: RunRequest, background_tasks: BackgroundTasks) -> RunResponse:
        """Submit a task.  Returns a ``job_id`` immediately; execution is async."""
        job_id = str(uuid.uuid4())
        response = RunResponse(job_id=job_id, status="queued")
        _jobs[job_id] = response
        background_tasks.add_task(_execute_job, job_id, req.task, engine_factory)
        return response

    # ── GET /status/{job_id} ──────────────────────────────────────────────────

    @app.get("/status/{job_id}", response_model=RunResponse)
    async def get_status(job_id: str) -> RunResponse:
        """Poll the status of a previously submitted job."""
        if job_id not in _jobs:
            raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
        return _jobs[job_id]

    # ── GET /stream/{job_id} — SSE ────────────────────────────────────────────

    @app.get("/stream/{job_id}")
    async def stream_events(job_id: str) -> StreamingResponse:
        """Stream agent events in real time via Server-Sent Events (SSE).

        Connect with ``EventSource`` in the browser or ``httpx`` with
        ``stream=True``.  Each SSE message is a JSON-encoded ``GantryEvent``::

            data: {"type": "act", "step": 2, "data": {"tools_executed": ["shell_run"]}}

        A final ``event: done`` message signals end-of-stream.
        """
        if job_id not in _jobs:
            raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")

        # Queue is created in _execute_job.  If the job already finished before
        # the client connected, we still drain the buffered events.
        queue = _job_queues.get(job_id)
        if queue is None:
            # Job was never started or cleaned up — return 410 Gone
            raise HTTPException(
                status_code=410,
                detail="Event stream no longer available.",
            )

        async def _sse_generator() -> Any:
            while True:
                payload = await queue.get()
                if payload is None:  # sentinel: job ended
                    yield "event: done\ndata: {}\n\n"
                    break
                yield f"data: {json.dumps(payload)}\n\n"

        return StreamingResponse(
            _sse_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    # ── POST /resume/{job_id} ─────────────────────────────────────────────────

    @app.post("/resume/{job_id}", response_model=RunResponse, status_code=202)
    async def resume_job(
        job_id: str,
        req: ResumeRequest,
        background_tasks: BackgroundTasks,
    ) -> RunResponse:
        """Resume a suspended job.

        The job must be in ``"suspended"`` status (i.e. it was created with an
        engine that has ``enable_suspension=True`` and triggered a HITL pause).
        """
        if job_id not in _jobs:
            raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
        job = _jobs[job_id]
        if job.status != "suspended":
            raise HTTPException(
                status_code=409,
                detail=f"Job is not suspended (current status: {job.status}).",
            )
        thread_id = job.thread_id
        if not thread_id:
            raise HTTPException(status_code=500, detail="No thread_id stored for this job.")

        job.status = "running"
        background_tasks.add_task(_resume_job, job_id, thread_id, req.approved)
        return job

    # ── GET /health ───────────────────────────────────────────────────────────

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


# ── background task helpers ───────────────────────────────────────────────────

async def _cleanup_job(job_id: str) -> None:
    """Remove a job from all state dicts after the TTL expires."""
    await asyncio.sleep(_JOB_TTL)
    _jobs.pop(job_id, None)
    _job_queues.pop(job_id, None)
    _suspended_engines.pop(job_id, None)


async def _execute_job(
    job_id: str,
    task: str,
    engine_factory: Callable[[], Any],
) -> None:
    """Background coroutine: run the engine and stream events to the SSE queue."""
    queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
    _job_queues[job_id] = queue

    engine = engine_factory()

    # Wrap the engine's on_event so events flow to the SSE queue too
    original_cb = getattr(engine, "_event_cb", None)

    async def _streaming_cb(event: Any) -> None:
        payload = {
            "type": event.event_type,
            "step": event.step,
            "data": event.data,
        }
        await queue.put(payload)
        if original_cb is not None:
            from gantrygraph._utils import ensure_awaitable

            await ensure_awaitable(original_cb, event)

    engine._event_cb = _streaming_cb
    engine._compiled = None  # force rebuild with new callback

    _jobs[job_id].status = "running"
    try:
        from gantrygraph.engine.engine import AgentSuspended

        result = await engine.arun(task)
        _jobs[job_id].status = "completed"
        _jobs[job_id].result = result
    except AgentSuspended as sus:
        _jobs[job_id].status = "suspended"
        _jobs[job_id].thread_id = sus.thread_id
        _suspended_engines[job_id] = engine  # dedicated dict — no type abuse
    except Exception as exc:
        _jobs[job_id].status = "failed"
        _jobs[job_id].error = str(exc)
    finally:
        await queue.put(None)  # signal SSE stream end
        asyncio.create_task(_cleanup_job(job_id))  # schedule TTL cleanup


async def _resume_job(
    job_id: str,
    thread_id: str,
    approved: bool,
) -> None:
    """Background coroutine: resume a suspended engine run."""
    engine: Any = _suspended_engines.pop(job_id, None)

    queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
    _job_queues[job_id] = queue

    if engine is None:
        # Engine was cleaned up by TTL before the resume arrived.
        _jobs[job_id].status = "failed"
        _jobs[job_id].error = "Suspended engine expired before resume was called."
        await queue.put(None)
        asyncio.create_task(_cleanup_job(job_id))
        return

    try:
        from gantrygraph.engine.engine import AgentSuspended

        result = await engine.resume(thread_id, approved=approved)
        _jobs[job_id].status = "completed"
        _jobs[job_id].result = result
        _jobs[job_id].thread_id = None
    except AgentSuspended as sus:
        # Suspended again — store engine for next resume
        _jobs[job_id].status = "suspended"
        _jobs[job_id].thread_id = sus.thread_id
        _suspended_engines[job_id] = engine
    except Exception as exc:
        _jobs[job_id].status = "failed"
        _jobs[job_id].error = str(exc)
    finally:
        await queue.put(None)
        asyncio.create_task(_cleanup_job(job_id))
