"""Unit tests for cloud serve module using FastAPI TestClient."""
from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

# Import only if fastapi is available (cloud extra)
try:
    from httpx import ASGITransport, AsyncClient

    from gantrygraph.cloud.serve import _build_app, _job_queues, _jobs, _suspended_engines
    _HAS_CLOUD = True
except ImportError:
    _HAS_CLOUD = False

pytestmark = pytest.mark.skipif(not _HAS_CLOUD, reason="Requires [cloud] extra")


@pytest.fixture(autouse=True)
def clear_jobs() -> None:
    """Reset global job store between tests."""
    _jobs.clear()
    _job_queues.clear()
    _suspended_engines.clear()


def _make_engine_factory(answer: str = "done") -> Any:
    def factory() -> Any:
        engine = MagicMock()
        engine.arun = AsyncMock(return_value=answer)
        return engine
    return factory


@pytest.mark.asyncio
async def test_health_endpoint() -> None:
    app = _build_app(_make_engine_factory())
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_run_returns_job_id() -> None:
    app = _build_app(_make_engine_factory())
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/run", json={"task": "do something"})
    assert resp.status_code == 202
    data = resp.json()
    assert "job_id" in data
    assert data["status"] == "queued"


@pytest.mark.asyncio
async def test_status_unknown_job_returns_404() -> None:
    app = _build_app(_make_engine_factory())
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/status/nonexistent-id")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_status_queued_job_returns_200() -> None:
    app = _build_app(_make_engine_factory())
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        post_resp = await client.post("/run", json={"task": "test"})
        job_id = post_resp.json()["job_id"]

        get_resp = await client.get(f"/status/{job_id}")
    assert get_resp.status_code == 200
    assert get_resp.json()["job_id"] == job_id


@pytest.mark.asyncio
async def test_job_completes_in_background() -> None:
    app = _build_app(_make_engine_factory(answer="final answer"))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        post_resp = await client.post("/run", json={"task": "complete me"})
        job_id = post_resp.json()["job_id"]

        # Give background task time to complete
        await asyncio.sleep(0.2)

        get_resp = await client.get(f"/status/{job_id}")
    data = get_resp.json()
    assert data["status"] == "completed"
    assert data["result"] == "final answer"


@pytest.mark.asyncio
async def test_failed_job_records_error() -> None:
    def failing_factory() -> Any:
        engine = MagicMock()
        engine.arun = AsyncMock(side_effect=RuntimeError("engine exploded"))
        return engine

    app = _build_app(failing_factory)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        post_resp = await client.post("/run", json={"task": "break"})
        job_id = post_resp.json()["job_id"]
        await asyncio.sleep(0.2)
        get_resp = await client.get(f"/status/{job_id}")

    data = get_resp.json()
    assert data["status"] == "failed"
    assert "engine exploded" in data["error"]


@pytest.mark.asyncio
async def test_multiple_jobs_tracked_independently() -> None:
    app = _build_app(_make_engine_factory())
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r1 = await client.post("/run", json={"task": "task 1"})
        r2 = await client.post("/run", json={"task": "task 2"})

    id1 = r1.json()["job_id"]
    id2 = r2.json()["job_id"]
    assert id1 != id2
    assert id1 in _jobs
    assert id2 in _jobs


# ── SSE streaming ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_stream_unknown_job_returns_404() -> None:
    app = _build_app(_make_engine_factory())
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/stream/nonexistent-id")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_stream_endpoint_delivers_done_event() -> None:
    """SSE stream must terminate with event: done after job completes."""
    app = _build_app(_make_engine_factory(answer="streaming answer"))
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        # Submit job
        post = await client.post("/run", json={"task": "stream me"})
        job_id = post.json()["job_id"]

        # Give background task time to start and set up the queue
        await asyncio.sleep(0.05)

        # Consume the SSE stream
        lines: list[str] = []
        async with client.stream("GET", f"/stream/{job_id}") as resp:
            assert resp.status_code == 200
            assert "text/event-stream" in resp.headers["content-type"]
            async for line in resp.aiter_lines():
                lines.append(line)
                if "event: done" in line:
                    break

    assert any("event: done" in ln for ln in lines)


@pytest.mark.asyncio
async def test_stream_delivers_agent_events() -> None:
    """Events emitted by the engine must appear in the SSE stream."""
    from gantrygraph.core.events import GantryEvent

    # Engine factory that emits one custom event then returns
    def _factory() -> Any:
        engine = MagicMock()

        async def _arun(task: str, *, thread_id=None) -> str:
            if hasattr(engine, "_event_cb") and engine._event_cb:
                from gantrygraph._utils import ensure_awaitable
                await ensure_awaitable(
                    engine._event_cb, GantryEvent("act", 1, {"tools_executed": ["shell_run"]})
                )
            return "done"

        engine.arun = _arun
        engine._event_cb = None
        engine._compiled = None
        return engine

    app = _build_app(_factory)
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        post = await client.post("/run", json={"task": "track events"})
        job_id = post.json()["job_id"]
        await asyncio.sleep(0.1)

        lines: list[str] = []
        async with client.stream("GET", f"/stream/{job_id}") as resp:
            async for line in resp.aiter_lines():
                lines.append(line)
                if "event: done" in line:
                    break

    data_lines = [ln for ln in lines if ln.startswith("data:")]
    # At least one data line with the act event
    assert any("act" in ln for ln in data_lines)


# ── Suspension / resume ───────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_resume_unknown_job_returns_404() -> None:
    app = _build_app(_make_engine_factory())
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/resume/no-such-job", json={"approved": True})
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_resume_non_suspended_job_returns_409() -> None:
    from gantrygraph.cloud.serve import RunResponse

    app = _build_app(_make_engine_factory())
    # Manually insert a completed job
    _jobs["completed-job"] = RunResponse(job_id="completed-job", status="completed")
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/resume/completed-job", json={"approved": True})
    assert resp.status_code == 409


@pytest.mark.asyncio
async def test_suspend_then_resume_completes_job() -> None:
    """Simulate a suspended job and verify that /resume/{job_id} completes it."""
    from gantrygraph.engine.engine import AgentSuspended

    call_count = 0

    def _suspending_factory() -> Any:
        nonlocal call_count
        engine = MagicMock()

        async def _arun(task: str, *, thread_id: str | None = None) -> str:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise AgentSuspended(thread_id="test-thread-123", data={"tool": "rm -rf"})
            return "completed after resume"

        async def _resume(thread_id: str, *, approved: bool = True) -> str:
            return "completed after resume"

        engine.arun = _arun
        engine.resume = _resume
        engine._event_cb = None
        engine._compiled = None
        return engine

    app = _build_app(_suspending_factory)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # Submit and let it suspend
        post = await client.post("/run", json={"task": "risky task"})
        job_id = post.json()["job_id"]
        await asyncio.sleep(0.3)

        # Should now be suspended
        status_resp = await client.get(f"/status/{job_id}")
        assert status_resp.json()["status"] == "suspended"
        assert status_resp.json()["thread_id"] == "test-thread-123"

        # Resume it
        resume_resp = await client.post(f"/resume/{job_id}", json={"approved": True})
        assert resume_resp.status_code == 202

        await asyncio.sleep(0.3)

        final = await client.get(f"/status/{job_id}")
    assert final.json()["status"] == "completed"
    assert final.json()["result"] == "completed after resume"


@pytest.mark.asyncio
async def test_resume_expired_engine_returns_failed() -> None:
    """If the suspended engine was cleaned up before resume, job goes to failed."""
    from gantrygraph.cloud.serve import RunResponse

    app = _build_app(_make_engine_factory())
    # Insert a job in "suspended" state but with NO engine in _suspended_engines
    _jobs["stale-job"] = RunResponse(
        job_id="stale-job",
        status="suspended",
        thread_id="some-thread",
    )
    # Do NOT put anything in _suspended_engines — simulates TTL expiry

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resume_resp = await client.post("/resume/stale-job", json={"approved": True})
        assert resume_resp.status_code == 202

        await asyncio.sleep(0.2)

        final = await client.get("/status/stale-job")

    assert final.json()["status"] == "failed"
    assert "expired" in final.json()["error"]
