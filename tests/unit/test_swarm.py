"""Unit tests for swarm module."""
from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

from gantrygraph.swarm.supervisor import GantrySupervisor
from gantrygraph.swarm.worker import ClawWorker, WorkerResult, WorkerSpec

# ── WorkerResult ──────────────────────────────────────────────────────────────

def test_worker_result_success() -> None:
    r = WorkerResult(worker_id=0, task="do x", answer="done")
    assert r.success is True
    assert r.error is None


def test_worker_result_failure() -> None:
    r = WorkerResult(worker_id=1, task="fail", error="something broke")
    assert r.success is False
    assert r.answer is None


def test_worker_result_default_metadata() -> None:
    r = WorkerResult(worker_id=0, task="t")
    assert r.metadata == {}


# ── ClawWorker ────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_worker_run_success() -> None:
    mock_engine = MagicMock()
    mock_engine.arun = AsyncMock(return_value="task result")

    worker = ClawWorker(worker_id=0, engine_factory=lambda: mock_engine)
    result = await worker.run("do something")

    assert result.success
    assert result.answer == "task result"
    assert result.worker_id == 0
    assert result.task == "do something"


@pytest.mark.asyncio
async def test_worker_run_captures_exception() -> None:
    mock_engine = MagicMock()
    mock_engine.arun = AsyncMock(side_effect=RuntimeError("engine blew up"))

    worker = ClawWorker(worker_id=1, engine_factory=lambda: mock_engine)
    result = await worker.run("broken task")

    assert not result.success
    assert result.error == "engine blew up"
    assert result.answer is None


@pytest.mark.asyncio
async def test_worker_factory_called_per_run() -> None:
    """Each run() creates a fresh engine via factory."""
    call_count = 0

    def factory() -> Any:
        nonlocal call_count
        call_count += 1
        engine = MagicMock()
        engine.arun = AsyncMock(return_value="ok")
        return engine

    worker = ClawWorker(worker_id=0, engine_factory=factory)
    await worker.run("t1")
    await worker.run("t2")
    assert call_count == 2


# ── GantrySupervisor ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_supervisor_runs_workers_concurrently() -> None:
    """Supervisor spawns multiple workers and synthesises their results."""
    decompose_response = AIMessage(content="1. Subtask A\n2. Subtask B")
    synthesize_response = AIMessage(content="Final synthesized answer.")

    llm = FakeMessagesListChatModel(
        responses=[decompose_response, synthesize_response]
    )

    call_log: list[str] = []

    def worker_factory() -> Any:
        engine = MagicMock()
        engine.arun = AsyncMock(side_effect=lambda task: call_log.append(task) or f"done: {task}")  # type: ignore[misc]
        return engine

    supervisor = GantrySupervisor(
        llm=llm,
        worker_factory=worker_factory,
        max_workers=4,
    )
    result = await supervisor.run("Analyse some data")
    assert isinstance(result, str)
    assert len(result) > 0
    # Both subtasks were dispatched
    assert len(call_log) == 2


@pytest.mark.asyncio
async def test_supervisor_falls_back_to_single_task_on_empty_decompose() -> None:
    """If decompose returns nothing, the whole task is treated as one subtask."""
    empty_decompose = AIMessage(content="")
    synth = AIMessage(content="Synthesized.")

    llm = FakeMessagesListChatModel(responses=[empty_decompose, synth])

    def factory() -> Any:
        engine = MagicMock()
        engine.arun = AsyncMock(return_value="worker answer")
        return engine

    supervisor = GantrySupervisor(llm=llm, worker_factory=factory, max_workers=2)
    result = await supervisor.run("Single big task")
    assert isinstance(result, str)


@pytest.mark.asyncio
async def test_supervisor_max_workers_caps_parallel() -> None:
    """No more than max_workers workers are spawned."""
    many_subtasks = "\n".join(f"{i+1}. Task {i+1}" for i in range(10))
    decompose_response = AIMessage(content=many_subtasks)
    synth = AIMessage(content="All done.")

    llm = FakeMessagesListChatModel(responses=[decompose_response, synth])
    call_log: list[str] = []

    def factory() -> Any:
        engine = MagicMock()
        engine.arun = AsyncMock(side_effect=lambda t: call_log.append(t) or "ok")  # type: ignore[misc]
        return engine

    supervisor = GantrySupervisor(llm=llm, worker_factory=factory, max_workers=3)
    await supervisor.run("Big task")
    assert len(call_log) == 3  # capped at max_workers


@pytest.mark.asyncio
async def test_supervisor_handles_worker_failure() -> None:
    """Failed workers are captured gracefully — supervisor still synthesises."""
    decompose = AIMessage(content="1. Task A\n2. Task B")
    synth = AIMessage(content="Partial results synthesized.")
    llm = FakeMessagesListChatModel(responses=[decompose, synth])

    call_count = [0]

    def factory() -> Any:
        engine = MagicMock()
        call_count[0] += 1
        if call_count[0] % 2 == 0:
            engine.arun = AsyncMock(side_effect=RuntimeError("worker failed"))
        else:
            engine.arun = AsyncMock(return_value="success")
        return engine

    supervisor = GantrySupervisor(llm=llm, worker_factory=factory, max_workers=2)
    result = await supervisor.run("Mixed success task")
    assert isinstance(result, str)


# ── WorkerSpec ────────────────────────────────────────────────────────────────

def test_worker_spec_construction() -> None:
    mock_engine = MagicMock()
    spec = WorkerSpec(name="analyst", engine=mock_engine, description="Analyses data.")
    assert spec.name == "analyst"
    assert spec.engine is mock_engine
    assert spec.description == "Analyses data."


def test_worker_spec_default_description() -> None:
    spec = WorkerSpec(name="worker", engine=MagicMock())
    assert spec.description == ""


def test_worker_spec_importable_from_gantrygraph() -> None:
    from gantrygraph import WorkerSpec as WS  # noqa: F401
    assert WS is WorkerSpec


def test_worker_spec_importable_from_gantrygraph_swarm() -> None:
    from gantrygraph.swarm import WorkerSpec as WS  # noqa: F401
    assert WS is WorkerSpec


# ── GantrySupervisor constructor validation ─────────────────────────────────────

def test_supervisor_requires_factory_or_workers() -> None:
    llm = FakeMessagesListChatModel(responses=[])
    with pytest.raises(ValueError, match="worker_factory"):
        GantrySupervisor(llm=llm)


def test_supervisor_rejects_both_factory_and_workers() -> None:
    llm = FakeMessagesListChatModel(responses=[])
    with pytest.raises(ValueError, match="not both"):
        GantrySupervisor(
            llm=llm,
            worker_factory=lambda: MagicMock(),
            workers=[WorkerSpec(name="w", engine=MagicMock())],
        )


# ── Heterogeneous worker path ─────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_supervisor_routes_subtasks_to_specialists() -> None:
    """Supervisor with WorkerSpec list routes subtasks and synthesises."""
    # LLM returns routing assignments then synthesis
    route_response = AIMessage(
        content="[analyst] Analyse the dataset\n[writer] Write the report"
    )
    synth_response = AIMessage(content="Combined result.")
    llm = FakeMessagesListChatModel(responses=[route_response, synth_response])

    analyst_log: list[str] = []
    writer_log: list[str] = []

    analyst_engine = MagicMock()
    analyst_engine.arun = AsyncMock(
        side_effect=lambda t: analyst_log.append(t) or "analysis done"  # type: ignore[misc]
    )
    writer_engine = MagicMock()
    writer_engine.arun = AsyncMock(
        side_effect=lambda t: writer_log.append(t) or "report written"  # type: ignore[misc]
    )

    supervisor = GantrySupervisor(
        llm=llm,
        workers=[
            WorkerSpec(name="analyst", engine=analyst_engine, description="Analyses data."),
            WorkerSpec(name="writer",  engine=writer_engine,  description="Writes reports."),
        ],
    )
    result = await supervisor.run("Analyse data and write a report")

    assert isinstance(result, str)
    assert len(result) > 0
    assert len(analyst_log) == 1
    assert len(writer_log) == 1


@pytest.mark.asyncio
async def test_supervisor_fallback_on_unknown_worker_name() -> None:
    """Unknown worker name in LLM output falls back to the first worker."""
    route_response = AIMessage(content="[nonexistent_worker] Do something")
    synth_response = AIMessage(content="Done.")
    llm = FakeMessagesListChatModel(responses=[route_response, synth_response])

    fallback_log: list[str] = []
    fallback_engine = MagicMock()
    fallback_engine.arun = AsyncMock(
        side_effect=lambda t: fallback_log.append(t) or "ok"  # type: ignore[misc]
    )

    supervisor = GantrySupervisor(
        llm=llm,
        workers=[
            WorkerSpec(name="first", engine=fallback_engine, description="Fallback worker."),
        ],
    )
    await supervisor.run("Some task")
    assert len(fallback_log) == 1


@pytest.mark.asyncio
async def test_supervisor_worker_metadata_in_result() -> None:
    """WorkerResult metadata contains worker_name for traceability."""
    route_response = AIMessage(content="[specialist] Subtask A")
    synth_response = AIMessage(content="Synthesized.")
    llm = FakeMessagesListChatModel(responses=[route_response, synth_response])

    captured_results: list[WorkerResult] = []

    engine = MagicMock()
    engine.arun = AsyncMock(return_value="specialist answer")

    class _TracingSupervisor(GantrySupervisor):
        async def _synthesize(
            self, original_task: str, results: list[WorkerResult]
        ) -> str:
            captured_results.extend(results)
            return await super()._synthesize(original_task, results)

    supervisor = _TracingSupervisor(
        llm=llm,
        workers=[WorkerSpec(name="specialist", engine=engine, description="Does stuff.")],
    )
    await supervisor.run("task")
    assert captured_results[0].metadata["worker_name"] == "specialist"


@pytest.mark.asyncio
async def test_supervisor_specs_run_concurrently() -> None:
    """WorkerSpec engines are called concurrently, not sequentially."""
    import asyncio as _asyncio

    route_response = AIMessage(
        content="[slow1] First subtask\n[slow2] Second subtask"
    )
    synth_response = AIMessage(content="Done.")
    llm = FakeMessagesListChatModel(responses=[route_response, synth_response])

    started: list[float] = []

    async def _slow(_: str) -> str:
        started.append(_asyncio.get_event_loop().time())
        await _asyncio.sleep(0.05)
        return "ok"

    e1 = MagicMock()
    e1.arun = AsyncMock(side_effect=_slow)
    e2 = MagicMock()
    e2.arun = AsyncMock(side_effect=_slow)

    supervisor = GantrySupervisor(
        llm=llm,
        workers=[
            WorkerSpec(name="slow1", engine=e1, description="First."),
            WorkerSpec(name="slow2", engine=e2, description="Second."),
        ],
    )
    await supervisor.run("parallel task")
    assert len(started) == 2
    assert abs(started[1] - started[0]) < 0.04
