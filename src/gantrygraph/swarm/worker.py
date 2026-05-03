"""AgentWorker — a single agent worker within a swarm."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class WorkerResult:
    """Outcome from a single AgentWorker execution."""

    worker_name: str
    task: str
    result: str | None = None
    error: str | None = None
    steps_taken: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return self.error is None and self.result is not None


@dataclass
class WorkerSpec:
    """Named specialist worker with its own pre-configured engine.

    Use ``WorkerSpec`` when different subtasks require different tools,
    LLMs, or configurations.  Pass a list of ``WorkerSpec`` instances
    to ``GantrySupervisor`` instead of a ``worker_factory``.

    The supervisor LLM reads all ``description`` fields and automatically
    routes each decomposed subtask to the most appropriate specialist.

    Args:
        name:        Short identifier used in routing (e.g. ``"analyst"``).
        engine:      A fully-configured ``GantryEngine`` instance.
        description: One-sentence description of what this worker can do.
                     Used by the supervisor LLM to route subtasks.

    Example::

        from gantrygraph import GantryEngine
        from gantrygraph.actions import ShellTools, FileSystemTools
        from gantrygraph.swarm import GantrySupervisor, WorkerSpec
        from langchain_anthropic import ChatAnthropic

        llm = ChatAnthropic(model="claude-sonnet-4-6")

        supervisor = GantrySupervisor(
            llm=llm,
            workers=[
                WorkerSpec(
                    name="shell_expert",
                    engine=GantryEngine(llm=llm, tools=[ShellTools(workspace="/tmp")]),
                    description="Runs shell commands, explores the filesystem, executes scripts.",
                ),
                WorkerSpec(
                    name="file_editor",
                    engine=GantryEngine(llm=llm, tools=[FileSystemTools(workspace="/tmp")]),
                    description="Reads, writes, and edits files.",
                ),
            ],
        )
        result = await supervisor.arun(
            "Explore /tmp, find all .log files, then read their first 10 lines."
        )
    """

    name: str
    engine: Any  # GantryEngine — kept as Any to avoid circular import
    description: str = ""


class AgentWorker:
    """Wraps a ``GantryEngine`` for use inside a ``GantrySupervisor``.

    Workers are stateless — each ``run()`` call creates a fresh engine
    instance from the provided factory so workers don't share memory.

    Example (usually not used directly — see ``GantrySupervisor``)::

        worker = AgentWorker(worker_name="worker_0", engine_factory=lambda: GantryEngine(...))
        result = await worker.run("Summarise page 1")
    """

    def __init__(
        self,
        worker_name: str,
        engine_factory: Any,  # Callable[[], GantryEngine]
    ) -> None:
        self._name = worker_name
        self._engine_factory = engine_factory

    async def run(self, task: str) -> WorkerResult:
        engine = self._engine_factory()
        try:
            answer = await engine.arun(task)
            return WorkerResult(worker_name=self._name, task=task, result=answer)
        except Exception as exc:
            return WorkerResult(
                worker_name=self._name,
                task=task,
                error=str(exc),
            )


ClawWorker = AgentWorker  # backward compat alias
