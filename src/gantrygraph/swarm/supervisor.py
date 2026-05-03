"""GantrySupervisor — decomposes a task and runs workers in parallel."""
from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Callable
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from gantrygraph._utils import _run_sync
from gantrygraph.swarm.worker import AgentWorker, WorkerResult, WorkerSpec

logger = logging.getLogger(__name__)


class GantrySupervisor:
    """Decompose a task into parallel subtasks and synthesize the results.

    The supervisor uses the LLM twice:

    1. **Decompose** — break the input task into N independent subtasks,
       optionally assigning each to a named specialist worker.
    2. **Synthesize** — merge all worker answers into a final response.

    Worker agents run concurrently via ``asyncio.gather``.

    **Homogeneous workers (all workers share the same tools):**

    .. code-block:: python

        from gantrygraph.swarm import GantrySupervisor
        from gantrygraph import GantryEngine
        from langchain_anthropic import ChatAnthropic

        llm = ChatAnthropic(model="claude-sonnet-4-6")
        supervisor = GantrySupervisor(
            llm=llm,
            worker_factory=lambda: GantryEngine(llm=llm, tools=[...]),
            max_workers=4,
        )
        result = await supervisor.arun("Analyse these 10 documents and summarise findings")

    **Heterogeneous workers (each worker has its own tools):**

    .. code-block:: python

        from gantrygraph.swarm import GantrySupervisor, WorkerSpec
        from gantrygraph import GantryEngine
        from gantrygraph.actions import ShellTools, FileSystemTools
        from langchain_anthropic import ChatAnthropic

        llm = ChatAnthropic(model="claude-sonnet-4-6")
        supervisor = GantrySupervisor(
            llm=llm,
            workers=[
                WorkerSpec(
                    name="shell_expert",
                    engine=GantryEngine(llm=llm, tools=[ShellTools(workspace="/tmp")]),
                    description="Runs shell commands and explores the filesystem.",
                ),
                WorkerSpec(
                    name="file_editor",
                    engine=GantryEngine(llm=llm, tools=[FileSystemTools(workspace="/tmp")]),
                    description="Reads, writes, and edits files.",
                ),
            ],
        )
        result = await supervisor.arun(
            "Find all .log files in /tmp and summarise their contents."
        )

    When ``workers`` is provided the supervisor asks the LLM to assign
    each subtask to the most appropriate specialist by name.  If a
    subtask is not assigned (or the name is unrecognised), it falls back
    to the first worker in the list.
    """

    def __init__(
        self,
        *,
        llm: BaseChatModel,
        worker_factory: Callable[[], Any] | None = None,
        workers: list[WorkerSpec] | None = None,
        max_workers: int = 5,
    ) -> None:
        if worker_factory is None and workers is None:
            raise ValueError("Provide either worker_factory= or workers=.")
        if worker_factory is not None and workers is not None:
            raise ValueError("Provide worker_factory= OR workers=, not both.")
        self._llm = llm
        self._worker_factory = worker_factory
        self._workers = workers or []
        self._max_workers = max_workers

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self, task: str) -> str:
        """Synchronous entry point.  Blocks until the task completes."""
        return _run_sync(self.arun(task))

    async def arun(self, task: str) -> str:
        """Decompose *task*, run workers concurrently, and synthesise results."""
        logger.info("Supervisor decomposing task: %s", task[:80])
        if self._workers:
            return await self._run_with_specs(task)
        return await self._run_with_factory(task)

    # ── Homogeneous path (worker_factory) ─────────────────────────────────────

    async def _run_with_factory(self, task: str) -> str:
        subtasks = await self._decompose(task)
        active_subtasks = subtasks[: self._max_workers]
        logger.info(
            "Spawning %d factory workers for %d subtasks",
            len(active_subtasks), len(subtasks),
        )
        workers = [
            AgentWorker(worker_name=str(i), engine_factory=self._worker_factory)
            for i in range(len(active_subtasks))
        ]
        results: list[WorkerResult] = await asyncio.gather(
            *[w.run(t) for w, t in zip(workers, active_subtasks, strict=False)],
            return_exceptions=False,
        )
        return await self._synthesize(task, results)

    # ── Heterogeneous path (WorkerSpec) ───────────────────────────────────────

    async def _run_with_specs(self, task: str) -> str:
        assignments = await self._decompose_with_routing(task)
        logger.info(
            "Routing %d subtasks to %d specialist workers",
            len(assignments), len(self._workers),
        )
        results: list[WorkerResult] = await asyncio.gather(
            *[
                self._run_spec(spec, subtask)
                for subtask, spec in assignments
            ],
            return_exceptions=False,
        )
        return await self._synthesize(task, results)

    async def _run_spec(self, spec: WorkerSpec, task: str) -> WorkerResult:
        logger.info("Worker [%s] starting subtask: %s", spec.name, task[:60])
        try:
            answer = await spec.engine.arun(task)
            return WorkerResult(worker_name=spec.name, task=task, result=answer)
        except Exception as exc:
            logger.warning("Worker [%s] failed: %s", spec.name, exc)
            return WorkerResult(worker_name=spec.name, task=task, error=str(exc))

    # ── LLM helpers ───────────────────────────────────────────────────────────

    async def _decompose(self, task: str) -> list[str]:
        """Ask the LLM to break *task* into parallel subtasks."""
        prompt = (
            "Break the following task into independent subtasks that can be worked on in parallel. "
            "Return ONLY a numbered list, one subtask per line, no extra text.\n\n"
            f"Task: {task}"
        )
        response = await self._llm.ainvoke(
            [SystemMessage(content="You are a task decomposition assistant."),
             HumanMessage(content=prompt)]
        )
        lines = str(response.content).strip().split("\n")
        subtasks: list[str] = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if line and line[0].isdigit():
                line = line.lstrip("0123456789.)- ").strip()
            if line:
                subtasks.append(line)
        return subtasks or [task]

    async def _decompose_with_routing(self, task: str) -> list[tuple[str, WorkerSpec]]:
        """Decompose *task* and assign each subtask to a named specialist.

        Returns a list of ``(subtask_description, WorkerSpec)`` pairs.
        The LLM is asked to produce lines in the format::

            [worker_name] subtask description

        Unrecognised worker names fall back to the first worker.
        """
        worker_list = "\n".join(
            f"- {w.name}: {w.description}" for w in self._workers
        )
        prompt = (
            f"You have these specialist workers:\n{worker_list}\n\n"
            "Break the following task into independent subtasks and assign each "
            "to the most appropriate worker.\n"
            "Return ONLY a list, one subtask per line, in this exact format:\n"
            "[worker_name] subtask description\n\n"
            f"Task: {task}"
        )
        response = await self._llm.ainvoke(
            [SystemMessage(content="You are a task decomposition and routing assistant."),
             HumanMessage(content=prompt)]
        )
        lines = str(response.content).strip().split("\n")
        worker_map = {w.name: w for w in self._workers}
        fallback = self._workers[0]
        assignments: list[tuple[str, WorkerSpec]] = []

        for line in lines:
            line = line.strip()
            if not line:
                continue
            match_result = re.match(r"^\[([^\]]+)\]\s*(.*)", line)
            if match_result:
                name, subtask = match_result.group(1).strip(), match_result.group(2).strip()
                spec = worker_map.get(name, fallback)
            else:
                subtask = line.lstrip("0123456789.)- ").strip()
                spec = fallback
            if subtask:
                assignments.append((subtask, spec))

        return assignments or [(task, fallback)]

    async def _synthesize(self, original_task: str, results: list[WorkerResult]) -> str:
        """Ask the LLM to merge worker results into a final answer."""
        summaries: list[str] = []
        for r in results:
            if r.success:
                summaries.append(f"Worker {r.worker_name} ({r.task[:50]}):\n{r.result}")
            else:
                summaries.append(f"Worker {r.worker_name} failed: {r.error}")

        combined = "\n\n---\n\n".join(summaries)
        prompt = (
            f"Original task: {original_task}\n\n"
            f"Worker results:\n\n{combined}\n\n"
            "Synthesize the above results into a single, coherent final answer."
        )
        response = await self._llm.ainvoke(
            [SystemMessage(content="You are a synthesis assistant."),
             HumanMessage(content=prompt)]
        )
        return str(response.content)
