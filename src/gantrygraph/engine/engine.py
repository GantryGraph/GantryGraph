"""GantryEngine — the main entry point for the gantrygraph framework."""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel
    from langchain_core.tools import BaseTool
    from langgraph.graph.state import CompiledStateGraph

    from gantrygraph.core.base_mcp import BaseMCPConnector
    from gantrygraph.core.base_perception import BasePerception
    from gantrygraph.core.events import GantryEvent
    from gantrygraph.memory.base import BaseMemory
    from gantrygraph.security.policies import (
        ApprovalCallback,
        BudgetPolicy,
        EventCallback,
        GuardrailPolicy,
        WorkspacePolicy,
    )

from gantrygraph._utils import _run_sync

logger = logging.getLogger(__name__)

_AnyTool = Any  # BaseAction | BaseMCPConnector | BaseTool


class AgentSuspended(Exception):
    """Raised by ``arun()`` when the agent is suspended awaiting human approval.

    Resume execution with ``GantryEngine.resume(thread_id, approved=True/False)``.
    """

    def __init__(self, thread_id: str, data: Any = None) -> None:
        super().__init__(f"Agent suspended (thread_id={thread_id})")
        self.thread_id = thread_id
        self.data = data


class GantryEngine:
    """Autonomous agent engine backed by LangGraph.

    Composes a perception source, a set of tools/MCP connectors, and an LLM
    into a self-correcting ``observe → think → act → review`` loop.

    Args:
        llm:               Any LangChain ``BaseChatModel``.
        perception:        How the agent observes its environment.
        tools:             ``BaseAction``, ``BaseMCPConnector``, or bare
                           ``BaseTool`` instances.  All are flattened into one
                           tool registry.
        approval_callback: Called before every tool execution.  Return
                           ``True`` to allow, ``False`` to deny.
        on_event:          Called after each node transition with a
                           ``GantryEvent``.  Use for logging, tracing, OTel, etc.
        max_steps:         Hard upper bound on act-node executions.
        guardrail:         Optional ``GuardrailPolicy`` for per-tool approval.
        system_prompt:     Prepended as a ``SystemMessage`` before the task.
        memory:            Optional long-term memory backend.  Past experiences
                           are recalled at the start of each run and the result
                           is stored automatically on completion.
        checkpointer:      LangGraph checkpointer for state persistence.  When
                           provided, ``enable_suspension`` defaults to ``True``.
        enable_suspension: Use LangGraph ``interrupt()`` inside ``act_node``
                           instead of ``approval_callback`` for HITL.  Requires
                           a *checkpointer* to be set (auto-creates
                           ``MemorySaver`` if none provided).
        budget:            Optional ``BudgetPolicy``.  Enforces ``max_steps``
                           (caps the engine-level limit) and ``max_wall_seconds``
                           (wall-clock timeout per run via ``asyncio.wait_for``).
                           ``max_tokens`` is stored but not currently enforced
                           by gantrygraph; configure token limits on the LLM itself.
        workspace_policy:  Optional ``WorkspacePolicy``.  When set, automatically
                           adds ``FileSystemTools`` and ``ShellTool`` locked to
                           ``workspace_path``.  Equivalent to passing those tools
                           in the ``tools`` list, but more declarative.

    Example::

        agent = GantryEngine(
            llm=ChatAnthropic(model="claude-sonnet-4-6"),
            tools=[ShellTool(workspace="/tmp")],
            on_event=lambda e: print(e),
            max_steps=20,
        )
        result = agent.run("List the 5 largest files in /tmp")
    """

    def __init__(
        self,
        *,
        llm: BaseChatModel,
        perception: BasePerception | None = None,
        tools: list[_AnyTool] | None = None,
        approval_callback: ApprovalCallback | None = None,
        on_event: EventCallback | None = None,
        max_steps: int = 50,
        guardrail: GuardrailPolicy | None = None,
        system_prompt: str | None = None,
        memory: BaseMemory | None = None,
        checkpointer: Any = None,
        enable_suspension: bool = False,
        budget: BudgetPolicy | None = None,
        workspace_policy: WorkspacePolicy | None = None,
    ) -> None:
        self._llm = llm
        self._perception = perception
        self._raw_tools: list[_AnyTool] = tools or []
        self._approval_cb = approval_callback
        self._event_cb = on_event
        self._max_steps = max_steps
        self._guardrail = guardrail
        self._system_prompt = system_prompt
        self._memory = memory
        self._compiled: CompiledStateGraph[Any] | None = None
        self._budget = budget
        self._workspace_policy = workspace_policy

        # BudgetPolicy caps max_steps
        if budget is not None:
            self._max_steps = min(self._max_steps, budget.max_steps)

        # Suspension / checkpointing
        self._use_interrupt = enable_suspension or checkpointer is not None
        if self._use_interrupt and checkpointer is None:
            from langgraph.checkpoint.memory import MemorySaver

            checkpointer = MemorySaver()
        self._checkpointer = checkpointer

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self, task: str, *, thread_id: str | None = None) -> str:
        """Synchronous entry point.  Blocks until the task completes."""
        return _run_sync(self.arun(task, thread_id=thread_id))

    async def arun(self, task: str, *, thread_id: str | None = None) -> str:
        """Primary async entry point.

        Enters the resource lifecycle, runs the graph to completion, stores
        the result in long-term memory (if configured), and returns the final
        answer as a string.

        When suspension is enabled and an approval is needed, raises
        :exc:`AgentSuspended` with the ``thread_id`` to resume from.
        """
        if self._checkpointer is not None and thread_id is None:
            thread_id = str(uuid.uuid4())

        async with self._lifecycle():
            compiled = self._build()
            initial_state = self._initial_state(task)
            config = {"configurable": {"thread_id": thread_id}} if thread_id else {}
            logger.info("Starting task: %s", task[:80])

            try:
                from langgraph.errors import GraphInterrupt

                coro = compiled.ainvoke(initial_state, config=config)  # type: ignore[call-overload]
                if self._budget is not None and self._budget.max_wall_seconds is not None:
                    try:
                        final_state: dict[str, Any] = await asyncio.wait_for(
                            coro, timeout=self._budget.max_wall_seconds
                        )
                    except TimeoutError as exc:
                        raise TimeoutError(
                            f"Agent exceeded budget wall time of {self._budget.max_wall_seconds}s"
                        ) from exc
                else:
                    final_state = await coro
            except GraphInterrupt as gi:
                logger.info("Agent suspended at thread_id=%s", thread_id)
                raise AgentSuspended(thread_id=thread_id or "", data=gi.args) from gi

            result = self._extract_answer(final_state)

            if self._memory is not None:
                await self._memory.add(
                    f"Task: {task}\nResult: {result}",
                    {"task": task, "steps": final_state.get("step_count", 0)},
                )

            return result

    async def resume(
        self,
        thread_id: str,
        *,
        approved: bool = True,
    ) -> str:
        """Resume a suspended agent run.

        Args:
            thread_id:  The ``thread_id`` from the :exc:`AgentSuspended`
                        exception raised by :meth:`arun`.
            approved:   Decision to pass back to the interrupted node.
                        ``True`` → execute the tool, ``False`` → deny it.

        Returns:
            The agent's final answer after resuming.

        Raises:
            :exc:`AgentSuspended` again if the agent suspends a second time.
        """
        if self._checkpointer is None:
            raise RuntimeError(
                "resume() requires a checkpointer.  Pass checkpointer= or "
                "enable_suspension=True when creating GantryEngine."
            )
        from langgraph.types import Command

        config = {"configurable": {"thread_id": thread_id}}
        async with self._lifecycle():
            compiled = self._build()
            try:
                from langgraph.errors import GraphInterrupt

                final_state: dict[str, Any] = await compiled.ainvoke(  # type: ignore[call-overload]
                    Command(resume=approved), config=config
                )
            except GraphInterrupt as gi:
                raise AgentSuspended(thread_id=thread_id, data=gi.args) from gi

        return self._extract_answer(final_state)

    async def astream_events(self, task: str) -> AsyncIterator[GantryEvent]:
        """Stream GantryEvents as they are emitted during execution."""
        queue: asyncio.Queue[GantryEvent | None] = asyncio.Queue()

        original_cb = self._event_cb

        async def _queuing_cb(event: GantryEvent) -> None:
            if original_cb:
                await _run_sync_safe(original_cb, event)
            await queue.put(event)

        self._event_cb = _queuing_cb
        self._compiled = None  # force rebuild with new callback

        async def _run() -> None:
            try:
                await self.arun(task)
            finally:
                await queue.put(None)  # always signal end, even if arun() raises

        run_task = asyncio.create_task(_run())

        try:
            while True:
                event = await queue.get()
                if event is None:
                    break
                yield event
        finally:
            self._event_cb = original_cb
            self._compiled = None
            await run_task

    def get_graph(self) -> CompiledStateGraph[Any]:
        """Return the compiled LangGraph — the official escape hatch for loop customisation.

        Use this when the default ``observe → think → act → review`` topology
        does not fit your use case.  The returned ``CompiledStateGraph`` is a
        standard LangGraph object; you can call ``ainvoke``, ``astream``,
        ``get_state``, etc. on it directly.

        **Pattern A — inspect or stream the existing graph:**

        .. code-block:: python

            compiled = agent.get_graph()

            # Invoke directly with a custom initial state
            result = await compiled.ainvoke({
                "task": "my task",
                "messages": [],
                "step_count": 0,
                "is_done": False,
            })

            # Stream individual node outputs
            async for chunk in compiled.astream(initial_state):
                print(chunk)

        **Pattern B — build a fully custom loop using gantrygraph's node primitives:**

        .. code-block:: python

            from functools import partial
            from gantrygraph.engine import (
                act_node, observe_node, review_node,
                should_continue, think_node,
            )
            from gantrygraph.core.state import GantryState
            from langgraph.graph import END, START, StateGraph

            async def my_pre_act_hook(state: GantryState) -> dict:
                \"\"\"Validate tool calls before execution.\"\"\"
                return {}

            graph = StateGraph(GantryState)
            graph.add_node("observe",  partial(observe_node, perception=None, on_event=None))
            graph.add_node("think",    partial(think_node,   bound_llm=my_llm, on_event=None))
            graph.add_node("pre_act",  my_pre_act_hook)       # custom hook
            graph.add_node("act",      partial(act_node, tool_map=tool_map,
                                               approval_callback=None, guardrail=None,
                                               on_event=None, use_interrupt=False))
            graph.add_node("review",   review_node)
            graph.add_edge(START,      "observe")
            graph.add_edge("observe",  "think")
            graph.add_edge("think",    "pre_act")
            graph.add_edge("pre_act",  "act")
            graph.add_edge("act",      "review")
            graph.add_conditional_edges(
                "review",
                partial(should_continue, max_steps=30),
                {"observe": "observe", END: END},
            )
            compiled = graph.compile()

        All node functions are exported from :mod:`gantrygraph.engine` and accept only
        keyword-only arguments (bound via ``functools.partial``), so they remain
        pure and testable in isolation.
        """
        return self._build()

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _build(self) -> CompiledStateGraph[Any]:
        if self._compiled is not None:
            return self._compiled

        tool_list = self._collect_tools()
        tool_map = {t.name: t for t in tool_list}
        if tool_list:
            try:
                bound_llm: BaseChatModel = self._llm.bind_tools(tool_list)  # type: ignore[assignment]
            except NotImplementedError:
                bound_llm = self._llm
        else:
            bound_llm = self._llm

        from gantrygraph.engine.graph import build_graph

        self._compiled = build_graph(
            perception=self._perception,
            bound_llm=bound_llm,
            tool_map=tool_map,
            approval_callback=self._approval_cb,
            guardrail=self._guardrail,
            on_event=self._event_cb,
            max_steps=self._max_steps,
            memory=self._memory,
            use_interrupt=self._use_interrupt,
            checkpointer=self._checkpointer,
        )
        return self._compiled

    def _collect_tools(self) -> list[BaseTool]:
        from langchain_core.tools import BaseTool

        from gantrygraph.core.base_action import BaseAction
        from gantrygraph.core.base_mcp import BaseMCPConnector

        tools: list[BaseTool] = []

        # workspace_policy auto-creates locked FileSystemTools + ShellTools
        if self._workspace_policy is not None:
            from gantrygraph.actions.filesystem import FileSystemTools
            from gantrygraph.actions.shell import ShellTools

            tools.extend(
                FileSystemTools(workspace=self._workspace_policy.workspace_path).get_tools()
            )
            tools.extend(ShellTools(workspace=self._workspace_policy.workspace_path).get_tools())

        for item in self._raw_tools:
            if isinstance(item, BaseMCPConnector):
                tools.extend(item.get_tools())
            elif isinstance(item, BaseAction):
                tools.extend(item.get_tools())
            elif isinstance(item, BaseTool):
                tools.append(item)
            else:
                raise TypeError(
                    f"Unexpected tool type {type(item).__name__}. "
                    "Expected BaseAction, BaseMCPConnector, or BaseTool."
                )
        return tools

    def _initial_state(self, task: str) -> dict[str, Any]:
        from langchain_core.messages import SystemMessage

        messages = []
        if self._system_prompt:
            messages.append(SystemMessage(content=self._system_prompt))
        messages.append(
            SystemMessage(
                content=(
                    "You are an autonomous agent. Complete the following task step by step. "
                    "When you are done, respond with a final summary and do not call any more"
                    f" tools.\n\nTask: {task}"
                )
            )
        )
        return {
            "task": task,
            "messages": messages,
            "step_count": 0,
            "is_done": False,
        }

    @staticmethod
    def _extract_answer(final_state: dict[str, Any]) -> str:
        from langchain_core.messages import AIMessage

        messages = final_state.get("messages", [])
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content:
                return str(msg.content)
        return "(no answer produced)"

    @asynccontextmanager
    async def _lifecycle(self) -> AsyncIterator[None]:
        from gantrygraph.core.base_action import BaseAction
        from gantrygraph.core.base_mcp import BaseMCPConnector

        mcp_connectors: list[BaseMCPConnector] = [
            t for t in self._raw_tools if isinstance(t, BaseMCPConnector)
        ]
        entered: list[BaseMCPConnector] = []
        try:
            for connector in mcp_connectors:
                await connector.__aenter__()
                entered.append(connector)

            self._compiled = None
            yield

        finally:
            for connector in reversed(entered):
                try:
                    await connector.__aexit__(None, None, None)
                except Exception as exc:
                    logger.warning("Error closing MCP connector: %s", exc)

            if self._perception is not None:
                try:
                    await self._perception.close()
                except Exception as exc:
                    logger.warning("Error closing perception: %s", exc)

            for item in self._raw_tools:
                if isinstance(item, BaseAction):
                    try:
                        await item.close()
                    except Exception as exc:
                        logger.warning("Error closing action: %s", exc)

            self._compiled = None


async def _run_sync_safe(cb: Any, event: Any) -> None:
    from gantrygraph._utils import ensure_awaitable

    await ensure_awaitable(cb, event)
