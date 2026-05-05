"""LangGraph node functions for the gantrygraph agent loop.

Each node is a standalone async function that receives ``GantryState`` and
returns a *partial* state update dict.  Nodes are bound to engine-level
configuration (perception, llm, tools, callbacks) via ``functools.partial``
in ``graph.py`` — this keeps the functions pure and trivially testable.

Loop structure::

    memory_recall → observe → think → act → review → (should_continue)
                                                           ↓           ↓
                                                        observe       END
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Literal

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel
    from langchain_core.tools import BaseTool

    from gantrygraph.core.base_perception import BasePerception
    from gantrygraph.core.state import GantryState
    from gantrygraph.memory.base import BaseMemory
    from gantrygraph.security.policies import ApprovalCallback, EventCallback, GuardrailPolicy

from gantrygraph._utils import ensure_awaitable

logger = logging.getLogger(__name__)


# ── memory recall ─────────────────────────────────────────────────────────────


async def memory_recall_node(
    state: GantryState,
    *,
    memory: BaseMemory | None,
) -> dict[str, Any]:
    """Inject relevant past experiences from long-term memory (first step only)."""
    if memory is None or state["step_count"] > 0:
        return {}
    results = await memory.search(state["task"], k=3)
    if not results:
        return {}
    lines = "\n".join(f"- {r.text[:300]}" for r in results)
    return {"messages": [HumanMessage(content=f"[Relevant past experiences]\n{lines}")]}


# ── observe ──────────────────────────────────────────────────────────────────


async def observe_node(
    state: GantryState,
    *,
    perception: BasePerception | None,
    on_event: EventCallback | None,
) -> dict[str, Any]:
    """Capture the current environment and append it as a HumanMessage."""
    from gantrygraph.core.events import GantryEvent, PerceptionResult

    if perception is not None:
        result: PerceptionResult = await perception.observe()
    else:
        result = PerceptionResult()

    content = result.to_message_content()
    observation = HumanMessage(content=content)  # type: ignore[arg-type]  # multimodal content list

    if on_event:
        await ensure_awaitable(
            on_event,
            GantryEvent(
                "observe",
                state["step_count"],
                {
                    "width": result.width,
                    "height": result.height,
                    "screenshot_b64": result.screenshot_b64,
                },
            ),
        )

    return {
        "messages": [observation],
        "last_observation": result.model_dump(),
    }


# ── think ────────────────────────────────────────────────────────────────────


async def think_node(
    state: GantryState,
    *,
    bound_llm: BaseChatModel,
    on_event: EventCallback | None,
) -> dict[str, Any]:
    """Invoke the LLM with the full message history and get the next action."""
    from gantrygraph.core.events import GantryEvent

    response: AIMessage = await bound_llm.ainvoke(state["messages"])

    tool_names = [tc["name"] for tc in response.tool_calls] if response.tool_calls else []
    logger.debug("think step=%d tools=%s", state["step_count"], tool_names)

    emit_type: Literal["think", "done"] = "done" if not tool_names else "think"
    if on_event:
        await ensure_awaitable(
            on_event,
            GantryEvent(emit_type, state["step_count"], {"tool_calls": tool_names}),
        )

    return {"messages": [response]}


# ── act ──────────────────────────────────────────────────────────────────────


async def act_node(
    state: GantryState,
    *,
    tool_map: dict[str, BaseTool],
    approval_callback: ApprovalCallback | None,
    guardrail: GuardrailPolicy | None,
    on_event: EventCallback | None,
    use_interrupt: bool = False,
) -> dict[str, Any]:
    """Execute tool calls from the last AIMessage.

    For each tool call:
    1. Check the guardrail approval list — pause and ask if required.
    2. Locate the tool in the registry — return an error message if not found.
    3. Execute the tool — catch all exceptions and return them as error messages
       so the LLM can self-correct on the next think step.

    When *use_interrupt* is ``True`` and a checkpointer is configured on the
    graph, tool calls that need approval use LangGraph's ``interrupt()`` to
    suspend execution and persist state.  Resume via
    ``GantryEngine.resume(thread_id, approved=True)``.
    """
    from gantrygraph.core.events import GantryEvent

    last_msg = state["messages"][-1] if state["messages"] else None
    if not isinstance(last_msg, AIMessage) or not last_msg.tool_calls:
        return {}

    tool_messages: list[ToolMessage] = []
    executed_names: list[str] = []

    for tool_call in last_msg.tool_calls:
        name: str = tool_call["name"]
        args: dict[str, Any] = tool_call["args"]
        call_id: str = tool_call["id"] or ""

        # ── Approval gate ────────────────────────────────────────────────────
        needs_approval = approval_callback is not None or (
            guardrail is not None and name in guardrail.requires_approval
        )
        if needs_approval:
            if use_interrupt:
                # Deep suspension: persist state to checkpointer and pause.
                # Execution resumes when engine.resume(thread_id, approved=...) is called.
                from langgraph.types import interrupt

                approved: bool = interrupt(
                    {"tool": name, "args": args, "message": f"Approve '{name}'?"}
                )
            elif approval_callback is not None:
                approved = await ensure_awaitable(approval_callback, name, args)
            else:
                # Guardrail hit but no callback — deny by default
                tool_messages.append(
                    ToolMessage(
                        content=(
                            f"Action '{name}' requires approval but no"
                            " approval_callback was provided."
                        ),
                        tool_call_id=call_id,
                        status="error",
                    )
                )
                continue

            if not approved:
                logger.info("Tool '%s' denied", name)
                tool_messages.append(
                    ToolMessage(
                        content=f"Action '{name}' was denied.",
                        tool_call_id=call_id,
                        status="error",
                    )
                )
                continue

        # ── Tool lookup ──────────────────────────────────────────────────────
        tool = tool_map.get(name)
        if tool is None:
            tool_messages.append(
                ToolMessage(
                    content=(
                        f"Tool '{name}' is not available. Available tools: {list(tool_map.keys())}"
                    ),
                    tool_call_id=call_id,
                    status="error",
                )
            )
            continue

        # ── Execute ──────────────────────────────────────────────────────────
        try:
            output = await tool.ainvoke(args)
            tool_messages.append(ToolMessage(content=str(output), tool_call_id=call_id))
            executed_names.append(name)
            logger.debug("Tool '%s' succeeded", name)
        except Exception as exc:
            logger.warning("Tool '%s' raised: %s", name, exc)
            tool_messages.append(
                ToolMessage(
                    content=f"Tool '{name}' failed with error: {exc}. Try a different approach.",
                    tool_call_id=call_id,
                    status="error",
                )
            )

    if on_event:
        await ensure_awaitable(
            on_event,
            GantryEvent("act", state["step_count"], {"tools_executed": executed_names}),
        )

    has_error = any(r.status == "error" for r in tool_messages)
    new_consecutive = state.get("consecutive_errors", 0) + 1 if has_error else 0

    return {
        "messages": tool_messages,
        "step_count": state["step_count"] + 1,
        "last_error": str(tool_messages[-1].content) if has_error else None,
        "consecutive_errors": new_consecutive,
    }


# ── review ───────────────────────────────────────────────────────────────────


def review_node(state: GantryState) -> dict[str, Any]:
    """Decide whether the task is complete.

    Termination condition: the last message is an AIMessage with no tool calls,
    meaning the LLM chose to stop calling tools and produce a final answer.
    This is a pure function — no I/O.
    """
    last_msg = state["messages"][-1] if state["messages"] else None
    is_done = isinstance(last_msg, AIMessage) and not last_msg.tool_calls
    return {"is_done": is_done}


# ── routing ──────────────────────────────────────────────────────────────────


def should_continue(
    state: GantryState,
    *,
    max_steps: int,
    max_consecutive_errors: int = 5,
) -> Literal["observe", "__end__"]:
    """Conditional edge: loop back to observe, or terminate."""
    from langgraph.graph import END

    if state["is_done"]:
        logger.debug("Terminating: is_done=True at step %d", state["step_count"])
        return END  # type: ignore[return-value]
    if state["step_count"] >= max_steps:
        logger.warning("Terminating: max_steps=%d reached", max_steps)
        return END  # type: ignore[return-value]
    if state.get("consecutive_errors", 0) >= max_consecutive_errors:
        logger.warning(
            "Terminating: %d consecutive tool errors (bot detection / stuck state?)",
            state["consecutive_errors"],
        )
        return END  # type: ignore[return-value]
    return "observe"
