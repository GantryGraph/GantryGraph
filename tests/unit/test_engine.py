"""Unit tests for engine nodes — tested in isolation with mock LLMs."""
from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

from gantrygraph.core.events import GantryEvent, PerceptionResult
from gantrygraph.core.state import GantryState
from gantrygraph.engine.nodes import (
    act_node,
    observe_node,
    review_node,
    should_continue,
    think_node,
)
from gantrygraph.security.policies import GuardrailPolicy

# ── Helpers ──────────────────────────────────────────────────────────────────

def _base_state(**overrides: Any) -> GantryState:
    state: GantryState = {
        "task": "test task",
        "messages": [],
        "step_count": 0,
        "is_done": False,
    }
    state.update(overrides)  # type: ignore[typeddict-item]
    return state


class _MockPerception:
    def __init__(self, result: PerceptionResult) -> None:
        self._result = result

    async def observe(self) -> PerceptionResult:
        return self._result


# ── observe_node ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_observe_node_no_perception() -> None:
    state = _base_state()
    update = await observe_node(state, perception=None, event_cb=None)
    assert "messages" in update
    assert len(update["messages"]) == 1
    msg = update["messages"][0]
    assert isinstance(msg, HumanMessage)


@pytest.mark.asyncio
async def test_observe_node_with_screenshot() -> None:
    perception = _MockPerception(PerceptionResult(screenshot_b64="abc"))
    state = _base_state()
    update = await observe_node(state, perception=perception, event_cb=None)  # type: ignore[arg-type]
    assert "last_observation" in update
    assert update["last_observation"]["screenshot_b64"] == "abc"


@pytest.mark.asyncio
async def test_observe_node_emits_event() -> None:
    events: list[GantryEvent] = []
    state = _base_state()
    await observe_node(state, perception=None, event_cb=lambda e: events.append(e))
    assert len(events) == 1
    assert events[0].event_type == "observe"


# ── think_node ───────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_think_node_appends_ai_message(mock_llm_done_immediately: Any) -> None:
    state = _base_state(messages=[HumanMessage(content="hello")])
    update = await think_node(state, llm_with_tools=mock_llm_done_immediately, event_cb=None)
    assert "messages" in update
    assert isinstance(update["messages"][0], AIMessage)


@pytest.mark.asyncio
async def test_think_node_emits_event(mock_llm_done_immediately: Any) -> None:
    events: list[GantryEvent] = []
    state = _base_state(messages=[HumanMessage(content="hello")])
    await think_node(
        state,
        llm_with_tools=mock_llm_done_immediately,
        event_cb=lambda e: events.append(e),
    )
    # No tool calls → LLM is done, event_type is "done"
    assert events[0].event_type == "done"


# ── act_node ─────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_act_node_executes_tool() -> None:
    @tool
    async def echo(text: str) -> str:
        """Echo text back."""
        return f"ECHO: {text}"

    tool_call = {"name": "echo", "args": {"text": "hi"}, "id": "c1", "type": "tool_call"}
    ai_msg = AIMessage(content="", tool_calls=[tool_call])
    state = _base_state(messages=[ai_msg])

    update = await act_node(
        state,
        tool_map={"echo": echo},
        approval_cb=None,
        guardrail=None,
        event_cb=None,
    )
    assert "messages" in update
    tool_msgs = [m for m in update["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_msgs) == 1
    assert "ECHO: hi" in tool_msgs[0].content  # type: ignore[operator]


@pytest.mark.asyncio
async def test_act_node_unknown_tool_returns_error() -> None:
    tool_call = {"name": "nonexistent", "args": {}, "id": "c2", "type": "tool_call"}
    ai_msg = AIMessage(content="", tool_calls=[tool_call])
    state = _base_state(messages=[ai_msg])

    update = await act_node(
        state,
        tool_map={},
        approval_cb=None,
        guardrail=None,
        event_cb=None,
    )
    tool_msgs = [m for m in update["messages"] if isinstance(m, ToolMessage)]
    assert tool_msgs[0].status == "error"
    assert "nonexistent" in tool_msgs[0].content  # type: ignore[operator]


@pytest.mark.asyncio
async def test_act_node_increments_step_count() -> None:
    @tool
    async def noop() -> str:
        """Does nothing."""
        return "ok"

    tool_call = {"name": "noop", "args": {}, "id": "c3", "type": "tool_call"}
    ai_msg = AIMessage(content="", tool_calls=[tool_call])
    state = _base_state(messages=[ai_msg], step_count=3)

    update = await act_node(
        state,
        tool_map={"noop": noop},
        approval_cb=None,
        guardrail=None,
        event_cb=None,
    )
    assert update["step_count"] == 4


@pytest.mark.asyncio
async def test_act_node_approval_denied() -> None:
    @tool
    async def dangerous() -> str:
        """Dangerous tool."""
        return "boom"

    tool_call = {"name": "dangerous", "args": {}, "id": "c4", "type": "tool_call"}
    ai_msg = AIMessage(content="", tool_calls=[tool_call])
    state = _base_state(messages=[ai_msg])

    update = await act_node(
        state,
        tool_map={"dangerous": dangerous},
        approval_cb=lambda name, args: False,  # always deny
        guardrail=None,
        event_cb=None,
    )
    tool_msgs = [m for m in update["messages"] if isinstance(m, ToolMessage)]
    assert tool_msgs[0].status == "error"
    assert "denied" in tool_msgs[0].content  # type: ignore[operator]


@pytest.mark.asyncio
async def test_act_node_approval_allowed() -> None:
    @tool
    async def safe() -> str:
        """Safe tool."""
        return "safe result"

    tool_call = {"name": "safe", "args": {}, "id": "c5", "type": "tool_call"}
    ai_msg = AIMessage(content="", tool_calls=[tool_call])
    state = _base_state(messages=[ai_msg])

    update = await act_node(
        state,
        tool_map={"safe": safe},
        approval_cb=lambda name, args: True,
        guardrail=None,
        event_cb=None,
    )
    tool_msgs = [m for m in update["messages"] if isinstance(m, ToolMessage)]
    assert tool_msgs[0].status != "error"


@pytest.mark.asyncio
async def test_act_node_guardrail_blocks_without_callback() -> None:
    @tool
    async def risky() -> str:
        """Risky tool."""
        return "risky"

    tool_call = {"name": "risky", "args": {}, "id": "c6", "type": "tool_call"}
    ai_msg = AIMessage(content="", tool_calls=[tool_call])
    state = _base_state(messages=[ai_msg])

    guardrail = GuardrailPolicy(requires_approval={"risky"})

    update = await act_node(
        state,
        tool_map={"risky": risky},
        approval_cb=None,  # no callback provided
        guardrail=guardrail,
        event_cb=None,
    )
    tool_msgs = [m for m in update["messages"] if isinstance(m, ToolMessage)]
    assert tool_msgs[0].status == "error"
    assert "approval" in tool_msgs[0].content.lower()  # type: ignore[operator]


@pytest.mark.asyncio
async def test_act_node_tool_exception_is_caught() -> None:
    @tool
    async def broken() -> str:
        """Broken tool."""
        raise RuntimeError("something went wrong")

    tool_call = {"name": "broken", "args": {}, "id": "c7", "type": "tool_call"}
    ai_msg = AIMessage(content="", tool_calls=[tool_call])
    state = _base_state(messages=[ai_msg])

    update = await act_node(
        state,
        tool_map={"broken": broken},
        approval_cb=None,
        guardrail=None,
        event_cb=None,
    )
    tool_msgs = [m for m in update["messages"] if isinstance(m, ToolMessage)]
    assert tool_msgs[0].status == "error"
    assert "something went wrong" in tool_msgs[0].content  # type: ignore[operator]


@pytest.mark.asyncio
async def test_act_node_no_tool_calls_returns_empty() -> None:
    ai_msg = AIMessage(content="Done.")
    state = _base_state(messages=[ai_msg])
    update = await act_node(
        state, tool_map={}, approval_cb=None, guardrail=None, event_cb=None
    )
    assert update == {}


# ── review_node ──────────────────────────────────────────────────────────────

def test_review_marks_done_when_no_tool_calls() -> None:
    state = _base_state(messages=[AIMessage(content="All done!")])
    update = review_node(state)
    assert update["is_done"] is True


def test_review_marks_not_done_when_tool_calls_pending() -> None:
    ai_msg = AIMessage(
        content="",
        tool_calls=[{"name": "x", "args": {}, "id": "1", "type": "tool_call"}],
    )
    state = _base_state(messages=[ai_msg])
    update = review_node(state)
    assert update["is_done"] is False


def test_review_empty_messages_marks_not_done() -> None:
    # review_node is reached after think+act; empty messages is an unusual
    # state that shouldn't terminate the loop prematurely.
    state = _base_state(messages=[])
    update = review_node(state)
    assert update["is_done"] is False


# ── should_continue ──────────────────────────────────────────────────────────

def test_should_continue_loops_when_not_done() -> None:
    state = _base_state(step_count=3, is_done=False)
    result = should_continue(state, max_steps=10)
    assert result == "observe"


def test_should_continue_ends_when_done() -> None:
    state = _base_state(step_count=1, is_done=True)
    result = should_continue(state, max_steps=10)
    assert result == "__end__"


def test_should_continue_ends_at_max_steps() -> None:
    state = _base_state(step_count=10, is_done=False)
    result = should_continue(state, max_steps=10)
    assert result == "__end__"


def test_should_continue_allows_one_more_before_max() -> None:
    state = _base_state(step_count=9, is_done=False)
    result = should_continue(state, max_steps=10)
    assert result == "observe"


# ── BudgetPolicy ──────────────────────────────────────────────────────────────

def test_budget_policy_caps_max_steps() -> None:
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
    from langchain_core.messages import AIMessage

    from gantrygraph.engine.engine import GantryEngine
    from gantrygraph.security.policies import BudgetPolicy

    llm = FakeMessagesListChatModel(responses=[AIMessage(content="done")])
    agent = GantryEngine(llm=llm, max_steps=100, budget=BudgetPolicy(max_steps=10))
    assert agent._max_steps == 10


def test_budget_policy_does_not_raise_max_steps() -> None:
    """Budget max_steps lower than engine max_steps wins."""
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
    from langchain_core.messages import AIMessage

    from gantrygraph.engine.engine import GantryEngine
    from gantrygraph.security.policies import BudgetPolicy

    llm = FakeMessagesListChatModel(responses=[AIMessage(content="done")])
    agent = GantryEngine(llm=llm, max_steps=5, budget=BudgetPolicy(max_steps=100))
    assert agent._max_steps == 5  # engine wins — budget is higher


@pytest.mark.asyncio
async def test_budget_policy_wall_seconds_raises_timeout() -> None:
    """max_wall_seconds breached → TimeoutError raised from arun()."""
    import asyncio as _asyncio

    from langchain_core.messages import AIMessage

    from gantrygraph.engine.engine import GantryEngine
    from gantrygraph.security.policies import BudgetPolicy

    class _SlowLLM:
        """Fake LLM that sleeps forever, simulating a stalled model call."""
        async def ainvoke(self, messages: Any, **kwargs: Any) -> AIMessage:
            await _asyncio.sleep(60)
            return AIMessage(content="done")

        def bind_tools(self, tools: Any) -> _SlowLLM:
            return self

    agent = GantryEngine(
        llm=_SlowLLM(),  # type: ignore[arg-type]
        budget=BudgetPolicy(max_steps=50, max_wall_seconds=0.05),
    )
    with pytest.raises((TimeoutError, Exception)):
        await agent.arun("will timeout")


# ── WorkspacePolicy ───────────────────────────────────────────────────────────

def test_workspace_policy_registers_tools() -> None:
    """workspace_policy auto-adds FileSystemTools + ShellTool to the registry."""
    import tempfile

    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
    from langchain_core.messages import AIMessage

    from gantrygraph.engine.engine import GantryEngine
    from gantrygraph.security.policies import WorkspacePolicy

    llm = FakeMessagesListChatModel(responses=[AIMessage(content="done")])

    with tempfile.TemporaryDirectory() as tmpdir:
        agent = GantryEngine(
            llm=llm,
            workspace_policy=WorkspacePolicy(workspace_path=tmpdir),
        )
        tools = agent._collect_tools()

    tool_names = {t.name for t in tools}
    assert "file_read" in tool_names
    assert "file_write" in tool_names
    assert "shell_run" in tool_names


def test_workspace_policy_tools_prepended_before_raw_tools() -> None:
    """Workspace tools are prepended so they appear first in the registry."""
    import tempfile

    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
    from langchain_core.messages import AIMessage
    from langchain_core.tools import tool

    from gantrygraph.engine.engine import GantryEngine
    from gantrygraph.security.policies import WorkspacePolicy

    @tool
    def custom_tool() -> str:
        """A custom tool."""
        return "custom"

    llm = FakeMessagesListChatModel(responses=[AIMessage(content="done")])
    with tempfile.TemporaryDirectory() as tmpdir:
        agent = GantryEngine(
            llm=llm,
            tools=[custom_tool],
            workspace_policy=WorkspacePolicy(workspace_path=tmpdir),
        )
        tools = agent._collect_tools()

    names = [t.name for t in tools]
    # custom_tool appears AFTER workspace tools
    assert names[-1] == "custom_tool"
    assert "file_read" in names


# ── astream_events exception safety ──────────────────────────────────────────

@pytest.mark.asyncio
async def test_astream_events_terminates_on_engine_exception() -> None:
    """If arun() raises, astream_events must not hang — it must propagate."""
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
    from langchain_core.messages import AIMessage

    from gantrygraph.engine.engine import GantryEngine

    llm = FakeMessagesListChatModel(responses=[AIMessage(content="done")])
    agent = GantryEngine(llm=llm)

    # Monkey-patch arun to raise immediately
    async def _failing_arun(task: str, *, thread_id: str | None = None) -> str:
        raise RuntimeError("simulated engine failure")

    agent.arun = _failing_arun  # type: ignore[method-assign]

    collected: list[Any] = []
    with pytest.raises((RuntimeError, Exception)):
        async for event in agent.astream_events("task"):
            collected.append(event)
    # Stream terminates (no hang) and we reach this line


# ── lvm module importability ──────────────────────────────────────────────────

def test_lvm_importable_from_gantrygraph() -> None:
    from gantrygraph import BaseVisionProvider, ClaudeVision  # noqa: F401
    assert ClaudeVision is not None
    assert BaseVisionProvider is not None


def test_lvm_importable_from_gantrygraph_lvm() -> None:
    from gantrygraph.lvm import BaseVisionProvider, ClaudeVision  # noqa: F401
    assert ClaudeVision is not None


def test_claude_vision_is_base_vision_provider() -> None:
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
    from langchain_core.messages import AIMessage

    from gantrygraph.lvm import BaseVisionProvider, ClaudeVision

    llm = FakeMessagesListChatModel(responses=[AIMessage(content="ok")])
    # suppress the "not a claude model" warning for fake LLMs
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        provider = ClaudeVision(llm)  # type: ignore[arg-type]
    assert isinstance(provider, BaseVisionProvider)
