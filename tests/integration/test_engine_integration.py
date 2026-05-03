"""Integration tests — full agent loop with mock LLM and real tools."""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.tools import tool

from gantrygraph import GantryEngine
from gantrygraph.core.events import GantryEvent


@pytest.mark.asyncio
async def test_engine_completes_immediately(mock_llm_done_immediately: Any) -> None:
    """Agent that produces no tool calls should terminate in one cycle."""
    agent = GantryEngine(llm=mock_llm_done_immediately, max_steps=5)
    result = await agent.arun("Just say hello")
    assert isinstance(result, str)
    assert len(result) > 0


@pytest.mark.asyncio
async def test_engine_runs_one_tool_then_finishes(mock_llm_one_shot: Any) -> None:
    """Agent that calls one tool and then stops."""

    @tool
    async def shell_run(command: str) -> str:
        """Run a shell command."""
        return f"output of: {command}"

    agent = GantryEngine(
        llm=mock_llm_one_shot,
        tools=[shell_run],  # type: ignore[list-item]
        max_steps=5,
    )
    result = await agent.arun("Echo hello")
    assert isinstance(result, str)


@pytest.mark.asyncio
async def test_engine_emits_events(mock_llm_done_immediately: Any) -> None:
    """on_event callback is called at least once."""
    events: list[GantryEvent] = []

    agent = GantryEngine(
        llm=mock_llm_done_immediately,
        on_event=lambda e: events.append(e),
        max_steps=5,
    )
    await agent.arun("Do something")
    assert len(events) > 0
    event_types = {e.event_type for e in events}
    assert "observe" in event_types
    # When the LLM produces a final answer (no tool calls), event is "done" not "think"
    assert "done" in event_types or "think" in event_types


@pytest.mark.asyncio
async def test_engine_respects_max_steps() -> None:
    """Agent that never finishes is capped at max_steps."""
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
    from langchain_core.messages import AIMessage

    # LLM that always calls a tool and never declares done
    infinite_responses = [
        AIMessage(
            content="",
            tool_calls=[{"name": "noop", "args": {}, "id": f"c{i}", "type": "tool_call"}],
        )
        for i in range(20)
    ]

    @tool
    async def noop() -> str:
        """Does nothing."""
        return "ok"

    agent = GantryEngine(
        llm=FakeMessagesListChatModel(responses=infinite_responses),
        tools=[noop],  # type: ignore[list-item]
        max_steps=3,
    )
    result = await agent.arun("Keep going forever")
    assert isinstance(result, str)


@pytest.mark.asyncio
async def test_engine_approval_callback_can_deny(mock_llm_one_shot: Any) -> None:
    """approval_callback returning False prevents tool execution."""

    @tool
    async def shell_run(command: str) -> str:
        """Run a shell command."""
        raise RuntimeError("This should not be reached")

    agent = GantryEngine(
        llm=mock_llm_one_shot,
        tools=[shell_run],  # type: ignore[list-item]
        approval_callback=lambda name, args: False,
        max_steps=5,
    )
    # Should not raise even though tool execution was denied
    result = await agent.arun("Do something dangerous")
    assert isinstance(result, str)


def test_engine_run_sync(mock_llm_done_immediately: Any) -> None:
    """Synchronous .run() should work from a regular Python script."""
    agent = GantryEngine(llm=mock_llm_done_immediately, max_steps=3)
    result = agent.run("Sync task")
    assert isinstance(result, str)


@pytest.mark.asyncio
async def test_engine_get_graph_returns_compiled(mock_llm_done_immediately: Any) -> None:
    """get_graph() returns a compiled LangGraph object."""
    from langgraph.graph.state import CompiledStateGraph

    agent = GantryEngine(llm=mock_llm_done_immediately)
    graph = agent.get_graph()
    assert isinstance(graph, CompiledStateGraph)


@pytest.mark.asyncio
async def test_engine_unknown_tool_type_raises() -> None:
    """Passing an unsupported tool type should raise TypeError."""
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
    from langchain_core.messages import AIMessage

    agent = GantryEngine(
        llm=FakeMessagesListChatModel(responses=[AIMessage(content="ok")]),
        tools=["not_a_tool"],  # type: ignore[list-item]
    )
    with pytest.raises(TypeError, match="Unexpected tool type"):
        await agent.arun("test")
