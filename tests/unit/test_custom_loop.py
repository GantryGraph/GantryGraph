"""Tests for the loop-customisation escape hatch.

Verifies that:
- node functions are importable from gantrygraph and gantrygraph.engine
- build_graph is importable and produces a working compiled graph
- get_graph() returns a functional CompiledStateGraph
- a developer can inject a custom node between standard nodes
- the custom graph runs end-to-end with a fake LLM
"""
from __future__ import annotations

from functools import partial
from typing import Any

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

from gantrygraph.core.state import GantryState


def _done_llm() -> FakeMessagesListChatModel:
    return FakeMessagesListChatModel(responses=[AIMessage(content="Task complete.")])


# ── Import surface ────────────────────────────────────────────────────────────

def test_node_functions_importable_from_gantrygraph() -> None:
    from gantrygraph import (  # noqa: F401
        act_node,
        memory_recall_node,
        observe_node,
        review_node,
        should_continue,
        think_node,
    )
    assert all(callable(f) for f in [
        act_node, memory_recall_node, observe_node,
        review_node, should_continue, think_node,
    ])


def test_node_functions_importable_from_gantrygraph_engine() -> None:
    from gantrygraph.engine import (  # noqa: F401
        act_node,
        build_graph,
        memory_recall_node,
        observe_node,
        review_node,
        should_continue,
        think_node,
    )
    assert callable(build_graph)


def test_build_graph_importable_from_gantrygraph() -> None:
    from gantrygraph import build_graph
    assert callable(build_graph)


# ── get_graph() returns a working CompiledStateGraph ─────────────────────────

def test_get_graph_returns_compiled_state_graph() -> None:
    from langgraph.graph.state import CompiledStateGraph

    from gantrygraph import GantryEngine

    agent = GantryEngine(llm=_done_llm())
    graph = agent.get_graph()
    assert isinstance(graph, CompiledStateGraph)


@pytest.mark.asyncio
async def test_get_graph_ainvoke_produces_result() -> None:
    """The compiled graph can be invoked directly, bypassing GantryEngine.arun."""
    from gantrygraph import GantryEngine

    agent = GantryEngine(llm=_done_llm(), max_steps=3)
    compiled = agent.get_graph()

    initial: dict[str, Any] = {
        "task": "hello",
        "messages": [],
        "step_count": 0,
        "is_done": False,
    }
    result = await compiled.ainvoke(initial)  # type: ignore[call-overload]
    assert isinstance(result, dict)
    assert "messages" in result


# ── Custom node injected between think and act ────────────────────────────────

@pytest.mark.asyncio
async def test_custom_node_injected_between_think_and_act() -> None:
    """Developer injects a validation node; the loop completes normally."""
    from langgraph.graph import END, START, StateGraph

    from gantrygraph.engine import (
        act_node,
        observe_node,
        review_node,
        should_continue,
        think_node,
    )

    hook_calls: list[int] = []

    async def pre_act_hook(state: GantryState) -> dict[str, Any]:
        """Custom node: record that it was called."""
        hook_calls.append(state["step_count"])
        return {}

    llm = _done_llm()
    bound_llm = llm  # no tools needed for this test

    graph: StateGraph = StateGraph(GantryState)  # type: ignore[type-arg]
    graph.add_node("observe",  partial(observe_node,  perception=None, on_event=None))
    graph.add_node("think",    partial(think_node,    bound_llm=bound_llm,
                                                       on_event=None))
    graph.add_node("pre_act",  pre_act_hook)
    graph.add_node("act",      partial(act_node,      tool_map={}, approval_callback=None,
                                                       guardrail=None, on_event=None,
                                                       use_interrupt=False))
    graph.add_node("review",   review_node)

    graph.add_edge(START,      "observe")
    graph.add_edge("observe",  "think")
    graph.add_edge("think",    "pre_act")   # ← custom routing
    graph.add_edge("pre_act",  "act")
    graph.add_edge("act",      "review")
    graph.add_conditional_edges(
        "review",
        partial(should_continue, max_steps=5),
        {"observe": "observe", END: END},
    )
    compiled = graph.compile()

    initial: dict[str, Any] = {
        "task": "test custom loop",
        "messages": [],
        "step_count": 0,
        "is_done": False,
    }
    result = await compiled.ainvoke(initial)  # type: ignore[call-overload]

    # The custom node was called at least once
    assert len(hook_calls) >= 1
    # The graph terminated normally
    assert result["is_done"] is True


@pytest.mark.asyncio
async def test_custom_node_can_short_circuit_loop() -> None:
    """A custom node that sets is_done=True stops the loop immediately."""
    from langgraph.graph import END, START, StateGraph

    from gantrygraph.engine import (
        act_node,
        observe_node,
        review_node,
        should_continue,
        think_node,
    )

    async def killswitch(state: GantryState) -> dict[str, Any]:
        """Immediately terminate: simulates a safety validator."""
        return {"is_done": True}

    llm = _done_llm()

    graph: StateGraph = StateGraph(GantryState)  # type: ignore[type-arg]
    graph.add_node("observe",    partial(observe_node, perception=None, on_event=None))
    graph.add_node("think",      partial(think_node,   bound_llm=llm, on_event=None))
    graph.add_node("killswitch", killswitch)
    graph.add_node("act",        partial(act_node,     tool_map={}, approval_callback=None,
                                                        guardrail=None, on_event=None,
                                                        use_interrupt=False))
    graph.add_node("review",     review_node)

    graph.add_edge(START,        "observe")
    graph.add_edge("observe",    "think")
    graph.add_edge("think",      "killswitch")
    graph.add_edge("killswitch", "act")
    graph.add_edge("act",        "review")
    graph.add_conditional_edges(
        "review",
        partial(should_continue, max_steps=50),
        {"observe": "observe", END: END},
    )
    compiled = graph.compile()

    result = await compiled.ainvoke({  # type: ignore[call-overload]
        "task": "should stop early",
        "messages": [],
        "step_count": 0,
        "is_done": False,
    })
    assert result["is_done"] is True
    # Only 1 iteration — killswitch fired on step 0
    assert result["step_count"] <= 1


# ── on_event callback flows through custom graph ──────────────────────────────

@pytest.mark.asyncio
async def test_event_callback_works_in_custom_graph() -> None:
    """on_event callbacks passed to node partials still fire in a custom graph."""
    from langgraph.graph import END, START, StateGraph

    from gantrygraph.core.events import GantryEvent
    from gantrygraph.engine import act_node, observe_node, review_node, should_continue, think_node

    received: list[GantryEvent] = []

    def capture(event: GantryEvent) -> None:
        received.append(event)

    llm = _done_llm()

    graph: StateGraph = StateGraph(GantryState)  # type: ignore[type-arg]
    graph.add_node("observe", partial(observe_node, perception=None, on_event=capture))
    graph.add_node("think",   partial(think_node,   bound_llm=llm, on_event=capture))
    graph.add_node("act",     partial(act_node,     tool_map={}, approval_callback=None,
                                                     guardrail=None, on_event=capture,
                                                     use_interrupt=False))
    graph.add_node("review",  review_node)
    graph.add_edge(START,     "observe")
    graph.add_edge("observe", "think")
    graph.add_edge("think",   "act")
    graph.add_edge("act",     "review")
    graph.add_conditional_edges(
        "review",
        partial(should_continue, max_steps=5),
        {"observe": "observe", END: END},
    )
    compiled = graph.compile()

    await compiled.ainvoke({  # type: ignore[call-overload]
        "task": "event test",
        "messages": [],
        "step_count": 0,
        "is_done": False,
    })

    event_types = {e.event_type for e in received}
    assert "observe" in event_types
    assert "done" in event_types or "think" in event_types
