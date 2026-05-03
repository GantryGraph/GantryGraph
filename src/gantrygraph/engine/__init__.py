"""LangGraph-based agent engine.

Public surface
--------------
``GantryEngine``      — the high-level facade; covers 95% of use cases.
``build_graph``     — assemble a compiled StateGraph from raw primitives;
                      use this when you want a *different* loop topology.
Node functions      — pure async functions, each a self-contained graph node.
                      Import them, wrap with ``functools.partial``, and wire
                      them however you like into a custom StateGraph.

Custom-loop quick-start::

    from functools import partial
    from gantrygraph.engine import (
        act_node, observe_node, review_node, should_continue, think_node,
    )
    from gantrygraph.core.state import GantryState
    from langgraph.graph import END, START, StateGraph

    async def my_validate_node(state: GantryState) -> dict:
        last = state["messages"][-1] if state["messages"] else None
        if last and "rm -rf" in str(last):
            return {"is_done": True}
        return {}

    graph: StateGraph = StateGraph(GantryState)  # type: ignore[type-arg]
    graph.add_node("observe",  partial(observe_node,  perception=None, event_cb=None))
    graph.add_node("think",    partial(think_node,    llm_with_tools=my_llm, event_cb=None))
    graph.add_node("validate", my_validate_node)
    graph.add_node("act",      partial(act_node, tool_map=tool_map, approval_cb=None,
                                       guardrail=None, event_cb=None, use_interrupt=False))
    graph.add_node("review",   review_node)
    graph.add_edge(START,      "observe")
    graph.add_edge("observe",  "think")
    graph.add_edge("think",    "validate")
    graph.add_edge("validate", "act")
    graph.add_edge("act",      "review")
    graph.add_conditional_edges(
        "review",
        partial(should_continue, max_steps=30),
        {"observe": "observe", END: END},
    )
    compiled = graph.compile()
"""

from gantrygraph.engine.engine import GantryEngine
from gantrygraph.engine.graph import build_graph
from gantrygraph.engine.nodes import (
    act_node,
    memory_recall_node,
    observe_node,
    review_node,
    should_continue,
    think_node,
)

__all__ = [
    # Facade
    "GantryEngine",
    # Graph builder
    "build_graph",
    # Node functions — use with functools.partial to build custom loops
    "memory_recall_node",
    "observe_node",
    "think_node",
    "act_node",
    "review_node",
    "should_continue",
]
