"""Assembles the LangGraph StateGraph from node functions and configuration."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Any

from langgraph.graph import END, START, StateGraph

if TYPE_CHECKING:
    from typing import Any

    from langchain_core.language_models import BaseChatModel
    from langchain_core.tools import BaseTool
    from langgraph.graph.state import CompiledStateGraph

    from gantrygraph.core.base_perception import BasePerception
    from gantrygraph.memory.base import BaseMemory
    from gantrygraph.security.policies import ApprovalCallback, EventCallback, GuardrailPolicy

from gantrygraph.engine.nodes import (
    act_node,
    memory_recall_node,
    observe_node,
    review_node,
    should_continue,
    think_node,
)


def build_graph(
    *,
    perception: BasePerception | None,
    bound_llm: BaseChatModel,
    tool_map: dict[str, BaseTool],
    approval_callback: ApprovalCallback | None,
    guardrail: GuardrailPolicy | None,
    on_event: EventCallback | None,
    max_steps: int,
    memory: BaseMemory | None = None,
    use_interrupt: bool = False,
    checkpointer: Any = None,
) -> CompiledStateGraph[Any]:
    """Build and compile the gantrygraph agent StateGraph.

    Nodes are bound to configuration via ``functools.partial`` so the node
    functions themselves remain pure and testable without constructing a full
    engine.

    Graph structure::

        START → memory_recall → observe → think → act → review → should_continue
                                                                       ↙         ↘
                                                                   observe        END
    """
    from gantrygraph.core.state import GantryState

    graph: StateGraph = StateGraph(GantryState)  # type: ignore[type-arg]

    graph.add_node(
        "memory_recall",
        partial(memory_recall_node, memory=memory),
    )
    graph.add_node(
        "observe",
        partial(observe_node, perception=perception, on_event=on_event),
    )
    graph.add_node(
        "think",
        partial(think_node, bound_llm=bound_llm, on_event=on_event),
    )
    graph.add_node(
        "act",
        partial(
            act_node,
            tool_map=tool_map,
            approval_callback=approval_callback,
            guardrail=guardrail,
            on_event=on_event,
            use_interrupt=use_interrupt,
        ),
    )
    graph.add_node("review", review_node)

    graph.add_edge(START, "memory_recall")
    graph.add_edge("memory_recall", "observe")
    graph.add_edge("observe", "think")
    graph.add_edge("think", "act")
    graph.add_edge("act", "review")
    graph.add_conditional_edges(
        "review",
        partial(should_continue, max_steps=max_steps),
        {"observe": "observe", END: END},
    )

    return graph.compile(checkpointer=checkpointer)
