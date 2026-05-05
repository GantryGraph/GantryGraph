from __future__ import annotations

from typing import Annotated, Any, NotRequired

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class GantryState(TypedDict):
    """LangGraph state dict for the gantrygraph agent loop.

    All graph nodes receive the full state and return a *partial* update dict.
    The ``messages`` field uses the ``add_messages`` reducer, so a node can
    append a new message by returning ``{"messages": [new_msg]}`` without
    reading the current list first.

    Fields
    ------
    task:             The original task string passed to ``GantryEngine.run()``.
    messages:         Full conversation history, auto-appended via reducer.
    step_count:       Number of act-node executions so far; used by budget guard.
    is_done:          Set to True by the review node to terminate the loop.
    last_error:       Most recent tool error message (for self-correction context).
    last_observation: Raw ``PerceptionResult.model_dump()`` from the last observe
                      node; stored so nodes can access it without re-capturing.
    """

    task: str
    messages: Annotated[list[AnyMessage], add_messages]
    step_count: int
    is_done: bool
    last_error: NotRequired[str | None]
    last_observation: NotRequired[Any]
    consecutive_errors: NotRequired[int]
