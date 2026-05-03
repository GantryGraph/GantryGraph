from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from pydantic import BaseModel, Field

ApprovalCallback = Callable[[str, dict[str, Any]], "Awaitable[bool] | bool"]
EventCallback = Callable[["GantryEvent"], "Awaitable[None] | None"]  # noqa: F821


class GuardrailPolicy(BaseModel):
    """Configures which tools require explicit human approval before execution.

    Pass this to ``GantryEngine`` together with an ``approval_callback`` to
    gate dangerous operations.

    Example::

        from gantrygraph.security import GuardrailPolicy

        policy = GuardrailPolicy(
            requires_approval={"shell_run", "file_delete"},
        )
        agent = GantryEngine(..., guardrail=policy,
                           approval_callback=my_slack_approval_fn)
    """

    requires_approval: set[str] = Field(
        default_factory=set,
        description="Tool names that must be approved before execution.",
    )

    model_config = {"arbitrary_types_allowed": True}


class WorkspacePolicy(BaseModel):
    """Restrict filesystem and shell operations to a specific directory.

    Pass to ``GantryEngine`` via ``workspace_policy=`` to automatically add
    ``FileSystemTools`` and ``ShellTool`` locked to ``workspace_path``.
    This is more declarative than listing the tools manually.

    Example::

        from gantrygraph import GantryEngine
        from gantrygraph.security import WorkspacePolicy

        agent = GantryEngine(
            llm=my_llm,
            workspace_policy=WorkspacePolicy(workspace_path="/home/user/project"),
        )
        # Equivalent to:
        # GantryEngine(llm=..., tools=[FileSystemTools("/home/user/project"),
        #                            ShellTool("/home/user/project")])

    Note:
        ``allow_read_outside`` and ``allow_write_outside`` are reserved for
        future fine-grained enforcement.  Currently the workspace boundary is
        enforced at the tool level (path traversal blocked in ``FileSystemTools``).
    """

    workspace_path: str
    allow_read_outside: bool = False
    allow_write_outside: bool = False


class BudgetPolicy(BaseModel):
    """Hard limits to prevent runaway costs and infinite loops.

    Pass to ``GantryEngine`` via ``budget=`` to enforce spending limits.

    Enforcement:
        - ``max_steps``: caps ``GantryEngine.max_steps`` (whichever is lower wins).
        - ``max_wall_seconds``: wraps the full ``arun()`` call in
          ``asyncio.wait_for``; raises ``TimeoutError`` on breach.
        - ``max_tokens``: stored but **not currently enforced** by gantrygraph.
          Configure token limits on the LLM itself (e.g. ``max_tokens=`` in
          ``ChatAnthropic``).

    Example::

        from gantrygraph import GantryEngine
        from gantrygraph.security import BudgetPolicy

        agent = GantryEngine(
            llm=my_llm,
            budget=BudgetPolicy(max_steps=30, max_wall_seconds=120.0),
        )
    """

    max_steps: int = Field(default=50, gt=0)
    max_tokens: int | None = Field(
        default=None, description="Approximate token cap (not enforced by gantrygraph)."
    )
    max_wall_seconds: float | None = Field(
        default=None, description="Wall-clock timeout per run in seconds."
    )


# Avoid circular import: GantryEvent is defined in gantrygraph.core.events
# The type annotation above is a forward reference (string literal) only.
from gantrygraph.core.events import GantryEvent  # noqa: E402 (must come after class definitions)

__all__ = [
    "GuardrailPolicy",
    "WorkspacePolicy",
    "BudgetPolicy",
    "ApprovalCallback",
    "EventCallback",
]
