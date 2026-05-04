from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

ApprovalCallback = Callable[[str, dict[str, Any]], "Awaitable[bool] | bool"]
EventCallback = Callable[["GantryEvent"], "Awaitable[None] | None"]  # noqa: F821


class BudgetExceededError(Exception):
    """Raised when an agent run exceeds its configured token budget.

    Only raised when ``BudgetPolicy.on_limit == "stop"`` (the default).
    """


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
    """Restrict filesystem and shell operations to one or more directories.

    Use the factory classmethods for the clearest intent:

    Example::

        from gantrygraph.security import WorkspacePolicy

        # Single directory (most common)
        policy = WorkspacePolicy.restricted("/home/user/project")

        # Multiple directories — separate input and output roots
        policy = WorkspacePolicy.multi_path(["/tmp/input", "/tmp/output"])

        # No restriction — development / trusted environments only
        policy = WorkspacePolicy.full_access()

        # Backward-compatible direct construction
        policy = WorkspacePolicy(workspace_path="/home/user/project")

    Note:
        When ``workspace_policy`` is passed to ``GantryEngine`` it automatically
        adds ``FileSystemTools`` and ``ShellTools`` locked to the allowed paths.
        This is more declarative than listing the tools manually.
    """

    workspace_path: str | None = Field(
        default=None,
        description="Deprecated: use allowed_paths or the factory classmethods.",
    )
    allowed_paths: list[str] = Field(
        default_factory=list,
        description="Directories the agent may read from and write to.",
    )
    unrestricted: bool = Field(
        default=False,
        description="Skip all path validation. Use only in trusted environments.",
    )
    allow_read_outside: bool = False
    allow_write_outside: bool = False

    @model_validator(mode="after")
    def _normalize_workspace_path(self) -> WorkspacePolicy:
        """Promote legacy workspace_path into allowed_paths."""
        if self.workspace_path and not self.allowed_paths:
            self.allowed_paths = [self.workspace_path]
        return self

    @classmethod
    def restricted(cls, path: str) -> WorkspacePolicy:
        """Lock to a single directory."""
        return cls(allowed_paths=[path])

    @classmethod
    def multi_path(cls, paths: list[str]) -> WorkspacePolicy:
        """Allow read/write access to multiple directories."""
        return cls(allowed_paths=paths)

    @classmethod
    def full_access(cls) -> WorkspacePolicy:
        """No path restrictions — for development and trusted environments only."""
        return cls(unrestricted=True)


class BudgetPolicy(BaseModel):
    """Hard limits to prevent runaway costs and infinite loops.

    Pass to ``GantryEngine`` via ``budget=`` to enforce spending limits.

    Enforcement:
        - ``max_steps``: caps ``GantryEngine.max_steps`` (whichever is lower wins).
        - ``max_wall_seconds``: wraps the full ``arun()`` call in
          ``asyncio.wait_for``; raises ``TimeoutError`` on breach.
        - ``max_tokens``: counts tokens via ``AIMessage.usage_metadata`` after
          each LLM call.  ``on_limit="stop"`` (default) raises
          ``BudgetExceededError``; ``on_limit="warn"`` logs a warning and
          continues.

    Example::

        from gantrygraph import GantryEngine
        from gantrygraph.security import BudgetPolicy

        # Hard stop at 10 000 tokens
        agent = GantryEngine(
            llm=my_llm,
            budget=BudgetPolicy(max_steps=30, max_tokens=10_000),
        )

        # Warn-only — log but never abort
        agent = GantryEngine(
            llm=my_llm,
            budget=BudgetPolicy(max_tokens=50_000, on_limit="warn"),
        )
    """

    max_steps: int = Field(default=50, gt=0)
    max_tokens: int | None = Field(
        default=None,
        description="Cumulative token cap across the full run.",
    )
    max_wall_seconds: float | None = Field(
        default=None,
        description="Wall-clock timeout per run in seconds.",
    )
    on_limit: Literal["stop", "warn"] = Field(
        default="stop",
        description=(
            "'stop' raises BudgetExceededError when max_tokens is hit; "
            "'warn' logs a warning and continues."
        ),
    )


# Avoid circular import: GantryEvent is defined in gantrygraph.core.events
# The type annotation above is a forward reference (string literal) only.
from gantrygraph.core.events import GantryEvent  # noqa: E402 (must come after class definitions)

__all__ = [
    "GuardrailPolicy",
    "WorkspacePolicy",
    "BudgetPolicy",
    "BudgetExceededError",
    "ApprovalCallback",
    "EventCallback",
]
