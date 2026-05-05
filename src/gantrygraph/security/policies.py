from __future__ import annotations

import re
from collections.abc import Awaitable, Callable
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

# ── Shell command denylist ────────────────────────────────────────────────────

_DEFAULT_DENY_PATTERNS: list[str] = [
    # Recursive deletion at filesystem root or home
    r"rm\s+-[a-zA-Z]*r[a-zA-Z]*\s+(/|~|\$HOME\b|\$\{HOME\})",
    # Disk wipe via dd
    r"\bdd\s+if=/dev/(zero|urandom)\s+of=/dev/",
    # Read SSH private keys
    r"(cat|less|more|head|tail|tee)\s+~?/\.ssh/id_",
    # Read system credential files
    r"(cat|less|more|head|tail)\s+/etc/(shadow|sudoers)\b",
    # Fork bomb
    r":\s*\(\s*\)\s*\{[^}]*\|",
    # Remote code execution via pipe-to-shell
    r"(curl|wget)\s+[^\|]+\|\s*(ba)?sh\b",
    r"(curl|wget)\s+[^\|]+\|\s*python3?\b",
]

_STRICT_DENY_PATTERNS: list[str] = [
    # Filesystem formatting / partition editing
    r"\bmkfs\b",
    r"\bfdisk\b",
    r"\bparted\b",
    # Write to block devices
    r">\s*/dev/sd[a-z]\b",
    r">\s*/dev/nvme\d",
    # Broad recursive permission change (chmod -R 777)
    r"chmod\s+(-R\s+)?0?777\b",
    # Environment credential dumps (alone on line)
    r"^\s*env\s*$",
    r"^\s*printenv\s*$",
]

ApprovalCallback = Callable[[str, dict[str, Any]], "Awaitable[bool] | bool"]
EventCallback = Callable[["GantryEvent"], "Awaitable[None] | None"]  # noqa: F821


class BudgetExceededError(Exception):
    """Raised when an agent run exceeds its configured token budget.

    Only raised when ``BudgetPolicy.on_limit == "stop"`` (the default).
    """


class ShellDenylist(BaseModel):
    """Regex-based filter that intercepts dangerous shell commands before execution.

    Applied by ``ShellTools`` **before** the subprocess is created. Blocking
    happens at the Python level — the OS never sees the command.

    Three built-in profiles:

    ``ShellDenylist.default()``
        Blocks catastrophic-but-rare commands (filesystem wipes, SSH key reads,
        fork bombs, curl-pipe-bash). Safe for all production use cases.

    ``ShellDenylist.strict()``
        Adds filesystem formatting, block-device writes, recursive chmod 777,
        and credential-env dumps on top of the default set.

    ``ShellDenylist.permissive()``
        No patterns — the developer takes full responsibility.  Use only in
        air-gapped or fully trusted environments.

    Custom patterns::

        from gantrygraph.security import ShellDenylist

        denylist = ShellDenylist(
            patterns=[
                *ShellDenylist.default().patterns,
                r"my-internal-forbidden-cmd",
            ],
            on_match="warn",   # log instead of blocking
        )
        tools = ShellTools(workspace="/app", denylist=denylist)
    """

    patterns: list[str] = Field(default_factory=list)
    on_match: Literal["block", "warn"] = "block"

    @classmethod
    def default(cls) -> ShellDenylist:
        """Secure baseline — blocks catastrophic commands only."""
        return cls(patterns=_DEFAULT_DENY_PATTERNS)

    @classmethod
    def strict(cls) -> ShellDenylist:
        """Extended set — also blocks filesystem formatting and credential dumps."""
        return cls(patterns=_DEFAULT_DENY_PATTERNS + _STRICT_DENY_PATTERNS)

    @classmethod
    def permissive(cls) -> ShellDenylist:
        """No patterns — developer takes full responsibility."""
        return cls(patterns=[])

    def check(self, command: str) -> str | None:
        """Return the first matched pattern string if the command is denied, else ``None``."""
        for pattern in self.patterns:
            if re.search(pattern, command, re.IGNORECASE | re.MULTILINE):
                return pattern
        return None


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
    "ShellDenylist",
    "ApprovalCallback",
    "EventCallback",
]
