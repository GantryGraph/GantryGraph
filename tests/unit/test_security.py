"""Unit tests for security policies and their enforcement in act_node."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool

from gantrygraph.engine.nodes import act_node
from gantrygraph.security.policies import BudgetPolicy, GuardrailPolicy, WorkspacePolicy

# ── GuardrailPolicy ───────────────────────────────────────────────────────────


def test_guardrail_policy_default_empty() -> None:
    p = GuardrailPolicy()
    assert p.requires_approval == set()


def test_guardrail_policy_requires_approval_set() -> None:
    p = GuardrailPolicy(requires_approval={"rm", "format_disk"})
    assert "rm" in p.requires_approval
    assert "format_disk" in p.requires_approval


def test_guardrail_policy_unknown_tool_not_blocked() -> None:
    p = GuardrailPolicy(requires_approval={"dangerous"})
    assert "safe_tool" not in p.requires_approval


# ── WorkspacePolicy ───────────────────────────────────────────────────────────


def test_workspace_policy_defaults() -> None:
    p = WorkspacePolicy(workspace_path="/tmp/project")
    assert p.workspace_path == "/tmp/project"
    assert p.allow_read_outside is False
    assert p.allow_write_outside is False


def test_workspace_policy_legacy_promotes_to_allowed_paths() -> None:
    p = WorkspacePolicy(workspace_path="/tmp/project")
    assert p.allowed_paths == ["/tmp/project"]


def test_workspace_policy_custom() -> None:
    p = WorkspacePolicy(
        workspace_path="/home/user",
        allow_read_outside=True,
        allow_write_outside=False,
    )
    assert p.allow_read_outside is True
    assert p.allow_write_outside is False


def test_workspace_policy_restricted_factory() -> None:
    p = WorkspacePolicy.restricted("/tmp/work")
    assert p.allowed_paths == ["/tmp/work"]
    assert not p.unrestricted


def test_workspace_policy_multi_path_factory() -> None:
    p = WorkspacePolicy.multi_path(["/tmp/input", "/tmp/output"])
    assert p.allowed_paths == ["/tmp/input", "/tmp/output"]
    assert not p.unrestricted


def test_workspace_policy_full_access_factory() -> None:
    p = WorkspacePolicy.full_access()
    assert p.unrestricted is True
    assert p.allowed_paths == []


# ── BudgetPolicy ──────────────────────────────────────────────────────────────


def test_budget_policy_defaults() -> None:
    p = BudgetPolicy()
    assert p.max_steps == 50
    assert p.max_tokens is None
    assert p.max_wall_seconds is None
    assert p.on_limit == "stop"


def test_budget_policy_custom() -> None:
    p = BudgetPolicy(max_steps=10, max_tokens=5000, max_wall_seconds=30.0)
    assert p.max_steps == 10
    assert p.max_tokens == 5000
    assert p.max_wall_seconds == 30.0


def test_budget_policy_on_limit_warn() -> None:
    p = BudgetPolicy(max_tokens=1000, on_limit="warn")
    assert p.on_limit == "warn"


def test_budget_policy_max_steps_must_be_positive() -> None:
    with pytest.raises(ValueError):
        BudgetPolicy(max_steps=0)


# ── BudgetExceededError ───────────────────────────────────────────────────────


def test_budget_exceeded_error_is_exception() -> None:
    from gantrygraph.security.policies import BudgetExceededError

    err = BudgetExceededError("Token budget of 1,000 exceeded (1,200 used).")
    assert isinstance(err, Exception)
    assert "1,000" in str(err)


# ── safe_path_multi ───────────────────────────────────────────────────────────


def test_safe_path_multi_single_allowed(tmp_path: Path) -> None:
    from gantrygraph._utils import safe_path_multi

    result = safe_path_multi([tmp_path], "subdir/file.txt")
    assert str(result).startswith(str(tmp_path.resolve()))


def test_safe_path_multi_blocks_traversal(tmp_path: Path) -> None:
    from gantrygraph._utils import safe_path_multi

    with pytest.raises(PermissionError, match="escapes the workspace"):
        safe_path_multi([tmp_path], "../../etc/passwd")


def test_safe_path_multi_allows_any_path_when_unrestricted() -> None:
    from gantrygraph._utils import safe_path_multi

    result = safe_path_multi(None, "/tmp/anywhere.txt")
    assert result.name == "anywhere.txt"


def test_safe_path_multi_second_allowed_path_accepted(tmp_path: Path) -> None:
    from gantrygraph._utils import safe_path_multi

    other = tmp_path / "other"
    other.mkdir()
    # Absolute path inside the second allowed directory should be accepted
    target = other / "file.txt"
    result = safe_path_multi([tmp_path / "first", other], str(target))
    assert result == target.resolve()


# ── GuardrailPolicy enforcement in act_node ───────────────────────────────────


def _state(**overrides: Any) -> Any:
    base = {
        "task": "t",
        "messages": [],
        "step_count": 0,
        "is_done": False,
    }
    base.update(overrides)
    return base


@pytest.mark.asyncio
async def test_guardrail_blocks_listed_tool_without_callback() -> None:
    """Tool in requires_approval + no callback → denied."""

    @tool
    async def dangerous_delete() -> str:
        """Dangerous delete."""
        return "deleted"

    guardrail = GuardrailPolicy(requires_approval={"dangerous_delete"})
    tool_call = {
        "name": "dangerous_delete",
        "args": {},
        "id": "c1",
        "type": "tool_call",
    }
    ai_msg = AIMessage(content="", tool_calls=[tool_call])
    state = _state(messages=[ai_msg])

    update = await act_node(
        state,
        tool_map={"dangerous_delete": dangerous_delete},
        approval_callback=None,
        guardrail=guardrail,
        on_event=None,
    )
    tool_msgs = [m for m in update["messages"] if isinstance(m, ToolMessage)]
    assert tool_msgs[0].status == "error"
    assert "approval" in tool_msgs[0].content.lower()  # type: ignore[operator]


@pytest.mark.asyncio
async def test_guardrail_allows_unlisted_tool() -> None:
    """Tool NOT in requires_approval runs without needing approval."""

    @tool
    async def safe_list() -> str:
        """Safe list."""
        return "files..."

    guardrail = GuardrailPolicy(requires_approval={"dangerous_delete"})
    tool_call = {"name": "safe_list", "args": {}, "id": "c2", "type": "tool_call"}
    ai_msg = AIMessage(content="", tool_calls=[tool_call])
    state = _state(messages=[ai_msg])

    update = await act_node(
        state,
        tool_map={"safe_list": safe_list},
        approval_callback=None,
        guardrail=guardrail,
        on_event=None,
    )
    tool_msgs = [m for m in update["messages"] if isinstance(m, ToolMessage)]
    assert tool_msgs[0].status != "error"
    assert "files" in tool_msgs[0].content  # type: ignore[operator]


@pytest.mark.asyncio
async def test_approval_callback_sync_deny() -> None:
    """Sync callback returning False → denied."""

    @tool
    async def risky() -> str:
        """Risky."""
        return "risky result"

    tool_call = {"name": "risky", "args": {}, "id": "c3", "type": "tool_call"}
    ai_msg = AIMessage(content="", tool_calls=[tool_call])
    state = _state(messages=[ai_msg])

    update = await act_node(
        state,
        tool_map={"risky": risky},
        approval_callback=lambda name, args: False,
        guardrail=None,
        on_event=None,
    )
    tool_msgs = [m for m in update["messages"] if isinstance(m, ToolMessage)]
    assert tool_msgs[0].status == "error"


@pytest.mark.asyncio
async def test_approval_callback_async_deny() -> None:
    """Async callback returning False → denied."""

    @tool
    async def risky() -> str:
        """Risky."""
        return "risky result"

    async def async_deny(name: str, args: dict) -> bool:
        return False

    tool_call = {"name": "risky", "args": {}, "id": "c4", "type": "tool_call"}
    ai_msg = AIMessage(content="", tool_calls=[tool_call])
    state = _state(messages=[ai_msg])

    update = await act_node(
        state,
        tool_map={"risky": risky},
        approval_callback=async_deny,
        guardrail=None,
        on_event=None,
    )
    tool_msgs = [m for m in update["messages"] if isinstance(m, ToolMessage)]
    assert tool_msgs[0].status == "error"


@pytest.mark.asyncio
async def test_approval_callback_allow_executes_tool() -> None:
    """Callback returning True → tool executes."""

    @tool
    async def approved_tool() -> str:
        """Approved."""
        return "approved result"

    tool_call = {"name": "approved_tool", "args": {}, "id": "c5", "type": "tool_call"}
    ai_msg = AIMessage(content="", tool_calls=[tool_call])
    state = _state(messages=[ai_msg])

    update = await act_node(
        state,
        tool_map={"approved_tool": approved_tool},
        approval_callback=lambda name, args: True,
        guardrail=None,
        on_event=None,
    )
    tool_msgs = [m for m in update["messages"] if isinstance(m, ToolMessage)]
    assert "approved result" in tool_msgs[0].content  # type: ignore[operator]


# ── Workspace path traversal via _utils.safe_path ────────────────────────────


def test_safe_path_blocks_double_dot(tmp_path: Path) -> None:
    from gantrygraph._utils import safe_path

    with pytest.raises(PermissionError, match="escapes the workspace"):
        safe_path(tmp_path, "../../etc/passwd")


def test_safe_path_blocks_absolute_escape(tmp_path: Path) -> None:
    from gantrygraph._utils import safe_path

    with pytest.raises(PermissionError):
        safe_path(tmp_path, "/etc/passwd")


def test_safe_path_allows_subdirectory(tmp_path: Path) -> None:
    from gantrygraph._utils import safe_path

    result = safe_path(tmp_path, "subdir/file.txt")
    assert str(result).startswith(str(tmp_path.resolve()))
