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


def test_workspace_policy_custom() -> None:
    p = WorkspacePolicy(
        workspace_path="/home/user",
        allow_read_outside=True,
        allow_write_outside=False,
    )
    assert p.allow_read_outside is True
    assert p.allow_write_outside is False


# ── BudgetPolicy ──────────────────────────────────────────────────────────────

def test_budget_policy_defaults() -> None:
    p = BudgetPolicy()
    assert p.max_steps == 50
    assert p.max_tokens is None
    assert p.max_wall_seconds is None


def test_budget_policy_custom() -> None:
    p = BudgetPolicy(max_steps=10, max_tokens=5000, max_wall_seconds=30.0)
    assert p.max_steps == 10
    assert p.max_tokens == 5000
    assert p.max_wall_seconds == 30.0


def test_budget_policy_max_steps_must_be_positive() -> None:
    with pytest.raises(ValueError):
        BudgetPolicy(max_steps=0)


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
        approval_cb=None,
        guardrail=guardrail,
        event_cb=None,
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
        approval_cb=None,
        guardrail=guardrail,
        event_cb=None,
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
        approval_cb=lambda name, args: False,
        guardrail=None,
        event_cb=None,
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
        approval_cb=async_deny,
        guardrail=None,
        event_cb=None,
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
        approval_cb=lambda name, args: True,
        guardrail=None,
        event_cb=None,
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
