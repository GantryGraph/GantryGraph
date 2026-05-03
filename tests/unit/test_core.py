"""Unit tests for src/gantrygraph/core — ABCs, PerceptionResult, GantryEvent, GantryState."""

from __future__ import annotations

from types import TracebackType

import pytest
from langchain_core.tools import BaseTool

from gantrygraph.core.base_action import BaseAction
from gantrygraph.core.base_mcp import BaseMCPConnector
from gantrygraph.core.base_perception import BasePerception
from gantrygraph.core.events import GantryEvent, PerceptionResult
from gantrygraph.core.state import GantryState

# ── BasePerception ───────────────────────────────────────────────────────────


def test_base_perception_cannot_be_instantiated() -> None:
    with pytest.raises(TypeError):
        BasePerception()  # type: ignore[abstract]


def test_concrete_perception_must_implement_observe() -> None:
    class Incomplete(BasePerception):  # type: ignore[abstract]
        pass

    with pytest.raises(TypeError):
        Incomplete()  # type: ignore[abstract]


def test_concrete_perception_valid() -> None:
    class MockPerception(BasePerception):
        async def observe(self) -> PerceptionResult:
            return PerceptionResult()

    p = MockPerception()
    assert isinstance(p, BasePerception)


@pytest.mark.asyncio
async def test_base_perception_close_is_noop() -> None:
    class MockPerception(BasePerception):
        async def observe(self) -> PerceptionResult:
            return PerceptionResult()

    p = MockPerception()
    await p.close()  # should not raise


# ── BaseAction ───────────────────────────────────────────────────────────────


def test_base_action_cannot_be_instantiated() -> None:
    with pytest.raises(TypeError):
        BaseAction()  # type: ignore[abstract]


def test_concrete_action_valid() -> None:
    class MockAction(BaseAction):
        def get_tools(self) -> list[BaseTool]:
            return []

    a = MockAction()
    assert isinstance(a, BaseAction)
    assert a.get_tools() == []


@pytest.mark.asyncio
async def test_base_action_close_is_noop() -> None:
    class MockAction(BaseAction):
        def get_tools(self) -> list[BaseTool]:
            return []

    await MockAction().close()  # should not raise


# ── BaseMCPConnector ─────────────────────────────────────────────────────────


def test_base_mcp_cannot_be_instantiated() -> None:
    with pytest.raises(TypeError):
        BaseMCPConnector()  # type: ignore[abstract]


def test_concrete_mcp_valid() -> None:
    class MockMCP(BaseMCPConnector):
        async def __aenter__(self) -> MockMCP:
            return self

        async def __aexit__(
            self,
            exc_type: type[BaseException] | None,
            exc_val: BaseException | None,
            exc_tb: TracebackType | None,
        ) -> None:
            pass

        def get_tools(self) -> list[BaseTool]:
            return []

    m = MockMCP()
    assert isinstance(m, BaseMCPConnector)
    assert m.get_tools() == []


@pytest.mark.asyncio
async def test_mcp_connector_context_manager() -> None:
    class MockMCP(BaseMCPConnector):
        entered = False
        exited = False

        async def __aenter__(self) -> MockMCP:
            MockMCP.entered = True
            return self

        async def __aexit__(
            self,
            exc_type: type[BaseException] | None,
            exc_val: BaseException | None,
            exc_tb: TracebackType | None,
        ) -> None:
            MockMCP.exited = True

        def get_tools(self) -> list[BaseTool]:
            return []

    async with MockMCP() as m:
        assert MockMCP.entered
        assert m.get_tools() == []
    assert MockMCP.exited


# ── PerceptionResult ─────────────────────────────────────────────────────────


def test_perception_result_defaults() -> None:
    r = PerceptionResult()
    assert r.screenshot_b64 is None
    assert r.accessibility_tree is None
    assert r.url is None
    assert r.width == 1920
    assert r.height == 1080
    assert r.metadata == {}


def test_perception_result_to_message_content_empty() -> None:
    r = PerceptionResult()
    content = r.to_message_content()
    assert len(content) == 1
    assert content[0]["type"] == "text"
    assert "no observation" in content[0]["text"]


def test_perception_result_to_message_content_screenshot_only() -> None:
    r = PerceptionResult(screenshot_b64="abc123")
    content = r.to_message_content()
    assert any(c["type"] == "image_url" for c in content)
    img_block = next(c for c in content if c["type"] == "image_url")
    assert "abc123" in img_block["image_url"]["url"]
    assert img_block["image_url"]["url"].startswith("data:image/png;base64,")


def test_perception_result_to_message_content_tree_only() -> None:
    r = PerceptionResult(accessibility_tree="button: Click me")
    content = r.to_message_content()
    assert len(content) == 1
    assert content[0]["type"] == "text"
    assert "Click me" in content[0]["text"]


def test_perception_result_to_message_content_both() -> None:
    r = PerceptionResult(screenshot_b64="xyz", accessibility_tree="nav: menu")
    content = r.to_message_content()
    assert len(content) == 2
    types = {c["type"] for c in content}
    assert types == {"text", "image_url"}


def test_perception_result_serializable() -> None:
    r = PerceptionResult(screenshot_b64="xyz", width=1024, height=768)
    d = r.model_dump()
    assert d["width"] == 1024
    assert d["height"] == 768
    assert d["screenshot_b64"] == "xyz"


def test_perception_result_round_trip() -> None:
    original = PerceptionResult(
        screenshot_b64="base64data",
        url="https://example.com",
        width=800,
        height=600,
        metadata={"monitor": 1},
    )
    restored = PerceptionResult(**original.model_dump())
    assert restored == original


# ── GantryEvent ────────────────────────────────────────────────────────────────


def test_gantry_event_creation() -> None:
    ev = GantryEvent(event_type="think", step=3, data={"tokens": 500})
    assert ev.event_type == "think"
    assert ev.step == 3
    assert ev.data["tokens"] == 500


def test_gantry_event_default_data() -> None:
    ev = GantryEvent(event_type="done", step=10)
    assert ev.data == {}


def test_gantry_event_all_types() -> None:
    for event_type in ("observe", "think", "act", "review", "error", "done"):
        ev = GantryEvent(event_type=event_type, step=0)  # type: ignore[arg-type]
        assert ev.event_type == event_type


# ── GantryState ────────────────────────────────────────────────────────────────


def test_gantry_state_minimal() -> None:
    state: GantryState = {
        "task": "do something",
        "messages": [],
        "step_count": 0,
        "is_done": False,
    }
    assert state["task"] == "do something"
    assert state["step_count"] == 0
    assert state["is_done"] is False
    assert state["messages"] == []


def test_gantry_state_with_optional_fields() -> None:
    state: GantryState = {
        "task": "test",
        "messages": [],
        "step_count": 5,
        "is_done": True,
        "last_error": "Tool not found",
        "last_observation": {"width": 1920, "height": 1080},
    }
    assert state["last_error"] == "Tool not found"
    assert state["last_observation"]["width"] == 1920


# ── _utils ───────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_ensure_awaitable_with_sync_fn() -> None:
    from gantrygraph._utils import ensure_awaitable

    def sync_fn(x: int) -> int:
        return x * 2

    result = await ensure_awaitable(sync_fn, 5)
    assert result == 10


@pytest.mark.asyncio
async def test_ensure_awaitable_with_async_fn() -> None:
    from gantrygraph._utils import ensure_awaitable

    async def async_fn(x: int) -> int:
        return x * 3

    result = await ensure_awaitable(async_fn, 4)
    assert result == 12


def test_safe_path_allows_valid_path() -> None:
    from pathlib import Path

    from gantrygraph._utils import safe_path

    workspace = Path("/tmp/workspace")
    result = safe_path(workspace, "subdir/file.txt")
    assert str(result).startswith(str(workspace.resolve()))


def test_safe_path_blocks_traversal() -> None:
    from pathlib import Path

    from gantrygraph._utils import safe_path

    workspace = Path("/tmp/workspace")
    with pytest.raises(PermissionError, match="escapes the workspace"):
        safe_path(workspace, "../../etc/passwd")
