"""Unit tests for perception module."""

from __future__ import annotations

import base64
import io
import sys
from unittest.mock import patch

import pytest

from gantrygraph.core.events import PerceptionResult

# ── DesktopScreen ─────────────────────────────────────────────────────────────


def _make_small_png() -> bytes:
    """Generate a tiny valid 2×2 PNG in memory."""
    from PIL import Image

    img = Image.new("RGB", (2, 2), color=(255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_desktop_screen_is_base_perception() -> None:
    from gantrygraph.core.base_perception import BasePerception
    from gantrygraph.perception.desktop import DesktopScreen

    d = DesktopScreen()
    assert isinstance(d, BasePerception)


@pytest.mark.asyncio
async def test_desktop_screen_observe_returns_perception_result() -> None:
    from gantrygraph.perception.desktop import DesktopScreen

    small_png = _make_small_png()

    with patch.object(DesktopScreen, "_capture_sync", return_value=(small_png, 2, 2)):
        d = DesktopScreen()
        result = await d.observe()

    assert isinstance(result, PerceptionResult)
    assert result.screenshot_b64 is not None
    assert result.width == 2
    assert result.height == 2


@pytest.mark.asyncio
async def test_desktop_screen_b64_is_valid_png() -> None:
    from PIL import Image

    from gantrygraph.perception.desktop import DesktopScreen

    small_png = _make_small_png()

    with patch.object(DesktopScreen, "_capture_sync", return_value=(small_png, 2, 2)):
        d = DesktopScreen()
        result = await d.observe()

    raw = base64.b64decode(result.screenshot_b64)  # type: ignore[arg-type]
    img = Image.open(io.BytesIO(raw))
    assert img.format == "PNG"


def test_resize_preserving_aspect_no_op_small() -> None:
    from PIL import Image

    from gantrygraph.perception.desktop import _resize_preserving_aspect

    img = Image.new("RGB", (100, 100))
    resized = _resize_preserving_aspect(img, (1920, 1080))
    assert resized.width == 100
    assert resized.height == 100


def test_resize_preserving_aspect_scales_down() -> None:
    from PIL import Image

    from gantrygraph.perception.desktop import _resize_preserving_aspect

    img = Image.new("RGB", (3840, 2160))  # 4K
    resized = _resize_preserving_aspect(img, (1920, 1080))
    assert resized.width <= 1920
    assert resized.height <= 1080


def test_resize_preserving_aspect_maintains_ratio() -> None:
    from PIL import Image

    from gantrygraph.perception.desktop import _resize_preserving_aspect

    img = Image.new("RGB", (2000, 1000))  # 2:1 ratio
    resized = _resize_preserving_aspect(img, (1000, 1000))
    ratio = resized.width / resized.height
    assert abs(ratio - 2.0) < 0.05


# ── PerceptionResult.to_message_content (integration with desktop) ────────────


@pytest.mark.asyncio
async def test_desktop_result_produces_valid_message_content() -> None:
    from gantrygraph.perception.desktop import DesktopScreen

    small_png = _make_small_png()

    with patch.object(DesktopScreen, "_capture_sync", return_value=(small_png, 2, 2)):
        d = DesktopScreen()
        result = await d.observe()

    content = result.to_message_content()
    assert len(content) == 1
    assert content[0]["type"] == "image_url"
    url = content[0]["image_url"]["url"]
    assert url.startswith("data:image/png;base64,")


# ── DesktopAXTree ─────────────────────────────────────────────────────────────

try:
    from gantrygraph.perception.desktop_ax import DesktopAXTree as _DesktopAXTree
    from gantrygraph.perception.desktop_ax import _serialize_node

    _HAS_DESKTOP_AX = True
except ImportError:
    _HAS_DESKTOP_AX = False


class _FakeAXNode:
    """Minimal stand-in for an atomacos AXUIElement."""

    def __init__(
        self,
        role: str,
        title: str = "",
        value: str = "",
        description: str = "",
        children: list[object] | None = None,
        enabled: bool = True,
        focused: bool = False,
    ) -> None:
        self.AXRole = role
        self.AXTitle = title
        self.AXValue = value
        self.AXDescription = description
        self.AXChildren = children or []
        self.AXEnabled = enabled
        self.AXFocused = focused


@pytest.mark.skipif(not _HAS_DESKTOP_AX, reason="atomacos not installed")
def test_serialize_node_basic() -> None:
    lines: list[str] = []
    node = _FakeAXNode("AXButton", title="Save")
    _serialize_node(node, depth=0, max_depth=4, max_children=10, max_text=100, lines=lines)
    assert any("AXButton" in ln for ln in lines)
    assert any("Save" in ln for ln in lines)


@pytest.mark.skipif(not _HAS_DESKTOP_AX, reason="atomacos not installed")
def test_serialize_node_text_area_includes_value() -> None:
    lines: list[str] = []
    node = _FakeAXNode("AXTextArea", value="hello world")
    _serialize_node(node, depth=0, max_depth=4, max_children=10, max_text=100, lines=lines)
    assert any("hello world" in ln for ln in lines)


@pytest.mark.skipif(not _HAS_DESKTOP_AX, reason="atomacos not installed")
def test_serialize_node_skips_unknown_role() -> None:
    lines: list[str] = []
    node = _FakeAXNode("AXUnknown")
    _serialize_node(node, depth=0, max_depth=4, max_children=10, max_text=100, lines=lines)
    assert lines == []


@pytest.mark.skipif(not _HAS_DESKTOP_AX, reason="atomacos not installed")
def test_serialize_node_respects_max_depth() -> None:
    deep = _FakeAXNode("AXButton", title="deep")
    mid = _FakeAXNode("AXGroup", children=[deep])
    root = _FakeAXNode("AXWindow", children=[mid])
    lines: list[str] = []
    _serialize_node(root, depth=0, max_depth=1, max_children=10, max_text=100, lines=lines)
    # max_depth=1 means we render root (depth 0) and its children (depth 1), but not deeper
    assert not any("deep" in ln for ln in lines)


@pytest.mark.skipif(not _HAS_DESKTOP_AX, reason="atomacos not installed")
def test_serialize_node_overflow_message() -> None:
    children = [_FakeAXNode("AXButton", title=str(i)) for i in range(10)]
    root = _FakeAXNode("AXGroup", children=children)
    lines: list[str] = []
    _serialize_node(root, depth=0, max_depth=4, max_children=3, max_text=100, lines=lines)
    assert any("more children" in ln for ln in lines)


@pytest.mark.skipif(not _HAS_DESKTOP_AX, reason="atomacos not installed")
def test_serialize_node_focused_tag() -> None:
    lines: list[str] = []
    node = _FakeAXNode("AXTextField", focused=True)
    _serialize_node(node, depth=0, max_depth=4, max_children=10, max_text=100, lines=lines)
    assert any("focused" in ln for ln in lines)


@pytest.mark.skipif(not _HAS_DESKTOP_AX, reason="atomacos not installed")
def test_desktop_ax_tree_import_error_without_package(monkeypatch: pytest.MonkeyPatch) -> None:
    import gantrygraph.perception.desktop_ax as _mod

    monkeypatch.setattr(_mod, "_HAS_ATOMACOS", False)
    with pytest.raises(ImportError, match="atomacos"):
        _DesktopAXTree()


@pytest.mark.skipif(not _HAS_DESKTOP_AX, reason="atomacos not installed")
@pytest.mark.asyncio
async def test_desktop_ax_tree_observe_returns_perception_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import gantrygraph.perception.desktop_ax as _mod

    fake_tree = "AXApplication 'TestApp'\n  AXWindow 'Main'"
    monkeypatch.setattr(_mod, "_HAS_ATOMACOS", True)
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(_mod, "_build_tree", lambda *a, **kw: fake_tree)

    perception = _DesktopAXTree()
    result = await perception.observe()

    assert isinstance(result, PerceptionResult)
    assert result.accessibility_tree == fake_tree
    assert result.screenshot_b64 is None
    content = result.to_message_content()
    assert content[0]["type"] == "text"
    assert "AXApplication" in content[0]["text"]


# ── WebPage import guard ──────────────────────────────────────────────────────


def test_web_page_raises_without_extra() -> None:
    import importlib
    import unittest.mock

    with unittest.mock.patch.dict(sys.modules, {"playwright": None, "playwright.async_api": None}):
        import gantrygraph.perception.web as wp

        importlib.reload(wp)
        assert not wp._HAS_PLAYWRIGHT
        with pytest.raises(ImportError, match="browser"):
            wp.WebPage()
