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
