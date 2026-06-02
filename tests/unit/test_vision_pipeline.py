"""Unit tests for gantrygraph.vision.pipeline.

Three fronts:
  1. Compression filters — Downsample, Grayscale, ConvertToWebP
  2. SetOfMarkAnnotator — isolated with a mock Playwright page
  3. browser_click_som action — error paths + coordinate / selector dispatch
"""

from __future__ import annotations

import io
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from PIL import Image

from gantrygraph.vision.pipeline import (
    ConvertToWebP,
    Downsample,
    Grayscale,
    ImageFilter,
    PerceptionPipeline,
    SetOfMarkAnnotator,
)

# ── image helpers ─────────────────────────────────────────────────────────────


def _make_png(
    width: int = 200,
    height: int = 150,
    color: tuple[int, int, int] = (80, 120, 200),
) -> bytes:
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_noise_png(width: int = 400, height: int = 300, seed: int = 42) -> bytes:
    """Random-noise image.

    PNG (lossless DEFLATE) can't compress random pixels — the file is nearly raw size.
    WebP (lossy DCT) discards the noise via quantisation and wins on file size.
    This makes the size-comparison test deterministic and reliable.
    """
    import random

    rng = random.Random(seed)
    img = Image.new("RGB", (width, height))
    img.putdata(
        [
            (rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255))
            for _ in range(width * height)
        ]
    )
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _open(image_bytes: bytes) -> Image.Image:
    return Image.open(io.BytesIO(image_bytes))


# ── SoM test helpers ──────────────────────────────────────────────────────────


def _fake_page(elements: list[dict[str, Any]]) -> MagicMock:
    """Mock Playwright Page whose evaluate() immediately returns *elements*."""
    page = MagicMock()
    page.evaluate = AsyncMock(return_value=elements)
    return page


def _five_elements() -> list[dict[str, Any]]:
    return [
        {
            "tag": "button",
            "text": "Submit",
            "x": 10,
            "y": 10,
            "w": 80,
            "h": 30,
            "selector": "#submit",
        },
        {"tag": "a", "text": "Home", "x": 100, "y": 5, "w": 60, "h": 20, "selector": None},
        {
            "tag": "input",
            "text": "Email",
            "x": 10,
            "y": 50,
            "w": 150,
            "h": 25,
            "selector": "#email",
        },
        {"tag": "button", "text": "Cancel", "x": 10, "y": 90, "w": 80, "h": 30, "selector": None},
        {"tag": "a", "text": "About", "x": 100, "y": 90, "w": 50, "h": 20, "selector": "#about"},
    ]


# ═══════════════════════════════════════════════════════════════════════════════
# 1 — Compression filters
# ═══════════════════════════════════════════════════════════════════════════════


class TestDownsample:
    @pytest.mark.asyncio
    async def test_reduces_width_to_max(self) -> None:
        result = await Downsample(max_width=640).process(_make_png(1920, 1080), {})
        assert _open(result).width == 640

    @pytest.mark.asyncio
    async def test_preserves_aspect_ratio(self) -> None:
        result = await Downsample(max_width=960).process(_make_png(1920, 1080), {})
        img = _open(result)
        assert img.width == 960
        assert img.height == 540  # 1920×1080 halved

    @pytest.mark.asyncio
    async def test_noop_when_already_narrow(self) -> None:
        png = _make_png(400, 300)
        result = await Downsample(max_width=1280).process(png, {})
        assert result == png  # exact same bytes

    @pytest.mark.asyncio
    async def test_does_not_write_ctx(self) -> None:
        ctx: dict[str, Any] = {}
        await Downsample(max_width=100).process(_make_png(200, 200), ctx)
        assert ctx == {}

    @pytest.mark.asyncio
    async def test_output_is_decodable(self) -> None:
        result = await Downsample(max_width=80).process(_make_png(320, 240), {})
        img = _open(result)
        assert img.size == (80, 60)


class TestGrayscale:
    @pytest.mark.asyncio
    async def test_all_pixels_are_grey(self) -> None:
        png = _make_png(10, 10, color=(80, 120, 200))
        result = await Grayscale().process(png, {})
        pixels = list(_open(result).convert("RGB").getdata())
        assert all(r == g == b for r, g, b in pixels)

    @pytest.mark.asyncio
    async def test_preserves_dimensions(self) -> None:
        result = await Grayscale().process(_make_png(64, 48), {})
        assert _open(result).size == (64, 48)

    @pytest.mark.asyncio
    async def test_output_is_decodable_png(self) -> None:
        result = await Grayscale().process(_make_png(20, 20), {})
        _open(result)  # must not raise

    @pytest.mark.asyncio
    async def test_does_not_write_ctx(self) -> None:
        ctx: dict[str, Any] = {}
        await Grayscale().process(_make_png(10, 10), ctx)
        assert ctx == {}


class TestConvertToWebP:
    @pytest.mark.asyncio
    async def test_output_format_is_webp(self) -> None:
        result = await ConvertToWebP(quality=85).process(_make_png(100, 100), {})
        assert _open(result).format == "WEBP"

    @pytest.mark.asyncio
    async def test_preserves_dimensions(self) -> None:
        result = await ConvertToWebP(quality=85).process(_make_png(320, 240), {})
        assert _open(result).size == (320, 240)

    @pytest.mark.asyncio
    async def test_reduces_file_size_vs_png(self) -> None:
        # Random noise: PNG ≈ raw size (DEFLATE fails on noise).
        # WebP lossy at quality=50 discards noise via DCT quantisation → much smaller.
        png = _make_noise_png(400, 300)
        webp = await ConvertToWebP(quality=50).process(png, {})
        assert len(webp) < len(png), (
            f"WebP ({len(webp)} B) should be smaller than noise PNG ({len(png)} B)"
        )

    @pytest.mark.asyncio
    async def test_does_not_write_ctx(self) -> None:
        ctx: dict[str, Any] = {}
        await ConvertToWebP().process(_make_png(10, 10), ctx)
        assert ctx == {}


# ═══════════════════════════════════════════════════════════════════════════════
# 2 — PerceptionPipeline
# ═══════════════════════════════════════════════════════════════════════════════


class TestPerceptionPipeline:
    @pytest.mark.asyncio
    async def test_filters_run_in_order(self) -> None:
        """Downsample first → result is 640 wide; ConvertToWebP second → format is WEBP."""
        pipeline = PerceptionPipeline([Downsample(max_width=640), ConvertToWebP(quality=80)])
        result = await pipeline.run(_make_png(1280, 720))
        img = _open(result)
        assert img.width == 640
        assert img.format == "WEBP"

    @pytest.mark.asyncio
    async def test_last_ctx_populated_after_run(self) -> None:
        class _Flag(ImageFilter):
            async def process(self, b: bytes, ctx: dict[str, Any]) -> bytes:
                ctx["hit"] = True
                return b

        pipeline = PerceptionPipeline([_Flag()])
        await pipeline.run(_make_png())
        assert pipeline.last_ctx.get("hit") is True

    @pytest.mark.asyncio
    async def test_ctx_threaded_through_all_filters(self) -> None:
        """Each filter can read what the previous one wrote."""

        class _Append(ImageFilter):
            def __init__(self, val: int) -> None:
                self.val = val

            async def process(self, b: bytes, ctx: dict[str, Any]) -> bytes:
                ctx.setdefault("log", []).append(self.val)
                return b

        pipeline = PerceptionPipeline([_Append(1), _Append(2), _Append(3)])
        await pipeline.run(_make_png())
        assert pipeline.last_ctx["log"] == [1, 2, 3]

    @pytest.mark.asyncio
    async def test_initial_ctx_values_are_visible_to_filters(self) -> None:
        class _ReadSeed(ImageFilter):
            async def process(self, b: bytes, ctx: dict[str, Any]) -> bytes:
                ctx["saw"] = ctx.get("seed")
                return b

        pipeline = PerceptionPipeline([_ReadSeed()])
        await pipeline.run(_make_png(), ctx={"seed": 42})
        assert pipeline.last_ctx["saw"] == 42

    @pytest.mark.asyncio
    async def test_som_map_empty_without_som_filter(self) -> None:
        pipeline = PerceptionPipeline([Downsample(max_width=100)])
        await pipeline.run(_make_png(200, 200))
        assert pipeline.som_map == {}

    @pytest.mark.asyncio
    async def test_empty_pipeline_returns_unchanged_image(self) -> None:
        png = _make_png()
        result = await PerceptionPipeline([]).run(png)
        assert result == png


# ═══════════════════════════════════════════════════════════════════════════════
# 3a — SetOfMarkAnnotator (isolated, no real browser)
# ═══════════════════════════════════════════════════════════════════════════════


class TestSetOfMarkAnnotator:
    @pytest.mark.asyncio
    async def test_noop_when_no_page_in_ctx(self) -> None:
        png = _make_png(200, 150)
        result = await SetOfMarkAnnotator().process(png, {})
        assert result == png

    @pytest.mark.asyncio
    async def test_som_map_has_correct_count(self) -> None:
        ctx: dict[str, Any] = {"page": _fake_page(_five_elements())}
        await SetOfMarkAnnotator().process(_make_png(300, 200), ctx)
        assert len(ctx["som_map"]) == 5

    @pytest.mark.asyncio
    async def test_som_map_ids_are_sequential_from_one(self) -> None:
        ctx: dict[str, Any] = {"page": _fake_page(_five_elements())}
        await SetOfMarkAnnotator().process(_make_png(300, 200), ctx)
        assert set(ctx["som_map"].keys()) == {1, 2, 3, 4, 5}

    @pytest.mark.asyncio
    async def test_som_map_entries_have_required_keys(self) -> None:
        ctx: dict[str, Any] = {"page": _fake_page(_five_elements())}
        await SetOfMarkAnnotator().process(_make_png(300, 200), ctx)
        for entry in ctx["som_map"].values():
            assert "tag" in entry
            assert "text" in entry
            assert "selector" in entry
            assert "cx" in entry
            assert "cy" in entry

    @pytest.mark.asyncio
    async def test_selector_preserved_when_present(self) -> None:
        ctx: dict[str, Any] = {"page": _fake_page(_five_elements())}
        await SetOfMarkAnnotator().process(_make_png(300, 200), ctx)
        assert ctx["som_map"][1]["selector"] == "#submit"

    @pytest.mark.asyncio
    async def test_selector_is_none_when_absent(self) -> None:
        ctx: dict[str, Any] = {"page": _fake_page(_five_elements())}
        await SetOfMarkAnnotator().process(_make_png(300, 200), ctx)
        assert ctx["som_map"][2]["selector"] is None

    @pytest.mark.asyncio
    async def test_centre_coordinates_within_bounding_box(self) -> None:
        # Element 1: x=10, y=10, w=80, h=30 → cx in [10,90], cy in [10,40]
        ctx: dict[str, Any] = {"page": _fake_page(_five_elements())}
        await SetOfMarkAnnotator().process(_make_png(300, 200), ctx)
        elem = ctx["som_map"][1]
        assert 10 <= elem["cx"] <= 90
        assert 10 <= elem["cy"] <= 40

    @pytest.mark.asyncio
    async def test_tag_and_text_match_input(self) -> None:
        ctx: dict[str, Any] = {"page": _fake_page(_five_elements())}
        await SetOfMarkAnnotator().process(_make_png(300, 200), ctx)
        assert ctx["som_map"][1]["tag"] == "button"
        assert ctx["som_map"][1]["text"] == "Submit"
        assert ctx["som_map"][3]["tag"] == "input"

    @pytest.mark.asyncio
    async def test_output_is_valid_png_same_size(self) -> None:
        ctx: dict[str, Any] = {"page": _fake_page(_five_elements())}
        result = await SetOfMarkAnnotator().process(_make_png(300, 200), ctx)
        img = _open(result)
        assert img.format == "PNG"
        assert img.size == (300, 200)

    @pytest.mark.asyncio
    async def test_zero_elements_returns_unchanged_image_no_crash(self) -> None:
        ctx: dict[str, Any] = {"page": _fake_page([])}
        result = await SetOfMarkAnnotator().process(_make_png(100, 80), ctx)
        assert ctx["som_map"] == {}
        assert _open(result).size == (100, 80)

    @pytest.mark.asyncio
    async def test_max_elements_cap_respected(self) -> None:
        many = [
            {
                "tag": "button",
                "text": f"b{i}",
                "x": i * 3,
                "y": 0,
                "w": 20,
                "h": 15,
                "selector": None,
            }
            for i in range(20)
        ]
        ctx: dict[str, Any] = {"page": _fake_page(many)}
        await SetOfMarkAnnotator(max_elements=7).process(_make_png(300, 200), ctx)
        assert len(ctx["som_map"]) == 7

    @pytest.mark.asyncio
    async def test_out_of_bounds_elements_do_not_crash(self) -> None:
        """Elements positioned outside the image must be skipped, not raise."""
        oob = [
            {"tag": "button", "text": "x", "x": 9999, "y": 9999, "w": 80, "h": 30, "selector": None}
        ]
        ctx: dict[str, Any] = {"page": _fake_page(oob)}
        result = await SetOfMarkAnnotator().process(_make_png(100, 80), ctx)
        _open(result)  # must not raise

    @pytest.mark.asyncio
    async def test_fixed_color_applied_to_all_boxes(self) -> None:
        """Smoke-test that a fixed color is accepted without error."""
        ctx: dict[str, Any] = {"page": _fake_page(_five_elements())}
        result = await SetOfMarkAnnotator(color=(255, 0, 0)).process(_make_png(300, 200), ctx)
        assert len(ctx["som_map"]) == 5
        _open(result)

    @pytest.mark.asyncio
    async def test_som_pipeline_integration_som_then_downsample(self) -> None:
        """SoM runs at full resolution; Downsample halves the width afterwards."""
        page = _fake_page(_five_elements())
        pipeline = PerceptionPipeline([SetOfMarkAnnotator(), Downsample(max_width=150)])
        result = await pipeline.run(_make_png(300, 200), ctx={"page": page})
        assert len(pipeline.som_map) == 5
        assert _open(result).width == 150


# ═══════════════════════════════════════════════════════════════════════════════
# 3b — browser_click_som action tool
# ═══════════════════════════════════════════════════════════════════════════════


def _make_bt(pipeline: PerceptionPipeline | None) -> Any:
    """Return a BrowserTools skeleton with only the attributes click_som needs."""
    from gantrygraph.actions.browser import BrowserTools

    bt = BrowserTools.__new__(BrowserTools)
    bt._vision_pipeline = pipeline
    bt._web_page = None
    return bt


def _som_pipeline(*entries: tuple[int, str, str | None, int, int]) -> PerceptionPipeline:
    """Build a PerceptionPipeline whose last_ctx["som_map"] is pre-populated.

    Each entry is (id, tag, selector, cx, cy).
    """
    pipeline = PerceptionPipeline([])
    pipeline.last_ctx = {
        "som_map": {
            elem_id: {"tag": tag, "text": tag, "selector": selector, "cx": cx, "cy": cy}
            for elem_id, tag, selector, cx, cy in entries
        }
    }
    return pipeline


class TestBrowserClickSom:
    # ── get_tools() presence ──────────────────────────────────────────────────

    def test_no_pipeline_nine_tools_no_som(self) -> None:
        bt = _make_bt(None)
        tools = bt.get_tools()
        assert len(tools) == 9
        assert not any(t.name == "browser_click_som" for t in tools)

    def test_with_pipeline_ten_tools_includes_som(self) -> None:
        bt = _make_bt(PerceptionPipeline([]))
        tools = bt.get_tools()
        assert len(tools) == 10
        assert any(t.name == "browser_click_som" for t in tools)

    # ── error paths (no page call needed) ────────────────────────────────────

    @pytest.mark.asyncio
    async def test_empty_som_map_returns_observe_hint(self) -> None:
        bt = _make_bt(PerceptionPipeline([]))  # no filters → empty som_map
        tool = bt._click_som_tool()
        result = await tool.ainvoke({"element_id": 1})
        assert "observe" in result.lower() or "empty" in result.lower()

    @pytest.mark.asyncio
    async def test_missing_id_returns_error_with_available_ids(self) -> None:
        pipeline = _som_pipeline(
            (1, "button", "#a", 10, 10),
            (2, "a", None, 50, 20),
            (3, "input", "#c", 30, 40),
        )
        bt = _make_bt(pipeline)
        mock_page = MagicMock()
        tool = bt._click_som_tool()
        with patch.object(bt, "_ensure_browser", new=AsyncMock(return_value=mock_page)):
            result = await tool.ainvoke({"element_id": 99})
        assert "99" in result
        # at least one valid id should appear in the error
        assert any(str(i) in result for i in [1, 2, 3])

    # ── successful click paths ────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_click_by_selector_when_available(self) -> None:
        pipeline = _som_pipeline((7, "button", "#submit", 100, 200))
        bt = _make_bt(pipeline)

        mock_page = MagicMock()
        mock_page.click = AsyncMock()
        mock_page.mouse = MagicMock()
        mock_page.mouse.click = AsyncMock()

        # Tool must be created INSIDE the patch so its closure captures the mock.
        with patch.object(bt, "_ensure_browser", new=AsyncMock(return_value=mock_page)):
            tool = bt._click_som_tool()
            result = await tool.ainvoke({"element_id": 7})

        mock_page.click.assert_called_once_with("#submit", timeout=5000)
        mock_page.mouse.click.assert_not_called()
        assert "7" in result

    @pytest.mark.asyncio
    async def test_no_selector_falls_through_to_coords(self) -> None:
        pipeline = _som_pipeline((2, "a", None, 55, 33))
        bt = _make_bt(pipeline)

        mock_page = MagicMock()
        mock_page.click = AsyncMock()
        mock_page.mouse = MagicMock()
        mock_page.mouse.click = AsyncMock()

        with patch.object(bt, "_ensure_browser", new=AsyncMock(return_value=mock_page)):
            tool = bt._click_som_tool()
            result = await tool.ainvoke({"element_id": 2})

        mock_page.click.assert_not_called()
        mock_page.mouse.click.assert_called_once_with(55, 33)
        assert "55" in result and "33" in result

    @pytest.mark.asyncio
    async def test_broken_selector_falls_back_to_coords(self) -> None:
        pipeline = _som_pipeline((3, "button", "#broken", 120, 80))
        bt = _make_bt(pipeline)

        mock_page = MagicMock()
        mock_page.click = AsyncMock(side_effect=Exception("element not found"))
        mock_page.mouse = MagicMock()
        mock_page.mouse.click = AsyncMock()

        with patch.object(bt, "_ensure_browser", new=AsyncMock(return_value=mock_page)):
            tool = bt._click_som_tool()
            result = await tool.ainvoke({"element_id": 3})

        mock_page.mouse.click.assert_called_once_with(120, 80)
        assert "(120, 80)" in result

    @pytest.mark.asyncio
    async def test_result_contains_element_tag_and_text(self) -> None:
        pipeline = PerceptionPipeline([])
        pipeline.last_ctx = {
            "som_map": {
                5: {"tag": "input", "text": "Username", "selector": "#user", "cx": 60, "cy": 50},
            }
        }
        bt = _make_bt(pipeline)

        mock_page = MagicMock()
        mock_page.click = AsyncMock()
        mock_page.mouse = MagicMock()
        mock_page.mouse.click = AsyncMock()

        with patch.object(bt, "_ensure_browser", new=AsyncMock(return_value=mock_page)):
            tool = bt._click_som_tool()
            result = await tool.ainvoke({"element_id": 5})

        assert "input" in result.lower() or "Username" in result
