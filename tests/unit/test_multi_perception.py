"""Unit tests for MultiPerception."""
from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock

import pytest

from gantrygraph.core.base_perception import BasePerception
from gantrygraph.core.events import PerceptionResult
from gantrygraph.perception.multi import MultiPerception

# ── Helpers ───────────────────────────────────────────────────────────────────

def _perception(
    *,
    screenshot_b64: str | None = None,
    accessibility_tree: str | None = None,
    url: str | None = None,
    width: int = 1920,
    height: int = 1080,
    metadata: dict[str, Any] | None = None,
) -> BasePerception:
    result = PerceptionResult(
        screenshot_b64=screenshot_b64,
        accessibility_tree=accessibility_tree,
        url=url,
        width=width,
        height=height,
        metadata=metadata or {},
    )
    mock = AsyncMock(spec=BasePerception)
    mock.observe = AsyncMock(return_value=result)
    mock.close = AsyncMock()
    return mock  # type: ignore[return-value]


# ── Construction ──────────────────────────────────────────────────────────────

def test_multi_perception_requires_at_least_one_source() -> None:
    with pytest.raises(ValueError, match="at least one"):
        MultiPerception([])


def test_multi_perception_is_base_perception() -> None:
    mp = MultiPerception([_perception()])
    assert isinstance(mp, BasePerception)


# ── Screenshot merge ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_first_screenshot_wins() -> None:
    mp = MultiPerception([
        _perception(screenshot_b64="FIRST", width=800, height=600),
        _perception(screenshot_b64="SECOND", width=1920, height=1080),
    ])
    result = await mp.observe()
    assert result.screenshot_b64 == "FIRST"
    assert result.width == 800
    assert result.height == 600


@pytest.mark.asyncio
async def test_screenshot_from_second_when_first_has_none() -> None:
    mp = MultiPerception([
        _perception(accessibility_tree="tree only"),
        _perception(screenshot_b64="ONLY_SCREENSHOT", width=1280, height=720),
    ])
    result = await mp.observe()
    assert result.screenshot_b64 == "ONLY_SCREENSHOT"
    assert result.width == 1280


@pytest.mark.asyncio
async def test_no_screenshot_when_none_provided() -> None:
    mp = MultiPerception([
        _perception(accessibility_tree="text only"),
        _perception(accessibility_tree="more text"),
    ])
    result = await mp.observe()
    assert result.screenshot_b64 is None


# ── Accessibility tree merge ──────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_trees_are_concatenated_with_labels() -> None:
    mp = MultiPerception([
        _perception(accessibility_tree="desktop state"),
        _perception(accessibility_tree="browser state"),
    ])
    result = await mp.observe()
    assert result.accessibility_tree is not None
    assert "desktop state" in result.accessibility_tree
    assert "browser state" in result.accessibility_tree
    assert "source 1" in result.accessibility_tree
    assert "source 2" in result.accessibility_tree


@pytest.mark.asyncio
async def test_empty_trees_are_skipped() -> None:
    mp = MultiPerception([
        _perception(accessibility_tree=None),
        _perception(accessibility_tree="only this one"),
    ])
    result = await mp.observe()
    assert result.accessibility_tree is not None
    assert "source 1" not in result.accessibility_tree
    assert "only this one" in result.accessibility_tree


@pytest.mark.asyncio
async def test_no_tree_when_all_none() -> None:
    mp = MultiPerception([
        _perception(screenshot_b64="img"),
    ])
    result = await mp.observe()
    assert result.accessibility_tree is None


# ── URL + metadata merge ──────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_first_url_wins() -> None:
    mp = MultiPerception([
        _perception(url="https://first.example.com"),
        _perception(url="https://second.example.com"),
    ])
    result = await mp.observe()
    assert result.url == "https://first.example.com"


@pytest.mark.asyncio
async def test_metadata_merged() -> None:
    mp = MultiPerception([
        _perception(metadata={"source": "desktop", "fps": 30}),
        _perception(metadata={"source": "browser", "url": "https://x.com"}),
    ])
    result = await mp.observe()
    assert result.metadata["fps"] == 30
    assert result.metadata["url"] == "https://x.com"
    assert result.metadata["source"] == "browser"  # later source wins


# ── Concurrency ───────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_sources_are_called_concurrently() -> None:
    """Both observe() calls must overlap — total time ≈ max(delays), not sum."""
    started: list[float] = []

    class SlowPerception(BasePerception):
        async def observe(self) -> PerceptionResult:
            started.append(asyncio.get_event_loop().time())
            await asyncio.sleep(0.05)
            return PerceptionResult(accessibility_tree="slow")

    mp = MultiPerception([SlowPerception(), SlowPerception()])
    await mp.observe()
    # Both started before either finished
    assert len(started) == 2
    assert abs(started[1] - started[0]) < 0.04


# ── close() propagates ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_close_calls_all_sources() -> None:
    s1 = _perception()
    s2 = _perception()
    mp = MultiPerception([s1, s2])
    await mp.close()
    s1.close.assert_awaited_once()  # type: ignore[attr-defined]
    s2.close.assert_awaited_once()  # type: ignore[attr-defined]


# ── Importable from gantrygraph and gantrygraph.perception ──────────────────────────────────

def test_importable_from_gantrygraph() -> None:
    from gantrygraph import MultiPerception as MP  # noqa: F401
    assert MP is MultiPerception


def test_importable_from_gantrygraph_perception() -> None:
    from gantrygraph.perception import MultiPerception as MP  # noqa: F401
    assert MP is MultiPerception
