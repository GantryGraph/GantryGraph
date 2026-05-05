"""Integration tests for browser stealth mode and WebPage/BrowserTools lifecycle.

Requires the [browser] extra and a Chromium installation::

    pip install 'gantrygraph[browser]'
    playwright install chromium

Tests are skipped automatically if Playwright is not installed.
"""

from __future__ import annotations

import pytest

try:
    from playwright.async_api import async_playwright

    _HAS_PLAYWRIGHT = True
except ImportError:
    _HAS_PLAYWRIGHT = False

pytestmark = pytest.mark.skipif(not _HAS_PLAYWRIGHT, reason="playwright not installed")


# ── WebPage stealth ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_webpage_stealth_patches_navigator_webdriver() -> None:
    """navigator.webdriver must be undefined (not True) when stealth=True."""
    from gantrygraph.perception.web import WebPage

    async with WebPage(headless=True, stealth=True) as web:
        page = web.page
        assert page is not None
        result = await page.evaluate("() => navigator.webdriver")
        assert result is None or result is False


@pytest.mark.asyncio
async def test_webpage_stealth_sets_realistic_useragent() -> None:
    from gantrygraph.perception.web import WebPage

    async with WebPage(headless=True, stealth=True) as web:
        page = web.page
        assert page is not None
        ua: str = await page.evaluate("() => navigator.userAgent")
        assert "HeadlessChrome" not in ua
        assert "Chrome" in ua


@pytest.mark.asyncio
async def test_webpage_stealth_plugins_non_empty() -> None:
    from gantrygraph.perception.web import WebPage

    async with WebPage(headless=True, stealth=True) as web:
        page = web.page
        assert page is not None
        count: int = await page.evaluate("() => navigator.plugins.length")
        assert count > 0


@pytest.mark.asyncio
async def test_webpage_stealth_languages_non_empty() -> None:
    from gantrygraph.perception.web import WebPage

    async with WebPage(headless=True, stealth=True) as web:
        page = web.page
        assert page is not None
        langs: list[str] = await page.evaluate("() => Array.from(navigator.languages)")
        assert len(langs) > 0
        assert "en" in langs[0]


@pytest.mark.asyncio
async def test_webpage_no_stealth_webdriver_exposed() -> None:
    """Without stealth, Playwright sets navigator.webdriver = true."""
    from gantrygraph.perception.web import WebPage

    async with WebPage(headless=True, stealth=False) as web:
        page = web.page
        assert page is not None
        result = await page.evaluate("() => navigator.webdriver")
        assert result is True


@pytest.mark.asyncio
async def test_webpage_observe_returns_result() -> None:
    """observe() works end-to-end with stealth enabled."""
    from gantrygraph.perception.web import WebPage

    async with WebPage(headless=True, stealth=True) as web:
        result = await web.observe()
        assert result.screenshot_b64 is not None
        assert len(result.screenshot_b64) > 0


# ── BrowserTools stealth ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_browser_tools_stealth_patches_webdriver() -> None:
    from gantrygraph.actions.browser import BrowserTools

    bt = BrowserTools(headless=True, stealth=True)
    try:
        page = await bt._ensure_browser()
        result = await page.evaluate("() => navigator.webdriver")
        assert result is None or result is False
    finally:
        await bt.close()


@pytest.mark.asyncio
async def test_browser_tools_navigate_succeeds() -> None:
    from gantrygraph.actions.browser import BrowserTools

    bt = BrowserTools(headless=True, stealth=True)
    try:
        tools = {t.name: t for t in bt.get_tools()}
        result: str = await tools["browser_navigate"].ainvoke({"url": "about:blank"})
        assert "about:blank" in result
    finally:
        await bt.close()


@pytest.mark.asyncio
async def test_browser_tools_get_url() -> None:
    from gantrygraph.actions.browser import BrowserTools

    bt = BrowserTools(headless=True, stealth=True)
    try:
        tools = {t.name: t for t in bt.get_tools()}
        await tools["browser_navigate"].ainvoke({"url": "about:blank"})
        url: str = await tools["browser_get_url"].ainvoke({})
        assert "about:blank" in url
    finally:
        await bt.close()


# ── Shared browser (WebPage + BrowserTools) ───────────────────────────────────


@pytest.mark.asyncio
async def test_shared_browser_uses_same_page() -> None:
    """BrowserTools(web_page=...) shares the exact same Page object as WebPage."""
    from gantrygraph.actions.browser import BrowserTools
    from gantrygraph.perception.web import WebPage

    web = WebPage(headless=True, stealth=True)
    bt = BrowserTools(headless=True, stealth=True, web_page=web)
    try:
        perception_page = await web._ensure_page()
        action_page = await bt._ensure_browser()
        assert perception_page is action_page
    finally:
        await web.close()


# ── Lifecycle: close is idempotent ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_webpage_close_idempotent() -> None:
    from gantrygraph.perception.web import WebPage

    web = WebPage(headless=True, stealth=True)
    await web._ensure_page()
    await web.close()
    await web.close()  # second close must not raise


@pytest.mark.asyncio
async def test_browser_tools_close_idempotent() -> None:
    from gantrygraph.actions.browser import BrowserTools

    bt = BrowserTools(headless=True, stealth=True)
    await bt._ensure_browser()
    await bt.close()
    await bt.close()  # second close must not raise
