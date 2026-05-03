"""Web page perception via Playwright — screenshot + accessibility tree.

Requires the ``[browser]`` extra::

    pip install gantrygraph[browser]
    playwright install chromium
"""

from __future__ import annotations

import base64
import json
from typing import Literal

from gantrygraph.core.base_perception import BasePerception
from gantrygraph.core.events import PerceptionResult

try:
    from playwright.async_api import (  # type: ignore[import-not-found]
        Browser,
        Page,
        async_playwright,
    )

    _HAS_PLAYWRIGHT = True
except ImportError:
    _HAS_PLAYWRIGHT = False

_INSTALL_MSG = (
    "WebPage requires the [browser] extra: "
    "pip install 'gantrygraph[browser]' && playwright install chromium"
)


class WebPage(BasePerception):
    """Capture a web page's screenshot and accessibility tree.

    ``WebPage`` manages its own browser lifecycle and is an async context
    manager for standalone usage.  When passed to ``GantryEngine``, the
    engine calls ``close()`` automatically.

    Example::

        # Standalone
        async with WebPage(url="https://example.com") as w:
            result = await w.observe()

        # With engine
        agent = GantryEngine(
            ...,
            perception=WebPage(url="https://example.com"),
        )
    """

    def __init__(
        self,
        url: str | None = None,
        browser_type: Literal["chromium", "firefox", "webkit"] = "chromium",
        headless: bool = True,
        include_screenshot: bool = True,
        include_accessibility: bool = True,
    ) -> None:
        if not _HAS_PLAYWRIGHT:
            raise ImportError(_INSTALL_MSG)
        self._url = url
        self._browser_type = browser_type
        self._headless = headless
        self._include_screenshot = include_screenshot
        self._include_accessibility = include_accessibility
        self._browser: Browser | None = None
        self._page: Page | None = None
        self._playwright_ctx = None

    async def __aenter__(self) -> WebPage:
        await self._launch()
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.close()

    async def _launch(self) -> None:
        ctx = async_playwright()
        self._playwright_ctx = await ctx.__aenter__()
        launcher = getattr(self._playwright_ctx, self._browser_type)
        self._browser = await launcher.launch(headless=self._headless)
        self._page = await self._browser.new_page()
        if self._url:
            await self._page.goto(self._url, wait_until="domcontentloaded")

    async def close(self) -> None:
        if self._browser:
            await self._browser.close()
            self._browser = None
            self._page = None
        if self._playwright_ctx:
            await self._playwright_ctx.__aexit__(None, None, None)
            self._playwright_ctx = None

    async def _ensure_page(self) -> Page:
        """Return the active Playwright page, launching the browser if needed."""
        if self._page is None:
            await self._launch()
        assert self._page is not None
        return self._page

    async def observe(self) -> PerceptionResult:
        page = await self._ensure_page()

        screenshot_b64: str | None = None
        accessibility_tree: str | None = None

        if self._include_screenshot:
            png_bytes = await page.screenshot(type="png")
            screenshot_b64 = base64.b64encode(png_bytes).decode("ascii")

        if self._include_accessibility:
            snapshot = await page.accessibility.snapshot()
            if snapshot:
                accessibility_tree = json.dumps(snapshot, indent=2)

        viewport = page.viewport_size or {"width": 1280, "height": 720}
        return PerceptionResult(
            screenshot_b64=screenshot_b64,
            accessibility_tree=accessibility_tree,
            url=page.url,
            width=viewport["width"],
            height=viewport["height"],
        )

    @property
    def page(self) -> Page | None:
        """Direct access to the Playwright Page for advanced use."""
        return self._page
