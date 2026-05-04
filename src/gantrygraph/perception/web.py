"""Web page perception via Playwright — screenshot + accessibility tree.

Requires the ``[browser]`` extra::

    pip install gantrygraph[browser]
    playwright install chromium
"""

from __future__ import annotations

import base64
import json
from typing import TYPE_CHECKING, Literal

from gantrygraph.core.base_perception import BasePerception
from gantrygraph.core.events import PerceptionResult

if TYPE_CHECKING:
    from playwright.async_api import BrowserContext, Playwright

try:
    from playwright.async_api import (
        Browser,
        BrowserContext,
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

# Keep in sync with actions/browser.py — same patches applied by both.
_STEALTH_INIT_SCRIPT = """
(() => {
    Object.defineProperty(navigator, 'webdriver', { get: () => undefined });

    if (!window.chrome) {
        window.chrome = { runtime: {}, loadTimes: function(){}, csi: function(){}, app: {} };
    }

    if (navigator.plugins.length === 0) {
        Object.defineProperty(navigator, 'plugins', {
            get: () => ['PDF Viewer', 'Chrome PDF Viewer', 'Chromium PDF Viewer',
                        'Microsoft Edge PDF Viewer', 'WebKit built-in PDF'].map(name => ({ name })),
        });
    }

    Object.defineProperty(navigator, 'languages', { get: () => ['en-US', 'en'] });

    if (window.navigator.permissions) {
        const _origQuery = window.navigator.permissions.query.bind(window.navigator.permissions);
        window.navigator.permissions.query = (params) =>
            params.name === 'notifications'
                ? Promise.resolve({ state: 'default', onchange: null })
                : _origQuery(params);
    }
})();
"""


def _downscale_png(png_bytes: bytes, max_size: tuple[int, int]) -> bytes:
    """Resize PNG bytes to fit within *max_size*, preserving aspect ratio."""
    import io

    from PIL import Image

    img = Image.open(io.BytesIO(png_bytes))
    max_w, max_h = max_size
    if img.width <= max_w and img.height <= max_h:
        return png_bytes
    ratio = min(max_w / img.width, max_h / img.height)
    new_w, new_h = int(img.width * ratio), int(img.height * ratio)
    img = img.resize((new_w, new_h), resample=3)  # LANCZOS = 3
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


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
        stealth: bool = True,
        include_screenshot: bool = True,
        include_accessibility: bool = True,
        vision_mode: Literal["high", "low"] = "high",
    ) -> None:
        if not _HAS_PLAYWRIGHT:
            raise ImportError(_INSTALL_MSG)
        self._url = url
        self._browser_type = browser_type
        self._headless = headless
        self._stealth = stealth
        self._include_screenshot = include_screenshot
        self._include_accessibility = include_accessibility
        self._vision_mode = vision_mode
        self._browser: Browser | None = None
        self._context: BrowserContext | None = None
        self._page: Page | None = None
        self._playwright_ctx: Playwright | None = None

    async def __aenter__(self) -> WebPage:
        await self._launch()
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.close()

    async def _launch(self) -> None:
        self._playwright_ctx = await async_playwright().start()
        launcher = getattr(self._playwright_ctx, self._browser_type)
        launch_args = (
            ["--disable-blink-features=AutomationControlled"] if self._stealth else []
        )
        self._browser = await launcher.launch(
            headless=self._headless, args=launch_args
        )
        if self._stealth:
            self._context = await self._browser.new_context(
                user_agent=(
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/125.0.0.0 Safari/537.36"
                ),
                viewport={"width": 1366, "height": 768},
                locale="en-US",
                timezone_id="America/New_York",
                extra_http_headers={"Accept-Language": "en-US,en;q=0.9"},
            )
            await self._context.add_init_script(_STEALTH_INIT_SCRIPT)
            self._page = await self._context.new_page()
        else:
            self._page = await self._browser.new_page()
        if self._url:
            await self._page.goto(self._url, wait_until="domcontentloaded")

    async def close(self) -> None:
        if self._context:
            await self._context.close()
            self._context = None
        if self._browser:
            await self._browser.close()
            self._browser = None
            self._page = None
        if self._playwright_ctx:
            await self._playwright_ctx.stop()
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
            if self._vision_mode == "low":
                png_bytes = _downscale_png(png_bytes, (1280, 720))
            screenshot_b64 = base64.b64encode(png_bytes).decode("ascii")

        if self._include_accessibility:
            accessibility = getattr(page, "accessibility", None)
            if accessibility is not None:
                snapshot = await accessibility.snapshot()
                if snapshot:
                    accessibility_tree = json.dumps(snapshot, indent=2)

        viewport = page.viewport_size or {"width": 1366, "height": 768}
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
