"""Playwright-based browser automation tools.

Requires the ``[browser]`` extra::

    pip install gantrygraph[browser]
    playwright install chromium
"""

from __future__ import annotations

import asyncio
import os
import random
from typing import TYPE_CHECKING, Any, Literal

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

from gantrygraph import _stealth
from gantrygraph.core.base_action import BaseAction

if TYPE_CHECKING:
    from playwright.async_api import Browser, BrowserContext, Page, Playwright

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
    "BrowserTools requires the [browser] extra: "
    "pip install 'gantrygraph[browser]' && playwright install chromium"
)


class BrowserTools(BaseAction):
    """Playwright-powered browser tools for reliable web automation.

    Clicking by element selector is more robust than pixel-coordinate
    clicking when the agent needs to interact with web pages, because
    it handles dynamic layouts, scrolling, and responsive designs.

    The browser is started lazily on first use and must be closed via
    ``await tools.close()``.  ``GantryEngine`` calls this automatically.

    **Standalone usage:**

    .. code-block:: python

        from gantrygraph.actions import BrowserTools
        agent = GantryEngine(..., tools=[BrowserTools(headless=True)])

    **Shared browser with WebPage perception (recommended):**

    Pass a ``WebPage`` instance so both perception and actions operate
    on the *same* browser page.  Without this, ``WebPage`` and
    ``BrowserTools`` each open an independent browser — actions in one
    are invisible to the other.

    .. code-block:: python

        from gantrygraph.perception import WebPage
        from gantrygraph.actions import BrowserTools

        web = WebPage(url="https://example.com")
        agent = GantryEngine(
            llm=my_llm,
            perception=web,
            tools=[BrowserTools(web_page=web)],
        )

    Or use the ``browser_agent`` preset which wires this up automatically::

        from gantrygraph.presets import browser_agent
        agent = browser_agent(llm, start_url="https://example.com")

    **Persistent profile (e.g. WhatsApp Web, logged-in sites):**

    When ``profile_dir`` is set, the browser stores cookies, localStorage
    and IndexedDB across sessions.  Log in once manually (or let the agent
    handle the QR scan / login flow on the first run) and subsequent runs
    will reuse the saved session.

    .. code-block:: python

        tools = BrowserTools(
            headless=False,          # show window for first login
            profile_dir="~/.gantrygraph/profiles/whatsapp",
        )
    """

    def __init__(
        self,
        browser_type: Literal["chromium", "firefox", "webkit"] = "chromium",
        headless: bool = True,
        stealth: bool = True,
        profile_dir: str | None = None,
        web_page: Any = None,  # WebPage | None — Any avoids circular import
    ) -> None:
        if not _HAS_PLAYWRIGHT:
            raise ImportError(_INSTALL_MSG)
        self._browser_type = browser_type
        self._headless = headless
        self._stealth = stealth
        self._profile_dir = profile_dir
        self._web_page = web_page
        self._browser: Browser | None = None
        self._context: BrowserContext | None = None
        self._page: Page | None = None
        self._playwright_ctx: Playwright | None = None

    async def _ensure_browser(self) -> Page:
        if self._web_page is not None:
            page: Page = await self._web_page._ensure_page()
            return page
        if self._page is None:
            self._playwright_ctx = await async_playwright().start()
            launcher = getattr(self._playwright_ctx, self._browser_type)

            if self._profile_dir is not None:
                # Persistent profile — launch_persistent_context returns a context directly.
                profile_path = os.path.expanduser(self._profile_dir)
                os.makedirs(profile_path, exist_ok=True)
                launch_kw: dict[str, Any] = {
                    "headless": self._headless,
                    "args": _stealth.LAUNCH_ARGS if self._stealth else [],
                }
                if self._stealth:
                    launch_kw.update(_stealth.context_kwargs())
                self._context = await launcher.launch_persistent_context(
                    profile_path, **launch_kw
                )
                if self._stealth:
                    await _stealth.apply_to_context(self._context)
                self._page = (
                    self._context.pages[0]
                    if self._context.pages
                    else await self._context.new_page()
                )
            else:
                self._browser = await launcher.launch(
                    headless=self._headless,
                    args=_stealth.LAUNCH_ARGS if self._stealth else [],
                )
                if self._stealth:
                    self._context = await self._browser.new_context(
                        **_stealth.context_kwargs()
                    )
                    await _stealth.apply_to_context(self._context)
                    self._page = await self._context.new_page()
                else:
                    self._page = await self._browser.new_page()

        assert self._page is not None
        return self._page

    async def close(self) -> None:
        if self._web_page is not None:
            return  # lifecycle managed by the shared WebPage
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

    def get_tools(self) -> list[BaseTool]:
        return [
            self._navigate_tool(),
            self._click_tool(),
            self._click_text_tool(),
            self._fill_tool(),
            self._get_text_tool(),
            self._get_url_tool(),
            self._scroll_tool(),
            self._evaluate_tool(),
            self._wait_for_selector_tool(),
        ]

    def _navigate_tool(self) -> BaseTool:
        ensure = self._ensure_browser

        class _Args(BaseModel):
            url: str = Field(description="Full URL to navigate to.")

        async def _navigate(url: str) -> str:
            page = await ensure()
            response = await page.goto(url, wait_until="domcontentloaded")
            status = response.status if response else "unknown"
            return f"Navigated to {url} (status {status})."

        return StructuredTool.from_function(
            coroutine=_navigate,
            name="browser_navigate",
            description="Open a URL in the browser.",
            args_schema=_Args,
        )

    def _click_tool(self) -> BaseTool:
        ensure = self._ensure_browser

        class _Args(BaseModel):
            selector: str = Field(description="CSS or XPath selector of the element to click.")

        async def _click(selector: str) -> str:
            page = await ensure()
            await asyncio.sleep(random.uniform(0.05, 0.25))
            await page.click(selector, timeout=5000)
            await asyncio.sleep(random.uniform(0.05, 0.15))
            return f"Clicked element matching '{selector}'."

        return StructuredTool.from_function(
            coroutine=_click,
            name="browser_click",
            description="Click an element on the page by CSS/XPath selector.",
            args_schema=_Args,
        )

    def _fill_tool(self) -> BaseTool:
        ensure = self._ensure_browser

        class _Args(BaseModel):
            selector: str = Field(description="CSS selector of the input field.")
            value: str = Field(description="Text to fill into the field.")

        async def _fill(selector: str, value: str) -> str:
            page = await ensure()
            await asyncio.sleep(random.uniform(0.1, 0.3))
            await page.fill(selector, value, timeout=5000)
            await asyncio.sleep(random.uniform(0.05, 0.2))
            return f"Filled '{selector}' with '{value}'."

        return StructuredTool.from_function(
            coroutine=_fill,
            name="browser_fill",
            description="Fill a form field with text.",
            args_schema=_Args,
        )

    def _get_text_tool(self) -> BaseTool:
        ensure = self._ensure_browser

        class _Args(BaseModel):
            selector: str = Field(
                default="body",
                description="CSS selector — defaults to page body.",
            )

        async def _get_text(selector: str = "body") -> str:
            page = await ensure()
            element = await page.query_selector(selector)
            if element is None:
                return f"No element found for selector '{selector}'."
            text: str = await element.inner_text()
            return text[:4000]

        return StructuredTool.from_function(
            coroutine=_get_text,
            name="browser_get_text",
            description="Get the visible text content of an element or the whole page.",
            args_schema=_Args,
        )

    def _get_url_tool(self) -> BaseTool:
        ensure = self._ensure_browser

        async def _get_url() -> str:
            page = await ensure()
            return str(page.url)

        return StructuredTool.from_function(
            coroutine=_get_url,
            name="browser_get_url",
            description="Get the current URL of the browser.",
        )

    def _click_text_tool(self) -> BaseTool:
        ensure = self._ensure_browser

        class _Args(BaseModel):
            text: str = Field(description="Visible text of the button or link to click.")

        async def _click_text(text: str) -> str:
            page = await ensure()
            await asyncio.sleep(random.uniform(0.05, 0.25))
            clicked: bool = await page.evaluate(
                """(text) => {
                    const all = Array.from(document.querySelectorAll('button, a, [role="button"]'));
                    const el = all.find(e => e.innerText.trim() === text
                                           || e.textContent.trim() === text);
                    if (!el) return false;
                    el.click();
                    return true;
                }""",
                text,
            )
            if not clicked:
                return f"No clickable element found with text '{text}'."
            await asyncio.sleep(random.uniform(0.1, 0.3))
            return f"Clicked element with text '{text}'."

        return StructuredTool.from_function(
            coroutine=_click_text,
            name="browser_click_text",
            description=(
                "Click a button, link, or interactive element by its visible text label. "
                "Use this when browser_click fails with a selector — it searches the whole "
                "page for any clickable element whose text matches exactly (e.g. 'Rifiuta tutto', "
                "'Accept all', 'Submit')."
            ),
            args_schema=_Args,
        )

    def _evaluate_tool(self) -> BaseTool:
        ensure = self._ensure_browser

        class _Args(BaseModel):
            script: str = Field(
                description=(
                    "JavaScript expression to evaluate. Return value is serialised to string."
                )
            )

        async def _evaluate(script: str) -> str:
            page = await ensure()
            result = await page.evaluate(script)
            return str(result)[:2000]

        return StructuredTool.from_function(
            coroutine=_evaluate,
            name="browser_evaluate",
            description=(
                "Execute a JavaScript expression in the current page and return the result. "
                "Use as a last resort when CSS selectors fail — e.g. to inspect the DOM, "
                "click elements by text, or read hidden attributes."
            ),
            args_schema=_Args,
        )

    def _scroll_tool(self) -> BaseTool:
        ensure = self._ensure_browser

        class _Args(BaseModel):
            direction: Literal["down", "up", "top", "bottom"] = Field(
                default="down",
                description=(
                    "Scroll direction: 'down' one viewport, 'up' one viewport,"
                    " 'top' to start, 'bottom' to end."
                ),
            )
            amount: int = Field(
                default=600,
                description=(
                    "Pixels to scroll for 'up'/'down' (ignored for 'top'/'bottom'). Default 600."
                ),
            )

        async def _scroll(direction: str = "down", amount: int = 600) -> str:
            page = await ensure()
            if direction == "bottom":
                await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            elif direction == "top":
                await page.evaluate("window.scrollTo(0, 0)")
            elif direction == "up":
                await page.evaluate(f"window.scrollBy(0, -{amount})")
            else:
                await page.evaluate(f"window.scrollBy(0, {amount})")
            await asyncio.sleep(random.uniform(0.1, 0.2))
            scroll_y: int = await page.evaluate("window.scrollY")
            return f"Scrolled {direction}. Current scroll position: {scroll_y}px from top."

        return StructuredTool.from_function(
            coroutine=_scroll,
            name="browser_scroll",
            description=(
                "Scroll the page. Use 'down'/'up' to move one viewport at a time, "
                "'bottom' to jump to end of page, 'top' to return to start. "
                "Essential for reading long pages, infinite-scroll feeds, or "
                "revealing content below the fold."
            ),
            args_schema=_Args,
        )

    def _wait_for_selector_tool(self) -> BaseTool:
        ensure = self._ensure_browser

        class _Args(BaseModel):
            selector: str = Field(description="CSS or XPath selector to wait for.")
            timeout_ms: int = Field(
                default=10000,
                description="Maximum wait time in milliseconds (default 10 000).",
            )

        async def _wait_for_selector(selector: str, timeout_ms: int = 10000) -> str:
            page = await ensure()
            try:
                await page.wait_for_selector(selector, timeout=timeout_ms)
                return f"Element '{selector}' is now visible."
            except Exception as exc:
                return f"Timeout waiting for '{selector}': {exc}"

        return StructuredTool.from_function(
            coroutine=_wait_for_selector,
            name="browser_wait_for_selector",
            description=(
                "Wait until a CSS/XPath selector becomes visible on the page. "
                "Call this before browser_click when the page is still loading or "
                "elements appear after JavaScript rendering."
            ),
            args_schema=_Args,
        )
