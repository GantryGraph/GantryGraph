"""Playwright-based browser automation tools.

Requires the ``[browser]`` extra::

    pip install gantrygraph[browser]
    playwright install chromium
"""

from __future__ import annotations

from typing import Any, Literal

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

from gantrygraph.core.base_action import BaseAction

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
    """

    def __init__(
        self,
        browser_type: Literal["chromium", "firefox", "webkit"] = "chromium",
        headless: bool = True,
        web_page: Any = None,  # WebPage | None — Any avoids circular import
    ) -> None:
        if not _HAS_PLAYWRIGHT:
            raise ImportError(_INSTALL_MSG)
        self._browser_type = browser_type
        self._headless = headless
        self._web_page = web_page  # shared WebPage for coordinated perception+action
        self._browser: Browser | None = None
        self._page: Page | None = None
        self._playwright_ctx = None

    async def _ensure_browser(self) -> Page:
        # If sharing a WebPage, delegate browser lifecycle to it
        if self._web_page is not None:
            return await self._web_page._ensure_page()
        if self._page is None:
            ctx = async_playwright()
            self._playwright_ctx = await ctx.__aenter__()
            launcher = getattr(self._playwright_ctx, self._browser_type)
            self._browser = await launcher.launch(headless=self._headless)
            self._page = await self._browser.new_page()
        return self._page

    async def close(self) -> None:
        if self._web_page is not None:
            return  # lifecycle managed by the shared WebPage
        if self._browser:
            await self._browser.close()
            self._browser = None
            self._page = None
        if self._playwright_ctx:
            await self._playwright_ctx.__aexit__(None, None, None)
            self._playwright_ctx = None

    def get_tools(self) -> list[BaseTool]:
        return [
            self._navigate_tool(),
            self._click_tool(),
            self._fill_tool(),
            self._get_text_tool(),
            self._get_url_tool(),
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
            await page.click(selector, timeout=5000)
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
            await page.fill(selector, value, timeout=5000)
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
            return text[:4000]  # cap to avoid huge token usage

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
