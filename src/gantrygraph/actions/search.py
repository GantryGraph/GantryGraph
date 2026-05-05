"""Web search tool via Tavily API.

Requires the ``[search]`` extra::

    pip install 'gantrygraph[search]'

Get a free API key at https://tavily.com (1 000 free queries/month).
Pass it as ``TAVILY_API_KEY`` env var or directly to ``WebSearchTool``.
"""

from __future__ import annotations

import os
from typing import Literal

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

from gantrygraph.core.base_action import BaseAction

try:
    from tavily import AsyncTavilyClient

    _HAS_TAVILY = True
except ImportError:
    _HAS_TAVILY = False

_INSTALL_MSG = "WebSearchTool requires the [search] extra: pip install 'gantrygraph[search]'"


class WebSearchTool(BaseAction):
    """Web search via Tavily API — works where scraping search engines fails.

    Search engines (Google, Bing, DuckDuckGo) block Playwright bots with
    CAPTCHAs.  ``WebSearchTool`` uses the Tavily search API instead, which
    returns structured results (title, URL, snippet) without any browser.

    **Setup:**

    .. code-block:: bash

        pip install 'gantrygraph[search]'
        export TAVILY_API_KEY=tvly-...   # free at https://tavily.com

    **Usage:**

    .. code-block:: python

        from gantrygraph.actions.search import WebSearchTool

        agent = GantryEngine(
            llm=my_llm,
            tools=[WebSearchTool()],            # reads TAVILY_API_KEY from env
        )
        # Or alongside browser tools:
        agent = GantryEngine(
            llm=my_llm,
            tools=[WebSearchTool(), BrowserTools()],
        )

    The agent will call ``web_search`` for queries and ``BrowserTools`` to
    navigate to specific pages it finds.
    """

    def __init__(
        self,
        api_key: str | None = None,
        max_results: int = 5,
        search_depth: Literal["basic", "advanced"] = "basic",
    ) -> None:
        if not _HAS_TAVILY:
            raise ImportError(_INSTALL_MSG)
        self._api_key = api_key or os.environ.get("TAVILY_API_KEY", "")
        if not self._api_key:
            raise ValueError(
                "Tavily API key required. Pass api_key= or set TAVILY_API_KEY env var. "
                "Get a free key at https://tavily.com"
            )
        self._max_results = max_results
        self._search_depth = search_depth

    def get_tools(self) -> list[BaseTool]:
        return [self._search_tool()]

    def _search_tool(self) -> BaseTool:
        api_key = self._api_key
        max_results = self._max_results
        search_depth = self._search_depth

        class _Args(BaseModel):
            query: str = Field(description="Search query.")
            max_results: int = Field(
                default=max_results,
                description="Number of results to return (default 5).",
            )

        async def _search(query: str, max_results: int = max_results) -> str:
            client = AsyncTavilyClient(api_key=api_key)
            response = await client.search(
                query=query,
                search_depth=search_depth,
                max_results=max_results,
                include_answer=True,
            )
            lines: list[str] = []
            if response.get("answer"):
                lines.append(f"Summary: {response['answer']}\n")
            for i, r in enumerate(response.get("results", []), 1):
                lines.append(
                    f"{i}. {r.get('title', 'No title')}\n"
                    f"   URL: {r.get('url', '')}\n"
                    f"   {r.get('content', '')[:300]}"
                )
            return "\n\n".join(lines) if lines else "No results found."

        return StructuredTool.from_function(
            coroutine=_search,
            name="web_search",
            description=(
                "Search the web and return a list of relevant results with titles, URLs, "
                "and snippets. Use this instead of navigating to Google/Bing/DuckDuckGo — "
                "search engines block automated browsers. "
                "Then use browser_navigate to open specific pages from the results."
            ),
            args_schema=_Args,
        )
