from __future__ import annotations

from abc import ABC, abstractmethod

from langchain_core.tools import BaseTool


class BaseAction(ABC):
    """Abstract base class for an action set — a bundle of LangChain tools.

    Subclass this to expose a group of related tools to the agent.
    ``GantryEngine`` collects tools from all registered action sets,
    flattens them into one registry, and binds them to the LLM via
    ``llm.bind_tools()``.

    Grouping tools into a ``BaseAction`` subclass is the recommended
    pattern when your tools share state (e.g., an HTTP session, a DB
    connection) or when you want them to appear as a logical unit.
    For single stateless tools, use the ``@gantry_tool`` decorator instead.

    Optionally override ``close()`` to release resources on shutdown.
    GantryEngine calls it automatically at the end of every run.

    **Minimal example — HTTP client tools:**

    .. code-block:: python

        import httpx
        from langchain_core.tools import StructuredTool
        from gantrygraph.core.base_action import BaseAction

        class HTTPTools(BaseAction):
            def __init__(self, base_url: str) -> None:
                self._client = httpx.AsyncClient(base_url=base_url)

            def get_tools(self) -> list:
                async def http_get(path: str) -> str:
                    \"\"\"Make a GET request to *path* and return the response body.\"\"\"
                    r = await self._client.get(path)
                    return r.text

                async def http_post(path: str, body: str) -> str:
                    \"\"\"POST *body* (JSON string) to *path*, return the response.\"\"\"
                    r = await self._client.post(path, content=body)
                    return r.text

                return [
                    StructuredTool.from_function(coroutine=http_get,  name="http_get"),
                    StructuredTool.from_function(coroutine=http_post, name="http_post"),
                ]

            async def close(self) -> None:
                await self._client.aclose()

        agent = GantryEngine(llm=my_llm, tools=[HTTPTools("https://api.example.com")])

    **Using @gantry_tool for simple stateless tools:**

    .. code-block:: python

        from gantrygraph import gantry_tool

        @gantry_tool
        def calculator(expression: str) -> str:
            \"\"\"Evaluate a Python math expression and return the result.\"\"\"
            return str(eval(expression))  # noqa: S307

        agent = GantryEngine(llm=my_llm, tools=[calculator])
    """

    @abstractmethod
    def get_tools(self) -> list[BaseTool]:
        """Return the LangChain tools this action set provides."""
        ...

    async def close(self) -> None:  # noqa: B027 — intentional no-op default
        """Optional cleanup hook called by GantryEngine on shutdown."""
