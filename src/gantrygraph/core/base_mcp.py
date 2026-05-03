from __future__ import annotations

from abc import ABC, abstractmethod
from types import TracebackType

from langchain_core.tools import BaseTool


class BaseMCPConnector(ABC):
    """Abstract base class for MCP server connections.

    MCP connectors own a subprocess (or network connection) lifetime.
    They *must* be used as async context managers: the server starts in
    ``__aenter__`` and shuts down in ``__aexit__``.

    ``GantryEngine`` enters all connectors automatically inside its
    ``_lifecycle()`` context manager before building the graph, so user
    code never needs to manage the lifecycle manually.

    Implement this ABC when you need to wrap an MCP-compatible server that
    is not covered by the built-in ``MCPClient``.

    **Standalone usage (e.g., in scripts):**

    .. code-block:: python

        async with MCPClient("npx -y @mcp/github") as client:
            tools = client.get_tools()   # list[BaseTool]
            print([t.name for t in tools])

    **Passing to GantryEngine (lifecycle managed automatically):**

    .. code-block:: python

        agent = GantryEngine(
            llm=my_llm,
            tools=[MCPClient("npx -y @mcp/github")],
        )
        agent.run("Open a pull request")

    **Custom connector example (Python-based MCP server):**

    .. code-block:: python

        from types import TracebackType
        from langchain_core.tools import BaseTool, StructuredTool
        from gantrygraph.core.base_mcp import BaseMCPConnector

        class InProcessMCPConnector(BaseMCPConnector):
            \"\"\"Wraps an in-process FastMCP server (no subprocess).\"\"\"

            def __init__(self) -> None:
                self._tools: list[BaseTool] = []

            async def __aenter__(self) -> "InProcessMCPConnector":
                # spin up in-process server, discover tools
                self._tools = [
                    StructuredTool.from_function(
                        func=lambda x: x.upper(), name="shout",
                        description="Convert text to uppercase.",
                    )
                ]
                return self

            async def __aexit__(
                self,
                exc_type: type[BaseException] | None,
                exc_val: BaseException | None,
                exc_tb: TracebackType | None,
            ) -> None:
                self._tools = []   # cleanup

            def get_tools(self) -> list[BaseTool]:
                return self._tools
    """

    @abstractmethod
    async def __aenter__(self) -> BaseMCPConnector:
        """Start the MCP server subprocess and initialise the client session."""
        ...

    @abstractmethod
    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Shut down the MCP server subprocess."""
        ...

    @abstractmethod
    def get_tools(self) -> list[BaseTool]:
        """Return dynamically generated StructuredTool instances.

        Only valid *after* ``__aenter__`` has been called.
        """
        ...
