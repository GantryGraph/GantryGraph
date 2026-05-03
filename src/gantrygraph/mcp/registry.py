"""MCPToolRegistry — manages multiple MCPClient instances as one group."""
from __future__ import annotations

from contextlib import AsyncExitStack
from types import TracebackType

from langchain_core.tools import BaseTool

from gantrygraph.core.base_mcp import BaseMCPConnector
from gantrygraph.mcp.client import MCPClient


class MCPToolRegistry(BaseMCPConnector):
    """Manage multiple ``MCPClient`` instances as a single pluggable unit.

    Useful when an agent needs tools from several MCP servers — pass
    the registry to ``GantryEngine`` instead of individual clients.

    Example::

        registry = MCPToolRegistry([
            MCPClient("npx -y @mcp/github"),
            MCPClient("npx -y @mcp/sqlite ./db.sqlite"),
        ])
        agent = GantryEngine(..., tools=[registry])
    """

    def __init__(self, clients: list[MCPClient]) -> None:
        self._clients = clients
        self._stack = AsyncExitStack()

    async def __aenter__(self) -> MCPToolRegistry:
        for client in self._clients:
            await self._stack.enter_async_context(client)
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self._stack.aclose()

    def get_tools(self) -> list[BaseTool]:
        return [tool for client in self._clients for tool in client.get_tools()]

    def __repr__(self) -> str:
        return f"MCPToolRegistry({len(self._clients)} clients, {len(self.get_tools())} tools)"
