"""Integration tests — real MCP server subprocess via mock_mcp_server.py."""
from __future__ import annotations

import sys

import pytest

from gantrygraph.mcp.client import MCPClient
from gantrygraph.mcp.registry import MCPToolRegistry

SERVER_CMD = f"{sys.executable} tests/integration/fixtures/mock_mcp_server.py"


@pytest.mark.asyncio
async def test_mcp_client_connects_and_discovers_tools() -> None:
    async with MCPClient(SERVER_CMD) as client:
        tools = client.get_tools()
        tool_names = {t.name for t in tools}
        assert "echo" in tool_names
        assert "add" in tool_names


@pytest.mark.asyncio
async def test_mcp_client_echo_tool() -> None:
    async with MCPClient(SERVER_CMD) as client:
        tools = {t.name: t for t in client.get_tools()}
        result = await tools["echo"].ainvoke({"message": "hello from claw"})
        assert "ECHO: hello from claw" in result


@pytest.mark.asyncio
async def test_mcp_client_add_tool() -> None:
    async with MCPClient(SERVER_CMD) as client:
        tools = {t.name: t for t in client.get_tools()}
        result = await tools["add"].ainvoke({"a": 3, "b": 7})
        assert "10" in result


@pytest.mark.asyncio
async def test_mcp_registry_with_two_clients() -> None:
    registry = MCPToolRegistry([
        MCPClient(SERVER_CMD),
        MCPClient(SERVER_CMD),
    ])
    async with registry:
        tools = registry.get_tools()
        # Two servers, each with 2 tools = 4 total
        assert len(tools) == 4


@pytest.mark.asyncio
async def test_mcp_client_repr_after_enter() -> None:
    async with MCPClient(SERVER_CMD) as client:
        r = repr(client)
        assert "MCPClient" in r
        assert "echo" in r


@pytest.mark.asyncio
async def test_mcp_client_tools_cleared_after_exit() -> None:
    client = MCPClient(SERVER_CMD)
    async with client:
        assert len(client.get_tools()) > 0
    assert client.get_tools() == []
