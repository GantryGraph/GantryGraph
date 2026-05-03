"""Unit tests for MCP module — schema conversion and tool generation."""
from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.tools import BaseTool

from gantrygraph.mcp.client import MCPClient, _json_schema_to_pydantic, _make_langchain_tool

# ── _json_schema_to_pydantic ─────────────────────────────────────────────────

def test_json_schema_to_pydantic_simple() -> None:
    schema = {
        "type": "object",
        "properties": {
            "message": {"type": "string", "description": "A message"},
        },
        "required": ["message"],
    }
    model = _json_schema_to_pydantic("echo", schema)
    instance = model(message="hello")
    assert instance.message == "hello"  # type: ignore[attr-defined]


def test_json_schema_to_pydantic_optional_field() -> None:
    schema = {
        "type": "object",
        "properties": {
            "limit": {"type": "integer", "default": 10},
        },
        "required": [],
    }
    model = _json_schema_to_pydantic("list_items", schema)
    # Should work without providing limit (has default)
    instance = model()
    assert instance.limit is None or instance.limit == 10  # type: ignore[attr-defined]


def test_json_schema_to_pydantic_empty_schema() -> None:
    schema: dict[str, Any] = {}
    model = _json_schema_to_pydantic("no_args_tool", schema)
    # No fields — should still be a valid Pydantic model
    instance = model()
    assert instance is not None


def test_json_schema_to_pydantic_multiple_types() -> None:
    schema = {
        "properties": {
            "name": {"type": "string"},
            "count": {"type": "integer"},
            "active": {"type": "boolean"},
            "score": {"type": "number"},
        },
        "required": ["name", "count", "active", "score"],
    }
    model = _json_schema_to_pydantic("complex", schema)
    instance = model(name="test", count=5, active=True, score=3.14)  # type: ignore[call-arg]
    assert instance.name == "test"  # type: ignore[attr-defined]
    assert instance.count == 5  # type: ignore[attr-defined]


# ── _make_langchain_tool ──────────────────────────────────────────────────────

def _make_mock_mcp_tool(name: str, description: str, schema: dict[str, Any]) -> Any:
    tool = MagicMock()
    tool.name = name
    tool.description = description
    tool.inputSchema = schema
    return tool


@pytest.mark.asyncio
async def test_make_langchain_tool_calls_session() -> None:
    mock_session = MagicMock()
    content_block = MagicMock()
    content_block.text = "result text"
    call_result = MagicMock()
    call_result.isError = False
    call_result.content = [content_block]
    mock_session.call_tool = AsyncMock(return_value=call_result)

    mcp_tool = _make_mock_mcp_tool(
        "echo",
        "Echo a message",
        {"properties": {"message": {"type": "string"}}, "required": ["message"]},
    )
    lc_tool = _make_langchain_tool(mcp_tool, mock_session)
    assert isinstance(lc_tool, BaseTool)
    assert lc_tool.name == "echo"

    result = await lc_tool.ainvoke({"message": "hi"})
    assert "result text" in result
    mock_session.call_tool.assert_called_once_with("echo", arguments={"message": "hi"})


@pytest.mark.asyncio
async def test_make_langchain_tool_raises_on_error() -> None:
    mock_session = MagicMock()
    error_result = MagicMock()
    error_result.isError = True
    error_result.content = [MagicMock(text="something broke")]
    mock_session.call_tool = AsyncMock(return_value=error_result)

    mcp_tool = _make_mock_mcp_tool("broken_tool", "A broken tool", {})
    lc_tool = _make_langchain_tool(mcp_tool, mock_session)

    with pytest.raises(RuntimeError, match="broken_tool"):
        await lc_tool.ainvoke({})


def test_make_langchain_tool_fallback_description() -> None:
    mock_session = MagicMock()
    mcp_tool = MagicMock()
    mcp_tool.name = "mystery_tool"
    mcp_tool.description = None
    mcp_tool.inputSchema = {}

    lc_tool = _make_langchain_tool(mcp_tool, mock_session)
    assert "mystery_tool" in lc_tool.description


# ── MCPClient lifecycle (mocked) ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_mcp_client_get_tools_empty_before_enter() -> None:
    """get_tools() returns empty list before __aenter__."""
    client = MCPClient("echo test")
    assert client.get_tools() == []


@pytest.mark.asyncio
async def test_mcp_client_context_manager_lifecycle() -> None:
    """Full lifecycle: enter → tools available → exit → tools cleared."""
    mock_tool = MagicMock()
    mock_tool.name = "test_tool"
    mock_tool.description = "A test"
    mock_tool.inputSchema = {}

    mock_list_result = MagicMock()
    mock_list_result.tools = [mock_tool]

    mock_session = AsyncMock()
    mock_session.initialize = AsyncMock()
    mock_session.list_tools = AsyncMock(return_value=mock_list_result)
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=None)

    mock_streams = (AsyncMock(), AsyncMock())

    with (
        patch("gantrygraph.mcp.client.stdio_client") as mock_stdio,
        patch("gantrygraph.mcp.client.ClientSession", return_value=mock_session),
    ):
        mock_stdio.return_value.__aenter__ = AsyncMock(return_value=mock_streams)
        mock_stdio.return_value.__aexit__ = AsyncMock(return_value=None)

        client = MCPClient("echo server")
        async with client as c:
            tools = c.get_tools()
            assert len(tools) == 1
            assert tools[0].name == "test_tool"

        assert c.get_tools() == []
