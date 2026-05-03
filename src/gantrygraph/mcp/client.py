"""MCP client — connects to an MCP server and exposes its tools as LangChain tools."""
from __future__ import annotations

import shlex
from contextlib import AsyncExitStack
from typing import Any

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field, create_model

from gantrygraph.core.base_mcp import BaseMCPConnector

try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    _HAS_MCP = True
except ImportError:
    _HAS_MCP = False

_INSTALL_MSG = "mcp is a core dependency — run: pip install 'gantrygraph'"

# JSON Schema → Python type map
_TYPE_MAP: dict[str, type] = {
    "string": str,
    "integer": int,
    "boolean": bool,
    "number": float,
    "array": list,
    "object": dict,
}


class MCPClient(BaseMCPConnector):
    """Connect to an MCP server process and expose its tools as LangChain tools.

    The server is started as a subprocess via stdio transport when the client
    is used as an async context manager.  ``GantryEngine`` manages this lifecycle
    automatically.

    Example — standalone::

        async with MCPClient("npx -y @modelcontextprotocol/server-filesystem /tmp") as c:
            tools = c.get_tools()
            result = await tools[0].ainvoke({"path": "/tmp"})

    Example — with engine (lifecycle managed automatically)::

        agent = GantryEngine(
            ...,
            tools=[MCPClient("npx -y @mcp/github")],
        )
    """

    def __init__(
        self,
        server_command: str,
        env: dict[str, str] | None = None,
    ) -> None:
        if not _HAS_MCP:
            raise ImportError(_INSTALL_MSG)
        self._command = server_command
        self._env = env
        self._tools: list[BaseTool] = []
        self._exit_stack = AsyncExitStack()
        self._session: ClientSession | None = None

    async def __aenter__(self) -> MCPClient:
        parts = shlex.split(self._command)
        params = StdioServerParameters(
            command=parts[0],
            args=parts[1:],
            env=self._env,
        )
        read, write = await self._exit_stack.enter_async_context(stdio_client(params))
        self._session = ClientSession(read, write)
        await self._exit_stack.enter_async_context(self._session)
        await self._session.initialize()

        result = await self._session.list_tools()
        self._tools = [
            _wrap_mcp_tool(t, self._session) for t in result.tools
        ]
        return self

    async def __aexit__(self, *args: object) -> None:
        await self._exit_stack.aclose()
        self._session = None
        self._tools = []

    def get_tools(self) -> list[BaseTool]:
        return list(self._tools)

    def __repr__(self) -> str:
        tool_names = [t.name for t in self._tools]
        return f"MCPClient(command={self._command!r}, tools={tool_names})"


def _wrap_mcp_tool(
    mcp_tool: Any,  # mcp.types.Tool
    session: Any,   # ClientSession
) -> StructuredTool:
    """Convert a single MCP tool descriptor into a LangChain StructuredTool."""
    name = mcp_tool.name
    description = mcp_tool.description or f"MCP tool: {name}"
    input_schema = mcp_tool.inputSchema or {}

    args_schema = _json_schema_to_pydantic(name, input_schema)

    async def _invoke(**kwargs: Any) -> str:
        result = await session.call_tool(name, arguments=kwargs)
        if result.isError:
            raise RuntimeError(f"MCP tool '{name}' returned an error: {result.content}")
        parts: list[str] = []
        for block in result.content:
            if hasattr(block, "text"):
                parts.append(block.text)
            else:
                parts.append(str(block))
        return "\n".join(parts)

    return StructuredTool.from_function(
        coroutine=_invoke,
        name=name,
        description=description,
        args_schema=args_schema,
    )


def _json_schema_to_pydantic(tool_name: str, schema: dict[str, Any]) -> type[BaseModel]:
    """Dynamically build a Pydantic model from an MCP tool's JSON Schema."""
    fields: dict[str, Any] = {}
    properties: dict[str, Any] = schema.get("properties", {})
    required: set[str] = set(schema.get("required", []))

    for prop_name, prop_schema in properties.items():
        json_type = prop_schema.get("type", "string")
        py_type: type = _TYPE_MAP.get(json_type, str)
        desc: str = prop_schema.get("description", "")

        if prop_name in required:
            fields[prop_name] = (py_type, Field(description=desc))
        else:
            default = prop_schema.get("default", None)
            fields[prop_name] = (py_type | None, Field(default=default, description=desc))

    if not fields:
        # Tool accepts no arguments — create a model with no fields
        return create_model(f"{tool_name}Args")

    return create_model(f"{tool_name}Args", **fields)
