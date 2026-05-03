# MCP Integration

[Model Context Protocol](https://modelcontextprotocol.io) (MCP) is an open standard
for connecting LLMs to external tools and data sources.
GantryGraph wraps any MCP server in a `BaseMCPConnector` that starts a subprocess,
discovers tools automatically, and shuts down cleanly when the run ends.

## MCPClient — single server

```python
from gantrygraph import GantryEngine
from gantrygraph.mcp import MCPClient
from langchain_anthropic import ChatAnthropic

agent = GantryEngine(
    llm=ChatAnthropic(model="claude-sonnet-4-6"),
    tools=[
        MCPClient("npx -y @modelcontextprotocol/server-filesystem /tmp"),
    ],
    max_steps=20,
)

result = agent.run("List every file modified today and show me the 3 largest.")
```

The subprocess starts automatically when `arun()` is called and is terminated
when the run completes — even on exceptions.

## MCPToolRegistry — multiple servers

```python
from gantrygraph.mcp import MCPToolRegistry

registry = MCPToolRegistry([
    MCPClient("npx -y @modelcontextprotocol/server-filesystem /workspace"),
    MCPClient("npx -y @modelcontextprotocol/server-github"),
    MCPClient("npx -y @modelcontextprotocol/server-postgres postgresql://..."),
])

agent = GantryEngine(llm=..., tools=[registry])
```

All tools from all servers are flattened into a single list and de-duplicated
by name. Useful when you want one `GantryEngine` to straddle multiple domains.

## mcp_agent preset

```python
from gantrygraph.presets import mcp_agent
from langchain_anthropic import ChatAnthropic

agent = mcp_agent(
    ChatAnthropic(model="claude-sonnet-4-6"),
    "npx -y @modelcontextprotocol/server-github",
    "npx -y @modelcontextprotocol/server-slack",
)
result = agent.run("List the open PRs in the monorepo and post a summary to #engineering.")
```

## Lifecycle management

MCP connectors are async context managers.
`GantryEngine` enters and exits them automatically in `_lifecycle()`.
If you use the graph directly, wrap with an explicit `async with`:

```python
async with MCPClient("npx -y @mcp/github") as client:
    tools = client.get_tools()
    # use tools...
```

## Writing an in-process connector

For tools you want to expose without a subprocess:

```python
from langchain_core.tools import BaseTool
from gantrygraph import BaseMCPConnector

class InProcessConnector(BaseMCPConnector):
    """Exposes internal services as MCP-style tools."""

    async def __aenter__(self) -> "InProcessConnector":
        await self._service.connect()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self._service.disconnect()

    def get_tools(self) -> list[BaseTool]:
        return [my_tool_1, my_tool_2]
```
