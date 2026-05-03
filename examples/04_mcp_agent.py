"""Example 4 — MCP tool servers.

Connects to the official MCP filesystem server and uses it to explore
a directory. The subprocess is started and stopped automatically.

Requires npx (Node.js):
    npm install -g npm   # ensure npx is available

Run:
    ANTHROPIC_API_KEY=sk-ant-... python examples/04_mcp_agent.py
"""
import asyncio

from langchain_anthropic import ChatAnthropic

from gantrygraph.presets import mcp_agent


async def main() -> None:
    agent = mcp_agent(
        ChatAnthropic(model="claude-sonnet-4-6"),
        "npx -y @modelcontextprotocol/server-filesystem /tmp",
        max_steps=10,
    )
    result = await agent.arun(
        "Create a file /tmp/hello.txt with the content 'Hello from gantrygraph!', "
        "then read it back and confirm the content is correct."
    )
    print(result)


asyncio.run(main())
