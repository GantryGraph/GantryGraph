"""Example 2 — Custom tools with @gantry_tool.

Demonstrates how to expose any Python function (sync or async) as an
agent tool without touching LangChain directly.

Run:
    ANTHROPIC_API_KEY=sk-ant-... python examples/02_custom_tools.py
"""
import asyncio
import json
from pathlib import Path

from langchain_anthropic import ChatAnthropic

from gantrygraph import GantryEngine, gantry_tool


@gantry_tool
def list_python_files(directory: str = ".") -> str:
    """List all Python files in a directory."""
    files = list(Path(directory).rglob("*.py"))
    return json.dumps([str(f) for f in files[:20]])


@gantry_tool
async def count_lines(file_path: str) -> str:
    """Count the number of lines in a file."""
    try:
        content = Path(file_path).read_text()
        return str(len(content.splitlines()))
    except FileNotFoundError:
        return f"File not found: {file_path}"


async def main() -> None:
    agent = GantryEngine(
        llm=ChatAnthropic(model="claude-sonnet-4-6"),
        tools=[list_python_files, count_lines],
        max_steps=10,
    )
    result = await agent.arun(
        "List the Python files in the src/ directory "
        "and tell me which one has the most lines of code."
    )
    print(result)


asyncio.run(main())
