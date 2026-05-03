"""Example 7 — Heterogeneous swarm with specialist workers.

Shows how to configure a GantrySupervisor with multiple WorkerSpec instances,
each carrying its own tool set.  The supervisor LLM decomposes the task and
routes subtasks to the most appropriate specialist automatically.

This example uses stub engines so no real API key is needed for the workers;
only the supervisor LLM is called.

Run:
    ANTHROPIC_API_KEY=sk-ant-... python examples/07_swarm_specialists.py
"""
from __future__ import annotations

import asyncio

from langchain_anthropic import ChatAnthropic

from gantrygraph import GantryEngine
from gantrygraph.actions import FileSystemTools, ShellTool
from gantrygraph.swarm import GantrySupervisor, WorkerSpec


async def main() -> None:
    llm = ChatAnthropic(model="claude-sonnet-4-6")

    supervisor = GantrySupervisor(
        llm=llm,
        workers=[
            WorkerSpec(
                name="shell_expert",
                engine=GantryEngine(
                    llm=llm,
                    tools=[ShellTool(workspace="/tmp", allowed_commands=["ls", "find", "echo"])],
                    max_steps=5,
                ),
                description=(
                    "Explores the filesystem with shell commands, "
                    "finds files, reads directory listings."
                ),
            ),
            WorkerSpec(
                name="file_editor",
                engine=GantryEngine(
                    llm=llm,
                    tools=[FileSystemTools(workspace="/tmp")],
                    max_steps=5,
                ),
                description=(
                    "Reads, writes, and edits file contents. "
                    "Best for summarising or modifying specific files."
                ),
            ),
        ],
    )

    result = await supervisor.run(
        "Find all .log files under /tmp, then read each one "
        "and produce a one-line summary of its content."
    )
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
