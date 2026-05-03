"""Example 3 — Real-time streaming of agent events.

Shows how to consume the async event stream — useful for building
progress UIs, WebSocket handlers, or live logging pipelines.

Run:
    ANTHROPIC_API_KEY=sk-ant-... python examples/03_streaming.py
"""
import asyncio

from langchain_anthropic import ChatAnthropic

from gantrygraph.actions.filesystem import FileSystemTools
from gantrygraph.actions.shell import ShellTool
from gantrygraph import GantryEngine

ICONS = {
    "observe": "👁 ",
    "think":   "💭",
    "act":     "⚡",
    "review":  "🔍",
    "done":    "✅",
    "error":   "❌",
}


async def main() -> None:
    agent = GantryEngine(
        llm=ChatAnthropic(model="claude-sonnet-4-6"),
        tools=[
            FileSystemTools(workspace="."),
            ShellTool(workspace="."),
        ],
        max_steps=15,
    )

    print("Starting agent…\n")
    async for event in agent.astream_events("How many Python files are in src/?"):
        icon = ICONS.get(event.event_type, "•")
        if event.event_type == "observe":
            screenshot = event.data.get("screenshot_b64")
            note = " [screenshot captured]" if screenshot else ""
            print(f"  {icon} step {event.step}: observe{note}")
        elif event.event_type == "act":
            tools = event.data.get("tools_executed", [])
            print(f"  {icon} step {event.step}: executed {tools}")
        elif event.event_type == "done":
            print(f"\n  {icon} Agent finished at step {event.step}")
        else:
            print(f"  {icon} step {event.step}: {event.event_type}")


asyncio.run(main())
