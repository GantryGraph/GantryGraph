"""Example 5 — Custom perception source.

Shows how to subclass BasePerception to feed any data source into the
agent's observe step.  Here we build a SystemStatsPerception that reads
CPU + memory stats from the OS and exposes them as a text observation.

No external service or display is required — runs anywhere Python 3.11+.

Run:
    ANTHROPIC_API_KEY=sk-ant-... python examples/05_custom_perception.py
"""
from __future__ import annotations

import asyncio

from langchain_anthropic import ChatAnthropic

from gantrygraph import GantryEngine
from gantrygraph.actions import ShellTool
from gantrygraph.core.base_perception import BasePerception
from gantrygraph.core.events import PerceptionResult


# ── Custom perception ─────────────────────────────────────────────────────────

class SystemStatsPerception(BasePerception):
    """Observe CPU, memory, and disk stats from the local OS.

    Returns a plain-text accessibility tree so the LLM understands the
    current machine state before deciding what action to take.
    """

    async def observe(self) -> PerceptionResult:
        proc = await asyncio.create_subprocess_shell(
            "echo 'CPU:' && top -bn1 | grep 'Cpu(s)' 2>/dev/null || "
            "ps -A -o %cpu | awk '{s+=$1} END {print s\"%\"}'; "
            "echo 'Memory:' && free -h 2>/dev/null || "
            "vm_stat | head -6; "
            "echo 'Disk:' && df -h / | tail -1",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        stdout, _ = await proc.communicate()
        tree = stdout.decode(errors="replace").strip() or "(stats unavailable)"
        return PerceptionResult(accessibility_tree=tree)


# ── Wire up the agent ─────────────────────────────────────────────────────────

async def main() -> None:
    agent = GantryEngine(
        llm=ChatAnthropic(model="claude-sonnet-4-6"),
        perception=SystemStatsPerception(),
        tools=[ShellTool(workspace="/tmp", allowed_commands=["echo", "ls", "df"])],
        max_steps=3,
        on_event=lambda e: print(f"[{e.event_type}] step={e.step}"),
    )
    result = await agent.arun(
        "Look at the current system stats and tell me "
        "how much free disk space is available on /."
    )
    print("\nAgent answer:", result)


if __name__ == "__main__":
    asyncio.run(main())
