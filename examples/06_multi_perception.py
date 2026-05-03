"""Example 6 — MultiPerception: desktop + web in a single agent.

Demonstrates how to point a GantryEngine at multiple perception sources
simultaneously.  Both sources are observed in parallel on every loop
iteration; their results are merged into one multimodal HumanMessage.

This example uses two stub sources so it runs without a real display or
browser.  Replace them with DesktopScreen() and WebPage(url) in practice.

Run:
    ANTHROPIC_API_KEY=sk-ant-... python examples/06_multi_perception.py
"""
from __future__ import annotations

import asyncio

from langchain_anthropic import ChatAnthropic

from gantrygraph import GantryEngine
from gantrygraph.core.base_perception import BasePerception
from gantrygraph.core.events import PerceptionResult
from gantrygraph.perception import MultiPerception


# ── Stub sources (replace with DesktopScreen / WebPage in production) ─────────

class FakeDesktop(BasePerception):
    async def observe(self) -> PerceptionResult:
        return PerceptionResult(
            accessibility_tree="[Desktop] Terminal window open. Cursor at /home/user.",
        )


class FakeBrowser(BasePerception):
    async def observe(self) -> PerceptionResult:
        return PerceptionResult(
            accessibility_tree="[Browser] URL: https://example.com — Page title: Example Domain",
            url="https://example.com",
        )


# ── Real usage would be: ──────────────────────────────────────────────────────
#
#   from gantrygraph.perception import DesktopScreen, WebPage
#
#   perception = MultiPerception([
#       DesktopScreen(max_resolution=(1280, 720)),
#       WebPage("https://example.com"),
#   ])


async def main() -> None:
    perception = MultiPerception([FakeDesktop(), FakeBrowser()])

    agent = GantryEngine(
        llm=ChatAnthropic(model="claude-sonnet-4-6"),
        perception=perception,
        max_steps=2,
        on_event=lambda e: print(f"[{e.event_type}] step={e.step}"),
    )

    result = await agent.arun(
        "Describe both what's on the desktop and what's currently open in the browser."
    )
    print("\nAgent answer:", result)


if __name__ == "__main__":
    asyncio.run(main())
