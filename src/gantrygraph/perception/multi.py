"""MultiPerception — combine several BasePerception sources into one."""

from __future__ import annotations

import asyncio
from typing import Any

from gantrygraph.core.base_perception import BasePerception
from gantrygraph.core.events import PerceptionResult


class MultiPerception(BasePerception):
    """Run multiple perception sources in parallel and merge their results.

    Pass any combination of ``BasePerception`` subclasses.  All ``observe()``
    calls are issued concurrently via ``asyncio.gather``; their results are
    merged into a single ``PerceptionResult``:

    - **screenshot_b64** — taken from the first source that returns one
      (usually the desktop camera).
    - **accessibility_tree** — all non-empty trees are concatenated with a
      ``--- <source N> ---`` separator so the LLM knows which source
      produced which text.
    - **url** — taken from the first source that returns one.
    - **width / height** — taken from the source that provided
      ``screenshot_b64``; fall back to the default ``1920 × 1080``.
    - **metadata** — all metadata dicts are merged (later sources win on
      key conflicts).

    **Example — desktop + browser in one agent:**

    .. code-block:: python

        from gantrygraph import GantryEngine
        from gantrygraph.perception import MultiPerception, DesktopScreen, WebPage

        agent = GantryEngine(
            llm=my_llm,
            perception=MultiPerception([
                DesktopScreen(max_resolution=(1280, 720)),
                WebPage("https://example.com"),
            ]),
        )
        await agent.arun("Describe what you see on the desktop and in the browser.")

    **Example — three custom sources:**

    .. code-block:: python

        perception = MultiPerception([
            DesktopScreen(),
            TerminalPerception(),   # custom BasePerception subclass
            APIStatusPerception("https://api.example.com/health"),
        ])
        agent = GantryEngine(llm=my_llm, perception=perception)

    Sources are always called concurrently, so the total observation time
    equals that of the *slowest* single source, not the sum of all.
    """

    def __init__(self, sources: list[BasePerception]) -> None:
        if not sources:
            raise ValueError("MultiPerception requires at least one source.")
        self._sources = sources

    async def observe(self) -> PerceptionResult:
        results: list[PerceptionResult] = await asyncio.gather(
            *[s.observe() for s in self._sources],
            return_exceptions=False,
        )
        return self._merge(results)

    async def close(self) -> None:
        await asyncio.gather(*[s.close() for s in self._sources])

    # ── merge logic ───────────────────────────────────────────────────────────

    def _merge(self, results: list[PerceptionResult]) -> PerceptionResult:
        screenshot_b64: str | None = None
        width = 1920
        height = 1080
        url: str | None = None
        trees: list[str] = []
        metadata: dict[str, Any] = {}

        for i, r in enumerate(results):
            if screenshot_b64 is None and r.screenshot_b64:
                screenshot_b64 = r.screenshot_b64
                width = r.width
                height = r.height
            if url is None and r.url:
                url = r.url
            if r.accessibility_tree:
                label = f"--- source {i + 1} ---"
                trees.append(f"{label}\n{r.accessibility_tree}")
            metadata.update(r.metadata)

        return PerceptionResult(
            screenshot_b64=screenshot_b64,
            accessibility_tree="\n\n".join(trees) if trees else None,
            url=url,
            width=width,
            height=height,
            metadata=metadata,
        )
