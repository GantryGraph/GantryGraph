from __future__ import annotations

from abc import ABC, abstractmethod

from gantrygraph.core.events import PerceptionResult


class BasePerception(ABC):
    """Abstract base class for all perception sources.

    Subclass this to add new ways for an agent to observe its environment
    (desktop screenshot, web page, terminal output, REST API, database
    state — anything that can be turned into text or an image).

    The engine calls ``observe()`` once per loop iteration and attaches
    the result as a multimodal ``HumanMessage`` before invoking the LLM.
    If you return a ``screenshot_b64``, the LLM receives a vision block;
    if you return an ``accessibility_tree``, it receives a text block; you
    can return both at the same time.

    Optionally override ``close()`` to release resources on shutdown (file
    handles, network sockets, subprocesses).  GantryEngine calls it
    automatically at the end of every ``arun()`` / ``run()`` call.

    **Minimal example — observe a REST API:**

    .. code-block:: python

        from gantrygraph.core.base_perception import BasePerception
        from gantrygraph.core.events import PerceptionResult
        import httpx

        class APIStatusPerception(BasePerception):
            def __init__(self, url: str) -> None:
                self._url = url
                self._client = httpx.AsyncClient()

            async def observe(self) -> PerceptionResult:
                resp = await self._client.get(self._url)
                return PerceptionResult(
                    accessibility_tree=f"HTTP {resp.status_code}\\n{resp.text[:2000]}"
                )

            async def close(self) -> None:
                await self._client.aclose()

        agent = GantryEngine(
            llm=my_llm,
            perception=APIStatusPerception("https://api.example.com/status"),
        )

    **Terminal / subprocess example:**

    .. code-block:: python

        import asyncio
        from gantrygraph.core.base_perception import BasePerception
        from gantrygraph.core.events import PerceptionResult

        class TerminalPerception(BasePerception):
            async def observe(self) -> PerceptionResult:
                proc = await asyncio.create_subprocess_shell(
                    "ps aux --sort=-%cpu | head -20",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                stdout, _ = await proc.communicate()
                return PerceptionResult(accessibility_tree=stdout.decode())

    **Combining multiple sources — use MultiPerception:**

    .. code-block:: python

        from gantrygraph.perception import MultiPerception, DesktopScreen, WebPage

        agent = GantryEngine(
            llm=my_llm,
            perception=MultiPerception([DesktopScreen(), WebPage("https://example.com")]),
        )
    """

    @abstractmethod
    async def observe(self) -> PerceptionResult:
        """Capture the current environment state."""
        ...

    async def close(self) -> None:  # noqa: B027 — intentional no-op default
        """Optional cleanup hook called by GantryEngine on shutdown."""
