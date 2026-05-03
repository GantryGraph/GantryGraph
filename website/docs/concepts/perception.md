# Perception

Perception is how the agent observes its environment before each think step.
Every perception source implements `BasePerception` and returns a `PerceptionResult`.

## PerceptionResult

```python
class PerceptionResult(BaseModel):
    screenshot_b64: str | None   # base-64 PNG
    accessibility_tree: str | None
    url: str | None
    width: int
    height: int
    metadata: dict[str, Any]
```

`to_message_content()` converts it to a multimodal LangChain message block
(image + text) that gets appended to the conversation as a `HumanMessage`.

## Built-in sources

### DesktopScreen

Captures the primary monitor via `mss`. Runs in a thread pool so it
never blocks the event loop.

```python
from gantrygraph.perception import DesktopScreen

screen = DesktopScreen(max_resolution=(1280, 720))
```

Requires `pip install gantrygraph` (no extra needed — `mss` is a core dep).

### WebPage

Renders a URL via Playwright and captures both a screenshot and the
page's accessibility tree.

```python
from gantrygraph.perception import WebPage

page = WebPage(url="https://example.com", headless=True)
```

Requires `pip install 'gantrygraph[browser]' && playwright install chromium`.

!!! tip "Share the browser with BrowserTools"
    Pass `web_page=page` to `BrowserTools` so both perception and actions
    operate on the **same** Playwright `Page` object:
    ```python
    web = WebPage(url="https://app.example.com")
    agent = GantryEngine(
        perception=web,
        tools=[BrowserTools(web_page=web)],
        ...
    )
    ```

### MultiPerception

Combine multiple sources — screenshots AND DOM tree from the same agent:

```python
from gantrygraph import MultiPerception
from gantrygraph.perception import DesktopScreen, WebPage

agent = GantryEngine(
    perception=MultiPerception([
        DesktopScreen(),
        WebPage(url="https://dashboard.internal"),
    ]),
    ...
)
```

Results are merged: first screenshot wins, accessibility trees are
concatenated with `--- source N ---` labels.

## Writing a custom perception source

```python
import asyncio
from gantrygraph import BasePerception
from gantrygraph.core.events import PerceptionResult

class SystemStatsPerception(BasePerception):
    """Expose CPU and memory metrics to the agent."""

    async def observe(self) -> PerceptionResult:
        import psutil
        stats = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: {
                "cpu_percent": psutil.cpu_percent(),
                "mem_percent": psutil.virtual_memory().percent,
            },
        )
        return PerceptionResult(
            screenshot_b64=None,
            accessibility_tree=f"CPU: {stats['cpu_percent']}%\nMEM: {stats['mem_percent']}%",
            url=None,
            width=0,
            height=0,
            metadata=stats,
        )

    async def close(self) -> None:
        pass  # nothing to clean up
```

## Vision preprocessing

`BaseVisionProvider` wraps the screenshot-to-LLM path,
letting you swap vision backends without touching agent code:

```python
from gantrygraph import ClaudeVision, GantryEngine

agent = GantryEngine(
    llm=ChatAnthropic(...),
    perception=DesktopScreen(),
    # ClaudeVision pre-processes screenshots for Claude's vision API
)
```

Supported: `ClaudeVision` (built-in). GPT-4o and custom providers via
`BaseVisionProvider`.
