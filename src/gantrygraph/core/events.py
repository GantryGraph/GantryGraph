from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel


class PerceptionResult(BaseModel):
    """Serialisable snapshot from any BasePerception.

    Passed to the agent as a LangChain multimodal message via
    ``to_message_content()``.  Pydantic is used here so the result
    can be stored in ``GantryState.last_observation`` as a plain dict
    via ``.model_dump()``.
    """

    screenshot_b64: str | None = None
    accessibility_tree: str | None = None
    url: str | None = None
    width: int = 1920
    height: int = 1080
    metadata: dict[str, Any] = {}

    def to_message_content(self) -> list[dict[str, Any]]:
        """Convert to LangChain multimodal message content blocks."""
        parts: list[dict[str, Any]] = []
        if self.accessibility_tree:
            parts.append(
                {"type": "text", "text": f"Accessibility tree:\n{self.accessibility_tree}"}
            )
        if self.screenshot_b64:
            parts.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{self.screenshot_b64}"},
                }
            )
        if not parts:
            parts.append({"type": "text", "text": "(no observation available)"})
        return parts


@dataclass
class GantryEvent:
    """Emitted by GantryEngine at each state transition for observability.

    Pass an ``on_event`` callback to ``GantryEngine`` to receive these events.
    The callback may be either a plain function or an async coroutine.

    Example::

        def my_logger(event: GantryEvent) -> None:
            print(f"[step {event.step}] {event.event_type}: {event.data}")

        agent = GantryEngine(..., on_event=my_logger)
    """

    event_type: Literal["observe", "think", "act", "review", "error", "done"]
    step: int
    data: dict[str, Any] = field(default_factory=dict)
