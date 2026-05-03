"""Desktop screen capture via mss + Pillow."""

from __future__ import annotations

import asyncio
import base64
import io
from typing import Any

from gantrygraph.core.base_perception import BasePerception
from gantrygraph.core.events import PerceptionResult


class DesktopScreen(BasePerception):
    """Capture screenshots of the physical (or virtual) desktop.

    Uses ``mss`` for fast OS-native screen capture and ``Pillow`` for
    resizing, both of which are in the core dependency set.  Capture is
    dispatched to a thread pool to avoid blocking the async event loop.

    Example::

        vision = DesktopScreen(max_resolution=(1280, 720), monitor=1)
        agent = GantryEngine(..., perception=vision)
    """

    def __init__(
        self,
        max_resolution: tuple[int, int] = (1920, 1080),
        monitor: int = 1,
        png_quality: int = 85,
    ) -> None:
        self._max_resolution = max_resolution
        self._monitor = monitor
        self._png_quality = png_quality

    async def observe(self) -> PerceptionResult:
        loop = asyncio.get_event_loop()
        raw_bytes, w, h = await loop.run_in_executor(None, self._capture_sync)
        b64 = base64.b64encode(raw_bytes).decode("ascii")
        return PerceptionResult(screenshot_b64=b64, width=w, height=h)

    def _capture_sync(self) -> tuple[bytes, int, int]:
        import mss
        import mss.tools
        from PIL import Image  # type: ignore[import-untyped]

        with mss.mss() as sct:
            monitors = sct.monitors
            if self._monitor >= len(monitors):
                raise ValueError(
                    f"Monitor {self._monitor} not found. Available monitors: 0-{len(monitors) - 1}."
                )
            monitor = monitors[self._monitor]
            raw = sct.grab(monitor)
            png_bytes = mss.tools.to_png(raw.rgb, raw.size)

        img = Image.open(io.BytesIO(png_bytes or b""))
        img = _resize_preserving_aspect(img, self._max_resolution)

        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
        return buf.getvalue(), img.width, img.height


def _resize_preserving_aspect(img: Any, max_size: tuple[int, int]) -> Any:
    """Resize *img* so it fits within *max_size*, preserving aspect ratio."""
    from PIL import Image  # noqa: F401

    max_w, max_h = max_size
    if img.width <= max_w and img.height <= max_h:
        return img
    ratio = min(max_w / img.width, max_h / img.height)
    new_w = int(img.width * ratio)
    new_h = int(img.height * ratio)
    return img.resize((new_w, new_h), resample=3)  # LANCZOS = 3
