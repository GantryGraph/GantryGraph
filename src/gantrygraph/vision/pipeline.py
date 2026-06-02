"""Composable image preprocessing pipeline for vision-mode screenshots.

Attach a :class:`PerceptionPipeline` to :class:`~gantrygraph.perception.web.WebPage`
to transform screenshots before they are encoded and sent to the LLM::

    from gantrygraph.vision.pipeline import (
        PerceptionPipeline,
        SetOfMarkAnnotator,
        Downsample,
        ConvertToWebP,
    )

    pipeline = PerceptionPipeline([
        SetOfMarkAnnotator(),        # numbered boxes → agent uses IDs, not coords
        Downsample(max_width=1280),  # cap resolution
        ConvertToWebP(quality=85),   # shrink payload ~30 %
    ])

    web = WebPage(url="https://example.com", vision_pipeline=pipeline)
    tools = BrowserTools(web_page=web, vision_pipeline=pipeline)

Order matters: :class:`SetOfMarkAnnotator` should run *before* :class:`Downsample`
so that bounding boxes are derived from the full-resolution screenshot.
"""

from __future__ import annotations

import io
from abc import ABC, abstractmethod
from typing import Any

try:
    from PIL import Image, ImageDraw, ImageFont

    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False

_PIL_MSG = "Vision pipeline filters require Pillow (core dep): pip install gantrygraph"


# ── abstract base ─────────────────────────────────────────────────────────────


class ImageFilter(ABC):
    """Base class for every step in a :class:`PerceptionPipeline`.

    Subclass this to build custom filters — e.g. domain-specific crop regions,
    watermark removal, or specialised annotation styles.
    """

    @abstractmethod
    async def process(self, image_bytes: bytes, ctx: dict[str, Any]) -> bytes:
        """Transform *image_bytes* and return the result.

        *ctx* is a shared mutable dict for the current pipeline run.  Filters
        may read from it (e.g. consume ``ctx["page"]`` for DOM access) and write
        to it (e.g. :class:`SetOfMarkAnnotator` writes ``ctx["som_map"]``).
        """


# ── pipeline coordinator ──────────────────────────────────────────────────────


class PerceptionPipeline:
    """Execute a list of :class:`ImageFilter` steps in sequence.

    :class:`~gantrygraph.perception.web.WebPage` sets ``ctx["page"]`` before
    calling :meth:`run`, which allows filters like :class:`SetOfMarkAnnotator`
    to query the live DOM.  After every :meth:`run` call the accumulated ctx is
    stored in :attr:`last_ctx` so that companion tools (e.g. ``browser_click_som``)
    can read the element map without re-running the pipeline.

    Example::

        pipeline = PerceptionPipeline([SetOfMarkAnnotator(), Downsample(1280)])
        web = WebPage(url="...", vision_pipeline=pipeline)
        tools = BrowserTools(web_page=web, vision_pipeline=pipeline)
    """

    def __init__(self, filters: list[ImageFilter]) -> None:
        self.filters = filters
        self.last_ctx: dict[str, Any] = {}

    async def run(self, image_bytes: bytes, ctx: dict[str, Any] | None = None) -> bytes:
        """Run all filters in order and return the transformed image bytes."""
        working: dict[str, Any] = dict(ctx or {})
        for f in self.filters:
            image_bytes = await f.process(image_bytes, working)
        self.last_ctx = working
        return image_bytes

    @property
    def som_map(self) -> dict[int, dict[str, Any]]:
        """Element map written by the last :class:`SetOfMarkAnnotator` run.

        Keys are the integer IDs painted on the screenshot.  Values contain
        ``tag``, ``text``, ``selector`` (CSS ``#id`` if available), ``cx``,
        and ``cy`` (centre pixel coordinates).
        """
        raw = self.last_ctx.get("som_map", {})
        return {int(k): v for k, v in raw.items()}


# ── image filters ─────────────────────────────────────────────────────────────


class Downsample(ImageFilter):
    """Resize the screenshot so width ≤ *max_width*, preserving aspect ratio.

    Images already narrower than *max_width* are returned unchanged.
    Put this *after* :class:`SetOfMarkAnnotator` so annotations are drawn
    at full resolution.

    Args:
        max_width: Target maximum width in pixels (default 1280).
    """

    def __init__(self, max_width: int = 1280) -> None:
        if not _HAS_PIL:
            raise ImportError(_PIL_MSG)
        self.max_width = max_width

    async def process(self, image_bytes: bytes, ctx: dict[str, Any]) -> bytes:
        img = Image.open(io.BytesIO(image_bytes))
        if img.width <= self.max_width:
            return image_bytes
        ratio = self.max_width / img.width
        new_size = (self.max_width, int(img.height * ratio))
        resized = img.resize(new_size, resample=Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        resized.save(buf, format=img.format or "PNG", optimize=True)
        return buf.getvalue()


class Grayscale(ImageFilter):
    """Convert the screenshot to grayscale.

    Token count is unchanged for vision models (they bill by resolution, not
    colour depth), but greyscale screenshots compress better and produce more
    stable bytes for prompt-cache hit-rate.  Avoid for tasks where colour
    carries information (e.g. reading charts or error badges).
    """

    def __init__(self) -> None:
        if not _HAS_PIL:
            raise ImportError(_PIL_MSG)

    async def process(self, image_bytes: bytes, ctx: dict[str, Any]) -> bytes:
        img = Image.open(io.BytesIO(image_bytes))
        grey = img.convert("L").convert("RGB")
        buf = io.BytesIO()
        grey.save(buf, format=img.format or "PNG")
        return buf.getvalue()


class ConvertToWebP(ImageFilter):
    """Re-encode the screenshot as lossy WebP.

    WebP is typically 25–35 % smaller than PNG at the same perceptual quality,
    reducing upload latency.  Use *quality* ≥ 85 to keep UI text legible.

    Args:
        quality: WebP lossy quality 0–100 (default 85).
    """

    def __init__(self, quality: int = 85) -> None:
        if not _HAS_PIL:
            raise ImportError(_PIL_MSG)
        self.quality = quality

    async def process(self, image_bytes: bytes, ctx: dict[str, Any]) -> bytes:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="WEBP", quality=self.quality, method=4)
        return buf.getvalue()


# ── Set-of-Mark annotator ─────────────────────────────────────────────────────

# JavaScript injected into the page to collect bounding boxes for all
# interactive elements visible in the current viewport.
_SOM_JS = """
() => {
    const SELECTORS = [
        'a[href]', 'button', 'input:not([type="hidden"])',
        'select', 'textarea',
        '[role="button"]', '[role="link"]', '[role="checkbox"]',
        '[role="menuitem"]', '[role="tab"]', '[role="option"]',
    ].join(',');
    const els = Array.from(document.querySelectorAll(SELECTORS));
    const result = [];
    for (const el of els) {
        if (result.length >= 120) break;
        const r = el.getBoundingClientRect();
        if (r.width < 2 || r.height < 2) continue;
        if (r.top > window.innerHeight || r.bottom < 0) continue;
        if (r.left > window.innerWidth || r.right < 0) continue;
        const val = typeof el.value === 'string' ? el.value : '';
        const label = (
            el.innerText || val || el.placeholder ||
            el.getAttribute('aria-label') || el.getAttribute('title') || ''
        ).trim().slice(0, 60);
        result.push({
            tag: el.tagName.toLowerCase(),
            text: label,
            x: Math.round(r.left),
            y: Math.round(r.top),
            w: Math.round(r.width),
            h: Math.round(r.height),
            selector: el.id ? '#' + el.id : null,
        });
    }
    return result;
}
"""

_PALETTE: list[tuple[int, int, int]] = [
    (220, 50, 50),  # red
    (50, 140, 220),  # blue
    (50, 180, 80),  # green
    (200, 130, 0),  # amber
    (150, 50, 200),  # purple
]

# Fixed chip dimensions to avoid font-metric calls on Pillow's default font.
_CHIP_CHAR_W = 7  # pixels per digit
_CHIP_H = 15  # chip height


class SetOfMarkAnnotator(ImageFilter):
    """Overlay numbered bounding boxes on every interactive element in the viewport.

    The filter injects a small JavaScript snippet that reads ``getBoundingClientRect``
    for all interactive selectors, then draws the boxes in Python using PIL.
    No screenshot diffing or ML model is required.

    **Requires** ``ctx["page"]`` — a live Playwright
    :class:`~playwright.async_api.Page` object.  This is set automatically when
    the pipeline is attached to :class:`~gantrygraph.perception.web.WebPage`.
    On non-browser screenshots (desktop, no ``page`` in ctx) the filter is a
    no-op and the image is returned unchanged.

    After ``process()`` completes, ``ctx["som_map"]`` contains::

        {
          1: {"tag": "button", "text": "Accept", "selector": "#accept-btn",
              "cx": 342, "cy": 891},
          2: {"tag": "a", "text": "Home", "selector": None,
              "cx": 80, "cy": 24},
        }

    The companion ``browser_click_som`` action reads this map so the agent can
    call ``browser_click_som(element_id=1)`` instead of guessing pixel coords.
    Precision goes from "close enough" to 100 % for every labelled element.

    Args:
        box_alpha: Opacity of the translucent fill overlay (0–255).
            0 = outline only, 255 = fully opaque.  Default 30.
        max_elements: Maximum elements annotated per frame (default 100).
        color: Fixed ``(R, G, B)`` for all boxes.  ``None`` cycles through a
            5-colour palette so adjacent elements are visually distinct.
    """

    def __init__(
        self,
        box_alpha: int = 30,
        max_elements: int = 100,
        color: tuple[int, int, int] | None = None,
    ) -> None:
        if not _HAS_PIL:
            raise ImportError(_PIL_MSG)
        self.box_alpha = box_alpha
        self.max_elements = max_elements
        self._fixed_color = color

    async def process(self, image_bytes: bytes, ctx: dict[str, Any]) -> bytes:
        page = ctx.get("page")
        if page is None:
            return image_bytes

        raw: list[dict[str, Any]] = await page.evaluate(_SOM_JS)
        elements = raw[: self.max_elements]

        img = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw_ov = ImageDraw.Draw(overlay)
        draw = ImageDraw.Draw(img)
        font = ImageFont.load_default()

        som_map: dict[int, dict[str, Any]] = {}

        for idx, el in enumerate(elements, start=1):
            rgb = self._fixed_color or _PALETTE[(idx - 1) % len(_PALETTE)]
            r_c, g_c, b_c = rgb
            x, y, w, h = el["x"], el["y"], el["w"], el["h"]

            # Clamp box to image bounds
            x1 = max(x, 0)
            y1 = max(y, 0)
            x2 = min(x + w, img.width - 1)
            y2 = min(y + h, img.height - 1)
            if x2 <= x1 or y2 <= y1:
                continue

            # Translucent fill on a separate layer (composited below)
            draw_ov.rectangle([x1, y1, x2, y2], fill=(r_c, g_c, b_c, self.box_alpha))

            # 2-px solid border directly on the image
            draw.rectangle([x1, y1, x2, y2], outline=rgb, width=2)

            # ID chip — fixed-size, no font metric calls
            label = str(idx)
            chip_w = len(label) * _CHIP_CHAR_W + 6
            chip_x2 = min(x1 + chip_w, img.width - 1)
            chip_y2 = min(y1 + _CHIP_H, img.height - 1)
            draw.rectangle([x1, y1, chip_x2, chip_y2], fill=rgb)
            draw.text((x1 + 3, y1 + 2), label, fill=(255, 255, 255), font=font)

            som_map[idx] = {
                "tag": el["tag"],
                "text": el.get("text", ""),
                "selector": el.get("selector"),
                "cx": x1 + (x2 - x1) // 2,
                "cy": y1 + (y2 - y1) // 2,
            }

        composited = Image.alpha_composite(img, overlay).convert("RGB")
        ctx["som_map"] = som_map

        buf = io.BytesIO()
        composited.save(buf, format="PNG", optimize=True)
        return buf.getvalue()
