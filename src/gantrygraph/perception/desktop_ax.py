"""Desktop accessibility tree perception via macOS AXUIElement API.

Requires the ``[desktop-ax]`` extra (macOS only)::

    pip install 'gantrygraph[desktop-ax]'

Then grant Accessibility permission to your terminal / app in
System Settings → Privacy & Security → Accessibility.

Example::

    from gantrygraph import GantryEngine
    from gantrygraph.perception import DesktopAXTree
    from langchain_anthropic import ChatAnthropic

    # Interact with whichever app is in focus — zero vision tokens
    agent = GantryEngine(
        llm=ChatAnthropic(model="claude-sonnet-4-6"),
        perception=DesktopAXTree(),
        tools=[MouseKeyboardTools()],
        max_steps=20,
    )
    agent.run("Open the 'Meeting Notes' note and add a bullet with today's date.")

    # Target a specific app by name
    agent = GantryEngine(
        llm=...,
        perception=DesktopAXTree(app_name="Obsidian"),
    )

    # Belt-and-suspenders: AX tree + screenshot together (MultiPerception not needed)
    agent = GantryEngine(
        llm=...,
        perception=DesktopAXTree(include_screenshot=True),
    )
"""

from __future__ import annotations

import asyncio
import sys
from typing import Any

from gantrygraph.core.base_perception import BasePerception
from gantrygraph.core.events import PerceptionResult

try:
    import atomacos

    _HAS_ATOMACOS = True
except ImportError:
    _HAS_ATOMACOS = False

_INSTALL_MSG = (
    "DesktopAXTree requires atomacos (macOS only): "
    "pip install 'gantrygraph[desktop-ax]'\n"
    "Then grant Accessibility permission in System Settings → Privacy & Security → Accessibility."
)

# AX roles that carry meaningful text content in their value
_TEXT_ROLES = frozenset(
    {
        "AXTextField",
        "AXTextArea",
        "AXStaticText",
        "AXComboBox",
        "AXSearchField",
    }
)

# Roles we skip entirely — they never help the LLM reason about the UI
_SKIP_ROLES = frozenset(
    {
        "AXUnknown",
        "AXSeparator",
        "AXGrowArea",
        "AXImage",
        "AXHelpTag",
    }
)


def _ax_get(node: Any, attr: str, default: Any = None) -> Any:
    """Safe attribute read — AX attributes raise on missing, not return None."""
    try:
        return getattr(node, attr, default)
    except Exception:
        return default


def _serialize_node(
    node: Any,
    depth: int,
    max_depth: int,
    max_children: int,
    max_text: int,
    lines: list[str],
) -> None:
    role: str = _ax_get(node, "AXRole") or "AXUnknown"
    if role in _SKIP_ROLES:
        return

    indent = "  " * depth
    parts: list[str] = [role]

    title: str = _ax_get(node, "AXTitle") or ""
    if title:
        parts.append(repr(title[:60]))

    value = _ax_get(node, "AXValue")
    if isinstance(value, str) and value.strip() and role in _TEXT_ROLES:
        truncated = value[:max_text]
        if len(value) > max_text:
            truncated += "…"
        parts.append(repr(truncated))

    desc: str = _ax_get(node, "AXDescription") or ""
    if desc and desc != title:
        parts.append(f"({desc[:50]})")

    tags: list[str] = []
    if _ax_get(node, "AXFocused"):
        tags.append("focused")
    enabled = _ax_get(node, "AXEnabled")
    if enabled is False:
        tags.append("disabled")
    if tags:
        parts.append(f"[{', '.join(tags)}]")

    lines.append(indent + " ".join(parts))

    if depth >= max_depth:
        return

    try:
        children: list[Any] = _ax_get(node, "AXChildren") or []
        visible = [c for c in children if _ax_get(c, "AXRole") not in _SKIP_ROLES]
        for child in visible[:max_children]:
            _serialize_node(child, depth + 1, max_depth, max_children, max_text, lines)
        overflow = len(visible) - max_children
        if overflow > 0:
            lines.append(indent + f"  … ({overflow} more children)")
    except Exception:
        pass


def _build_tree(
    app_name: str | None,
    bundle_id: str | None,
    max_depth: int,
    max_children: int,
    max_text: int,
) -> str:
    if not _HAS_ATOMACOS:
        raise ImportError(_INSTALL_MSG)
    if sys.platform != "darwin":
        raise RuntimeError("DesktopAXTree is macOS-only.")

    try:
        if bundle_id:
            app = atomacos.getAppRefByBundleId(bundle_id)
        elif app_name:
            app = atomacos.getAppRefByLocalizedName(app_name)
        else:
            app = atomacos.getFrontmostApp()
    except atomacos.errors.Error as exc:
        return f"[AX error: {exc} — check Accessibility permission in System Settings]"

    lines: list[str] = []
    app_title: str = _ax_get(app, "AXTitle") or app_name or "unknown"
    lines.append(f"AXApplication '{app_title}'")

    try:
        windows: list[Any] = app.windows() or []
    except Exception:
        windows = []

    for win in windows:
        _serialize_node(win, depth=1, max_depth=max_depth,
                        max_children=max_children, max_text=max_text, lines=lines)

    return "\n".join(lines)


class DesktopAXTree(BasePerception):
    """Native macOS accessibility tree perception — zero vision tokens.

    Walks the AXUIElement tree of a running app and returns structured text
    describing every visible UI element.  The LLM can reason about buttons,
    text fields, labels, and content without ever seeing a screenshot.

    Equivalent to ``WebPage(include_accessibility=True)`` but for any native
    macOS application (Obsidian, VS Code, Finder, Safari, Mail, …).

    Args:
        app_name:         Localised app name (e.g. ``"Obsidian"``).
                          ``None`` targets whichever app is currently focused.
        bundle_id:        App bundle ID (e.g. ``"md.obsidian"``).
                          Takes precedence over *app_name* when both are set.
        include_screenshot: When ``True``, a desktop screenshot is appended to
                            the observation alongside the AX tree.  Useful as a
                            visual fallback for apps with limited AX support.
        max_depth:        Maximum nesting depth of the serialised tree (default 8).
        max_children:     Maximum children rendered per node (default 40).
        max_text_length:  Maximum characters kept from any single text value (default 400).

    Example::

        from gantrygraph.perception import DesktopAXTree
        from gantrygraph.actions import MouseKeyboardTools

        agent = GantryEngine(
            llm=ChatAnthropic(model="claude-sonnet-4-6"),
            perception=DesktopAXTree(app_name="Obsidian"),
            tools=[MouseKeyboardTools()],
            max_steps=20,
        )
        agent.run("Find the note titled 'Q2 Goals' and append a new bullet point.")
    """

    def __init__(
        self,
        app_name: str | None = None,
        bundle_id: str | None = None,
        include_screenshot: bool = False,
        max_depth: int = 8,
        max_children: int = 40,
        max_text_length: int = 400,
    ) -> None:
        if not _HAS_ATOMACOS:
            raise ImportError(_INSTALL_MSG)
        if sys.platform != "darwin":
            raise RuntimeError("DesktopAXTree is macOS-only.")
        self._app_name = app_name
        self._bundle_id = bundle_id
        self._include_screenshot = include_screenshot
        self._max_depth = max_depth
        self._max_children = max_children
        self._max_text = max_text_length

    async def observe(self) -> PerceptionResult:
        loop = asyncio.get_event_loop()
        tree: str = await loop.run_in_executor(
            None,
            lambda: _build_tree(
                self._app_name,
                self._bundle_id,
                self._max_depth,
                self._max_children,
                self._max_text,
            ),
        )

        screenshot_b64: str | None = None
        width, height = 1920, 1080

        if self._include_screenshot:
            import base64
            import io

            import mss
            import mss.tools
            from PIL import Image

            def _snap() -> tuple[bytes, int, int]:
                with mss.mss() as sct:
                    monitor = sct.monitors[1]
                    raw = sct.grab(monitor)
                    png = mss.tools.to_png(raw.rgb, raw.size)
                img = Image.open(io.BytesIO(png or b""))
                buf = io.BytesIO()
                img.save(buf, format="PNG", optimize=True)
                return buf.getvalue(), img.width, img.height

            png_bytes, width, height = await loop.run_in_executor(None, _snap)
            screenshot_b64 = base64.b64encode(png_bytes).decode("ascii")

        return PerceptionResult(
            accessibility_tree=tree,
            screenshot_b64=screenshot_b64,
            width=width,
            height=height,
        )
