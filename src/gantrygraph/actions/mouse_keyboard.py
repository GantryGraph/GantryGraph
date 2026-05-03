"""Mouse and keyboard tools via PyAutoGUI.

Requires the ``[desktop]`` extra::

    pip install gantrygraph[desktop]
"""
from __future__ import annotations

import asyncio
from typing import Literal

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

from gantrygraph.core.base_action import BaseAction

try:
    import pyautogui as _pag  # type: ignore[import-untyped]  # noqa: F401
    _HAS_PYAUTOGUI = True
except ImportError:
    _HAS_PYAUTOGUI = False

_INSTALL_MSG = (
    "MouseKeyboardTools requires the [desktop] extra: "
    "pip install 'gantrygraph[desktop]'"
)


class MouseKeyboardTools(BaseAction):
    """Physical mouse and keyboard control via PyAutoGUI.

    All PyAutoGUI calls are dispatched to a thread pool so they don't block
    the async event loop.

    Example::

        from gantrygraph.actions import MouseKeyboardTools
        agent = GantryEngine(..., tools=[MouseKeyboardTools(fail_safe=True)])
    """

    def __init__(self, fail_safe: bool = True, pause: float = 0.05) -> None:
        if not _HAS_PYAUTOGUI:
            raise ImportError(_INSTALL_MSG)
        import pyautogui
        pyautogui.FAILSAFE = fail_safe
        pyautogui.PAUSE = pause
        self._pag = pyautogui

    def get_tools(self) -> list[BaseTool]:
        return [
            self._click_tool(),
            self._type_tool(),
            self._scroll_tool(),
            self._hotkey_tool(),
            self._move_tool(),
        ]

    def _click_tool(self) -> BaseTool:
        pag = self._pag

        class _Args(BaseModel):
            x: int = Field(description="X coordinate in pixels.")
            y: int = Field(description="Y coordinate in pixels.")
            button: Literal["left", "right", "middle"] = Field(
                default="left", description="Mouse button to click."
            )
            clicks: int = Field(default=1, ge=1, le=3, description="Number of clicks.")

        async def _click(x: int, y: int, button: str = "left", clicks: int = 1) -> str:
            await asyncio.get_event_loop().run_in_executor(
                None, lambda: pag.click(x, y, button=button, clicks=clicks)
            )
            return f"Clicked ({x}, {y}) with {button} button × {clicks}."

        return StructuredTool.from_function(
            coroutine=_click,
            name="mouse_click",
            description="Click at pixel coordinates (x, y) on the screen.",
            args_schema=_Args,
        )

    def _type_tool(self) -> BaseTool:
        pag = self._pag

        class _Args(BaseModel):
            text: str = Field(description="Text to type.")
            interval: float = Field(
                default=0.02, ge=0.0, description="Delay between keystrokes in seconds."
            )

        async def _type(text: str, interval: float = 0.02) -> str:
            await asyncio.get_event_loop().run_in_executor(
                None, lambda: pag.typewrite(text, interval=interval)
            )
            return f"Typed {len(text)} characters."

        return StructuredTool.from_function(
            coroutine=_type,
            name="keyboard_type",
            description="Type text using the keyboard at the current cursor position.",
            args_schema=_Args,
        )

    def _scroll_tool(self) -> BaseTool:
        pag = self._pag

        class _Args(BaseModel):
            clicks: int = Field(
                description="Scroll amount. Positive = scroll up, negative = scroll down."
            )
            x: int | None = Field(default=None, description="X coordinate (current if None).")
            y: int | None = Field(default=None, description="Y coordinate (current if None).")

        async def _scroll(clicks: int, x: int | None = None, y: int | None = None) -> str:
            kwargs = {}
            if x is not None:
                kwargs["x"] = x
            if y is not None:
                kwargs["y"] = y
            await asyncio.get_event_loop().run_in_executor(
                None, lambda: pag.scroll(clicks, **kwargs)
            )
            direction = "up" if clicks > 0 else "down"
            return f"Scrolled {abs(clicks)} clicks {direction}."

        return StructuredTool.from_function(
            coroutine=_scroll,
            name="mouse_scroll",
            description="Scroll the mouse wheel up (positive) or down (negative).",
            args_schema=_Args,
        )

    def _hotkey_tool(self) -> BaseTool:
        pag = self._pag

        class _Args(BaseModel):
            keys: list[str] = Field(
                description="Keys to press simultaneously, e.g. ['ctrl', 'c'] for copy."
            )

        async def _hotkey(keys: list[str]) -> str:
            await asyncio.get_event_loop().run_in_executor(
                None, lambda: pag.hotkey(*keys)
            )
            return f"Pressed hotkey: {'+'.join(keys)}."

        return StructuredTool.from_function(
            coroutine=_hotkey,
            name="keyboard_hotkey",
            description="Press a keyboard shortcut (e.g. ['ctrl', 'c'] for copy).",
            args_schema=_Args,
        )

    def _move_tool(self) -> BaseTool:
        pag = self._pag

        class _Args(BaseModel):
            x: int = Field(description="Target X coordinate in pixels.")
            y: int = Field(description="Target Y coordinate in pixels.")
            duration: float = Field(
                default=0.1, ge=0.0, description="Movement duration in seconds."
            )

        async def _move(x: int, y: int, duration: float = 0.1) -> str:
            await asyncio.get_event_loop().run_in_executor(
                None, lambda: pag.moveTo(x, y, duration=duration)
            )
            return f"Moved mouse to ({x}, {y})."

        return StructuredTool.from_function(
            coroutine=_move,
            name="mouse_move",
            description="Move the mouse cursor to the specified coordinates.",
            args_schema=_Args,
        )
