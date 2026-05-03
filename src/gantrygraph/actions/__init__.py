"""Action tool implementations for gantrygraph."""

from gantrygraph.actions.filesystem import FileSystemTools
from gantrygraph.actions.shell import ShellTool

__all__ = [
    "FileSystemTools",
    "ShellTool",
    # Requires [desktop] extra:
    # "MouseKeyboardTools",
    # Requires [browser] extra:
    # "BrowserTools",
]


def __getattr__(name: str) -> object:
    if name == "MouseKeyboardTools":
        from gantrygraph.actions.mouse_keyboard import MouseKeyboardTools
        return MouseKeyboardTools
    if name == "BrowserTools":
        from gantrygraph.actions.browser import BrowserTools
        return BrowserTools
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
