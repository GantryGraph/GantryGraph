"""Perception implementations for gantrygraph.

All imports are lazy so ``import gantrygraph.perception`` never fails regardless of
which extras are installed.

Available classes:
- ``DesktopScreen``    — always available (``mss`` + ``pillow`` are core deps)
- ``WebPage``          — requires ``[browser]`` extra
- ``MultiPerception``  — always available; combines multiple sources in parallel
"""

__all__ = ["DesktopScreen", "MultiPerception", "WebPage"]


def __getattr__(name: str) -> object:
    if name == "DesktopScreen":
        from gantrygraph.perception.desktop import DesktopScreen

        return DesktopScreen
    if name == "MultiPerception":
        from gantrygraph.perception.multi import MultiPerception

        return MultiPerception
    if name == "WebPage":
        from gantrygraph.perception.web import WebPage

        return WebPage
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
