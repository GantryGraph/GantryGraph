"""Core abstractions for gantrygraph.

This package contains only ABCs and data types — no I/O, no side effects.
Import from here when building custom perception sources, action sets,
or MCP connectors.
"""

from gantrygraph.core.base_action import BaseAction
from gantrygraph.core.base_mcp import BaseMCPConnector
from gantrygraph.core.base_perception import BasePerception
from gantrygraph.core.events import GantryEvent, PerceptionResult
from gantrygraph.core.state import GantryState

__all__ = [
    "BasePerception",
    "BaseAction",
    "BaseMCPConnector",
    "GantryEvent",
    "PerceptionResult",
    "GantryState",
]
