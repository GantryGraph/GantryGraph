"""Abstract base class for long-term memory backends."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class MemoryResult:
    """A single retrieved memory entry with its relevance score."""

    text: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseMemory(ABC):
    """Pluggable long-term memory backend.

    Subclasses must implement :meth:`add` and :meth:`search`.
    All methods are async so implementations can use I/O-bound vector DBs.
    """

    @abstractmethod
    async def add(self, text: str, metadata: dict[str, Any] | None = None) -> None:
        """Persist *text* with optional *metadata* for future retrieval."""

    @abstractmethod
    async def search(self, query: str, k: int = 5) -> list[MemoryResult]:
        """Return up to *k* results most relevant to *query*, sorted by score desc."""

    async def close(self) -> None:  # noqa: B027 — intentional no-op default
        """Release any held resources (connections, files, etc.)."""
