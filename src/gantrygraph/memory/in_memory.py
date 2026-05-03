"""Pure-Python in-memory vector store — no external dependencies.

Uses character-trigram Jaccard overlap for similarity scoring.
Good for testing and low-volume use.  For production, use :class:`ChromaMemory`.
"""
from __future__ import annotations

from typing import Any

from gantrygraph.memory.base import BaseMemory, MemoryResult


def _trigrams(text: str) -> frozenset[str]:
    return frozenset(text[i : i + 3] for i in range(len(text) - 2))


class InMemoryStore(BaseMemory):
    """Ephemeral in-process memory backed by trigram Jaccard similarity.

    All entries are held in RAM and lost when the process exits.
    Thread-safe for single-threaded asyncio use (no locking needed).
    """

    def __init__(self) -> None:
        self._entries: list[tuple[str, dict[str, Any]]] = []

    async def add(self, text: str, metadata: dict[str, Any] | None = None) -> None:
        self._entries.append((text, metadata or {}))

    async def search(self, query: str, k: int = 5) -> list[MemoryResult]:
        if not self._entries:
            return []
        q_grams = _trigrams(query.lower())
        results: list[MemoryResult] = []
        for text, meta in self._entries:
            t_grams = _trigrams(text.lower())
            union = q_grams | t_grams
            score = len(q_grams & t_grams) / len(union) if union else 0.0
            if score > 0.0:
                results.append(MemoryResult(text=text, score=score, metadata=meta))
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:k]

    def __len__(self) -> int:
        return len(self._entries)


InMemoryVector = InMemoryStore  # backward compat alias
