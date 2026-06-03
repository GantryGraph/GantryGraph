"""MiniVecDb-backed vector memory with native TTL support.

Requires the ``[minivecdb]`` extra::

    pip install 'gantrygraph[minivecdb]'

You must also supply an embedding function that returns 384-dimensional
float vectors (e.g. ``OpenAIEmbeddings(model="text-embedding-3-small").embed_query``
or any ``SentenceTransformer`` model)::

    from gantrygraph.memory import MiniVecDbMemory
    from langchain_openai import OpenAIEmbeddings

    embed = OpenAIEmbeddings(model="text-embedding-3-small").embed_query
    memory = MiniVecDbMemory(embed_fn=embed, ttl_ms=300_000)   # 5-minute TTL
    agent = GantryEngine(llm=..., memory=memory)
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from typing import Any

try:
    from minivecdb import MiniVecDb as _MiniVecDb

    _HAS_MINIVECDB = True
except ImportError:
    _HAS_MINIVECDB = False

from gantrygraph.memory.base import BaseMemory, MemoryResult

_INSTALL_MSG = (
    "MiniVecDbMemory requires minivecdb: pip install 'gantrygraph[minivecdb]'\n"
    "You also need an embed_fn — e.g. OpenAIEmbeddings(...).embed_query or any "
    "callable that maps a string to a list[float] of 384 dimensions."
)


class MiniVecDbMemory(BaseMemory):
    """Semantic vector memory backed by MiniVecDb (Rust HNSW, 1-bit quantisation).

    Key properties vs the other built-in backends:

    * **TTL** — entries expire automatically after *ttl_ms* milliseconds,
      simulating the natural decay of working memory in a long agent loop.
    * **Ultra-light** — the Rust native extension adds ~50 KB and uses
      48 bytes per vector (32× less RAM than raw float32).
    * **Bring-your-own embedder** — pass any callable as *embed_fn*;
      LangChain ``Embeddings.embed_query`` works directly.

    Args:
        embed_fn:         ``(text: str) -> list[float]`` callable returning a
                          384-dimensional dense embedding.  Any LangChain
                          ``Embeddings.embed_query`` method works directly.
        capacity:         Maximum number of entries the store can hold.
        ttl_ms:           Milliseconds before an entry expires.  ``None``
                          disables expiry.  Expired entries are pruned lazily
                          on each :meth:`search` call.
        m:                HNSW graph fanout — higher = better recall,
                          more RAM (default 16).
        ef_construction:  HNSW build-time beam width (default 200).

    Example::

        from gantrygraph.memory import MiniVecDbMemory
        from langchain_openai import OpenAIEmbeddings

        embed = OpenAIEmbeddings(model="text-embedding-3-small").embed_query
        memory = MiniVecDbMemory(embed_fn=embed, ttl_ms=600_000)  # 10 min TTL
        agent = GantryEngine(llm=..., memory=memory, max_steps=50)
    """

    def __init__(
        self,
        embed_fn: Callable[[str], list[float]],
        *,
        capacity: int = 10_000,
        ttl_ms: int | None = None,
        m: int = 16,
        ef_construction: int = 200,
    ) -> None:
        if not _HAS_MINIVECDB:
            raise ImportError(_INSTALL_MSG)
        self._db = _MiniVecDb.with_capacity(capacity)
        self._embed = embed_fn
        self._ttl_ms = ttl_ms
        self._m = m
        self._ef = ef_construction
        # int id → (text, metadata)
        self._texts: dict[int, tuple[str, dict[str, Any]]] = {}
        # int id → insertion timestamp ms (for Python-side TTL pruning)
        self._inserted_at: dict[int, int] = {}
        self._counter: int = 0
        # HNSW index needs a rebuild after new inserts
        self._dirty: bool = False

    async def add(self, text: str, metadata: dict[str, Any] | None = None) -> None:
        loop = asyncio.get_event_loop()
        vector: list[float] = await loop.run_in_executor(None, self._embed, text)
        doc_id = self._counter
        self._counter += 1
        await loop.run_in_executor(None, lambda: self._db.insert(doc_id, vector))
        self._texts[doc_id] = (text, metadata or {})
        self._inserted_at[doc_id] = int(time.time() * 1000)
        self._dirty = True

    async def search(self, query: str, k: int = 5) -> list[MemoryResult]:
        if not self._texts:
            return []
        loop = asyncio.get_event_loop()
        if self._ttl_ms is not None:
            # Tombstone expired vectors in the Rust HNSW graph
            await loop.run_in_executor(None, lambda: self._db.run_gc(self._ttl_ms))
            # Mirror the expiry on the Python-side text store
            now = int(time.time() * 1000)
            expired = {
                did for did, ts in self._inserted_at.items()
                if now - ts > self._ttl_ms
            }
            for did in expired:
                self._texts.pop(did, None)
                self._inserted_at.pop(did, None)
            if not self._texts:
                return []
        # Rebuild HNSW graph if new entries were added since last search
        if self._dirty:
            await loop.run_in_executor(
                None, lambda: self._db.build_index(self._m, self._ef)
            )
            self._dirty = False
        vector: list[float] = await loop.run_in_executor(None, self._embed, query)
        n = min(k, len(self._texts))
        if n == 0:
            return []
        raw: list[tuple[int, float]] = await loop.run_in_executor(
            None, lambda: self._db.search(vector, n)
        )
        results: list[MemoryResult] = []
        for doc_id, score in raw:
            if doc_id in self._texts:
                text, meta = self._texts[doc_id]
                results.append(MemoryResult(text=text, score=score, metadata=meta))
        return results

    async def close(self) -> None:
        self._texts.clear()
        self._inserted_at.clear()
        self._dirty = False

    def __len__(self) -> int:
        return len(self._texts)
