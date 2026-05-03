"""ChromaDB-backed persistent vector memory.

Requires the ``[memory]`` extra::

    pip install 'gantrygraph[memory]'
"""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path
from typing import Any

try:
    import chromadb

    _HAS_CHROMA = True
except ImportError:
    _HAS_CHROMA = False

from gantrygraph.memory.base import BaseMemory, MemoryResult

_INSTALL_MSG = "ChromaMemory requires chromadb: pip install 'gantrygraph[memory]'"


class ChromaMemory(BaseMemory):
    """Persistent semantic memory backed by ChromaDB.

    Uses ChromaDB's built-in sentence-transformer embeddings when no custom
    embedding function is provided.

    Args:
        collection_name:    Name of the ChromaDB collection.
        persist_directory:  Directory for on-disk persistence.  ``None`` uses
                            an in-process ephemeral client (good for tests).
    """

    def __init__(
        self,
        collection_name: str = "gantry_memory",
        persist_directory: str | Path | None = None,
    ) -> None:
        if not _HAS_CHROMA:
            raise ImportError(_INSTALL_MSG)
        if persist_directory is not None:
            self._client = chromadb.PersistentClient(path=str(persist_directory))
        else:
            self._client = chromadb.EphemeralClient()
        self._col = self._client.get_or_create_collection(collection_name)

    async def add(self, text: str, metadata: dict[str, Any] | None = None) -> None:
        doc_id = str(uuid.uuid4())
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: self._col.add(
                documents=[text],
                ids=[doc_id],
                metadatas=[metadata or {}],
            ),
        )

    async def search(self, query: str, k: int = 5) -> list[MemoryResult]:
        loop = asyncio.get_event_loop()
        count: int = await loop.run_in_executor(None, lambda: len(self._col.get()["ids"]))
        if count == 0:
            return []
        n = min(k, count)
        raw = await loop.run_in_executor(
            None, lambda: self._col.query(query_texts=[query], n_results=n)
        )
        docs = (raw.get("documents") or [[]])[0]
        metas = (raw.get("metadatas") or [[]])[0]
        distances = (raw.get("distances") or [[]])[0]
        results: list[MemoryResult] = []
        for doc, meta, dist in zip(docs, metas, distances, strict=False):
            # ChromaDB returns L2 distance; convert to 0..1 similarity
            score = 1.0 / (1.0 + dist)
            results.append(MemoryResult(text=doc, score=score, metadata=dict(meta or {})))
        return results

    async def close(self) -> None:
        pass  # chromadb manages its own connection lifecycle
