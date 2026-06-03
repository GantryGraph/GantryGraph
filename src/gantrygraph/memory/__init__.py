"""Long-term semantic memory for gantrygraph agents.

Quick start::

    from gantrygraph.memory import InMemoryStore

    memory = InMemoryStore()
    agent = GantryEngine(llm=..., memory=memory)

For persistent vector search backed by ChromaDB::

    from gantrygraph.memory import ChromaMemory   # requires pip install 'gantrygraph[memory]'

    memory = ChromaMemory(persist_directory="/tmp/gantrygraph-memory")

For ultra-light Rust HNSW memory with TTL (time-to-live) support::

    from gantrygraph.memory import MiniVecDbMemory  # requires pip install 'gantrygraph[minivecdb]'

    memory = MiniVecDbMemory(embed_fn=my_embed, ttl_ms=300_000)  # 5-minute TTL
"""

from gantrygraph.memory.base import BaseMemory, MemoryResult
from gantrygraph.memory.in_memory import InMemoryStore, InMemoryVector

__all__ = ["BaseMemory", "MemoryResult", "InMemoryStore", "InMemoryVector"]

try:
    from gantrygraph.memory.chroma import ChromaMemory

    __all__ += ["ChromaMemory"]
except ImportError:
    pass

try:
    from gantrygraph.memory.minivecdb import MiniVecDbMemory

    __all__ += ["MiniVecDbMemory"]
except ImportError:
    pass
