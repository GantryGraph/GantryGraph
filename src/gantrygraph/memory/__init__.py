"""Long-term semantic memory for gantrygraph agents.

Quick start::

    from gantrygraph.memory import InMemoryStore

    memory = InMemoryStore()
    agent = GantryEngine(llm=..., memory=memory)

For persistent vector search backed by ChromaDB::

    from gantrygraph.memory import ChromaMemory   # requires pip install 'gantrygraph[memory]'

    memory = ChromaMemory(persist_directory="/tmp/gantrygraph-memory")
"""

from gantrygraph.memory.base import BaseMemory, MemoryResult
from gantrygraph.memory.in_memory import InMemoryStore, InMemoryVector

__all__ = ["BaseMemory", "MemoryResult", "InMemoryStore", "InMemoryVector"]

try:
    from gantrygraph.memory.chroma import ChromaMemory

    __all__ += ["ChromaMemory"]
except ImportError:
    pass
