"""Unit tests for the gantrygraph.memory module."""
from __future__ import annotations

import pytest

from gantrygraph.memory.base import BaseMemory, MemoryResult
from gantrygraph.memory.in_memory import InMemoryVector

# ── BaseMemory ABC ────────────────────────────────────────────────────────────

def test_base_memory_cannot_be_instantiated() -> None:
    with pytest.raises(TypeError):
        BaseMemory()  # type: ignore[abstract]


def test_memory_result_fields() -> None:
    r = MemoryResult(text="hello", score=0.9, metadata={"k": "v"})
    assert r.text == "hello"
    assert r.score == 0.9
    assert r.metadata == {"k": "v"}


def test_memory_result_default_metadata() -> None:
    r = MemoryResult(text="x", score=0.5)
    assert r.metadata == {}


# ── InMemoryVector ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_in_memory_empty_search_returns_nothing() -> None:
    mem = InMemoryVector()
    results = await mem.search("anything")
    assert results == []


@pytest.mark.asyncio
async def test_in_memory_add_and_retrieve() -> None:
    mem = InMemoryVector()
    await mem.add("React hook error in useEffect", {"lang": "js"})
    results = await mem.search("React useEffect hook")
    assert len(results) > 0
    assert "React" in results[0].text
    assert results[0].metadata == {"lang": "js"}


@pytest.mark.asyncio
async def test_in_memory_scores_sorted_descending() -> None:
    mem = InMemoryVector()
    await mem.add("Python list comprehension tutorial")
    await mem.add("JavaScript array map reduce")
    await mem.add("Python for loop iteration guide")
    results = await mem.search("Python iteration")
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


@pytest.mark.asyncio
async def test_in_memory_k_limits_results() -> None:
    mem = InMemoryVector()
    for i in range(10):
        await mem.add(f"Python programming tip number {i}")
    results = await mem.search("Python programming", k=3)
    assert len(results) <= 3


@pytest.mark.asyncio
async def test_in_memory_no_match_returns_empty() -> None:
    mem = InMemoryVector()
    await mem.add("hello world")
    results = await mem.search("zzzzzzz")
    assert results == []


@pytest.mark.asyncio
async def test_in_memory_len() -> None:
    mem = InMemoryVector()
    assert len(mem) == 0
    await mem.add("first")
    await mem.add("second")
    assert len(mem) == 2


@pytest.mark.asyncio
async def test_in_memory_close_is_no_op() -> None:
    mem = InMemoryVector()
    await mem.close()  # should not raise


@pytest.mark.asyncio
async def test_in_memory_metadata_preserved() -> None:
    mem = InMemoryVector()
    meta = {"task": "fix bug", "steps": 5}
    await mem.add("ZeroDivisionError in safe_divide", meta)
    results = await mem.search("ZeroDivisionError safe_divide")
    assert results[0].metadata == meta


# ── Engine integration with memory ───────────────────────────────────────────

@pytest.mark.asyncio
async def test_engine_stores_result_in_memory() -> None:
    """After arun(), the task+result is stored in memory."""
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
    from langchain_core.messages import AIMessage

    from gantrygraph import GantryEngine

    memory = InMemoryVector()
    llm = FakeMessagesListChatModel(
        responses=[AIMessage(content="Task completed successfully.")]
    )
    agent = GantryEngine(llm=llm, memory=memory, max_steps=5)
    await agent.arun("Fix the divide-by-zero bug")

    assert len(memory) == 1
    stored = memory._entries[0][0]
    assert "Fix the divide-by-zero bug" in stored
    assert "Task completed successfully" in stored


@pytest.mark.asyncio
async def test_engine_recalls_memory_on_second_run() -> None:
    """On the second run, memory results are injected into the context."""
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
    from langchain_core.messages import AIMessage, HumanMessage

    from gantrygraph import GantryEngine

    captured_messages: list = []

    class SpyLLM(FakeMessagesListChatModel):
        responses: list = [AIMessage(content="Done."), AIMessage(content="Done again.")]

        async def ainvoke(self, messages, **kwargs):  # type: ignore[override]
            captured_messages.extend(messages)
            return await super().ainvoke(messages, **kwargs)

    memory = InMemoryVector()
    await memory.add("Task: fix bug\nResult: added zero guard", {"task": "fix bug"})

    agent = GantryEngine(llm=SpyLLM(), memory=memory, max_steps=5)
    await agent.arun("fix zero division error")

    # The memory recall HumanMessage should appear in the messages passed to LLM
    texts = [m.content if isinstance(m.content, str) else str(m.content)
             for m in captured_messages if isinstance(m, HumanMessage)]
    assert any("past experiences" in t.lower() or "fix bug" in t.lower() for t in texts)
