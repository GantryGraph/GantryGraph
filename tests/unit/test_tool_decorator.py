"""Unit tests for the @gantry_tool decorator."""

from __future__ import annotations

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from langchain_core.tools import BaseTool


def _done_llm() -> FakeMessagesListChatModel:
    return FakeMessagesListChatModel(responses=[AIMessage(content="Done.")])


# ── Construction ──────────────────────────────────────────────────────────────


def test_sync_tool_returns_base_tool() -> None:
    from gantrygraph import gantry_tool

    @gantry_tool
    def my_tool(x: str) -> str:
        """A test tool."""
        return x

    assert isinstance(my_tool, BaseTool)


def test_async_tool_returns_base_tool() -> None:
    from gantrygraph import gantry_tool

    @gantry_tool
    async def my_async_tool(x: str) -> str:
        """An async test tool."""
        return x

    assert isinstance(my_async_tool, BaseTool)


def test_tool_name_defaults_to_function_name() -> None:
    from gantrygraph import gantry_tool

    @gantry_tool
    def search_orders(query: str) -> str:
        """Search orders."""
        return query

    assert search_orders.name == "search_orders"


def test_tool_description_defaults_to_docstring() -> None:
    from gantrygraph import gantry_tool

    @gantry_tool
    def lookup(key: str) -> str:
        """Search the company database."""
        return key

    assert lookup.description == "Search the company database."


def test_tool_name_override() -> None:
    from gantrygraph import gantry_tool

    @gantry_tool(name="custom_name")
    def some_func(x: str) -> str:
        """Does something."""
        return x

    assert some_func.name == "custom_name"


def test_tool_description_override() -> None:
    from gantrygraph import gantry_tool

    @gantry_tool(description="Overridden description.")
    def some_func(x: str) -> str:
        """Original docstring."""
        return x

    assert some_func.description == "Overridden description."


def test_missing_docstring_and_no_description_raises() -> None:
    from gantrygraph import gantry_tool

    with pytest.raises(ValueError, match="description"):

        @gantry_tool
        def no_doc(x: str) -> str:
            return x


# ── Invocation ────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_sync_tool_invocation() -> None:
    from gantrygraph import gantry_tool

    @gantry_tool
    def add(a: int, b: int) -> str:
        """Add two numbers and return the result as a string."""
        return str(a + b)

    result = await add.ainvoke({"a": 3, "b": 4})
    assert result == "7"


@pytest.mark.asyncio
async def test_async_tool_invocation() -> None:
    from gantrygraph import gantry_tool

    @gantry_tool
    async def greet(name: str) -> str:
        """Return a greeting for the given name."""
        return f"Hello, {name}!"

    result = await greet.ainvoke({"name": "world"})
    assert result == "Hello, world!"


# ── Integration with GantryEngine ───────────────────────────────────────────────


def test_gantry_tool_accepted_by_collect_tools() -> None:
    from gantrygraph import GantryEngine, gantry_tool

    @gantry_tool
    def my_action(query: str) -> str:
        """Run a query against the internal index."""
        return query

    engine = GantryEngine(llm=_done_llm(), tools=[my_action])
    collected = engine._collect_tools()
    assert my_action in collected


def test_gantry_tool_name_in_tool_registry() -> None:
    from gantrygraph import GantryEngine, gantry_tool

    @gantry_tool(name="ping_service")
    def ping(host: str) -> str:
        """Ping a host and return latency."""
        return f"pong {host}"

    engine = GantryEngine(llm=_done_llm(), tools=[ping])
    names = {t.name for t in engine._collect_tools()}
    assert "ping_service" in names


# ── Top-level export ──────────────────────────────────────────────────────────


def test_gantry_tool_importable_from_gantrygraph() -> None:
    from gantrygraph import gantry_tool

    assert callable(gantry_tool)
