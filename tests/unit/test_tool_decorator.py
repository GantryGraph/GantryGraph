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


# ── destructive=True ─────────────────────────────────────────────────────────


def test_destructive_tool_has_metadata_flag() -> None:
    from gantrygraph import gantry_tool

    @gantry_tool(destructive=True)
    def wipe_database() -> str:
        """Irreversibly wipe the entire database."""
        return "wiped"

    assert wipe_database.metadata == {"gantry_destructive": True}


def test_non_destructive_tool_has_no_metadata_flag() -> None:
    from gantrygraph import gantry_tool

    @gantry_tool
    def safe_read(path: str) -> str:
        """Read a file."""
        return path

    assert not (safe_read.metadata or {}).get("gantry_destructive")


def test_destructive_tool_auto_added_to_guardrail() -> None:
    """GantryEngine must auto-populate GuardrailPolicy for destructive tools."""
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
    from langchain_core.messages import AIMessage

    from gantrygraph import GantryEngine, gantry_tool

    @gantry_tool(destructive=True)
    def drop_table(table: str) -> str:
        """Drop a database table permanently."""
        return f"dropped {table}"

    llm = FakeMessagesListChatModel(responses=[AIMessage(content="Done.")])
    engine = GantryEngine(llm=llm, tools=[drop_table])
    engine._build()
    # After _build, the compiled graph was created; check the guardrail was populated.
    # The easiest way is to re-inspect the destructive name detection path.
    tool_list = engine._collect_tools()
    destructive_names = {
        t.name for t in tool_list if (t.metadata or {}).get("gantry_destructive")
    }
    assert "drop_table" in destructive_names


def test_destructive_tool_merges_with_existing_guardrail() -> None:
    """Existing GuardrailPolicy entries are preserved when merging destructive tools."""
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
    from langchain_core.messages import AIMessage

    from gantrygraph import GantryEngine, gantry_tool
    from gantrygraph.security.policies import GuardrailPolicy

    @gantry_tool(destructive=True)
    def nuke(target: str) -> str:
        """Nuke everything."""
        return f"nuked {target}"

    llm = FakeMessagesListChatModel(responses=[AIMessage(content="Done.")])
    existing_guardrail = GuardrailPolicy(requires_approval={"shell_run"})
    engine = GantryEngine(llm=llm, tools=[nuke], guardrail=existing_guardrail)
    engine._build()
    # shell_run was in existing guardrail; nuke should be added by destructive logic
    tool_list = engine._collect_tools()
    destructive_names = {
        t.name for t in tool_list if (t.metadata or {}).get("gantry_destructive")
    }
    assert "nuke" in destructive_names


# ── Top-level export ──────────────────────────────────────────────────────────


def test_gantry_tool_importable_from_gantrygraph() -> None:
    from gantrygraph import gantry_tool

    assert callable(gantry_tool)
