"""``@gantry_tool`` — turn any Python function into a GantryEngine tool.

Zero LangChain knowledge required.  Accepts both sync and async functions,
infers name and description from the function signature, and returns a
``BaseTool`` that ``GantryEngine`` accepts directly in its ``tools=`` list.

Quick start::

    from gantrygraph import gantry_tool

    @gantry_tool
    def search_db(query: str, limit: int = 10) -> str:
        \"\"\"Search the company database.\"\"\"
        return db.execute(query, limit)

    @gantry_tool
    async def call_api(url: str) -> str:
        \"\"\"Call an internal REST endpoint and return the response body.\"\"\"
        async with httpx.AsyncClient() as c:
            return (await c.get(url)).text

    agent = GantryEngine(llm=..., tools=[search_db, call_api])

Override name or description::

    @gantry_tool(name="db_search", description="Full-text search over orders.")
    def search(query: str) -> str:
        return db.fts(query)
"""
from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any, overload

from langchain_core.tools import BaseTool, StructuredTool


@overload
def gantry_tool(fn: Callable[..., Any]) -> BaseTool: ...


@overload
def gantry_tool(
    fn: None = None,
    *,
    name: str | None = None,
    description: str | None = None,
) -> Callable[[Callable[..., Any]], BaseTool]: ...


def gantry_tool(
    fn: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
) -> BaseTool | Callable[[Callable[..., Any]], BaseTool]:
    """Decorate a function (sync or async) and return a LangChain ``BaseTool``.

    Can be used bare (``@gantry_tool``) or with keyword arguments
    (``@gantry_tool(name="x", description="y")``).

    Args:
        fn:          The function to wrap.  Passed automatically when the
                     decorator is used bare.  Leave ``None`` when passing
                     keyword arguments.
        name:        Tool name visible to the LLM.  Defaults to the
                     function's ``__name__``.
        description: Tool description visible to the LLM.  Defaults to the
                     function's docstring.  **Required** if the function has
                     no docstring.

    Returns:
        A ``BaseTool`` instance (when used bare or the inner function is
        decorated), or a decorator factory (when keyword arguments are
        provided).

    Raises:
        ValueError: If neither a docstring nor an explicit ``description``
                    is provided.
    """

    def _wrap(func: Callable[..., Any]) -> BaseTool:
        tool_name = name or func.__name__
        tool_desc = description or inspect.getdoc(func) or ""
        if not tool_desc:
            raise ValueError(
                f"@gantry_tool on '{tool_name}' requires either a docstring"
                " or an explicit description= argument."
            )
        if inspect.iscoroutinefunction(func):
            return StructuredTool.from_function(
                coroutine=func,
                name=tool_name,
                description=tool_desc,
            )
        return StructuredTool.from_function(
            func=func,
            name=tool_name,
            description=tool_desc,
        )

    if fn is not None:
        return _wrap(fn)
    return _wrap
