# Tools & Actions

GantryGraph exposes three ways to give an agent capabilities:

| Mechanism | Use case |
|-----------|----------|
| `@gantry_tool` decorator | One-off functions you write |
| `BaseAction` subclass | Grouped tools sharing state (e.g. a browser instance) |
| `MCPClient` | External tool servers via Model Context Protocol |

## `@gantry_tool` — the fast path

```python
from gantrygraph import gantry_tool

@gantry_tool
async def search_docs(query: str) -> str:
    """Search the internal docs and return relevant snippets."""
    return await internal_search(query)
```

The decorator wraps the function in a `StructuredTool`, infers the
Pydantic schema from the type annotations, and sets the LLM-visible
description from the docstring.

Supports both sync and async functions.

## Built-in actions

### FileSystemTools

Sandboxed filesystem — all paths are validated against `workspace`.
Path traversal (`../../etc/passwd`) raises `PermissionError`.

```python
from gantrygraph.actions import FileSystemTools

tools = FileSystemTools(workspace="/my/project")
# Exposes: file_read, file_write, file_list, file_delete
```

### ShellTool

```python
from gantrygraph.actions import ShellTool

shell = ShellTool(
    workspace="/my/project",
    allowed_commands=["python", "pytest", "git"],  # None = allow all
    timeout=30.0,
)
# Exposes: shell_run
```

### MouseKeyboardTools

Requires `pip install 'gantrygraph[desktop]'`.

```python
from gantrygraph.actions import MouseKeyboardTools

tools = MouseKeyboardTools()
# Exposes: mouse_click, mouse_move, key_press, type_text, screenshot
```

### BrowserTools

Requires `pip install 'gantrygraph[browser]' && playwright install chromium`.

```python
from gantrygraph.actions import BrowserTools

tools = BrowserTools(headless=True)
# Exposes: browser_navigate, browser_click, browser_fill,
#          browser_get_text, browser_get_url
```

## Security policies

### WorkspacePolicy

Declarative shorthand — automatically creates `FileSystemTools` + `ShellTool`:

```python
from gantrygraph import GantryEngine
from gantrygraph.security import WorkspacePolicy

agent = GantryEngine(
    llm=...,
    workspace_policy=WorkspacePolicy(workspace_path="/app"),
)
```

### GuardrailPolicy

```python
from gantrygraph.security import GuardrailPolicy

guardrail = GuardrailPolicy(requires_approval={"shell_run", "file_delete"})

agent = GantryEngine(
    llm=...,
    tools=[shell, fs],
    guardrail=guardrail,
    approval_callback=my_slack_approval_fn,
)
```

## Writing a custom BaseAction

```python
from langchain_core.tools import BaseTool, StructuredTool
from gantrygraph import BaseAction

class DatabaseTools(BaseAction):
    def __init__(self, conn_string: str) -> None:
        self._db = connect(conn_string)

    def get_tools(self) -> list[BaseTool]:
        db = self._db

        async def _query(sql: str) -> str:
            rows = await db.fetch(sql)
            return str(rows)

        return [
            StructuredTool.from_function(
                coroutine=_query,
                name="db_query",
                description="Run a read-only SQL query and return results.",
            )
        ]

    async def close(self) -> None:
        await self._db.close()
```

`GantryEngine` calls `close()` automatically when the run finishes.
