# Contributing to gantrygraph

Thank you for wanting to improve gantrygraph.  
This document covers everything you need to add a new feature, fix a bug, or build a plugin — from architecture decisions to the exact commands to run before opening a PR.

---

## Table of contents

1. [Project philosophy](#1-project-philosophy)
2. [Architecture in 5 minutes](#2-architecture-in-5-minutes)
3. [Setting up the dev environment](#3-setting-up-the-dev-environment)
4. [How to add a new perception source](#4-how-to-add-a-new-perception-source)
5. [How to add a new action set](#5-how-to-add-a-new-action-set)
6. [How to add a new memory backend](#6-how-to-add-a-new-memory-backend)
7. [How to add a new LVM wrapper](#7-how-to-add-a-new-lvm-wrapper)
8. [Security guardrails — what you must not break](#8-security-guardrails--what-you-must-not-break)
9. [Optional extras — the import guard contract](#9-optional-extras--the-import-guard-contract)
10. [Testing conventions](#10-testing-conventions)
11. [Code style](#11-code-style)
12. [Opening a pull request](#12-opening-a-pull-request)

---

## 1. Project philosophy

gantrygraph is a **library, not a product**.  
The developer who imports it is the customer. Keep that in mind for every decision.

**Three principles that must never be violated:**

| Principle | Meaning |
|-----------|---------|
| `import gantrygraph` never fails | Core imports must succeed with only core deps installed. Optional extras are guarded at module level. |
| No hidden state | Agents are stateless objects. The only state is what the developer explicitly passes in (`checkpointer`, `memory`). |
| Composable, not opinionated | Every layer (`perception`, `actions`, `memory`) is pluggable via ABCs. The developer owns the LLM choice. |

---

## 2. Architecture in 5 minutes

```
gantrygraph/
  core/         Pure ABCs and shared types. No I/O, no imports of optional deps.
  engine/       LangGraph wiring. Two files: nodes.py (pure async fns) + graph.py (wires them).
  perception/   Observe the environment → PerceptionResult.
  actions/      Tools the agent can execute → list[BaseTool].
  mcp/          MCP subprocess lifecycle + dynamic StructuredTool generation.
  memory/       Long-term recall. add() / search() / close().
  security/     Policy objects passed into GantryEngine. Enforcement lives in act_node.
  swarm/        Multi-agent pattern built on top of GantryEngine.
  cloud/        FastAPI REST + SSE. No core dep on FastAPI.
  telemetry/    OTel span exporter. No core dep on opentelemetry.
  tool.py       @gantry_tool decorator.
  config.py     GantryConfig — declarative setup from YAML / env.
  presets.py    Opinionated factory functions for common scenarios.
```

**The agent loop** (in `engine/`):

```
START → memory_recall → observe → think → act → review → should_continue
                                                              │         │
                                                           observe     END
```

- `memory_recall`: injects past experiences from long-term memory (first step only).
- `observe`: calls `perception.observe()` → appends `HumanMessage` with screenshot/tree.
- `think`: calls `llm_with_tools.ainvoke(messages)` → gets `AIMessage` with tool calls.
- `act`: executes each tool call; applies guardrail and approval gate before each.
- `review`: pure fn — sets `is_done=True` when LLM stops calling tools.
- `should_continue`: conditional edge — loop or `END`.

Nodes are **pure async functions** bound to config via `functools.partial` in `graph.py`.  
This is the key design decision: nodes are testable in isolation without a full engine.

---

## 3. Setting up the dev environment

```bash
git clone https://github.com/GantryGraph/GantryGraph
cd gantrygraph

# Recommended: use uv
uv sync --all-extras

# Or pip
pip install -e ".[all,dev]"

# Copy env vars
cp .env.example .env
# Fill in at least ANTHROPIC_API_KEY for integration tests
```

**Run the full check suite before every commit:**

```bash
pytest tests/unit/         # ~2 s, no display needed
mypy src/gantrygraph --strict
ruff check src/ tests/
ruff format src/ tests/
```

---

## 4. How to add a new perception source

A perception source observes the environment and returns a `PerceptionResult`.

**Step 1** — Create `src/gantrygraph/perception/my_source.py`:

```python
from __future__ import annotations

from gantrygraph.core.base_perception import BasePerception
from gantrygraph.core.events import PerceptionResult


class MySource(BasePerception):
    """One-line description for the docs."""

    async def observe(self) -> PerceptionResult:
        # Do your I/O here.
        # Return PerceptionResult with whatever fields you populate:
        #   screenshot_b64: str | None   (base64 PNG)
        #   accessibility_tree: str | None
        #   url: str | None
        #   width: int, height: int
        #   metadata: dict[str, Any]
        return PerceptionResult(accessibility_tree="some text")

    async def close(self) -> None:
        # Release connections, browsers, etc.
        pass
```

**Step 2** — Expose via lazy import in `src/gantrygraph/perception/__init__.py`:

```python
if name == "MySource":
    from gantrygraph.perception.my_source import MySource
    return MySource
```

**Step 3** — If it requires an optional extra, guard the import at the top of the file:

```python
try:
    import my_dep
    _HAS_MY_DEP = True
except ImportError:
    _HAS_MY_DEP = False
```

Raise `ImportError` with a `pip install` hint inside `__init__` if `_HAS_MY_DEP` is `False`.

**Step 4** — Add a test in `tests/unit/test_perception.py` that mocks the I/O and asserts the returned `PerceptionResult` has valid fields.

**Rule:** `PerceptionResult.screenshot_b64` is always included in the `observe` event emitted by the engine. If your source has no screenshot, leave it `None` — do not omit the key.

---

## 5. How to add a new action set

An action set bundles related tools and exposes them as `list[BaseTool]`.

**Step 1** — Create `src/gantrygraph/actions/my_tools.py`:

```python
from __future__ import annotations

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

from gantrygraph.core.base_action import BaseAction


class MyTools(BaseAction):
    """Short description."""

    def __init__(self, *, my_config: str) -> None:
        self._config = my_config

    def get_tools(self) -> list[BaseTool]:
        return [self._do_thing_tool()]

    def _do_thing_tool(self) -> BaseTool:
        config = self._config

        class _Args(BaseModel):
            input: str = Field(description="The input string.")

        async def _run(input: str) -> str:
            """Do the thing with the input."""
            return f"{config}: {input}"

        return StructuredTool.from_function(
            coroutine=_run,
            name="do_thing",
            description="Does the thing.",
            args_schema=_Args,
        )

    async def close(self) -> None:
        pass  # clean up connections if needed
```

**Rule:** Tool names must be globally unique when the agent collects all tools. Use a prefix (`my_*`) to avoid collisions with built-in tool names (`file_read`, `shell_run`, `browser_navigate`, etc.).

**Step 2** — Add to `src/gantrygraph/actions/__init__.py` (lazy `__getattr__`).

**Step 3** — Add tests in `tests/unit/test_actions.py` that call `await tool.ainvoke({"input": "x"})`.

---

## 6. How to add a new memory backend

Memory backends persist experiences between runs.

```python
from gantrygraph.memory.base import BaseMemory, MemoryResult

class RedisMemory(BaseMemory):
    def __init__(self, url: str) -> None:
        import redis.asyncio as redis  # guard optional dep
        self._client = redis.from_url(url)

    async def add(self, text: str, metadata: dict | None = None) -> None:
        await self._client.rpush("gantrygraph:memory", text)

    async def search(self, query: str, k: int = 5) -> list[MemoryResult]:
        # Implement semantic search or keyword search.
        raw = await self._client.lrange("gantrygraph:memory", -k, -1)
        return [MemoryResult(text=r.decode(), score=1.0) for r in raw]

    async def close(self) -> None:
        await self._client.close()
```

Guard the optional dep with `try/except ImportError` at module level.  
Add `redis` to the appropriate extra in `pyproject.toml`.

---

## 7. How to add a new LVM wrapper

An LVM (Large Vision Model) wrapper preprocesses messages — overlaying a grid, resizing, adding annotations — before forwarding them to the underlying LLM.

```python
from gantrygraph.lvm.base import BaseVisionProvider
from langchain_core.messages import BaseMessage

class GridOverlayVision(BaseVisionProvider):
    """Overlay a numbered grid on screenshots for precise click targeting."""

    async def _preprocess(self, messages: list[BaseMessage]) -> list[BaseMessage]:
        # Modify multimodal messages in-place or return new ones.
        # Call super()._preprocess(messages) first if you want to chain.
        return [self._add_grid(m) for m in messages]
```

Pass it as `llm=`:

```python
agent = GantryEngine(
    llm=GridOverlayVision(ChatAnthropic(model="claude-sonnet-4-6")),
    perception=DesktopScreen(),
    tools=[MouseKeyboardTools()],
)
```

`BaseVisionProvider` implements `bind_tools()` and `ainvoke()` so it is a drop-in replacement for any `BaseChatModel`.

---

## 8. Security guardrails — what you must not break

The security invariants below must hold for every PR that touches `engine/`, `actions/`, or `security/`:

1. **Workspace restriction** — `FileSystemTools` and `ShellTool` must always call `safe_path()` before touching the filesystem.  `safe_path()` raises `PermissionError` on traversal attempts. Never bypass it.

2. **Approval gate** — `act_node` checks `GuardrailPolicy` and `approval_callback` **before** every tool execution. If you add a new path through `act_node`, you must include the gate.

3. **Error isolation** — tool exceptions must become `ToolMessage(status="error")` so the LLM can self-correct. They must never propagate out of `act_node` and crash the graph.

4. **`import gantrygraph` never fails** — never add a non-optional import at the top of `__init__.py`, `config.py`, `engine/`, `core/`, `memory/base.py`, or `security/`. Optional deps go behind `try/except ImportError` blocks.

---

## 9. Optional extras — the import guard contract

Every module that uses an optional package must follow this pattern:

```python
try:
    import optional_package  # type: ignore[import-not-found]
    _HAS_OPTIONAL = True
except ImportError:
    _HAS_OPTIONAL = False
```

And raise a helpful error on first use:

```python
_INSTALL_MSG = (
    "MyFeature requires the [myextra] extra: "
    "pip install 'gantrygraph[myextra]'"
)

class MyClass:
    def __init__(self) -> None:
        if not _HAS_OPTIONAL:
            raise ImportError(_INSTALL_MSG)
```

Add the new extra to `pyproject.toml` under `[project.optional-dependencies]`.

---

## 10. Testing conventions

| Test category | Location | Speed | Requires |
|---------------|----------|-------|----------|
| Unit | `tests/unit/` | ~2 s | Nothing real |
| Integration | `tests/integration/` | ~15 s | MCP subprocess |
| E2E | `tests/e2e/` | ~30 s | Real filesystem |

**Unit test rules:**

- Use `FakeMessagesListChatModel` from `langchain_core.language_models.fake_chat_models` — never a real LLM.
- Use `tmp_path` (pytest fixture) for any filesystem work.
- Never hit the network. Mock external calls with `unittest.mock`.
- Every new public symbol needs at least one test for the happy path and one for the error path.

**Async tests:**

```python
import pytest

@pytest.mark.asyncio
async def test_my_tool() -> None:
    result = await my_tool.ainvoke({"x": "hello"})
    assert result == "hello"
```

`asyncio_mode = "auto"` is set in `pyproject.toml`, so `@pytest.mark.asyncio` is optional but recommended for clarity.

---

## 11. Code style

- **mypy strict** — every file must pass `mypy src/gantrygraph --strict`. No `Any` escape hatches unless the external library is truly untyped and documented inline.
- **ruff** — `ruff check src/ tests/` and `ruff format src/ tests/` must both pass.
- **No comments explaining WHAT** — use self-documenting names. Add a comment only when the WHY is non-obvious (hidden constraint, subtle invariant, workaround for a specific upstream bug).
- **No `__future__` annotations on new files** — Python 3.11 supports native `X | Y` syntax. Use it.
- **Docstrings on public symbols** — one-line summary for tools (the LLM reads it), full docstring for public classes.

---

## 12. Opening a pull request

1. Fork the repository and create a feature branch: `git checkout -b feat/my-feature`.
2. Run the full check suite (see [§3](#3-setting-up-the-dev-environment)).
3. Open a PR against `main`. The title should be: `feat: add X`, `fix: Y`, `docs: Z`, etc.
4. The PR description must include:
   - What the change does.
   - Why it belongs in core (vs. a separate package).
   - Which tests cover it.
5. CI will run `pytest`, `mypy`, and `ruff` automatically via GitHub Actions.

**First-time contributors:** look for issues tagged `good first issue`.

---

Questions? Open a GitHub Discussion or file an issue.
