# gantrygraph

Autonomous desktop agent framework. Developers import it; it is not a product.

## Setup

```bash
uv sync --all-extras          # install with all optional deps
# fallback:
pip install -e ".[all,dev]"
```

## Commands

```bash
pytest tests/unit/            # unit tests (no real display, no real LLM needed)
pytest tests/integration/     # integration tests (requires MCP subprocess + display)
mypy src/gantrygraph --strict        # type checking
ruff check src/ tests/        # linting
ruff format src/ tests/       # formatting
```

## Architecture

```
src/gantrygraph/
  core/       – ABCs and shared types (no side effects, no I/O)
  engine/     – LangGraph graph wiring + GantryEngine facade
  perception/ – desktop screenshot (mss+PIL), web accessibility (playwright)
  actions/    – pyautogui tools, playwright browser, filesystem, shell
  mcp/        – MCP client + dynamic StructuredTool generation
  security/   – policy objects passed into GantryEngine
  swarm/      – multi-agent supervisor pattern
  cloud/      – FastAPI serve(), Dockerfile template
```

## Key design decisions

- `BasePerception`, `BaseAction`, `BaseMCPConnector` use ABC — engine needs `isinstance` checks and subclassers get IDE errors for missing methods
- `PerceptionResult`, `GuardrailPolicy`, `BudgetPolicy` use Pydantic — serialisable cross-module data
- `GantryState` uses TypedDict — hard requirement of LangGraph StateGraph
- `approval_callback` and `on_event` are Protocol callables — supports both sync and async via `ensure_awaitable`
- HITL runs *inside* `act_node` pre-execution — no checkpointer required for simple case; advanced users call `engine.get_graph()`
- MCP connectors are async context managers — they own a subprocess lifetime
- Optional extras are guarded at module-level with `try: import X` — `import gantrygraph` never fails; `from gantrygraph.perception import DesktopScreen` fails with a clear install hint if the extra is missing
- All graph nodes are `async def` — avoids blocking the event loop during I/O-heavy perception
- `GantryEngine.run()` calls `_run_sync(arun())` with a Jupyter-compatible fallback; `.arun()` is the primary method

## Optional extras

| Extra       | Installs            | Unlocks                          |
|-------------|---------------------|----------------------------------|
| `[desktop]` | `pyautogui`         | `MouseKeyboardTools`             |
| `[browser]` | `playwright`        | `BrowserTools`, `WebPage`        |
| `[cloud]`   | `fastapi`, `uvicorn`| `gantrygraph.cloud.serve()`             |
| `[dev]`     | test + type tools   | pytest, mypy, ruff               |
