# gantrygraph

**Autonomous agent framework for Python.** Screenshot → think → act. LangGraph inside. Zero boilerplate outside.

[![PyPI](https://img.shields.io/pypi/v/gantrygraph)](https://pypi.org/project/gantrygraph/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://pypi.org/project/gantrygraph/)

```python
from gantrygraph import GantryEngine, gantry_tool
from gantrygraph.perception import DesktopScreen
from gantrygraph.actions import MouseKeyboardTools
from langchain_anthropic import ChatAnthropic

@gantry_tool
async def read_jira(ticket_id: str) -> str:
    """Fetch a Jira ticket and return its description."""
    return await jira_client.get(ticket_id)

agent = GantryEngine(
    llm=ChatAnthropic(model="claude-sonnet-4-6"),
    perception=DesktopScreen(),
    tools=[MouseKeyboardTools(), read_jira],
    max_steps=50,
)
agent.run("Open PROJ-123 in Jira and submit the fix.")
```

Full docs at [gantrygraph.com/docs](https://gantrygraph.com/docs).

---

## Why gantrygraph?

| | gantrygraph | Raw LangGraph | AutoGen |
|---|---|---|---|
| Visual computer use (screenshot + click) | ✅ built-in | ❌ | ❌ |
| MCP tool servers | ✅ built-in | ❌ | ❌ |
| `@gantry_tool` — any function in 1 line | ✅ | ❌ | partial |
| Human-in-the-loop (suspend / resume) | ✅ | manual | partial |
| `import gantrygraph` never fails | ✅ | — | — |
| Strict-typed (mypy strict) | ✅ | partial | ❌ |

---

## Install

```bash
# Core only (no GUI, no browser, no cloud)
pip install gantrygraph

# Desktop automation (screenshot + mouse/keyboard)
pip install 'gantrygraph[desktop]'

# Web scraping / form filling
pip install 'gantrygraph[browser]'
playwright install chromium

# REST server (POST /run, SSE streaming)
pip install 'gantrygraph[cloud]'

# Everything
pip install 'gantrygraph[all]'
```

---

## Common patterns

### Add a custom tool

```python
from gantrygraph import GantryEngine, gantry_tool
from langchain_anthropic import ChatAnthropic

@gantry_tool
def search_orders(query: str, limit: int = 5) -> str:
    """Search the order management system by keyword."""
    rows = db.execute("SELECT * FROM orders WHERE ... LIMIT ?", query, limit)
    return "\n".join(str(r) for r in rows)

agent = GantryEngine(
    llm=ChatAnthropic(model="claude-sonnet-4-6"),
    tools=[search_orders],
)
agent.run("Find all overdue orders and summarise them.")
```

Use `@gantry_tool(destructive=True)` to tag a tool as destructive — the engine automatically gates it behind `approval_callback` without any manual guardrail config.

### Filesystem and shell with security

```python
from gantrygraph import GantryEngine
from gantrygraph.actions import FileSystemTools, ShellTools
from gantrygraph.security import WorkspacePolicy, ShellDenylist, BudgetPolicy
from langchain_anthropic import ChatAnthropic

agent = GantryEngine(
    llm=ChatAnthropic(model="claude-sonnet-4-6"),
    workspace_policy=WorkspacePolicy.restricted("/app"),   # auto-wires FS + shell
    tools=[
        ShellTools(
            workspace="/app",
            allowed_commands=["pytest", "ruff", "git"],
            denylist=ShellDenylist.strict(),               # blocks rm -rf, curl|bash, etc.
        ),
    ],
    budget=BudgetPolicy(max_steps=30, max_wall_seconds=120.0),
)
agent.run("Run the test suite, fix any lint errors, and commit the result.")
```

### Connect an MCP server

```python
from gantrygraph import GantryEngine
from gantrygraph.mcp import MCPClient
from langchain_anthropic import ChatAnthropic

async def main():
    async with MCPClient("npx -y @modelcontextprotocol/server-github") as mcp:
        agent = GantryEngine(
            llm=ChatAnthropic(model="claude-sonnet-4-6"),
            tools=[mcp],
            max_steps=20,
        )
        result = await agent.arun("Open a PR that adds a CHANGELOG entry for v1.2.0.")
        print(result)
```

---

## Architecture

```
gantrygraph/
  core/         ABCs and shared types — no I/O, no side effects
  engine/       LangGraph graph wiring (observe → think → act → review)
  perception/   Desktop screenshot (mss+PIL), web accessibility (Playwright)
  actions/      Mouse/keyboard (pyautogui), browser (Playwright), filesystem, shell
  mcp/          MCP client — dynamic StructuredTool generation from any MCP server
  memory/       InMemoryVector, ChromaDB
  security/     GuardrailPolicy, WorkspacePolicy, BudgetPolicy, ShellDenylist, GantrySecrets
  swarm/        Multi-agent supervisor pattern
  cloud/        FastAPI REST server + SSE streaming
  telemetry/    OpenTelemetry span exporter
  tool.py       @gantry_tool decorator
```

The agent loop is a LangGraph `StateGraph`:

```
START → memory_recall → observe → think → act → review
                                                   │
                              ┌────────────────────┘
                              ▼
                          is_done or max_steps?
                              │yes          │no
                             END          observe
```

Every node is a pure `async def`. Callbacks (`on_event`, `approval_callback`) are called inside the nodes and support both `def` and `async def` via `ensure_awaitable`.

---

## Security

GantryGraph ships with layered, opt-in security controls. The defaults are safe; every layer can be tuned or disabled as needed.

```python
import os
from gantrygraph import GantryEngine, gantry_tool
from gantrygraph.actions import ShellTools
from gantrygraph.security import (
    GuardrailPolicy, WorkspacePolicy, BudgetPolicy,
    ShellDenylist, GantrySecrets,
)

# Tag a custom tool as destructive — auto-added to the approval gate
@gantry_tool(destructive=True)
def drop_table(table: str) -> str:
    """Drop a database table permanently."""
    ...

agent = GantryEngine(
    llm=my_llm,
    tools=[
        ShellTools(
            workspace="/app",
            denylist=ShellDenylist.strict(),   # blocks rm -rf /, dd wipe, curl|bash, etc.
        ),
        drop_table,                             # auto-requires approval
    ],
    workspace_policy=WorkspacePolicy.restricted("/app"),
    guardrail=GuardrailPolicy(requires_approval={"shell_run"}),
    budget=BudgetPolicy(max_steps=50, max_tokens=20_000, max_wall_seconds=300),
    approval_callback=lambda tool, args: input(f"Allow {tool}({args})? [y/N] ") == "y",
    secrets=GantrySecrets({"DB_PASS": os.environ["DB_PASSWORD"]}),
)
```

| Layer | Class | What it does |
|-------|-------|-------------|
| Approval gate | `GuardrailPolicy` | Require human sign-off before listed tools run |
| Auto-approval | `@gantry_tool(destructive=True)` | Tag a tool as destructive — auto-added to the gate |
| Shell firewall | `ShellDenylist` | Block `rm -rf /`, fork bombs, `curl\|bash`, SSH key reads before the OS sees them |
| Blind secrets | `GantrySecrets` | Keep credentials out of the LLM context window |
| Path sandbox | `WorkspacePolicy` | Restrict file/shell tools to allowed directories |
| Cost cap | `BudgetPolicy` | Hard limit on steps, tokens, and wall-clock time |

---

## Development

```bash
git clone https://github.com/GantryGraph/GantryGraph
cd gantrygraph
pip install -e ".[all,dev]"

pytest tests/unit/           # fast, no display needed
pytest tests/integration/    # needs MCP subprocess
mypy src/gantrygraph --strict
ruff check src/ tests/
ruff format src/ tests/
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full contributor guide.

---

## License

MIT — see [LICENSE](LICENSE).
