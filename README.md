# gantrygraph

**Autonomous agent framework for Python.**  
Screenshot → think → act. LangGraph inside. Zero boilerplate outside.

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
pip install playwright
playwright install chromium

# REST server (POST /run, SSE streaming)
pip install 'gantrygraph[cloud]'

# Everything
pip install 'gantrygraph[all]'
```

---

## Quick-start guides

### 1 — Use a preset (zero configuration)

```python
from gantrygraph.presets import qa_agent
from langchain_anthropic import ChatAnthropic

agent = qa_agent(
    llm=ChatAnthropic(model="claude-sonnet-4-6"),
    workspace="/my/project",
)
result = agent.run("Run the test suite and fix any failures.")
print(result)
```

Available presets: `qa_agent`, `desktop_agent`, `browser_agent`, `mcp_agent`, `cloud_agent`.

---

### 2 — Add your own tools with `@gantry_tool`

```python
from gantrygraph import GantryEngine, gantry_tool
from langchain_anthropic import ChatAnthropic

@gantry_tool
def search_orders(query: str, limit: int = 5) -> str:
    """Search the order management system by keyword."""
    rows = db.execute("SELECT * FROM orders WHERE ... LIMIT ?", query, limit)
    return "\n".join(str(r) for r in rows)

@gantry_tool
async def send_slack(channel: str, message: str) -> str:
    """Post a message to a Slack channel."""
    await slack_client.chat_postMessage(channel=channel, text=message)
    return f"Sent to #{channel}."

agent = GantryEngine(
    llm=ChatAnthropic(model="claude-sonnet-4-6"),
    tools=[search_orders, send_slack],
)
agent.run("Find all overdue orders and notify #ops-alerts on Slack.")
```

---

### 3 — Stream events to a WebSocket

```python
import asyncio
from gantrygraph.presets import desktop_agent
from langchain_anthropic import ChatAnthropic

agent = desktop_agent(llm=ChatAnthropic(model="claude-sonnet-4-6"))

async def run_with_stream(websocket):
    async for event in agent.astream_events("Fill in the expense report"):
        if event.event_type == "observe":
            screenshot = event.data.get("screenshot_b64")
            if screenshot:
                await websocket.send_json({"type": "screen", "data": screenshot})
        elif event.event_type == "act":
            await websocket.send_json({
                "type": "action",
                "tools": event.data["tools_executed"],
            })
        elif event.event_type == "done":
            await websocket.send_json({"type": "done"})
```

---

### 4 — Persistent state across sessions (thread isolation)

```python
from langgraph.checkpoint.postgres import PostgresSaver
from gantrygraph import GantryEngine
from gantrygraph.actions.shell import ShellTool

# One checkpointer shared by the whole process
checkpointer = PostgresSaver("postgresql://user:pass@db/prod")

agent = GantryEngine(
    llm=my_llm,
    tools=[ShellTool(workspace="/app")],
    checkpointer=checkpointer,
)

# Each user gets their own isolated memory lane
result = agent.run("Deploy the staging branch.", thread_id="deploy-mario-2025")
```

Crash mid-run? `arun()` resumes exactly where it left off when you pass the same `thread_id`.

---

### 5 — MCP tool servers

```python
from gantrygraph.mcp import MCPClient
from gantrygraph.presets import mcp_agent
from langchain_anthropic import ChatAnthropic

agent = mcp_agent(
    ChatAnthropic(model="claude-sonnet-4-6"),
    "npx -y @modelcontextprotocol/server-filesystem /tmp",
    "npx -y @modelcontextprotocol/server-github",
)
# MCP subprocesses start automatically on first run
result = agent.run("Open a PR that adds a CHANGELOG entry for v1.2.0.")
```

---

### 6 — Load config from YAML or environment variables

**YAML:**
```yaml
# agent.yaml
max_steps: 30
workspace: /app
memory: in_memory
guardrail_requires_approval:
  - shell_run
  - file_delete
```

```python
from gantrygraph import GantryConfig
cfg = GantryConfig.from_yaml("agent.yaml")
agent = cfg.build(llm=my_llm)
```

**Environment variables** (copy `.env.example` → `.env`):
```bash
CLAW_MAX_STEPS=30
CLAW_WORKSPACE=/app
CLAW_MEMORY=in_memory
CLAW_GUARDRAIL_REQUIRES_APPROVAL=shell_run,file_delete
```
```python
cfg = GantryConfig.from_env()
agent = cfg.build(llm=my_llm)
```

---

### 7 — Deploy as a REST server

```python
# server.py
from gantrygraph import GantryEngine
from gantrygraph.actions.shell import ShellTool
from gantrygraph.cloud import serve

def make_agent():
    return GantryEngine(llm=my_llm, tools=[ShellTool(workspace="/app")])

serve(make_agent, host="0.0.0.0", port=8080)
```

```bash
# POST /run  →  { "job_id": "..." }
curl -X POST http://localhost:8080/run \
     -H 'Content-Type: application/json' \
     -d '{"task": "Run the test suite"}'

# GET /stream/{job_id}  →  Server-Sent Events
curl http://localhost:8080/stream/abc123
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
  security/     GuardrailPolicy, WorkspacePolicy, BudgetPolicy
  swarm/        Multi-agent supervisor pattern
  cloud/        FastAPI REST server + SSE streaming
  telemetry/    OpenTelemetry span exporter
  tool.py       @gantry_tool decorator
  config.py     GantryConfig — YAML / env-var driven setup
  presets.py    Ready-made factory functions
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

```python
from gantrygraph import GantryEngine
from gantrygraph.security.policies import GuardrailPolicy, WorkspacePolicy, BudgetPolicy

agent = GantryEngine(
    llm=my_llm,
    tools=[...],
    guardrail=GuardrailPolicy(
        requires_approval={"shell_run", "file_delete"},
    ),
    approval_callback=lambda tool, args: input(f"Allow {tool}({args})? [y/N] ") == "y",
)
```

| Policy | What it does |
|--------|-------------|
| `GuardrailPolicy` | Require human approval before specific tools run |
| `WorkspacePolicy` | Restrict file/shell tools to a declared directory |
| `BudgetPolicy` | Hard cap on steps and wall-clock time |

---

## Extending gantrygraph

### New perception source

```python
from gantrygraph.core.base_perception import BasePerception
from gantrygraph.core.events import PerceptionResult

class TerminalPerception(BasePerception):
    async def observe(self) -> PerceptionResult:
        output = await run_command("git status")
        return PerceptionResult(accessibility_tree=output)

agent = GantryEngine(llm=..., perception=TerminalPerception(), tools=[...])
```

### New action set

```python
from gantrygraph.core.base_action import BaseAction
from langchain_core.tools import BaseTool, StructuredTool

class DatabaseTools(BaseAction):
    def get_tools(self) -> list[BaseTool]:
        return [self._query_tool(), self._insert_tool()]

    def _query_tool(self) -> BaseTool:
        async def _run(sql: str) -> str:
            """Execute a read-only SQL query."""
            return str(await self._db.fetch(sql))
        return StructuredTool.from_function(coroutine=_run, name="db_query",
                                            description="Run a SELECT query.")
```

See `CONTRIBUTING.md` for the full contributor guide.

---

## Development

```bash
git clone https://github.com/GantryGraph/GantryGraph
cd gantrygraph
pip install -e ".[all,dev]"

# All checks
pytest tests/unit/           # fast, no display needed
pytest tests/integration/    # needs MCP subprocess
mypy src/gantrygraph --strict
ruff check src/ tests/
ruff format src/ tests/
```

Copy `.env.example` to `.env` and fill in your API keys before running the examples.

---

## License

MIT — see [LICENSE](LICENSE).
