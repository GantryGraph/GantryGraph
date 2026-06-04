# GantryGraph

**Give your AI agent eyes, hands, and memory — on any screen.**

GantryGraph is a Python framework that lets an LLM autonomously navigate websites, control desktop apps, run shell commands, and remember what it has done — without you writing a single line of browser or UI automation code.

[![PyPI](https://img.shields.io/pypi/v/gantrygraph)](https://pypi.org/project/gantrygraph/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://pypi.org/project/gantrygraph/)

---

## What can it do?

| Use case | How |
|---|---|
| Scrape a dynamic SPA, fill out a form, handle login | Browser automation + stealth mode |
| Open a legacy ERP that has no API and extract data | Desktop control (screenshot + mouse/keyboard) |
| Read macOS apps without screenshots — 10× cheaper | Native Accessibility API (AXUIElement) |
| Remember what it saw at step 2 when it reaches step 50 | Vector memory with TTL (entries auto-expire) |
| Connect to GitHub, Jira, Notion or any MCP service | MCP client — no wrapper code needed |
| Run 4 sub-agents in parallel and merge their results | Built-in swarm supervisor |
| Stop before a dangerous shell command and ask a human | Human-in-the-loop (suspend / resume) |

---

## Quickstart

```bash
pip install gantrygraph langchain-anthropic
```

```python
from gantrygraph import GantryEngine
from langchain_anthropic import ChatAnthropic

agent = GantryEngine(llm=ChatAnthropic(model="claude-sonnet-4-6"))
print(agent.run("What is the capital of France?"))
```

The agent runs an `observe → think → act → review` loop until the task is done.
`run()` is synchronous; use `arun()` in async code.

---

## Install by use case

```bash
# Core — MCP, swarm, security, memory (no GUI)
pip install gantrygraph

# Web scraping / browser automation
pip install 'gantrygraph[browser]'
playwright install chromium

# Desktop control — mouse, keyboard, screenshots
pip install 'gantrygraph[desktop]'

# macOS native app control — zero vision tokens (AX tree)
pip install 'gantrygraph[desktop-ax]'

# Persistent semantic memory (ChromaDB)
pip install 'gantrygraph[memory]'

# Ultra-light TTL memory — Rust HNSW, entries auto-expire
pip install 'gantrygraph[minivecdb]'

# REST server (POST /run, SSE streaming)
pip install 'gantrygraph[cloud]'

# Everything (except desktop-ax — macOS only)
pip install 'gantrygraph[all]'
```

---

## Examples

### Automate a website

```python
from gantrygraph import GantryEngine
from gantrygraph.perception import WebPage
from gantrygraph.actions import BrowserTools
from langchain_anthropic import ChatAnthropic

web = WebPage(url="https://app.example.com", headless=True)

agent = GantryEngine(
    llm=ChatAnthropic(model="claude-sonnet-4-6"),
    perception=web,
    tools=[BrowserTools(web_page=web)],
    max_steps=30,
)
agent.run("Log in, go to Reports, download the CSV for last month.")
```

The agent sees the page, clicks buttons, fills forms — exactly like a human would.
Stealth mode is on by default: no bot-detection walls.

---

### Control the desktop (any app, no API needed)

```python
from gantrygraph import GantryEngine
from gantrygraph.perception import DesktopScreen
from gantrygraph.actions import MouseKeyboardTools
from langchain_anthropic import ChatAnthropic

agent = GantryEngine(
    llm=ChatAnthropic(model="claude-sonnet-4-6"),
    perception=DesktopScreen(),
    tools=[MouseKeyboardTools()],
    max_steps=20,
)
agent.run("Open the invoicing software, find invoice #4821, and mark it as paid.")
```

Works with any GUI application — legacy ERP, accounting software, internal tools.
No API, no SDK, no reverse engineering.

---

### Read native macOS apps without screenshots (10× cheaper)

```bash
pip install 'gantrygraph[desktop-ax]'
# Grant: System Settings → Privacy & Security → Accessibility
```

```python
from gantrygraph.perception import DesktopAXTree

agent = GantryEngine(
    llm=ChatAnthropic(model="claude-sonnet-4-6"),
    perception=DesktopAXTree(app_name="Obsidian"),
    tools=[MouseKeyboardTools()],
    max_steps=20,
)
agent.run("Find the note 'Q2 Goals' and add a bullet point with today's KPIs.")
```

Instead of sending a screenshot to the vision model, `DesktopAXTree` reads the
macOS Accessibility API and sends structured text — buttons, labels, text fields.
~90% fewer tokens per step vs a screenshot.

---

### Add memory so the agent remembers across steps

Long tasks saturate the LLM context. Attach a memory store and the agent
automatically saves what it learns and recalls it when relevant.

```python
from gantrygraph.memory import MiniVecDbMemory
from langchain_openai import OpenAIEmbeddings

embed = OpenAIEmbeddings(model="text-embedding-3-small").embed_query

agent = GantryEngine(
    llm=...,
    memory=MiniVecDbMemory(
        embed_fn=embed,
        ttl_ms=300_000,   # entries expire after 5 min — no stale context
    ),
    max_steps=50,
)
```

`MiniVecDbMemory` is backed by a Rust HNSW engine — 48 bytes per vector,
32× less RAM than ChromaDB. TTL makes old observations decay automatically,
the same way human working memory does.

For persistent memory across runs use `ChromaMemory` (`pip install 'gantrygraph[memory]'`).

---

### Connect any external service via MCP

No wrapper code. No SDK. Any MCP-compatible server becomes a tool in one line.

```python
from gantrygraph.mcp import MCPClient

async with MCPClient("npx -y @modelcontextprotocol/server-github") as mcp:
    agent = GantryEngine(llm=..., tools=[mcp])
    await agent.arun("Open a PR that adds a CHANGELOG entry for v1.2.0.")
```

Works with GitHub, Jira, Notion, Slack, filesystem servers, databases — anything
in the [MCP ecosystem](https://modelcontextprotocol.io).

---

### Run multiple agents in parallel

```python
from gantrygraph.swarm import GantrySupervisor

supervisor = GantrySupervisor(
    llm=my_llm,
    worker_factory=lambda: GantryEngine(llm=my_llm, tools=[...]),
    max_workers=4,
)
result = supervisor.run("Analyse Q1–Q4 reports and produce an annual summary.")
```

`GantrySupervisor` decomposes the task, dispatches to parallel workers,
and synthesises the result. Each worker is a full `GantryEngine`.

---

### Add human approval before risky actions

```python
from gantrygraph.security import GuardrailPolicy

agent = GantryEngine(
    llm=...,
    guardrail=GuardrailPolicy(requires_approval={"file_delete", "shell_run"}),
    approval_callback=lambda tool, args: input(f"Allow {tool}? [y/N] ") == "y",
)
```

The agent pauses before any listed tool and waits for your answer.
For async / webhook flows, use `enable_suspension=True` with `agent.resume()`.

---

### Write a custom tool in 1 line

```python
from gantrygraph import GantryEngine, gantry_tool

@gantry_tool
async def get_invoice(invoice_id: str) -> str:
    """Fetch an invoice from the ERP system and return its details."""
    return await erp_client.fetch(invoice_id)

agent = GantryEngine(llm=..., tools=[get_invoice])
agent.run("Find all overdue invoices and send a summary by email.")
```

Any `def` or `async def` becomes an LLM-callable tool.
The docstring is the tool description — make it specific.

---

## Security layers

```python
from gantrygraph.security import (
    GuardrailPolicy, WorkspacePolicy,
    BudgetPolicy, ShellDenylist, GantrySecrets,
)
import os

agent = GantryEngine(
    llm=...,
    workspace_policy=WorkspacePolicy.restricted("/app"),           # sandbox FS + shell
    guardrail=GuardrailPolicy(requires_approval={"shell_run"}),    # human gate
    budget=BudgetPolicy(max_steps=50, max_wall_seconds=300),       # cost cap
    secrets=GantrySecrets({"DB_PASS": os.environ["DB_PASSWORD"]}), # keeps creds off-context
    tools=[ShellTools(denylist=ShellDenylist.strict())],            # blocks rm -rf, curl|bash
)
```

| Layer | What it does |
|---|---|
| `WorkspacePolicy` | Restrict file and shell tools to allowed directories |
| `GuardrailPolicy` | Require human sign-off before specific tools run |
| `BudgetPolicy` | Hard cap on steps, tokens, and wall-clock time |
| `ShellDenylist` | Block `rm -rf /`, fork bombs, `curl\|bash`, SSH key reads |
| `GantrySecrets` | Inject credentials without exposing them to the LLM |

---

## Why GantryGraph vs alternatives?

| | GantryGraph | Raw LangGraph | AutoGen |
|---|---|---|---|
| Visual computer use (screenshot + click) | ✅ built-in | ❌ | ❌ |
| macOS native app control (AX tree) | ✅ built-in | ❌ | ❌ |
| MCP tool servers | ✅ built-in | ❌ | ❌ |
| TTL vector memory (Rust HNSW) | ✅ built-in | ❌ | ❌ |
| Human-in-the-loop (suspend / resume) | ✅ | manual | partial |
| Stealth browser (bot-detection bypass) | ✅ | ❌ | ❌ |
| Multi-agent swarm | ✅ built-in | manual | ✅ |
| Security layers (sandbox, denylist, budget) | ✅ built-in | ❌ | ❌ |
| `@gantry_tool` — any function in 1 line | ✅ | ❌ | partial |
| `import gantrygraph` never fails | ✅ | — | — |
| Strict-typed (mypy strict) | ✅ | partial | ❌ |

---

## Architecture

```
gantrygraph/
  engine/       observe → think → act → review loop (LangGraph)
  perception/   DesktopScreen · DesktopAXTree · WebPage · MultiPerception
  actions/      BrowserTools · MouseKeyboardTools · FileSystemTools · ShellTools
  memory/       InMemoryStore · ChromaMemory · MiniVecDbMemory (Rust HNSW + TTL)
  mcp/          MCPClient — any MCP server → LangChain tools, automatically
  swarm/        GantrySupervisor — parallel worker agents
  security/     GuardrailPolicy · WorkspacePolicy · BudgetPolicy · ShellDenylist
  cloud/        FastAPI REST server + SSE streaming
  telemetry/    OpenTelemetry span exporter
  vision/       SetOfMarkAnnotator · Downsample · ConvertToWebP · Grayscale
  tool.py       @gantry_tool decorator
```

The loop:

```
START → memory_recall → observe → think → act → review
                                                  │
                             ┌────────────────────┘
                             ▼
                         is_done or max_steps?
                            │yes         │no
                           END         observe
```

---

## Full docs

**[gantrygraph.com](https://gantrygraph.com)** — quickstart, how-to guides, API reference.

---

## Development

```bash
git clone https://github.com/GantryGraph/GantryGraph
cd GantryGraph
pip install -e ".[all,dev]"

pytest tests/unit/           # fast, no display needed
pytest tests/integration/    # needs MCP subprocess + Playwright
mypy src/gantrygraph --strict
ruff check src/ tests/
```

See [CONTRIBUTING.md](CONTRIBUTING.md).

---

## License

MIT — see [LICENSE](LICENSE).
