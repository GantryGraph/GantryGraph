# Human-in-the-Loop

Human-in-the-loop (HITL) puts a human approval gate in front of specific tool
calls.  The agent pauses before executing a dangerous or irreversible action,
waits for a decision, and then continues — or skips the action if denied.

GantryGraph supports two HITL modes:

| Mode | How it works | Requires checkpointer |
|------|-------------|----------------------|
| **Callback** | `approval_callback` is called inline inside `act_node`. If it blocks (e.g. waits for user input), the process hangs. Best for CLIs and dev environments. | No |
| **Suspend / resume** | The agent raises `AgentSuspended`, releases all resources, and waits. Resume it later from any process. | Yes (auto-created if omitted) |

---

## Callback mode

### Inline approval

The simplest gate: a Python callable that returns `True` (allow) or `False` (deny).

```python
from gantrygraph import GantryEngine
from gantrygraph.actions import ShellTools
from gantrygraph.security import GuardrailPolicy
from langchain_anthropic import ChatAnthropic

async def my_approval(tool_name: str, args: dict) -> bool:
    print(f"\n⚠  Agent wants to run: {tool_name}({args})")
    answer = input("  Allow? [y/N] ")
    return answer.lower() == "y"

agent = GantryEngine(
    llm=ChatAnthropic(model="claude-sonnet-4-6"),
    tools=[ShellTools(workspace="/app")],
    guardrail=GuardrailPolicy(requires_approval={"shell_run"}),
    approval_callback=my_approval,
)

result = agent.run("Run the test suite and fix any failures.")
```

!!! tip "Sync callbacks work too"
    `approval_callback` can be a plain `def` — GantryGraph wraps it with
    `ensure_awaitable` internally.

### Tag tools as destructive

Use `@gantry_tool(destructive=True)` to mark a custom tool for automatic
inclusion in the guardrail — no manual `GuardrailPolicy` needed:

```python
from gantrygraph import GantryEngine, gantry_tool
from langchain_anthropic import ChatAnthropic

@gantry_tool(destructive=True)
def purge_cache(region: str) -> str:
    """Flush the CDN cache for a given region."""
    cdn.purge(region)
    return f"Cache purged in {region}."

agent = GantryEngine(
    llm=ChatAnthropic(model="claude-sonnet-4-6"),
    tools=[purge_cache],
    approval_callback=lambda name, args: input(f"Allow {name}? [y/N] ") == "y",
)
```

### HTTP approval webhook

`WebhookApprovalCallback` POSTs the tool call to any URL and waits for
`{"approved": bool}`. Works with Slack bots, PagerDuty, or custom dashboards
without adding extra dependencies.

```python
from gantrygraph import GantryEngine
from gantrygraph.actions import ShellTools
from gantrygraph.security import GuardrailPolicy, WebhookApprovalCallback
from langchain_anthropic import ChatAnthropic

agent = GantryEngine(
    llm=ChatAnthropic(model="claude-sonnet-4-6"),
    tools=[ShellTools(workspace="/app")],
    guardrail=GuardrailPolicy(requires_approval={"shell_run"}),
    approval_callback=WebhookApprovalCallback(
        url="https://myapp.example.com/gantry/approve",
        timeout=300.0,              # wait up to 5 minutes for a decision
        extra_headers={"X-API-Key": "secret"},
    ),
)
```

The endpoint receives:

```json
{"tool": "shell_run", "args": {"command": "rm -rf /tmp/old"}}
```

And must respond with:

```json
{"approved": true}
```

---

## Suspend / resume mode

Suspension externalises the approval wait — the agent raises `AgentSuspended`,
your server returns a response immediately, and a human approves later.
Pair this with a durable checkpointer so the state survives the wait.

### Basic example

```python
from gantrygraph import GantryEngine, AgentSuspended
from gantrygraph.actions import ShellTools
from gantrygraph.security import GuardrailPolicy
from langchain_anthropic import ChatAnthropic

agent = GantryEngine(
    llm=ChatAnthropic(model="claude-sonnet-4-6"),
    tools=[ShellTools(workspace="/app")],
    guardrail=GuardrailPolicy(requires_approval={"shell_run"}),
    enable_suspension=True,        # no checkpointer arg needed — MemorySaver is auto-created
)

try:
    result = await agent.arun("Deploy the new release.", thread_id="deploy-99")
except AgentSuspended as exc:
    print(f"Suspended at thread {exc.thread_id} — waiting for approval")
    # Save exc.thread_id somewhere, notify a human...

# Later (in the same process):
result = await agent.resume("deploy-99", approved=True)
print(result)
```

### With a persistent checkpointer (production)

Use SQLite or Postgres so the agent state survives process restarts:

=== "SQLite (dev / single-server)"

    ```python
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    from gantrygraph import GantryEngine, AgentSuspended
    from gantrygraph.actions import ShellTools
    from gantrygraph.security import GuardrailPolicy
    from langchain_anthropic import ChatAnthropic

    checkpointer = AsyncSqliteSaver.from_conn_string("gantry_state.db")

    agent = GantryEngine(
        llm=ChatAnthropic(model="claude-sonnet-4-6"),
        tools=[ShellTools(workspace="/app")],
        guardrail=GuardrailPolicy(requires_approval={"shell_run"}),
        checkpointer=checkpointer,
        enable_suspension=True,
    )
    ```

=== "Postgres (production)"

    ```python
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    from gantrygraph import GantryEngine, AgentSuspended
    from gantrygraph.actions import ShellTools
    from gantrygraph.security import GuardrailPolicy
    from langchain_anthropic import ChatAnthropic
    import os

    checkpointer = AsyncPostgresSaver.from_conn_string(
        os.environ["DATABASE_URL"]
    )
    await checkpointer.setup()

    agent = GantryEngine(
        llm=ChatAnthropic(model="claude-sonnet-4-6"),
        tools=[ShellTools(workspace="/app")],
        guardrail=GuardrailPolicy(requires_approval={"shell_run"}),
        checkpointer=checkpointer,
        enable_suspension=True,
    )
    ```

### FastAPI integration

A minimal REST server where `/run` starts the agent and `/approve/{thread_id}`
resumes it:

```python
from fastapi import FastAPI
from pydantic import BaseModel
from gantrygraph import GantryEngine, AgentSuspended
from gantrygraph.actions import ShellTools
from gantrygraph.security import GuardrailPolicy
from langchain_anthropic import ChatAnthropic

app = FastAPI()

agent = GantryEngine(
    llm=ChatAnthropic(model="claude-sonnet-4-6"),
    tools=[ShellTools(workspace="/app")],
    guardrail=GuardrailPolicy(requires_approval={"shell_run"}),
    enable_suspension=True,
)

class RunRequest(BaseModel):
    task: str
    thread_id: str

class ApproveRequest(BaseModel):
    approved: bool

@app.post("/run")
async def run(req: RunRequest):
    try:
        result = await agent.arun(req.task, thread_id=req.thread_id)
        return {"status": "done", "result": result}
    except AgentSuspended as exc:
        return {"status": "suspended", "thread_id": exc.thread_id}

@app.post("/approve/{thread_id}")
async def approve(thread_id: str, req: ApproveRequest):
    try:
        result = await agent.resume(thread_id, approved=req.approved)
        return {"status": "done", "result": result}
    except AgentSuspended as exc:
        return {"status": "suspended_again", "thread_id": exc.thread_id}
```

!!! warning "Multi-process deployments"
    `agent.resume()` requires the same checkpointer instance (or one pointed
    at the same backing store) that was used during `arun()`.
    In a multi-process deployment, all workers must share the Postgres DSN.

---

## Choosing between callback and suspend / resume

Use **callback mode** when:

- You are running interactively (CLI, notebook, dev environment).
- The approval can complete quickly (human is watching the terminal).
- You do not need state durability across restarts.

Use **suspend / resume** when:

- The approval may take minutes or hours (async Slack workflow, email review).
- You are running in a cloud environment where processes are ephemeral.
- You want to resume from a different process or server replica.

See the [State Persistence guide](persistence.md) for checkpointer setup details.
