# State Persistence & Resume

By default, `GantryEngine` keeps all state in memory — the moment the process
exits, the run is gone. Persistence lets you:

- **Survive restarts**: pick up a long-running task after a server reboot or crash.
- **Resume HITL gates**: suspend an agent waiting for human approval, shut the
  server down, and resume it hours later.
- **Audit & replay**: replay any past run step-by-step for debugging.

GantryGraph uses LangGraph's checkpointer abstraction, so any compatible
checkpointer works — SQLite for development, Postgres for production.

## Choosing a backend

| Backend | Extra | Best for |
|---------|-------|----------|
| In-memory (default) | none | Short scripts, unit tests |
| SQLite | `langgraph-checkpoint-sqlite` | Local dev, single-server deploys |
| Postgres | `langgraph-checkpoint-postgres` | Production, multi-replica |

## In-memory (default)

No extra setup required. State lives in a Python `dict` for the duration of
the process. If you don't pass `checkpointer=`, this is what you get.

```python
from gantrygraph import GantryEngine
from langchain_anthropic import ChatAnthropic

agent = GantryEngine(
    llm=ChatAnthropic(model="claude-sonnet-4-6"),
    max_steps=20,
)

result = await agent.arun("Summarise the README.")
```

!!! warning
    In-memory state is lost when the process exits. Do not rely on it
    for tasks that span multiple requests or server restarts.

## SQLite (local / dev)

=== "Install"

    ```bash
    pip install langgraph-checkpoint-sqlite
    ```

=== "Setup"

    ```python
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    from gantrygraph import GantryEngine
    from langchain_anthropic import ChatAnthropic

    checkpointer = AsyncSqliteSaver.from_conn_string("gantry_state.db")

    agent = GantryEngine(
        llm=ChatAnthropic(model="claude-sonnet-4-6"),
        checkpointer=checkpointer,
        max_steps=30,
    )
    ```

=== "Run with thread_id"

    ```python
    # First run — or resume if the thread already exists
    result = await agent.arun("Refactor the auth module.", thread_id="run-001")
    print(result)
    ```

    Pass the same `thread_id` across calls to resume from where the agent
    left off. If the thread is new, the run starts from scratch.

The SQLite file is created automatically in the current directory.
For a shared location use an absolute path:

```python
checkpointer = AsyncSqliteSaver.from_conn_string("/var/lib/myapp/gantry.db")
```

!!! tip
    SQLite is single-writer. For a single-server staging environment it works
    perfectly; for multi-process or multi-replica deploys, switch to Postgres.

## Postgres (production)

=== "Install"

    ```bash
    pip install langgraph-checkpoint-postgres
    ```

=== "Setup"

    ```python
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    from gantrygraph import GantryEngine
    from langchain_anthropic import ChatAnthropic

    checkpointer = AsyncPostgresSaver.from_conn_string(
        "postgresql://user:password@db-host:5432/gantry"
    )

    # Create checkpoint tables on first run (idempotent)
    await checkpointer.setup()

    agent = GantryEngine(
        llm=ChatAnthropic(model="claude-sonnet-4-6"),
        checkpointer=checkpointer,
        max_steps=50,
    )
    ```

=== "Run"

    ```python
    result = await agent.arun(
        "Run the full regression suite and file a GitHub issue for any failures.",
        thread_id="nightly-2026-05-07",
    )
    ```

!!! tip
    Store the connection string in an environment variable and read it with
    `os.environ["DATABASE_URL"]` rather than hard-coding credentials.

## Suspend / resume (HITL)

When `enable_suspension=True`, approval gates raise `AgentSuspended` instead
of blocking. Pair this with a persistent checkpointer so the agent state
survives the wait.

```python
from gantrygraph import GantryEngine, AgentSuspended
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langchain_anthropic import ChatAnthropic

checkpointer = AsyncSqliteSaver.from_conn_string("gantry_state.db")

agent = GantryEngine(
    llm=ChatAnthropic(model="claude-sonnet-4-6"),
    checkpointer=checkpointer,
    enable_suspension=True,
)

# --- In your request handler ---
try:
    result = await agent.arun("Deploy to production.", thread_id="deploy-42")
except AgentSuspended as exc:
    # Persist exc.thread_id, notify a human, and return early.
    save_pending_approval(exc.thread_id)

# --- Later, after human approval ---
result = await agent.resume(thread_id="deploy-42", approved=True)
```

!!! warning
    `agent.resume()` requires the same `checkpointer` instance (or one pointed
    at the same backing store) that was used during the original `arun()` call.
    In a multi-process deployment, make sure all workers share the Postgres DSN.

See the [HITL guide](../concepts/engine.md) for a deeper walkthrough of
approval callbacks and suspension mechanics.
