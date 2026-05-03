# Swarm / Multi-Agent

`GantrySupervisor` decomposes a complex task into subtasks and routes each
to a specialist `GantryEngine` worker. Workers run in parallel via
`asyncio.gather` and their results are synthesised by the LLM into a final answer.

## Homogeneous workers (same engine, N replicas)

```python
from gantrygraph.swarm import GantrySupervisor
from gantrygraph.presets import qa_agent
from langchain_anthropic import ChatAnthropic

llm = ChatAnthropic(model="claude-sonnet-4-6")

supervisor = GantrySupervisor(
    llm=llm,
    worker_factory=lambda: qa_agent(llm, workspace="/app"),
    n_workers=4,
)

result = await supervisor.arun(
    "Run tests, check coverage, lint the codebase, and update CHANGELOG."
)
```

The supervisor LLM splits the task into ≤ `n_workers` subtasks,
dispatches them in parallel, waits for all to finish, and synthesises
the results.

## Heterogeneous workers (different specialist agents)

Use `WorkerSpec` when each worker has a different toolset:

```python
from gantrygraph import GantryEngine, WorkerSpec
from gantrygraph.presets import qa_agent, browser_agent
from gantrygraph.swarm import GantrySupervisor
from langchain_anthropic import ChatAnthropic

llm = ChatAnthropic(model="claude-sonnet-4-6")

supervisor = GantrySupervisor(
    llm=llm,
    workers=[
        WorkerSpec(
            name="code_fixer",
            engine=qa_agent(llm, workspace="/app"),
            description="Fixes Python bugs, runs tests, commits changes.",
        ),
        WorkerSpec(
            name="web_researcher",
            engine=browser_agent(llm, start_url="https://google.com"),
            description="Browses the web and retrieves information.",
        ),
        WorkerSpec(
            name="db_analyst",
            engine=GantryEngine(llm=llm, tools=[DatabaseTools(CONN)]),
            description="Queries the production database for metrics.",
        ),
    ],
)

result = await supervisor.arun(
    "Fix the auth bug, research the OAuth 2.0 spec, and pull the last-week login stats."
)
```

The supervisor LLM assigns subtasks using `[worker_name] subtask` notation.
Each worker runs its subtask independently, then all results are synthesised.

## WorkerResult

Each worker returns a `WorkerResult`:

```python
from gantrygraph.swarm import WorkerResult

# result.results is a list[WorkerResult]
for r in supervisor_result.results:
    print(r.worker_name, r.result, r.metadata)
```

`metadata["worker_name"]` is always set for traceability.

## Scaling considerations

- Workers share no state — each has its own `GantryEngine` lifecycle.
- `asyncio.gather` is used, so all workers run concurrently in the same event loop.
- For CPU-bound workers or very long tasks, consider running each worker in a separate process.
