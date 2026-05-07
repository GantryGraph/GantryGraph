# The GantryEngine

`GantryEngine` is the main entry point for the framework.
It composes a perception source, a set of tools, and an LLM
into a self-correcting **observe → think → act → review** loop
backed by [LangGraph](https://github.com/langchain-ai/langgraph).

## The loop

```
START
  └─ memory_recall    retrieve relevant past experiences
       └─ observe     screenshot / DOM / custom sensor
            └─ think  LLM decides what to do
                 └─ act   execute tool calls (with guardrail gate)
                      └─ review   task done? → END, else → observe
```

Each node is a plain `async def` function bound to its configuration
via `functools.partial`. The graph is compiled once and cached on the engine.

## Creating an engine

```python
from gantrygraph import GantryEngine
from langchain_anthropic import ChatAnthropic

agent = GantryEngine(
    llm=ChatAnthropic(model="claude-sonnet-4-6"),
    tools=[...],
    perception=...,           # optional
    memory=...,               # optional
    guardrail=...,            # optional
    budget=...,               # optional
    max_steps=50,
    system_prompt="You are a QA engineer.",
    on_event=lambda e: print(e),
)
```

## Running a task

=== "Sync"
    ```python
    result = agent.run("List the 5 largest files in /tmp")
    ```

=== "Async"
    ```python
    result = await agent.arun("List the 5 largest files in /tmp")
    ```

=== "Stream events"
    ```python
    async for event in agent.astream_events("List the 5 largest files"):
        print(event.event_type, event.step, event.data)
    ```

## Token optimisation

GantryGraph ships four orthogonal cost controls that stack on top of each other.
Enabling all four can reduce token spend by **5–10×** on long browser tasks:

```python
agent = GantryEngine(
    llm=...,
    tools=[ShellTools(max_output_chars=2_000)],    # (2)
    perception_mode="auto",                         # (1)
    message_window=20,                              # (3)
    enable_caching=True,                            # (4)
)
```

1. **`perception_mode`** — `"auto"` (default) sends the accessibility tree
   text instead of a screenshot when the DOM is readable.  Saves ~80 % per
   observe step.  Set `"vision"` to always use screenshots (canvas apps,
   desktop GUIs).
2. **`max_output_chars`** on `ShellTools` — truncates stdout/stderr at
   2 000 characters.  Prevents a single `cat large_file.log` from
   consuming hundreds of thousands of tokens.
3. **`message_window`** — keeps only `messages[0]` (task prompt) + the last
   N messages in the context window sent to the LLM.  Prevents O(N²) token
   growth over long runs.
4. **`enable_caching`** — marks system messages with Anthropic's
   `cache_control: {"type": "ephemeral"}`.  Up to 90 % discount on input
   tokens for cached blocks within the 5-minute TTL.

See the [Cost Optimization](../how-to/cost-optimization.md) how-to guide for
a full before/after comparison.

## BudgetPolicy

Hard limits to prevent runaway cost:

```python
from gantrygraph import GantryEngine
from gantrygraph.security import BudgetPolicy

agent = GantryEngine(
    llm=...,
    budget=BudgetPolicy(
        max_steps=20,
        max_wall_seconds=60.0,   # TimeoutError after 60 s
    ),
)
```

## Human-in-the-loop

### Callback mode (no checkpointer needed)

```python
async def my_approval(tool_name: str, args: dict) -> bool:
    print(f"Approve {tool_name}({args})?")
    return input("[y/N] ").lower() == "y"

agent = GantryEngine(
    llm=...,
    tools=[shell_tool],
    approval_callback=my_approval,
    guardrail=GuardrailPolicy(requires_approval={"shell_run"}),
)
```

### Interrupt / resume mode (durable suspension)

```python
from gantrygraph import GantryEngine, AgentSuspended

agent = GantryEngine(llm=..., tools=[shell_tool], enable_suspension=True)

try:
    result = await agent.arun("Deploy to production", thread_id="t1")
except AgentSuspended as e:
    print("Suspended! Tool call is waiting for approval.")
    result = await agent.resume(e.thread_id, approved=True)
```

## Escape hatch — custom graph topology

```python
from functools import partial
from gantrygraph import GantryEngine
from gantrygraph.engine import act_node, observe_node, review_node, should_continue, think_node
from gantrygraph.core.state import GantryState
from langgraph.graph import END, START, StateGraph

compiled = agent.get_graph()   # returns the raw CompiledStateGraph
```

See the [API Reference](../api-reference.md) for full parameter documentation.
