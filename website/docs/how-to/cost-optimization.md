# Cost Optimization

Production AI agents call the LLM at every step — perception, planning, and action.
Without guardrails, token costs grow quickly: a 100-step task can easily exceed
100,000 input tokens per run.
GantryGraph exposes four orthogonal knobs that together cut costs by 90 %+ on
typical workloads.

## Summary

| Optimization | Default | Token savings |
|---|---|---|
| AXTree-first perception | `"auto"` | ~80% per observation |
| Shell output truncation | 2,000 chars | Prevents 100k+ token dumps |
| Message window | disabled | Cuts O(N²) to O(N) growth |
| Prompt caching | disabled | Up to 90% on input tokens |

## Before / after

=== "Unoptimized"

    ```python
    from gantrygraph import GantryEngine
    from gantrygraph.actions import ShellTools
    from langchain_anthropic import ChatAnthropic

    agent = GantryEngine(
        llm=ChatAnthropic(model="claude-sonnet-4-6"),
        tools=[ShellTools()],
        max_steps=50,
    )
    ```

=== "Optimized"

    ```python
    from gantrygraph import GantryEngine
    from gantrygraph.actions import ShellTools
    from langchain_anthropic import ChatAnthropic

    agent = GantryEngine(
        llm=ChatAnthropic(model="claude-sonnet-4-6"),
        tools=[ShellTools(max_output_chars=2000)],
        perception_mode="axtree",
        message_window=20,
        enable_caching=True,
        max_steps=50,
    )
    ```

---

## 1. AXTree-first perception (`perception_mode`)

**Parameter:** `perception_mode` on `GantryEngine`  
**Values:** `"auto"` (default), `"axtree"`, `"vision"`

By default (`"auto"` or `"axtree"`), GantryGraph sends the browser or desktop
accessibility tree as plain text instead of a full screenshot.
A screenshot sent as a vision message costs ~1,500 tokens per step;
the equivalent accessibility tree costs ~300 tokens — an **80% reduction** per
observation.

Switch to `"vision"` only when the task genuinely requires pixel-level detail
(e.g. CAPTCHA solving, image editing).

```python
from gantrygraph import GantryEngine
from gantrygraph.actions import BrowserTools
from langchain_anthropic import ChatAnthropic

agent = GantryEngine(
    llm=ChatAnthropic(model="claude-sonnet-4-6"),
    tools=[BrowserTools()],
    perception_mode="axtree",   # (1)
    max_steps=30,
)
```

1. `"auto"` applies the same behaviour automatically when an AXTree is available.
   Use `"axtree"` to enforce it explicitly and fail fast on vision-only setups.

!!! tip "When to use `\"vision\"`"
    Tasks that require reading a chart, verifying visual layout, or interacting
    with canvas elements should use `perception_mode="vision"`.
    Expect ~5× higher per-step cost compared to `"axtree"`.

---

## 2. Shell output truncation (`max_output_chars`)

**Parameter:** `max_output_chars` on `ShellTools`  
**Default:** `2000`

Shell commands like `cat`, `find`, or failing test suites can produce megabytes
of stdout/stderr.
`ShellTools` truncates any combined output that exceeds `max_output_chars` and
appends a clear hint:

```
[N chars truncated — use grep/head/tail for details]
```

This prevents a single tool call from injecting tens of thousands of tokens into
the conversation history.

```python
from gantrygraph import GantryEngine
from gantrygraph.actions import ShellTools
from langchain_anthropic import ChatAnthropic

agent = GantryEngine(
    llm=ChatAnthropic(model="claude-sonnet-4-6"),
    tools=[
        ShellTools(max_output_chars=2000),  # (1)
    ],
    max_steps=50,
)
```

1. Raise this value (e.g. `5000`) if the agent needs more context from long
   log files, or lower it to `500` for very token-sensitive pipelines.

!!! warning "Do not set `max_output_chars=0` or a very large value in production"
    Setting the limit too high defeats the purpose.
    The LLM rarely needs more than the first 2,000 characters to decide its next
    action — the truncation hint guides it to refine with `grep` or `head`.

---

## 3. Sliding message window (`message_window`)

**Parameter:** `message_window` on `GantryEngine`  
**Default:** `None` (full history)

Without a window, every step appends a new observe/think/act triple to the
conversation.
After N steps the context contains ~N messages, and the *next* LLM call pays
for all of them — O(N²) token growth over the lifetime of a long task.

Setting `message_window=20` keeps the system/task prompt (`messages[0]`) plus
the **last 20 messages**, bounding context to a fixed size regardless of how
many steps the agent runs.

```python
from gantrygraph import GantryEngine
from gantrygraph.actions import ShellTools
from langchain_anthropic import ChatAnthropic

agent = GantryEngine(
    llm=ChatAnthropic(model="claude-sonnet-4-6"),
    tools=[ShellTools()],
    message_window=20,  # (1)
    max_steps=100,
)
```

1. `messages[0]` (the system/task prompt) is always preserved regardless of the
   window size, so the agent never loses its objective.

!!! tip "Choosing a window size"
    A window of `10`–`20` messages covers the most recent action cycle and is
    sufficient for most tasks.
    Increase to `40`+ only for tasks with complex multi-step dependencies where
    the agent must recall decisions made many steps earlier.

---

## 4. Prompt caching (`enable_caching`)

**Parameter:** `enable_caching` on `GantryEngine`  
**Default:** `False`

When `enable_caching=True`, GantryGraph adds Anthropic's
`cache_control: {"type": "ephemeral"}` header to system messages.
Anthropic caches the system prompt for **5 minutes**; any request that hits the
cache pays the *cached input* rate — up to **90% cheaper** than the standard
input rate.

This is most effective for:

- Long system prompts that repeat verbatim across every step of a run.
- Multi-agent swarms where many workers share the same system prompt.
- Cloud deployments that serve many short tasks with the same base instruction.

```python
from gantrygraph import GantryEngine
from gantrygraph.actions import ShellTools
from langchain_anthropic import ChatAnthropic

agent = GantryEngine(
    llm=ChatAnthropic(model="claude-sonnet-4-6"),
    tools=[ShellTools()],
    enable_caching=True,  # (1)
    max_steps=50,
)
```

1. The cache TTL is 5 minutes on Anthropic's side.
   For runs that take longer than 5 minutes, only the first window of steps
   receives the cached rate; subsequent steps re-prime the cache automatically.

!!! warning "Provider compatibility"
    `enable_caching=True` is only effective with Anthropic models via
    `langchain_anthropic.ChatAnthropic`.
    Passing an OpenAI or other provider LLM silently ignores the flag —
    no error is raised, but no caching occurs.

---

## Recommended production config

!!! tip "Use all four optimizations together"

    ```python
    from gantrygraph import GantryEngine
    from gantrygraph.actions import ShellTools
    from langchain_anthropic import ChatAnthropic

    agent = GantryEngine(
        llm=ChatAnthropic(model="claude-sonnet-4-6"),
        tools=[
            ShellTools(max_output_chars=2000),
        ],
        perception_mode="axtree",
        message_window=20,
        enable_caching=True,
        max_steps=50,
    )

    result = agent.run("Your task here.")
    ```

    **Expected impact on a 50-step browser task:**

    | Before | After |
    |---|---|
    | ~75,000 input tokens | ~8,000 input tokens |
    | ~1,500 tokens/step (vision) | ~300 tokens/step (AXTree) |
    | O(N²) history growth | O(N) bounded history |
    | Full price on all tokens | Up to 90% cached-input discount |
