# Telemetry & Observability

GantryGraph emits structured events at every step of the agent loop —
`observe`, `think`, `act`, `review`, `error`, and `done`. You choose how
those events are handled by passing `telemetry=` to `GantryEngine`.

## Telemetry modes

| Mode | What it does |
|------|--------------|
| `None` (default) | Events are discarded. Zero overhead. |
| `"silent"` | Events are collected in memory but not printed or written anywhere. |
| `"stdout"` | Human-readable summary printed to `sys.stdout` at each step. |
| `"json"` | Each event appended as a JSON line to a `.jsonl` file. |
| `"langsmith"` | Sets `LANGCHAIN_TRACING_V2=true` and traces the run in LangSmith. |

All modes except `None` and `"silent"` are compatible with an additional
`on_event` callback for custom routing.

## `"stdout"` — development

The fastest way to see what the agent is doing:

```python
from gantrygraph import GantryEngine
from langchain_anthropic import ChatAnthropic

agent = GantryEngine(
    llm=ChatAnthropic(model="claude-sonnet-4-6"),
    telemetry="stdout",
    max_steps=10,
)

result = await agent.arun("List all .py files modified today.")
```

Output looks like:

```
[step=0] observe  — screenshot captured (1280×720)
[step=0] think    — selected tool: shell_run
[step=0] act      — shell_run: find . -name "*.py" -newer ...
[step=0] done     — result produced
```

!!! tip
    `"stdout"` adds minimal latency and is safe to leave on during local
    development. Strip it (or switch to `"json"`) before deploying to production.

## `"json"` — production logging

Appends one JSON line per event to a file. The default path is
`gantry-events.jsonl` in the current working directory; override it with
`log_file=`.

=== "Basic usage"

    ```python
    agent = GantryEngine(
        llm=ChatAnthropic(model="claude-sonnet-4-6"),
        telemetry="json",
    )
    ```

=== "Custom log path"

    ```python
    agent = GantryEngine(
        llm=ChatAnthropic(model="claude-sonnet-4-6"),
        telemetry="json",
        log_file="/var/log/gantry/events.jsonl",
    )
    ```

Each line has the following shape:

```json
{"ts": "2026-05-07T14:23:01.482Z", "step": 0, "type": "observe", "data": {"width": 1280, "height": 720}}
{"ts": "2026-05-07T14:23:02.019Z", "step": 0, "type": "think",   "data": {"tool": "shell_run", "reasoning": "..."}}
{"ts": "2026-05-07T14:23:02.341Z", "step": 0, "type": "act",     "data": {"tools_executed": ["shell_run"], "output": "..."}}
{"ts": "2026-05-07T14:23:02.890Z", "step": 0, "type": "done",    "data": {"result": "3 files found: ..."}}
```

!!! note
    `data` never contains `screenshot_b64`. Raw image bytes are stripped
    automatically before writing, so log files stay compact.

Tail the file in a separate terminal while the agent runs:

```bash
tail -f gantry-events.jsonl | python -m json.tool
```

Or pipe into `jq` for filtering:

```bash
tail -f gantry-events.jsonl | jq 'select(.type == "act")'
```

## `on_event` — custom callback

For Datadog, OpenTelemetry, Slack alerts, or any custom sink, pass a callable
to `on_event`. It receives a `GantryEvent` and can be sync or async.

```python
from gantrygraph import GantryEngine, GantryEvent
from langchain_anthropic import ChatAnthropic

async def send_to_datadog(event: GantryEvent) -> None:
    if event.event_type == "error":
        await datadog_client.metric(
            "gantry.error",
            value=1,
            tags=[f"step:{event.step}"],
        )

agent = GantryEngine(
    llm=ChatAnthropic(model="claude-sonnet-4-6"),
    on_event=send_to_datadog,
)
```

`GantryEvent` fields:

| Field | Type | Description |
|-------|------|-------------|
| `event_type` | `str` | One of `"observe"`, `"think"`, `"act"`, `"review"`, `"error"`, `"done"` |
| `step` | `int` | Zero-based step index within the current run |
| `data` | `dict` | Event payload (no raw image bytes) |

!!! tip
    A sync callback works too — GantryGraph wraps it with `ensure_awaitable`
    internally so you don't need to worry about the event loop.

## LangSmith integration

=== "Setup"

    ```bash
    pip install langsmith
    export LANGCHAIN_API_KEY="ls__..."
    export LANGCHAIN_PROJECT="my-agent"   # optional, defaults to "default"
    ```

=== "Enable"

    ```python
    agent = GantryEngine(
        llm=ChatAnthropic(model="claude-sonnet-4-6"),
        telemetry="langsmith",
    )

    result = await agent.arun("Summarise the last 10 commits.")
    ```

Setting `telemetry="langsmith"` flips `LANGCHAIN_TRACING_V2=true` in the
process environment. Every LangChain call — including the LLM and tool
invocations — is recorded in your LangSmith project automatically.

!!! warning
    LangSmith tracing sends data to Anthropic's LangChain partner service.
    Review your data-handling policies before enabling it in environments that
    process sensitive content.

## Combining `"json"` and `on_event`

Use both together to write a durable local log *and* forward high-priority
events to an external system simultaneously:

```python
from gantrygraph import GantryEngine, GantryEvent
from langchain_anthropic import ChatAnthropic

async def alert_on_error(event: GantryEvent) -> None:
    if event.event_type in ("error", "done"):
        await pagerduty.trigger(
            summary=f"GantryAgent {event.event_type} at step {event.step}",
            details=event.data,
        )

agent = GantryEngine(
    llm=ChatAnthropic(model="claude-sonnet-4-6"),
    telemetry="json",
    log_file="/var/log/gantry/events.jsonl",
    on_event=alert_on_error,
)
```

!!! tip
    This pattern gives you a complete JSONL audit trail for post-mortems while
    still getting real-time alerts for failures — without any extra infrastructure.
