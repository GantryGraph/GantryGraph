# Quickstart

Get a working autonomous agent in under 5 minutes.

## Install

```bash
pip install gantrygraph
# With all optional extras:
pip install 'gantrygraph[desktop,browser,cloud,dev]'
```

## Your first agent — no perception, no tools

The simplest possible agent: an LLM that reasons and responds.

```python
from gantrygraph import GantryEngine
from langchain_anthropic import ChatAnthropic

agent = GantryEngine(
    llm=ChatAnthropic(model="claude-sonnet-4-6"),
    max_steps=5,
)

result = agent.run("What is 17 * 23? Show your working.")
print(result)
```

## Add a custom tool

Turn any function into an agent tool with `@gantry_tool`.

```python
from gantrygraph import GantryEngine, gantry_tool
from langchain_anthropic import ChatAnthropic

@gantry_tool
async def get_weather(city: str) -> str:
    """Return the current weather for a city."""
    # Replace with a real API call
    return f"Sunny, 22°C in {city}"

agent = GantryEngine(
    llm=ChatAnthropic(model="claude-sonnet-4-6"),
    tools=[get_weather],
    max_steps=5,
)

print(agent.run("What's the weather in Milan and Tokyo?"))
```

## QA / code-repair agent

The `qa_agent` preset bundles filesystem + shell tools + memory.

```python
from gantrygraph.presets import qa_agent
from langchain_anthropic import ChatAnthropic

agent = qa_agent(
    llm=ChatAnthropic(model="claude-sonnet-4-6"),
    workspace="/my/project",
    shell_allowed_commands=["python", "pytest", "git"],
)

result = agent.run("Run the test suite, fix all failures, and commit the changes.")
print(result)
```

## Desktop agent (screenshot + mouse/keyboard)

```bash
pip install 'gantrygraph[desktop]'
```

```python
from gantrygraph.presets import desktop_agent
from langchain_anthropic import ChatAnthropic

agent = desktop_agent(
    llm=ChatAnthropic(model="claude-sonnet-4-6"),
)
agent.run("Open the terminal, run `ls -la`, and copy the output into a new text file.")
```

## Browser agent

```bash
pip install 'gantrygraph[browser]'
playwright install chromium
```

```python
from gantrygraph.presets import browser_agent
from langchain_anthropic import ChatAnthropic

agent = browser_agent(
    llm=ChatAnthropic(model="claude-sonnet-4-6"),
    start_url="https://news.ycombinator.com",
)
result = agent.run("Find the top 3 stories and summarise them.")
print(result)
```

## Next steps

- [Concepts: The Engine](concepts/engine.md) — understand the observe→think→act loop
- [Concepts: Tools & Actions](concepts/tools.md) — filesystem, shell, browser, MCP
- [How-to: MCP Integration](how-to/mcp.md) — connect GitHub, Notion, databases
- [How-to: Cloud Deploy](how-to/cloud-deploy.md) — REST API + Docker
