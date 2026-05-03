# Browser Agent

A browser agent controls a Chromium browser via Playwright.
It can navigate URLs, click elements, fill forms, and extract text.

## Install

```bash
pip install 'gantrygraph[browser]'
playwright install chromium
```

## Quick start — with perception

When `start_url` is provided, `WebPage` perception is added automatically
and **shares the same browser** as `BrowserTools`.
The agent sees the current page state at every loop iteration.

```python
from gantrygraph.presets import browser_agent
from langchain_anthropic import ChatAnthropic

agent = browser_agent(
    llm=ChatAnthropic(model="claude-sonnet-4-6"),
    start_url="https://github.com/trending",
)
result = agent.run("Find the top 3 trending Python repos and return their names and star counts.")
print(result)
```

## Without perception (tool-only mode)

The agent reads the page explicitly using `browser_get_text`:

```python
agent = browser_agent(llm=ChatAnthropic(...))
result = agent.run(
    "Go to https://news.ycombinator.com, "
    "find all links on the front page, and return the top 5 titles."
)
```

## Manual configuration

```python
from gantrygraph import GantryEngine
from gantrygraph.perception import WebPage
from gantrygraph.actions import BrowserTools
from langchain_anthropic import ChatAnthropic

web = WebPage(url="https://myapp.example.com/login", headless=False)

agent = GantryEngine(
    llm=ChatAnthropic(model="claude-sonnet-4-6"),
    perception=web,
    tools=[BrowserTools(web_page=web)],   # shared browser instance
    max_steps=30,
)

agent.run("Log in with user=admin pass=secret and navigate to the dashboard.")
```

## Available browser tools

| Tool | Description |
|------|-------------|
| `browser_navigate` | Open a URL |
| `browser_click` | Click a CSS/XPath selector |
| `browser_fill` | Fill a form field |
| `browser_get_text` | Get visible text of an element or the whole page |
| `browser_get_url` | Get the current URL |
