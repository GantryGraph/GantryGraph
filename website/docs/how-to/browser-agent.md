# Browser Agent

A browser agent controls a Chromium browser via Playwright.
It navigates URLs, clicks elements, fills forms, extracts text, and handles
JavaScript-heavy pages — including sites that require login.

## Install

```bash
pip install 'gantrygraph[browser]'
playwright install chromium
```

## Quick start

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

The agent reads the page explicitly with `browser_get_text`:

```python
agent = browser_agent(llm=ChatAnthropic(...))
result = agent.run(
    "Go to https://news.ycombinator.com, "
    "find all links on the front page, and return the top 5 titles."
)
```

## Manual configuration

Use `WebPage` and `BrowserTools` together so they share the same browser
instance — actions taken by the tools are immediately visible to perception.

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

---

## Stealth mode

By default `stealth=True` patches the browser to reduce bot-detection
signals. This is applied automatically — you do not need to configure it.

What it does under the hood:

- Removes `navigator.webdriver = true` (caught by every basic detector)
- Spoof `navigator.plugins`, `navigator.languages`, `hardwareConcurrency`, `deviceMemory`
- Fake `window.chrome` runtime object (absent in headless builds)
- Realistic user-agent (`Chrome/125`) instead of `HeadlessChrome`
- Consistent screen geometry and WebGL vendor/renderer
- Disables the `--enable-automation` Chromium flag

```python
from gantrygraph.actions import BrowserTools

# stealth=True is the default — this is explicit for clarity
tools = BrowserTools(headless=True, stealth=True)

# Disable only if you need to test how stealth-free headless behaves
tools_no_stealth = BrowserTools(headless=True, stealth=False)
```

!!! tip "Optional deeper stealth"
    Install `playwright-stealth` for additional patches on top of GantryGraph's
    built-in layer:
    ```bash
    pip install playwright-stealth
    ```
    When installed, it is applied automatically — no code change required.

---

## Persistent sessions — stay logged in

Sites like WhatsApp Web, Notion, and Google require a login.
Without persistence, the agent faces a fresh browser on every run and must
log in again — or scan a QR code.

`profile_dir` saves the entire Chromium profile (cookies, localStorage,
IndexedDB) to disk. Log in once; every subsequent run reuses the session.

### First run — log in manually

```python
from gantrygraph.actions import BrowserTools

tools = BrowserTools(
    headless=False,                                        # show the browser window
    profile_dir="~/.gantrygraph/profiles/whatsapp",       # where to save the session
)
```

Open the browser, complete the login or QR-code scan, then close it.
The profile is saved automatically.

### Subsequent runs — session restored automatically

```python
from gantrygraph import GantryEngine
from gantrygraph.actions import BrowserTools
from langchain_anthropic import ChatAnthropic

tools = BrowserTools(
    headless=True,                                         # can go headless now
    profile_dir="~/.gantrygraph/profiles/whatsapp",
)

agent = GantryEngine(
    llm=ChatAnthropic(model="claude-sonnet-4-6"),
    tools=[tools],
    max_steps=30,
)

result = agent.run(
    "Open WhatsApp Web, search for 'Mamma', "
    "and send: 'Ciao! I found the Cacio e Pepe recipe you wanted.'"
)
print(result)
```

The same pattern works for any site — Gmail, GitHub, Notion, Slack Web:

```python
BrowserTools(profile_dir="~/.gantrygraph/profiles/gmail")
BrowserTools(profile_dir="~/.gantrygraph/profiles/notion")
```

!!! tip "Use a separate profile per site"
    Each `profile_dir` is an isolated Chromium profile.
    Keep one directory per service so sessions never interfere.

!!! warning "First-run headless=False requirement"
    The first run must use `headless=False` so the browser window is visible for
    login. After the session is saved you can switch to `headless=True`.

---

## Perception mode

`perception_mode` controls whether the agent uses a screenshot (vision tokens)
or the accessibility tree (text tokens) for each observation.

| Mode | Behaviour | When to use |
|------|-----------|-------------|
| `"auto"` (default) | Text accessibility tree when available, screenshot as fallback | Most web tasks |
| `"axtree"` | Always accessibility tree, never screenshot | Maximum token savings |
| `"vision"` | Always screenshot | Canvas apps, PDFs, visual verification |

```python
from gantrygraph import GantryEngine
from gantrygraph.perception import WebPage
from gantrygraph.actions import BrowserTools
from langchain_anthropic import ChatAnthropic

web = WebPage(url="https://example.com")

agent = GantryEngine(
    llm=ChatAnthropic(model="claude-sonnet-4-6"),
    perception=web,
    tools=[BrowserTools(web_page=web)],
    perception_mode="axtree",   # ~80% cheaper per step vs screenshots
    message_window=20,          # cap context growth on long runs
    enable_caching=True,        # Anthropic prompt cache
    max_steps=50,
)
```

See the [Cost Optimization](cost-optimization.md) guide for a full breakdown.

---

## Handling cookie banners and popups

Use `browser_click_text` to dismiss cookie banners by their button label —
no need to find a brittle CSS selector:

```python
# The agent can call this tool directly:
# browser_click_text("Accept all")
# browser_click_text("Rifiuta tutto")
# browser_click_text("Reject non-essential")
```

This is particularly useful when the agent encounters a consent wall before
it can interact with the actual page content.

---

## Available tools

| Tool | Description |
|------|-------------|
| `browser_navigate` | Open a URL |
| `browser_click` | Click an element by CSS/XPath selector |
| `browser_click_text` | Click a button or link by its visible text label |
| `browser_fill` | Type text into a form field |
| `browser_get_text` | Get the visible text of an element or the full page |
| `browser_get_url` | Get the current URL |
| `browser_scroll` | Scroll the page up, down, to top, or to bottom |
| `browser_evaluate` | Run a JavaScript expression and return the result |
| `browser_wait_for_selector` | Wait until a CSS/XPath selector appears on the page |
