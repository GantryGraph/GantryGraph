# Desktop Agent

A desktop agent captures screenshots of the primary monitor and controls
the mouse and keyboard via `pyautogui`.

## Install

```bash
pip install 'gantrygraph[desktop]'
```

On Linux, a display server is required:

```bash
apt-get install -y xvfb
Xvfb :99 -screen 0 1280x720x24 &
export DISPLAY=:99
```

## Quick start

```python
from gantrygraph.presets import desktop_agent
from langchain_anthropic import ChatAnthropic

agent = desktop_agent(
    llm=ChatAnthropic(model="claude-sonnet-4-6"),
    max_resolution=(1280, 720),
)

agent.run("Open the terminal, type 'echo hello world', press Enter, and close it.")
```

## Manual configuration

```python
from gantrygraph import GantryEngine
from gantrygraph.perception import DesktopScreen
from gantrygraph.actions import MouseKeyboardTools
from langchain_anthropic import ChatAnthropic

agent = GantryEngine(
    llm=ChatAnthropic(model="claude-sonnet-4-6"),
    perception=DesktopScreen(max_resolution=(1920, 1080)),
    tools=[MouseKeyboardTools()],
    max_steps=50,
    system_prompt=(
        "You are a desktop automation agent. "
        "Always verify your action by observing the result before proceeding."
    ),
)
```

## Tips

- **Resolution**: Smaller screenshots cost fewer tokens. `(1024, 768)` is a good default for most tasks.
- **Retries**: GUI state can change unexpectedly. GantryGraph's self-correction loop retries failed tool calls automatically.
- **Guardrails**: Always set a `BudgetPolicy` for unattended desktop agents to prevent runaway loops.
