"""gantrygraph: Autonomous OS-level agent framework — visual computer use + MCP.

Quick start::

    from gantrygraph import GantryEngine
    from gantrygraph.perception import DesktopScreen
    from gantrygraph.actions import MouseKeyboardTools
    from gantrygraph.mcp import MCPClient
    from langchain_anthropic import ChatAnthropic

    agent = GantryEngine(
        llm=ChatAnthropic(model="claude-sonnet-4-6"),
        perception=DesktopScreen(max_resolution=(1024, 768)),
        tools=[MouseKeyboardTools(), MCPClient("npx -y @mcp/github")],
        on_event=lambda e: print(e),
        max_steps=50,
    )
    agent.run("Clone the repo and open a pull request")

Sub-packages are not eagerly imported here so that optional extras
(pyautogui, playwright, fastapi) are not required just to ``import gantrygraph``.
"""

from gantrygraph.config import GantryConfig
from gantrygraph.core.base_action import BaseAction
from gantrygraph.core.base_mcp import BaseMCPConnector
from gantrygraph.core.base_perception import BasePerception
from gantrygraph.core.events import GantryEvent, PerceptionResult
from gantrygraph.core.state import GantryState
from gantrygraph.engine.engine import AgentSuspended, GantryEngine
from gantrygraph.engine.graph import build_graph
from gantrygraph.engine.nodes import (
    act_node,
    memory_recall_node,
    observe_node,
    review_node,
    should_continue,
    think_node,
)
from gantrygraph.memory.base import BaseMemory, MemoryResult
from gantrygraph.memory.in_memory import InMemoryStore, InMemoryVector
from gantrygraph.perception.multi import MultiPerception
from gantrygraph.security.policies import (
    BudgetExceededError,
    BudgetPolicy,
    GuardrailPolicy,
    ShellDenylist,
    WorkspacePolicy,
)
from gantrygraph.security.secrets import GantrySecrets
from gantrygraph.swarm.worker import WorkerSpec
from gantrygraph.tool import gantry_tool
from gantrygraph.vision.base import BaseVisionProvider
from gantrygraph.vision.claude import ClaudeVision

__all__ = [
    # Engine
    "GantryEngine",
    "AgentSuspended",
    # Config-driven setup
    "GantryConfig",
    # Decorators
    "gantry_tool",
    # Graph customisation — escape hatch for custom loops
    "build_graph",
    "memory_recall_node",
    "observe_node",
    "think_node",
    "act_node",
    "review_node",
    "should_continue",
    # ABCs
    "BasePerception",
    "BaseAction",
    "BaseMCPConnector",
    "BaseMemory",
    # Events / state
    "GantryEvent",
    "PerceptionResult",
    "GantryState",
    "MemoryResult",
    # Perception
    "MultiPerception",
    # Vision preprocessing
    "BaseVisionProvider",
    "ClaudeVision",
    # Memory
    "InMemoryStore",
    "InMemoryVector",  # backward compat
    # Security
    "GuardrailPolicy",
    "WorkspacePolicy",
    "BudgetPolicy",
    "BudgetExceededError",
    "ShellDenylist",
    "GantrySecrets",
    # Swarm
    "WorkerSpec",
]
