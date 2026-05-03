"""Vision preprocessing providers for gantrygraph.

Use these to apply model-specific image transformations (grid overlays,
coordinate normalisation, downscaling) before screenshots reach the LLM.
Pass a provider as the ``llm`` argument to ``GantryEngine`` — it is a
drop-in replacement for any ``BaseChatModel``.

Quick start::

    from gantrygraph.vision import ClaudeVision
    from langchain_anthropic import ChatAnthropic

    provider = ClaudeVision(ChatAnthropic(model="claude-opus-4-7"))
    agent = GantryEngine(llm=provider, perception=DesktopScreen())
"""

from gantrygraph.vision.base import BaseVisionProvider
from gantrygraph.vision.claude import ClaudeVision

__all__ = ["BaseVisionProvider", "ClaudeVision"]
