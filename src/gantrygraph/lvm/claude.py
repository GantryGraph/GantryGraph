"""ClaudeVision — pass-through provider for Claude models.

Claude natively handles image coordinates and vision tasks, so no preprocessing
is needed.  This class adds a validation warning when wrapped LLM is not a
Claude model.
"""
from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING

from gantrygraph.lvm.base import BaseVisionProvider

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel
    from langchain_core.messages import BaseMessage

logger = logging.getLogger(__name__)

# Model name substrings that indicate a Claude model
_CLAUDE_IDENTIFIERS = ("claude", "anthropic")


class ClaudeVision(BaseVisionProvider):
    """Vision provider for Anthropic Claude models.

    Claude natively understands image content with pixel-accurate coordinates,
    so no grid overlay or downscaling is applied.  All messages are passed
    through unchanged.

    Warns (via :mod:`warnings`) if the wrapped LLM does not appear to be a
    Claude model, so developers get a clear signal if they accidentally wrap
    the wrong model.

    Example::

        from langchain_anthropic import ChatAnthropic
        from gantrygraph.lvm import ClaudeVision

        provider = ClaudeVision(ChatAnthropic(model="claude-opus-4-5"))
        agent = GantryEngine(llm=provider, ...)
    """

    def __init__(self, llm: BaseChatModel) -> None:
        super().__init__(llm)
        self._validate_model()

    def _validate_model(self) -> None:
        name = self.model_name.lower()
        if not any(ident in name for ident in _CLAUDE_IDENTIFIERS):
            # Also check class name for langchain_anthropic models
            class_name = type(self._llm).__name__.lower()
            if not any(ident in class_name for ident in _CLAUDE_IDENTIFIERS):
                warnings.warn(
                    f"ClaudeVision is wrapping '{self.model_name}' "
                    f"(class: {type(self._llm).__name__}), which does not appear "
                    "to be an Anthropic Claude model.  Use GridOverlayVision or "
                    "LocalVision for other models.",
                    UserWarning,
                    stacklevel=3,
                )

    async def _preprocess(
        self, messages: list[BaseMessage]
    ) -> list[BaseMessage]:
        """Pass messages through unchanged — Claude handles images natively."""
        return messages
