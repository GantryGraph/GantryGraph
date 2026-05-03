"""BaseVisionProvider — wraps any LangChain BaseChatModel with vision preprocessing."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel
    from langchain_core.messages import AIMessage, BaseMessage
    from langchain_core.tools import BaseTool


class BaseVisionProvider:
    """Wraps a BaseChatModel with vision-specific image preprocessing.

    Subclasses override ``_preprocess`` to modify messages (e.g. overlay grids,
    downscale images) before the underlying LLM receives them.

    Implements ``bind_tools(tools) -> self`` and
    ``async ainvoke(messages) -> AIMessage`` to be a drop-in replacement
    wherever a ``BaseChatModel`` is expected.

    Example::

        provider = GridOverlayVision(ChatAnthropic(model="claude-opus-4-5"))
        agent = GantryEngine(llm=provider, ...)
    """

    def __init__(self, llm: BaseChatModel) -> None:
        self._llm = llm

    # ── BaseChatModel compatibility ───────────────────────────────────────────

    def bind_tools(self, tools: list[BaseTool]) -> BaseVisionProvider:
        """Delegate tool binding to the wrapped LLM and return self."""
        try:
            self._llm = self._llm.bind_tools(tools)  # type: ignore[assignment]
        except NotImplementedError:
            pass
        return self

    async def ainvoke(self, messages: list[BaseMessage], **kwargs: Any) -> AIMessage:
        """Preprocess messages then forward to the wrapped LLM."""
        processed = await self._preprocess(messages)
        return await self._llm.ainvoke(processed, **kwargs)

    # ── Override in subclasses ────────────────────────────────────────────────

    async def _preprocess(self, messages: list[BaseMessage]) -> list[BaseMessage]:
        """Transform messages before forwarding to the LLM.  Default: pass-through."""
        return messages

    # ── Convenience properties ────────────────────────────────────────────────

    @property
    def model_name(self) -> str:
        """Return the underlying model name, or 'unknown' if not detectable."""
        return (
            getattr(self._llm, "model_name", None) or getattr(self._llm, "model", None) or "unknown"
        )
