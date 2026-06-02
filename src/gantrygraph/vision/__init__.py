"""Vision preprocessing for gantrygraph.

Two complementary APIs:

**LLM-level providers** — wrap a ``BaseChatModel`` and preprocess image
messages before they reach the model.  Pass as the ``llm`` argument to
``GantryEngine``::

    from gantrygraph.vision import ClaudeVision
    provider = ClaudeVision(ChatAnthropic(model="claude-opus-4-7"))
    agent = GantryEngine(llm=provider, ...)

**Screenshot pipelines** — transform raw PNG bytes after each ``observe()``
call, before they are base64-encoded.  Attach to ``WebPage`` and ``BrowserTools``::

    from gantrygraph.vision import PerceptionPipeline, SetOfMarkAnnotator, Downsample
    pipeline = PerceptionPipeline([SetOfMarkAnnotator(), Downsample(1280)])
    web = WebPage(url="...", vision_pipeline=pipeline)
    tools = BrowserTools(web_page=web, vision_pipeline=pipeline)
"""

from gantrygraph.vision.base import BaseVisionProvider
from gantrygraph.vision.claude import ClaudeVision
from gantrygraph.vision.pipeline import (
    ConvertToWebP,
    Downsample,
    Grayscale,
    ImageFilter,
    PerceptionPipeline,
    SetOfMarkAnnotator,
)

__all__ = [
    # LLM-level providers
    "BaseVisionProvider",
    "ClaudeVision",
    # Pipeline & filters
    "ImageFilter",
    "PerceptionPipeline",
    "Downsample",
    "Grayscale",
    "ConvertToWebP",
    "SetOfMarkAnnotator",
]
