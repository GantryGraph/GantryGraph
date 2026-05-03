"""Cloud deployment utilities for gantrygraph.

The Pydantic models (``RunRequest``, ``RunResponse``, ``ResumeRequest``) are
always importable.  The ``serve`` function and ``_build_app`` require the
``[cloud]`` extra::

    pip install 'gantrygraph[cloud]'
"""

# Models are importable without the cloud extra (pydantic is a core dep)
from gantrygraph.cloud.serve import ResumeRequest, RunRequest, RunResponse

__all__ = ["serve", "_build_app", "RunRequest", "RunResponse", "ResumeRequest"]


def __getattr__(name: str) -> object:
    if name in ("serve", "_build_app"):
        from gantrygraph.cloud.serve import _build_app, serve

        return serve if name == "serve" else _build_app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
