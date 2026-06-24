"""tokenslim-proxy — transparent context-compression HTTP proxy.

Wraps the ``tokenslim`` core: parse a provider request body, run its message
array through ``compress()``, forward to the real upstream preserving auth.
"""

from __future__ import annotations

from .app import create_app
from .compression import CompressionOutcome, compress_messages_body
from .config import ProxyConfig
from .responses import ResponsesOutcome, compress_responses_body

__version__ = "0.0.1"

__all__ = [
    "__version__",
    "create_app",
    "ProxyConfig",
    "compress_messages_body",
    "CompressionOutcome",
    "compress_responses_body",
    "ResponsesOutcome",
]
