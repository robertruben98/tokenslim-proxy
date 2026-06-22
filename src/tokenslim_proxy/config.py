"""Runtime configuration for the proxy, sourced from ``TOKENSLIM_PROXY_*`` env vars."""

from __future__ import annotations

import os
from dataclasses import dataclass

# Default upstream bases. Overridable so tests (and self-hosted gateways) can
# repoint the proxy without touching code.
DEFAULT_ANTHROPIC_BASE = "https://api.anthropic.com"
DEFAULT_OPENAI_BASE = "https://api.openai.com"


def _clean_base(url: str) -> str:
    """Normalize an upstream base URL by trimming any trailing slash."""
    return url.rstrip("/")


@dataclass(frozen=True)
class ProxyConfig:
    """Resolved proxy settings.

    Attributes:
        host: Interface uvicorn binds to.
        port: Port uvicorn binds to.
        anthropic_base: Upstream base URL for Anthropic requests.
        openai_base: Upstream base URL for OpenAI requests.
        upstream_timeout: Per-request timeout (seconds) for upstream calls.
        compression_enabled: Master switch; when off the proxy still forwards
            but does not rewrite bodies (useful for A/B and debugging).
        cache_prefix_stable: When ``True`` (default) each message is compressed
            in isolation so a message's compressed bytes never depend on its
            neighbours or position. This keeps the already-sent conversation
            prefix byte-identical turn-to-turn, so provider KV/prompt caches
            keep hitting. When ``False`` the whole array is compressed in one
            pass (M0 behaviour).
        anthropic_cache_breakpoint: When ``True``, mark the last prefix message
            of an Anthropic ``/v1/messages`` body with a ``cache_control``
            breakpoint at a stable boundary so the provider caches the prefix.
            Off by default (opt-in, since it consumes a cache-control slot).
    """

    host: str = "127.0.0.1"
    port: int = 8788
    anthropic_base: str = DEFAULT_ANTHROPIC_BASE
    openai_base: str = DEFAULT_OPENAI_BASE
    upstream_timeout: float = 60.0
    compression_enabled: bool = True
    cache_prefix_stable: bool = True
    anthropic_cache_breakpoint: bool = False

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> ProxyConfig:
        """Build a config from environment variables (defaults to ``os.environ``)."""
        e = os.environ if env is None else env
        return cls(
            host=e.get("TOKENSLIM_PROXY_HOST", cls.host),
            port=int(e.get("TOKENSLIM_PROXY_PORT", str(cls.port))),
            anthropic_base=_clean_base(
                e.get("TOKENSLIM_PROXY_ANTHROPIC_BASE", DEFAULT_ANTHROPIC_BASE)
            ),
            openai_base=_clean_base(e.get("TOKENSLIM_PROXY_OPENAI_BASE", DEFAULT_OPENAI_BASE)),
            upstream_timeout=float(
                e.get("TOKENSLIM_PROXY_UPSTREAM_TIMEOUT", str(cls.upstream_timeout))
            ),
            compression_enabled=_parse_bool(
                e.get("TOKENSLIM_PROXY_COMPRESSION", "1"), default=True
            ),
            cache_prefix_stable=_parse_bool(
                e.get("TOKENSLIM_PROXY_CACHE_PREFIX_STABLE", "1"), default=True
            ),
            anthropic_cache_breakpoint=_parse_bool(
                e.get("TOKENSLIM_PROXY_ANTHROPIC_CACHE_BREAKPOINT", "0"), default=False
            ),
        )


def _parse_bool(value: str, *, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}
