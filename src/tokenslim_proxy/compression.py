"""Thin adapter over the ``tokenslim`` core ``compress()`` for provider bodies.

The core operates on a message array (``[{"role", "content"}, ...]``). Both the
Anthropic ``/v1/messages`` and OpenAI ``/v1/chat/completions`` payloads carry
such an array under the ``messages`` key, so compression is the same operation:
pull the array, run it through the core, put it back. Everything else in the
body (model, temperature, tools, system, …) is forwarded untouched.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from tokenslim import compress as _core_compress


@dataclass(frozen=True)
class CompressionOutcome:
    """Result of compressing one request body."""

    body: dict[str, Any]
    orig_tokens: int
    new_tokens: int
    ratio: float
    changed: bool


def compress_messages_body(body: dict[str, Any], *, enabled: bool = True) -> CompressionOutcome:
    """Compress the ``messages`` array of a chat-style request body.

    The input ``body`` is never mutated; a shallow copy with a rewritten
    ``messages`` value is returned. Bodies without a list ``messages`` field are
    passed through unchanged with zeroed stats.

    Args:
        body: The parsed JSON request body.
        enabled: When ``False``, forward verbatim (stats still reported by core).
    """
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        return CompressionOutcome(body=body, orig_tokens=0, new_tokens=0, ratio=0.0, changed=False)

    new_messages, stats = _core_compress(messages, enabled=enabled)

    new_body = dict(body)
    new_body["messages"] = new_messages

    changed = stats.new_tokens < stats.orig_tokens
    return CompressionOutcome(
        body=new_body,
        orig_tokens=stats.orig_tokens,
        new_tokens=stats.new_tokens,
        ratio=stats.ratio,
        changed=changed,
    )
