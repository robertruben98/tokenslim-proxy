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


def _compress_stable(messages: list[Any]) -> tuple[list[Any], int, int]:
    """Compress each message in isolation, returning ``(messages, orig, new)``.

    Running the core one message at a time guarantees that a message's
    compressed bytes depend only on that message's own content — never on its
    position or its neighbours. Across conversation turns the shared prefix
    therefore compresses to identical bytes, so provider prompt/KV caches keep
    hitting (issue #7). Non-dict entries are passed through untouched.
    """
    out: list[Any] = []
    orig_total = 0
    new_total = 0
    for msg in messages:
        if not isinstance(msg, dict):
            out.append(msg)
            continue
        single, stats = _core_compress([msg])
        out.append(single[0])
        orig_total += stats.orig_tokens
        new_total += stats.new_tokens
    return out, orig_total, new_total


def compress_messages_body(
    body: dict[str, Any], *, enabled: bool = True, stable_prefix: bool = True
) -> CompressionOutcome:
    """Compress the ``messages`` array of a chat-style request body.

    The input ``body`` is never mutated; a shallow copy with a rewritten
    ``messages`` value is returned. Bodies without a list ``messages`` field are
    passed through unchanged with zeroed stats.

    Args:
        body: The parsed JSON request body.
        enabled: When ``False``, forward verbatim (stats still reported by core).
        stable_prefix: When ``True`` (default) compress each message in
            isolation so already-sent prefix bytes stay stable across turns
            (see :func:`_compress_stable`). When ``False`` use a single
            whole-array pass.
    """
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        return CompressionOutcome(body=body, orig_tokens=0, new_tokens=0, ratio=0.0, changed=False)

    if not enabled:
        new_messages, stats = _core_compress(messages, enabled=False)
        orig_tokens, new_tokens = stats.orig_tokens, stats.new_tokens
    elif stable_prefix:
        new_messages, orig_tokens, new_tokens = _compress_stable(messages)
    else:
        new_messages, stats = _core_compress(messages)
        orig_tokens, new_tokens = stats.orig_tokens, stats.new_tokens

    new_body = dict(body)
    new_body["messages"] = new_messages

    ratio = 0.0 if orig_tokens <= 0 else 1.0 - (new_tokens / orig_tokens)
    changed = new_tokens < orig_tokens
    return CompressionOutcome(
        body=new_body,
        orig_tokens=orig_tokens,
        new_tokens=new_tokens,
        ratio=ratio,
        changed=changed,
    )
