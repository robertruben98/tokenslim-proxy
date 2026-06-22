"""``tokenslim wrap <agent> [args…]`` — launch a coding agent through the proxy.

The proxy is API-compatible with the providers, so routing an agent through it
is just a matter of pointing the agent's *base URL* env vars at the proxy and
then exec'ing the agent unchanged. Each supported agent gets a small entry in
:data:`AGENTS` describing its binary and which base-URL vars to set:

    agent   binary    base-url env vars set to the proxy URL
    ─────   ───────   ──────────────────────────────────────────────
    claude  claude    ANTHROPIC_BASE_URL
    codex   codex     OPENAI_BASE_URL, OPENAI_API_BASE
    cursor  cursor    OPENAI_BASE_URL, OPENAI_API_BASE
    aider   aider     OPENAI_BASE_URL, OPENAI_API_BASE, ANTHROPIC_BASE_URL
    copilot copilot   OPENAI_BASE_URL, OPENAI_API_BASE

``aider`` talks to both providers depending on the ``--model`` chosen, so both
families' base URLs are repointed. The user's API keys are left untouched in the
environment — the proxy forwards them upstream verbatim.

This module only computes the environment and the argv; the actual process
replacement goes through the injectable :func:`_default_exec` so it is testable
without launching a real agent.
"""

from __future__ import annotations

import os
import shutil
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from .config import ProxyConfig

# Env var that all OpenAI-compatible SDKs/clients honor for the API host. We set
# both spellings: the official SDK uses OPENAI_BASE_URL, older tools and litellm
# (which aider/cursor lean on) use OPENAI_API_BASE.
_OPENAI_VARS = ("OPENAI_BASE_URL", "OPENAI_API_BASE")
_ANTHROPIC_VARS = ("ANTHROPIC_BASE_URL",)


@dataclass(frozen=True)
class Agent:
    """How to launch one coding agent through the proxy."""

    binary: str
    base_url_vars: tuple[str, ...]
    # Some agents want the OpenAI base to include the ``/v1`` suffix; default
    # off so the proxy's bare root is used (its routes already carry ``/v1``).
    extra: dict[str, str] = field(default_factory=dict)


# The agent → launch-recipe table. Adding an agent is one entry here.
AGENTS: dict[str, Agent] = {
    "claude": Agent(binary="claude", base_url_vars=_ANTHROPIC_VARS),
    "codex": Agent(binary="codex", base_url_vars=_OPENAI_VARS),
    "cursor": Agent(binary="cursor", base_url_vars=_OPENAI_VARS),
    "aider": Agent(binary="aider", base_url_vars=_OPENAI_VARS + _ANTHROPIC_VARS),
    "copilot": Agent(binary="copilot", base_url_vars=_OPENAI_VARS),
}


class WrapError(Exception):
    """A user-facing error (unknown agent, binary not installed)."""


def proxy_url(cfg: ProxyConfig) -> str:
    """The local proxy's base URL, derived from the proxy's bind config.

    A ``0.0.0.0`` bind is rewritten to ``127.0.0.1`` since that is the address a
    client actually connects to.
    """
    host = "127.0.0.1" if cfg.host in ("0.0.0.0", "") else cfg.host
    return f"http://{host}:{cfg.port}"


def build_env(
    agent: Agent, *, cfg: ProxyConfig, base_env: dict[str, str] | None = None
) -> dict[str, str]:
    """Return the child environment: ``base_env`` plus the agent's base-URL vars.

    The base URLs all point at the local proxy. API keys already present in
    ``base_env`` are preserved untouched so the proxy can forward them upstream.
    """
    env = dict(os.environ if base_env is None else base_env)
    url = proxy_url(cfg)
    for var in agent.base_url_vars:
        env[var] = url
    env.update(agent.extra)
    return env


def _default_exec(argv: list[str], env: dict[str, str]) -> int:
    """Replace the current process with the agent (never returns on success).

    Falls back to a subprocess only if ``execvpe`` is unavailable; returns the
    child's exit code in that case.
    """
    os.execvpe(argv[0], argv, env)  # noqa: S606 — intentional process handoff


def resolve_agent(name: str) -> Agent:
    """Look up an agent by name, raising :class:`WrapError` if unknown."""
    agent = AGENTS.get(name.lower())
    if agent is None:
        known = ", ".join(sorted(AGENTS))
        raise WrapError(f"Unknown agent '{name}'. Supported agents: {known}.")
    return agent


def wrap(
    name: str,
    agent_args: Sequence[str],
    *,
    cfg: ProxyConfig | None = None,
    exec_fn: Callable[[list[str], dict[str, str]], int] = _default_exec,
    base_env: dict[str, str] | None = None,
) -> int:
    """Resolve the agent, set its proxy env, and exec it with ``agent_args``.

    Args:
        name: Agent name (one of :data:`AGENTS`).
        agent_args: Extra args passed through to the agent binary verbatim.
        cfg: Proxy config (defaults to env-derived).
        exec_fn: Injectable process launcher — defaults to ``execvpe``. Tests
            pass a fake to capture the argv/env without launching anything.
        base_env: Starting environment (defaults to ``os.environ``).

    Raises:
        WrapError: Unknown agent, or its binary is not on ``PATH``.
    """
    cfg = cfg or ProxyConfig.from_env()
    agent = resolve_agent(name)

    if shutil.which(agent.binary) is None:
        raise WrapError(
            f"'{agent.binary}' is not installed or not on PATH. "
            f"Install the {name} CLI, then re-run `tokenslim wrap {name}`."
        )

    env = build_env(agent, cfg=cfg, base_env=base_env)
    argv = [agent.binary, *agent_args]
    return exec_fn(argv, env)


def _usage() -> str:
    return (
        "usage: tokenslim wrap <agent> [args…]\n"
        f"agents: {', '.join(sorted(AGENTS))}\n"
        "Points the agent's API base URL at the local tokenslim proxy."
    )


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for ``tokenslim wrap``.

    Expected argv (excluding the program name): ``wrap <agent> [args…]``. The
    leading ``wrap`` token is accepted (so the ``tokenslim`` dispatcher and a
    direct call both work) but optional.
    """
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] == "wrap":
        args = args[1:]

    if not args or args[0] in ("-h", "--help"):
        sys.stdout.write(_usage() + "\n")
        return 0 if args else 2

    name, *agent_args = args
    try:
        return wrap(name, agent_args)
    except WrapError as exc:
        sys.stderr.write(f"tokenslim wrap: {exc}\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
