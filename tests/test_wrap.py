"""`tokenslim wrap <agent>` tests — assert the env + argv, never launch anything.

The real exec is injected (``exec_fn``) so we capture what *would* be run, and
``shutil.which`` is monkeypatched so "binary installed?" is deterministic.
"""

from __future__ import annotations

import pytest

from tokenslim_proxy import wrap as wrapmod
from tokenslim_proxy.config import ProxyConfig
from tokenslim_proxy.wrap import WrapError, build_env, proxy_url, wrap


@pytest.fixture
def cfg() -> ProxyConfig:
    return ProxyConfig(host="127.0.0.1", port=8788)


@pytest.fixture
def installed(monkeypatch):
    """Pretend every agent binary is on PATH."""
    monkeypatch.setattr(wrapmod.shutil, "which", lambda _b: "/usr/bin/" + _b)


class _FakeExec:
    """Captures the argv/env it was called with instead of exec'ing."""

    def __init__(self) -> None:
        self.argv: list[str] | None = None
        self.env: dict[str, str] | None = None

    def __call__(self, argv: list[str], env: dict[str, str]) -> int:
        self.argv = argv
        self.env = env
        return 0


def test_proxy_url_rewrites_wildcard_bind():
    assert proxy_url(ProxyConfig(host="0.0.0.0", port=9000)) == "http://127.0.0.1:9000"
    assert proxy_url(ProxyConfig(host="127.0.0.1", port=8788)) == "http://127.0.0.1:8788"


def test_claude_sets_anthropic_base_url_and_execs_binary(cfg, installed):
    fake = _FakeExec()
    rc = wrap("claude", ["--dangerously-skip-permissions"], cfg=cfg, exec_fn=fake, base_env={})
    assert rc == 0
    # Exec'd the right binary with passed-through args.
    assert fake.argv == ["claude", "--dangerously-skip-permissions"]
    # Anthropic base repointed at the proxy; OpenAI vars not set for claude.
    assert fake.env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:8788"
    assert "OPENAI_BASE_URL" not in fake.env


def test_codex_sets_both_openai_base_vars(cfg, installed):
    fake = _FakeExec()
    wrap("codex", [], cfg=cfg, exec_fn=fake, base_env={})
    assert fake.argv == ["codex"]
    assert fake.env["OPENAI_BASE_URL"] == "http://127.0.0.1:8788"
    assert fake.env["OPENAI_API_BASE"] == "http://127.0.0.1:8788"
    assert "ANTHROPIC_BASE_URL" not in fake.env


def test_aider_repoints_both_providers(cfg, installed):
    fake = _FakeExec()
    wrap("aider", ["--model", "gpt-4o"], cfg=cfg, exec_fn=fake, base_env={})
    assert fake.argv == ["aider", "--model", "gpt-4o"]
    url = "http://127.0.0.1:8788"
    assert fake.env["OPENAI_BASE_URL"] == url
    assert fake.env["OPENAI_API_BASE"] == url
    assert fake.env["ANTHROPIC_BASE_URL"] == url


def test_existing_api_keys_are_preserved(cfg, installed):
    fake = _FakeExec()
    base = {"OPENAI_API_KEY": "sk-keep", "PATH": "/usr/bin", "HOME": "/home/x"}
    wrap("codex", [], cfg=cfg, exec_fn=fake, base_env=base)
    # Keys untouched (proxy forwards them upstream); base env carried through.
    assert fake.env["OPENAI_API_KEY"] == "sk-keep"
    assert fake.env["HOME"] == "/home/x"


def test_case_insensitive_agent_name(cfg, installed):
    fake = _FakeExec()
    wrap("Claude", [], cfg=cfg, exec_fn=fake, base_env={})
    assert fake.argv == ["claude"]


def test_unknown_agent_raises_clear_error(cfg, installed):
    fake = _FakeExec()
    with pytest.raises(WrapError) as ei:
        wrap("emacs", [], cfg=cfg, exec_fn=fake, base_env={})
    msg = str(ei.value)
    assert "Unknown agent 'emacs'" in msg
    assert "claude" in msg  # lists supported agents
    assert fake.argv is None  # never tried to exec


def test_missing_binary_raises_install_hint(cfg, monkeypatch):
    monkeypatch.setattr(wrapmod.shutil, "which", lambda _b: None)
    fake = _FakeExec()
    with pytest.raises(WrapError) as ei:
        wrap("cursor", [], cfg=cfg, exec_fn=fake, base_env={})
    msg = str(ei.value)
    assert "cursor" in msg
    assert "not installed" in msg
    assert fake.argv is None


def test_build_env_only_touches_base_url_vars(cfg):
    env = build_env(wrapmod.AGENTS["claude"], cfg=cfg, base_env={"FOO": "bar"})
    assert env["FOO"] == "bar"
    assert env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:8788"


# --- CLI dispatch (cli.main / wrap.main) -------------------------------------


def test_cli_dispatches_wrap(monkeypatch, installed):
    from tokenslim_proxy import cli

    captured = {}

    def fake_wrap_main(args):
        captured["args"] = list(args)
        return 0

    monkeypatch.setattr(cli.wrap_cmd, "main", fake_wrap_main)
    rc = cli.main(["wrap", "claude", "--foo"])
    assert rc == 0
    assert captured["args"] == ["claude", "--foo"]


def test_cli_unknown_command_returns_2(capsys):
    from tokenslim_proxy import cli

    rc = cli.main(["frobnicate"])
    assert rc == 2
    assert "unknown command" in capsys.readouterr().err


def test_wrap_main_no_args_prints_usage(capsys):
    rc = wrapmod.main([])
    assert rc == 2
    assert "usage: tokenslim wrap" in capsys.readouterr().out


def test_wrap_main_error_returns_1(monkeypatch, capsys):
    monkeypatch.setattr(wrapmod.shutil, "which", lambda _b: None)
    rc = wrapmod.main(["wrap", "aider"])
    assert rc == 1
    assert "tokenslim wrap:" in capsys.readouterr().err
