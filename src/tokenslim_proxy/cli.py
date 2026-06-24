"""``tokenslim`` umbrella CLI — dispatches subcommands.

Currently the only subcommand is ``wrap`` (launch a coding agent through the
proxy, see :mod:`tokenslim_proxy.wrap`). Kept as a thin dispatcher so more
subcommands (e.g. ``serve``) can slot in later without changing the entry point.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence

from . import wrap as wrap_cmd

_SUBCOMMANDS = ("wrap",)


def _usage() -> str:
    return (
        "usage: tokenslim <command> [args…]\n"
        f"commands: {', '.join(_SUBCOMMANDS)}\n"
        "  wrap <agent> [args…]   launch a coding agent routed through the proxy"
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch ``tokenslim <command>`` to the matching subcommand."""
    args = list(sys.argv[1:] if argv is None else argv)

    if not args or args[0] in ("-h", "--help"):
        sys.stdout.write(_usage() + "\n")
        return 0 if args else 2

    command, *rest = args
    if command == "wrap":
        return wrap_cmd.main(rest)

    sys.stderr.write(f"tokenslim: unknown command '{command}'\n{_usage()}\n")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
