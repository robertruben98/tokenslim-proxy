"""CLI entrypoint: ``python -m tokenslim_proxy`` / ``tokenslim-proxy``."""

from __future__ import annotations

import uvicorn

from .config import ProxyConfig


def main() -> None:
    cfg = ProxyConfig.from_env()
    uvicorn.run("tokenslim_proxy.app:app", host=cfg.host, port=cfg.port, factory=False)


if __name__ == "__main__":
    main()
