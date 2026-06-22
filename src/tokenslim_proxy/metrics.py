"""Minimal in-process counters exposed in Prometheus text format.

Kept dependency-free for P0; a real ``prometheus_client`` registry (histograms,
labels) can replace this without changing the ``/metrics`` route contract.
"""

from __future__ import annotations

from threading import Lock


class Metrics:
    """Cumulative compression counters, safe for concurrent requests."""

    def __init__(self) -> None:
        self._lock = Lock()
        self.requests = 0
        self.orig_tokens = 0
        self.new_tokens = 0

    def record(self, *, orig: int, new: int) -> None:
        with self._lock:
            self.requests += 1
            self.orig_tokens += orig
            self.new_tokens += new

    @property
    def saved_tokens(self) -> int:
        return self.orig_tokens - self.new_tokens

    def render(self) -> str:
        """Render the counters as Prometheus text exposition format."""
        with self._lock:
            lines = [
                "# HELP tokenslim_proxy_requests_total Chat requests processed.",
                "# TYPE tokenslim_proxy_requests_total counter",
                f"tokenslim_proxy_requests_total {self.requests}",
                "# HELP tokenslim_proxy_orig_tokens_total Tokens before compression.",
                "# TYPE tokenslim_proxy_orig_tokens_total counter",
                f"tokenslim_proxy_orig_tokens_total {self.orig_tokens}",
                "# HELP tokenslim_proxy_new_tokens_total Tokens after compression.",
                "# TYPE tokenslim_proxy_new_tokens_total counter",
                f"tokenslim_proxy_new_tokens_total {self.new_tokens}",
                "# HELP tokenslim_proxy_saved_tokens_total Tokens saved by compression.",
                "# TYPE tokenslim_proxy_saved_tokens_total counter",
                f"tokenslim_proxy_saved_tokens_total {self.saved_tokens}",
            ]
        return "\n".join(lines) + "\n"
