"""UI utilities for CLI: spinners, progress indicators."""

from __future__ import annotations

import sys
import threading
import time
from typing import Any, Optional


class Spinner:
    """Simple CLI spinner for long-running operations."""

    def __init__(self, message: str = "Processing"):
        self.message = message
        self._running = False
        self._thread: threading.Thread | None = None
        self._chars = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

    def _spin(self) -> None:
        idx = 0
        while self._running:
            char = self._chars[idx % len(self._chars)]
            sys.stderr.write(f"\r{char} {self.message}...")
            sys.stderr.flush()
            idx += 1
            time.sleep(0.1)
        sys.stderr.write("\r" + " " * (len(self.message) + 5) + "\r")
        sys.stderr.flush()

    def __enter__(self) -> "Spinner":
        self._running = True
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *args: Any) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=0.5)


class ProgressIndicator:
    """Progress indicator for operations with known step count."""

    def __init__(self, total: int, message: str = "Processing", width: int = 30):
        self.total = total
        self.current = 0
        self.message = message
        self.width = width
        self._start_time = time.time()

    def update(self, current: Optional[int] = None, extra: str = ""):
        """Update progress display."""
        if current is not None:
            self.current = current
        else:
            self.current += 1

        pct = self.current / self.total if self.total > 0 else 0
        filled = int(self.width * pct)
        bar = "█" * filled + "░" * (self.width - filled)

        elapsed = time.time() - self._start_time
        eta = ""
        if pct > 0 and pct < 1:
            remaining = (elapsed / pct) * (1 - pct)
            eta = f" ETA: {int(remaining)}s"

        extra_str = f" {extra}" if extra else ""
        sys.stderr.write(
            f"\r{self.message}: [{bar}] {self.current}/{self.total} ({pct:.0%}){eta}{extra_str}  "
        )
        sys.stderr.flush()

    def finish(self, message: str = "Done"):
        """Complete the progress indicator."""
        elapsed = time.time() - self._start_time
        sys.stderr.write(
            f"\r{self.message}: {message} ({elapsed:.1f}s)" + " " * 20 + "\n"
        )
        sys.stderr.flush()


__all__ = ["Spinner", "ProgressIndicator"]
