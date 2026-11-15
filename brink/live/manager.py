"""One live run at a time, in a background thread, shared by every visitor; one model job at a time."""

from __future__ import annotations

import threading
import time
from pathlib import Path

from brink.live.config import LiveConfig
from brink.live.pipeline import release_memory, run_live

MODEL_LOCK = threading.Lock()


class LiveManager:
    """Runs at most one live run at a time in a background thread; every visitor watches the same run."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.owner: str | None = None
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self.current: Path | None = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, cfg: LiveConfig, owner: str | None = None, **kw) -> Path:
        with self._lock:
            if self.running:
                return self.current
            if not MODEL_LOCK.acquire(blocking=False):
                raise RuntimeError("the model is busy with another job")
            self._stop.clear()
            run_dir = self.root / time.strftime("%Y%m%d-%H%M%S")
            run_dir.mkdir(parents=True, exist_ok=True)
            self.current, self.owner = run_dir, owner

            def work():
                try:
                    run_live(cfg, run_dir, should_stop=self._stop.is_set, **kw)
                except Exception:
                    pass
                finally:
                    release_memory()
                    MODEL_LOCK.release()

            self._thread = threading.Thread(target=work, name="brink-live", daemon=True)
            self._thread.start()
            return run_dir

    def stop(self, owner: str | None = None) -> bool:
        """Stop the current run. Only its owner can stop it (when it has one). Returns whether it was stopped."""
        if not self.running or (self.owner is not None and owner != self.owner):
            return False
        self._stop.set()
        return True

    def latest(self, *extra_roots: Path) -> Path | None:
        """The current run, else the newest run (with a status.json) under the root or `extra_roots`."""
        if self.current is not None:
            return self.current
        found = [p.parent for r in (self.root, *extra_roots) if Path(r).exists() for p in Path(r).rglob("status.json")]
        return max(found, key=lambda p: p.stat().st_mtime) if found else None
