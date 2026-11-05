"""Stage-by-stage progress of a live run, in status.json (the UI polls it)."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path

from brink.live.config import STAGE_TITLES, STAGES


class LiveStopped(Exception):
    pass


class Status:
    """Stage-by-stage progress in status.json (written atomically; the UI polls it)."""

    def __init__(self, out: Path, say: Callable[[str], None] | None = None):
        self.path = Path(out) / "status.json"
        self.say = say or (lambda m: print(m, flush=True))
        self.state = {
            "stages": {s: {"state": "pending", "seconds": None, "detail": ""} for s in STAGES},
            "log": [],
            "state": "running",
            "error": None,
            "started": time.time(),
            "finished": None,
        }
        self._save()

    def _save(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, indent=1))
        tmp.replace(self.path)

    def log(self, msg: str) -> None:
        self.say(f"[live] {msg}")
        self.state["log"] = (self.state["log"] + [f"{time.strftime('%H:%M:%S')}  {msg}"])[-200:]
        self._save()

    def detail(self, stage: str, text: str) -> None:
        self.state["stages"][stage]["detail"] = text
        self.log(f"{STAGE_TITLES[stage]}: {text}")

    @contextmanager
    def stage(self, name: str):
        s = self.state["stages"][name]
        s["state"], t0 = "running", time.time()
        self.log(f"{STAGE_TITLES[name]} …")
        try:
            yield
        except LiveStopped:
            s["state"] = "stopped"
            raise
        except BaseException:
            s["state"] = "failed"
            raise
        finally:
            s["seconds"] = round(time.time() - t0, 1)
            self._save()
        s["state"] = "done"
        self._save()

    def finish(self, state: str, error: str | None = None) -> None:
        self.state.update(state=state, error=error, finished=time.time())
        self._save()


def read_status(run_dir: Path) -> dict | None:
    try:
        return json.loads((Path(run_dir) / "status.json").read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None
