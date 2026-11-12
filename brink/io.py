"""File logging helpers with no heavy dependencies (used by the dashboard as well as training)."""

from __future__ import annotations

import csv
import json
import sys
import time
import traceback
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path


class CsvLog:
    """Append-only CSV with the header fixed by the first row (crash-safe, flushes every row)."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fields: list[str] | None = None
        if self.path.exists():
            self.path.unlink()

    def write(self, row: dict) -> None:
        new = self._fields is None
        if new:
            self._fields = list(row.keys())
        with self.path.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self._fields, extrasaction="ignore")
            if new:
                w.writeheader()
            w.writerow(row)


def read_csv(path: Path) -> list[dict]:
    with Path(path).open() as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in r.items():
            try:
                r[k] = float(v) if v not in ("", None) else None
            except ValueError:
                pass
    return rows


def write_jsonl(path: Path, records: Iterable[dict], mode: str = "a") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open(mode) as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_json(path: Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False))


REPORT = "REPORT.md"


def stage_finished(stage_dir: Path) -> bool:
    """An experiment stage counts as finished once its analysis wrote REPORT.md (it is written last)."""
    return (Path(stage_dir) / REPORT).is_file()


def finished_stages(results: Path | None) -> list[str]:
    """Names of the finished stage folders directly under `results`, sorted."""
    if results is None or not Path(results).is_dir():
        return []
    return sorted(p.name for p in Path(results).iterdir() if p.is_dir() and stage_finished(p))


def calibrated_threshold(summary_path: Path, signal: str = "S3") -> float | None:
    """The alarm threshold go / no-go calibrated on its control runs only, or None if it isn't there yet."""
    try:
        return float(json.loads(Path(summary_path).read_text())["calibration"][signal]["threshold"])
    except (FileNotFoundError, KeyError, TypeError, ValueError):
        return None


ERROR_FILE = "ERROR.md"


@contextmanager
def record_failure(out_dir: Path) -> Iterator[None]:
    """Append the traceback of a crash to `out_dir`/ERROR.md, then re-raise.

    A Kaggle session's log is only readable after the whole session ends; ERROR.md is pushed with the results, so a
    failed stage can be diagnosed (and fixed) while the session is still running the next one. Appends, so a
    subprocess's own traceback is kept above the parent's.
    """
    try:
        yield
    except Exception:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
        with (out / ERROR_FILE).open("a") as f:
            f.write(f"## {stamp}: `{' '.join(sys.argv)}`\n\n```\n{traceback.format_exc()}```\n\n")
        raise
