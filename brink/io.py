"""File logging helpers with no heavy dependencies (used by the dashboard as well as training)."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable
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
