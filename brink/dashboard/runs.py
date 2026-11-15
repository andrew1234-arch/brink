"""Load experiment run directories (complete or still being written) for the dashboard."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path


def _num(v):
    if v in ("", None):
        return None
    try:
        return float(v)
    except ValueError:
        return v


def read_rows(path: Path) -> list[dict]:
    """CSV rows as dicts of numbers. Tolerates a half-written last line from a live run."""
    path = Path(path)
    if not path.exists():
        return []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = []
        for r in reader:
            if None in r or any(v is None for v in r.values()):  # truncated line
                continue
            rows.append({k: _num(v) for k, v in r.items()})
    return [r for r in rows if isinstance(r.get("step"), float)]


def _json(path: Path):
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


@dataclass
class Condition:
    name: str
    probe: list[dict]
    evals: list[dict]
    train: list[dict]
    done: bool

    @property
    def steps(self) -> list[int]:
        return [int(r["step"]) for r in self.probe]

    @property
    def last_step(self) -> int | None:
        s = self.steps + [int(r["step"]) for r in self.evals]
        return max(s) if s else None


@dataclass
class Run:
    path: Path
    kind: str  # "go_nogo" | "gate" | "live" | "unknown"
    conditions: dict[str, Condition]
    fingerprint: dict | None = None
    summary: dict | None = None
    config: dict | None = None
    scores: list[dict] = field(default_factory=list)
    simulated: bool = False

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def analysis_cfg(self) -> dict:
        return (self.config or {}).get("analysis", {})


def is_run_dir(p: Path) -> bool:
    p = Path(p)
    return p.is_dir() and ((p / "fingerprint.json").exists() or any(p.glob("*/probe.csv")))


def discover_runs(root: Path) -> list[Path]:
    root = Path(root)
    if not root.exists():
        return []
    found = [root] if is_run_dir(root) else []
    found += sorted(p for p in root.rglob("*") if p.is_dir() and is_run_dir(p) and p != root)
    return found


def load_run(path: Path) -> Run:
    path = Path(path)
    conds = {}
    for d in sorted(p for p in path.iterdir() if p.is_dir() and (p / "probe.csv").exists()):
        conds[d.name] = Condition(
            d.name,
            read_rows(d / "probe.csv"),
            read_rows(d / "evals.csv"),
            read_rows(d / "train.csv"),
            (d / "DONE").exists(),
        )
    meta = _json(path / "meta.json") or {}
    scores = read_rows_any(path / "scores.csv")
    kind = meta.get("kind") or (
        "gate" if scores or (path / "subsets.json").exists() else ("go_nogo" if conds else "unknown")
    )
    return Run(
        path,
        kind,
        conds,
        _json(path / "fingerprint.json"),
        _json(path / "summary.json"),
        _json(path / "config.json"),
        scores,
        bool(meta.get("simulated")),
    )


def read_rows_any(path: Path) -> list[dict]:
    """Like read_rows, for CSVs without a 'step' column (scores.csv)."""
    path = Path(path)
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return [
            {k: _num(v) for k, v in r.items()}
            for r in csv.DictReader(f)
            if None not in r and all(v is not None for v in r.values())
        ]


STAGE_ORDER = ["go_nogo", "gate", "repair", "multilingual", "pipeline", "live", "circuit"]


def stage_reports(root: Path) -> str:
    """Every finished stage's REPORT.md (plus ROI.md when present) as one Markdown page, in programme order."""
    root = Path(root)
    if not root.is_dir():
        return "No results yet."
    stages = sorted(
        (p for p in root.iterdir() if p.is_dir() and (p / "REPORT.md").exists()),
        key=lambda p: (STAGE_ORDER.index(p.name) if p.name in STAGE_ORDER else len(STAGE_ORDER), p.name),
    )
    parts = [f"<!-- {p.name} -->\n" + (p / "REPORT.md").read_text().strip() for p in stages]
    if (root / "ROI.md").exists():
        parts.append((root / "ROI.md").read_text().strip())
    return "\n\n---\n\n".join(parts) if parts else "No finished stages yet."
