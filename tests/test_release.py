"""Release tooling: the Space build and the Hub staging must never include harmful text or drifted adapters."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _fake_results(root: Path) -> Path:
    r = root / "results"
    for cond in ("none", "rollback_guard"):
        d = r / "repair" / cond
        (d / "adapter").mkdir(parents=True)
        (d / "adapter" / "adapter_model.safetensors").write_bytes(b"weights")
        (d / "adapter" / "adapter_config.json").write_text("{}")
        (d / "evals.csv").write_text("step,asr\n0,0.1\n")
        (d / "generations.jsonl").write_text('{"response": "harmful text"}\n')
    (r / "repair" / "fingerprint.json").write_text("{}")
    (r / "repair" / "fingerprint.pt").write_bytes(b"t")
    (r / "repair" / "REPORT.md").write_text("# report")
    return r


def test_space_build_excludes_sensitive_files(tmp_path):
    from scripts.build_space import build

    out = build(tmp_path / "space", _fake_results(tmp_path), demo=True)
    names = {p.name for p in out.rglob("*") if p.is_file()}
    assert {"app.py", "dashboard.py", "README.md", "requirements.txt", "evals.csv", "REPORT.md"} <= names
    assert not names & {"generations.jsonl", "adapter_model.safetensors", "adapter_config.json", "fingerprint.pt"}
    assert (out / "results" / "simulated_go_nogo" / "meta.json").exists()
    assert "sdk: gradio" in (out / "README.md").read_text()


def test_hub_staging_policy(tmp_path):
    from scripts.publish_hub import collect

    stage = tmp_path / "stage"
    files = collect(_fake_results(tmp_path), stage, ["rollback_guard"])
    assert "fingerprints/repair/fingerprint.pt" in files
    assert any(f.startswith("adapters/repair/rollback_guard/") for f in files)
    assert not any("/none/adapter" in f for f in files)  # drifted adapter never published
    assert not any(f.endswith("generations.jsonl") for f in files)


def test_hub_refuses_drifted_adapters(tmp_path):
    r = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "publish_hub.py"),
            "--repo",
            "x/y",
            "--results",
            str(_fake_results(tmp_path)),
            "--adapters",
            "none",
        ],
        capture_output=True,
        text=True,
    )
    assert r.returncode != 0 and "refusing" in (r.stderr + r.stdout)


def test_pipeline_layout_and_work_dir(tmp_path):
    from scripts.publish_hub import collect

    r = _fake_results(tmp_path)
    for cond in ("naive", "brink"):
        d = r / "_work" / "pipeline" / "adapters" / cond
        d.mkdir(parents=True)
        (d / "adapter_model.safetensors").write_bytes(b"w")
    (r / "_work" / "pipeline" / "raw.jsonl").write_text('{"instruction": "x"}\n')
    (r / "_work" / "pipeline" / "manifest.json").write_text("{}")
    files = collect(r, tmp_path / "stage", ["brink"])
    assert any("adapters/brink/" in f for f in files)
    assert not any("/naive/" in f for f in files)  # the unprotected pipeline model stays private
    assert not any(f.endswith(("raw.jsonl", "manifest.json")) for f in files)
