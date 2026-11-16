"""Release tooling: the Space build and the Hub staging must never include harmful text or drifted adapters."""

from __future__ import annotations

import json
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

    out = build(tmp_path / "space", _fake_results(tmp_path), demo=False)
    names = {p.name for p in out.rglob("*") if p.is_file()}
    assert {"app.py", "dashboard.py", "live_tab.py", "README.md", "requirements.txt", "evals.csv", "REPORT.md"} <= names
    assert (out / "examples" / "finetune_sample.jsonl").exists()
    assert not names & {"generations.jsonl", "adapter_model.safetensors", "adapter_config.json", "fingerprint.pt"}
    assert not (out / "results" / "simulated_go_nogo").exists()
    assert "sdk: gradio" in (out / "README.md").read_text()


def test_space_build_includes_the_roi(tmp_path):
    from scripts.build_space import build

    r = _fake_results(tmp_path)
    (r / "ROI.md").write_text("# ROI")
    out = build(tmp_path / "space", r, demo=False)
    assert (out / "results" / "ROI.md").exists()


def test_space_demo_build_never_mixes_in_real_runs(tmp_path):
    from scripts.build_space import build

    out = build(tmp_path / "space", _fake_results(tmp_path), demo=True)
    assert (out / "results" / "simulated_go_nogo" / "meta.json").exists()
    assert not (out / "results" / "repair").exists()


def test_hub_staging_policy(tmp_path):
    from scripts.publish_hub import collect

    stage = tmp_path / "stage"
    files = collect(_fake_results(tmp_path), stage, ["rollback_guard"])
    assert "fingerprints/repair/fingerprint.pt" in files
    assert any(f.startswith("adapters/repair/rollback_guard/") for f in files)
    assert not any("/none/adapter" in f for f in files)
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
    assert "adapters/pipeline/brink/adapter_model.safetensors" in files
    assert not any("/naive/" in f for f in files)
    assert not any(f.endswith(("raw.jsonl", "manifest.json")) for f in files)


def test_split_release_keeps_adapters_out_of_the_dataset(tmp_path):
    from scripts.publish_hub import collect, split_release

    stage = tmp_path / "stage"
    collect(_fake_results(tmp_path), stage, ["rollback_guard"])
    model, data = split_release(stage)
    model_files = {str(p.relative_to(model)) for p in model.rglob("*") if p.is_file()}
    data_files = {str(p.relative_to(data)) for p in data.rglob("*") if p.is_file()}
    assert "fingerprints/repair/fingerprint.pt" in model_files
    assert any(f.startswith("adapters/repair/rollback_guard/") for f in model_files)
    assert "results/repair/REPORT.md" in data_files
    assert not any(f.startswith(("adapters/", "fingerprints/")) for f in data_files)
    assert not any(f.endswith("generations.jsonl") for f in model_files | data_files)
    assert "pretty_name" in (data / "README.md").read_text()


def test_static_space_is_one_page_of_the_finished_reports(tmp_path):
    from scripts.build_space import build

    r = _fake_results(tmp_path)
    (r / "repair" / "REPORT.md").write_text("# Repair: **DONE**\n\n| a | b |\n|---|---|\n| 1 | <script> |\n\n- one\n")
    (r / "repair" / "repair.png").write_bytes(b"png")
    out = build(tmp_path / "space", r, demo=False, sdk="static")
    names = {p.name for p in out.rglob("*") if p.is_file()}
    assert {"index.html", "README.md", "REPORT.md", "evals.csv", "repair.png"} <= names
    assert not names & {"app.py", "requirements.txt", "generations.jsonl", "adapter_model.safetensors"}
    assert "sdk: static" in (out / "README.md").read_text()
    page = (out / "index.html").read_text()
    assert "<strong>DONE</strong>" in page and "<td>1</td>" in page and "<li>one</li>" in page
    assert "brink gate" in page and 'id="use"' in page and 'class="chip ok">DONE' in page
    assert 'src="results/repair/repair.png"' in page
    assert "&lt;script&gt;" in page and "| <script> |" not in page and "SIMULATED" not in page
    site = {"favicon.ico", "favicon.svg", "apple-touch-icon.png", "og-image.png", "site.webmanifest", "robots.txt"}
    assert site | {"sitemap.xml"} <= names
    for name in site - {"site.webmanifest", "robots.txt"}:
        assert f'"{name}"' in page or f"/{name}" in page
    assert json.loads((out / "site.webmanifest").read_text())["icons"]
    assert "Sitemap:" in (out / "robots.txt").read_text()
    assert '"@context": "https://schema.org"' in page or '"@context":"https://schema.org"' in page


def test_static_demo_space_says_it_is_simulated(tmp_path):
    from scripts.build_space import build

    out = build(tmp_path / "space", _fake_results(tmp_path), demo=True, sdk="static")
    assert "SIMULATED" in (out / "index.html").read_text()
    assert not (out / "results" / "repair").exists()


def test_product_page_demo_replays_the_recorded_run(tmp_path):
    from scripts.static_site import demo_data, render

    live = tmp_path / "results" / "live"
    for cond, z in (("unprotected", [0.0, -4.0, -9.0]), ("brink", [0.0, 1.0, -1.0])):
        (live / cond).mkdir(parents=True)
        (live / cond / "probe.csv").write_text("step,z_S3\n" + "".join(f"{4 * i},{v}\n" for i, v in enumerate(z)))
    (live / "gate.csv").write_text("rank,sample,score,harmful,dropped\n1,3,0.4,1,1\n2,0,-0.1,0,0\n")
    summary = {
        "model": "m",
        "config": {"alarm_threshold": -3.0, "alarm_persistence": 2, "gate_drop_frac": 0.5},
        "before": {"asr": 0.2},
        "unprotected": {"asr": 0.9, "over_refusal": 0.0, "heldout_loss": 1.5},
        "brink": {"asr": 0.2, "over_refusal": 0.0, "heldout_loss": 1.6},
        "gate": {"harmful_caught": 1, "harmful_total": 1},
    }
    (live / "summary.json").write_text(json.dumps(summary))
    (live / "REPORT.md").write_text("# Brink, end to end")
    d = demo_data(tmp_path / "results")
    assert d["unprotected"] == {"steps": [0, 4, 8], "z": [0.0, -4.0, -9.0]} and d["gate"] == [[0.4, 1], [-0.1, 0]]
    assert d["asr_unprotected"] == "90 %" and d["threshold"] == -3.0
    page = render(tmp_path / "results", ["live"])
    assert "window.BRINK_DEMO" in page and 'id="alarm-chart"' in page and "90 % → 20 %" in page
    assert demo_data(tmp_path / "nothing") is None and "BRINK_DEMO =" not in render(tmp_path / "nothing", [])


def test_hub_model_card_and_collections(tmp_path):
    from scripts.publish_hub import collection_plan, model_card

    r = _fake_results(tmp_path)
    card = model_card("u/brink-artifacts", r, "u/brink-results")
    assert "license: cc-by-nc-4.0" in card and "base_model: CohereLabs/tiny-aya-global" in card
    assert 'PeftModel.from_pretrained(base, "u/brink-artifacts", subfolder="adapters/pipeline/brink")' in card
    assert "https://huggingface.co/datasets/u/brink-results" in card
    plan = collection_plan("Brink", "u/brink-artifacts", "u/brink-results", "u/brink")
    assert plan == {
        "Brink: models": [("u/brink-artifacts", "model"), ("u/brink", "space")],
        "Brink: datasets": [("u/brink-results", "dataset"), ("u/brink", "space")],
    }
    assert collection_plan("Brink", "u/a", None, None) == {"Brink": [("u/a", "model")]}
