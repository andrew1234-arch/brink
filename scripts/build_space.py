"""Assemble the monitor dashboard as a Hugging Face Space (Gradio SDK, CPU basic is enough).

    python scripts/build_space.py --out build/space                      # build only
    python scripts/build_space.py --out build/space --push user/brink    # build and upload (needs HF_TOKEN)

Copies the app, the `brink` package and every run folder under results/ (CSV / JSON / Markdown / PNG only:
never model replies, tensors or adapters). Without real runs, `--demo` bakes in the clearly-bannered
simulated runs instead.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ALLOWED = {".csv", ".json", ".md", ".png"}
SKIP_DIRS = {"adapter", "__pycache__"}

SPACE_README = """---
title: Brink safety monitor
emoji: 🛡️
colorFrom: blue
colorTo: red
sdk: gradio
sdk_version: 6.31.0
python_version: "3.12"
app_file: app.py
pinned: false
license: mit
short_description: Catches LLM safety collapse during fine-tuning
---

# Brink · safety monitor

Watches a model's refusal mechanism while it is fine-tuned and raises the alarm before its behaviour breaks.
Shows the alarm, the probe signals, the ground-truth attack success rate, per-layer erosion and the data-gate
results for every experiment run. The alarm rule can be changed live.

Code: https://github.com/andrew1234-arch/brink
"""

SPACE_APP = """import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from dashboard import CSS, build  # noqa: E402

demo = build(HERE / "results")
demo.launch(css=CSS, js="() => { document.body.classList.remove('dark'); }")
"""


def copy_results(src: Path, dst: Path) -> int:
    n = 0
    for f in src.rglob("*"):
        if f.is_file() and f.suffix in ALLOWED and not (set(f.relative_to(src).parts) & SKIP_DIRS):
            target = dst / f.relative_to(src)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, target)
            n += 1
    return n


def build(out: Path, results: Path, demo: bool) -> Path:
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    shutil.copy2(ROOT / "app" / "app.py", out / "dashboard.py")  # not app.py: the Space entry point is app.py
    shutil.copytree(ROOT / "brink", out / "brink", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy2(ROOT / "app" / "requirements.txt", out / "requirements.txt")
    (out / "README.md").write_text(SPACE_README)
    (out / "app.py").write_text(SPACE_APP)
    n = copy_results(results, out / "results") if results.exists() else 0
    if demo:
        from brink.dashboard.simulate import simulate_gate, simulate_go_nogo

        simulate_go_nogo(out / "results" / "simulated_go_nogo")
        simulate_gate(out / "results" / "simulated_gate")
    print(f"built {out} ({n} result files{', plus simulated demo runs' if demo else ''})")
    if n == 0 and not demo:
        print("warning: no result files found; the Space will show 'No runs found'. Add real runs or pass --demo.")
    return out


def main():
    import sys

    sys.path.insert(0, str(ROOT))
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "build" / "space"))
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--demo", action="store_true", help="include clearly-bannered simulated runs")
    ap.add_argument("--push", metavar="REPO_ID", help="upload to this Space (needs HF_TOKEN with write access)")
    ap.add_argument("--public", action="store_true", help="create the Space public (default: private)")
    args = ap.parse_args()
    out = build(Path(args.out), Path(args.results), args.demo)
    if args.push:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(args.push, repo_type="space", space_sdk="gradio", private=not args.public, exist_ok=True)
        api.upload_folder(
            folder_path=str(out),
            repo_id=args.push,
            repo_type="space",
            commit_message="Deploy Brink dashboard",
            delete_patterns=["results/**"],
        )
        print(f"pushed to https://huggingface.co/spaces/{args.push}")


if __name__ == "__main__":
    main()
