"""Assemble the monitor dashboard as a Hugging Face Space (Gradio SDK, CPU basic is enough).

    python scripts/build_space.py --out build/space                      # build only
    python scripts/build_space.py --out build/space --push user/brink    # build and upload (needs HF_TOKEN)

Copies the app, the `brink` package and every FINISHED experiment under results/ (one with a REPORT.md; CSV /
JSON / Markdown / PNG only: never model replies, tensors or adapters). Unfinished or failed stages are left out.
`--demo` builds with the clearly bannered simulated runs INSTEAD of results/: real and simulated runs are never
mixed on one Space.

The Gradio version comes from the pin in app/requirements.txt: the same file CI installs, so the deployed
version is the tested one. Automatic deploys from GitHub Actions go through scripts/deploy_space.py.

`--static` builds a static Space instead (free on Hugging Face, no server): one HTML page with every finished
stage's report, plots and logs. It shows the recorded results only; the live tabs need the Gradio app.
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from brink.io import finished_stages

REQUIREMENTS = ROOT / "app" / "requirements.txt"
ALLOWED = {".csv", ".json", ".md", ".png"}
SKIP_DIRS = {"adapter", "adapters", "_work", "__pycache__"}
SDKS = ("gradio", "static")
BRAND = ROOT / "brand"
SITE_FILES = (
    "favicon.ico",
    "favicon.svg",
    "favicon-16.png",
    "favicon-32.png",
    "favicon-48.png",
    "apple-touch-icon.png",
    "icon-192.png",
    "icon-512.png",
    "icon-512.webp",
    "icon-maskable-512.png",
    "logo.svg",
    "logo-dark.svg",
    "logo.png",
    "logo-dark.png",
    "logo.webp",
    "mark.svg",
    "og-image.png",
    "og-image.webp",
    "og-image.jpg",
)


def gradio_pin(requirements: Path = REQUIREMENTS) -> str:
    """The exact Gradio version pinned in app/requirements.txt (single source of truth)."""
    m = re.search(r"^gradio==([\w.]+)(?:\s|$)", requirements.read_text(), flags=re.MULTILINE)
    if not m:
        raise ValueError(f"{requirements} must pin gradio exactly (gradio==X.Y.Z)")
    return m.group(1)


STATIC_README = """---
title: Brink safety monitor
emoji: 🛡️
colorFrom: purple
colorTo: indigo
sdk: static
pinned: false
license: mit
short_description: Catches LLM safety collapse during fine-tuning
---

# Brink · keep a model safe while you fine-tune it

The recorded results of every finished Brink experiment on Tiny Aya: reports, plots and logs, including the
criteria that failed. This page runs nothing. The live app (*Run Brink*, *Gate your data*) is in the repository:
`make app`.

Code: https://github.com/andrew1234-arch/brink
"""


def space_readme(sdk: str = "gradio") -> str:
    if sdk == "static":
        return STATIC_README
    return f"""---
title: Brink safety monitor
emoji: 🛡️
colorFrom: purple
colorTo: indigo
sdk: gradio
sdk_version: {gradio_pin()}
python_version: "3.12"
app_file: app.py
suggested_hardware: t4-small
pinned: false
license: mit
short_description: Catches LLM safety collapse during fine-tuning
---

# Brink · keep a model safe while you fine-tune it

- **Run Brink:** the whole pipeline live on a real model. Brink fingerprints the model's refusal mechanism,
  gates a contaminated fine-tuning set, fine-tunes the model twice (unprotected, and with Brink's alarm and
  repair), then verifies both on held-out prompts. On a GPU it runs Tiny Aya; on CPU a 0.5B model.
- **Gate your data:** upload a fine-tuning file and get every sample's risk score and a cleaned file.
- **Experiments:** every experiment run, with the alarm rule adjustable live.

Live runs load AdvBench and HarmBench, which are gated on Hugging Face: accept their terms and add a read
token as the Space secret `HF_TOKEN`.

Code: https://github.com/andrew1234-arch/brink
"""


SPACE_APP = """import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from dashboard import CSS, FAVICON, MAX_UPLOAD, THEME, build  # noqa: E402

demo = build(HERE / "results")
demo.launch(
    css=CSS,
    theme=THEME,
    favicon_path=FAVICON,
    js="() => { document.body.classList.remove('dark'); }",
    max_file_size=MAX_UPLOAD,
)
"""


def copy_results(src: Path, dst: Path) -> tuple[int, list[str]]:
    """Copy the publishable files of every finished stage. Returns (files copied, stages copied)."""
    stages = finished_stages(src)
    n = 0
    for stage in stages:
        for f in (src / stage).rglob("*"):
            rel = f.relative_to(src)
            if f.is_file() and f.suffix in ALLOWED and not (set(rel.parts) & SKIP_DIRS):
                target = dst / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, target)
                n += 1
    for name in ("ROI.md", "roi.json"):
        if (src / name).is_file():
            dst.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src / name, dst / name)
            n += 1
    skipped = sorted(p.name for p in src.iterdir() if p.is_dir() and p.name not in stages)
    if skipped:
        print(f"left out unfinished stages (no REPORT.md): {', '.join(skipped)}")
    return n, stages


def build_static(out: Path, results: Path | None, demo: bool) -> Path:
    """The static Space: index.html, the results it shows, and the README. No app code."""
    from scripts.static_site import render

    if demo:
        from brink.dashboard.simulate import simulate_gate, simulate_go_nogo

        simulate_go_nogo(out / "results" / "simulated_go_nogo")
        simulate_gate(out / "results" / "simulated_gate")
        stages = finished_stages(out / "results")
    else:
        _, stages = copy_results(results, out / "results") if results is not None and results.is_dir() else (0, [])
    from scripts.static_site import manifest, robots, sitemap

    for name in SITE_FILES:
        shutil.copy2(BRAND / name, out / name)
    (out / "site.webmanifest").write_text(manifest(), encoding="utf-8")
    (out / "robots.txt").write_text(robots(), encoding="utf-8")
    (out / "sitemap.xml").write_text(sitemap(), encoding="utf-8")
    (out / "README.md").write_text(space_readme("static"))
    (out / "index.html").write_text(render(out / "results", stages, simulated=demo), encoding="utf-8")
    print(f"built {out} (static page, {len(stages)} {'SIMULATED' if demo else 'finished'} stage(s))")
    return out


def build(out: Path, results: Path | None, demo: bool, sdk: str = "gradio") -> Path:
    """Build the Space folder. With demo=True, `results` is ignored: the Space shows simulated runs only."""
    if sdk not in SDKS:
        raise ValueError(f"unknown Space SDK {sdk!r}: use one of {', '.join(SDKS)}")
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    if sdk == "static":
        return build_static(out, results, demo)
    shutil.copy2(ROOT / "app" / "app.py", out / "dashboard.py")
    shutil.copy2(ROOT / "app" / "live_tab.py", out / "live_tab.py")
    shutil.copytree(ROOT / "app" / "examples", out / "examples")
    shutil.copytree(BRAND, out / "brand")
    shutil.copytree(ROOT / "brink", out / "brink", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy2(REQUIREMENTS, out / "requirements.txt")
    (out / "README.md").write_text(space_readme())
    (out / "app.py").write_text(SPACE_APP)
    if demo:
        from brink.dashboard.simulate import simulate_gate, simulate_go_nogo

        simulate_go_nogo(out / "results" / "simulated_go_nogo")
        simulate_gate(out / "results" / "simulated_gate")
        print(f"built {out} with SIMULATED demo runs only")
        return out
    n, stages = copy_results(results, out / "results") if results is not None and results.is_dir() else (0, [])
    print(f"built {out} ({n} result files from {len(stages)} finished stage(s))")
    if n == 0:
        print("warning: no finished runs found; the Space will show 'No runs found'. Add real runs or pass --demo.")
    return out


def set_visibility(api, space_id: str, public: bool) -> None:
    """Applied on every deploy (not only at creation), so changing the setting takes effect next time."""
    api.update_repo_settings(space_id, private=not public, repo_type="space")


def push(
    out: Path, space_id: str, public: bool, api=None, message: str = "Deploy Brink dashboard", sdk: str = "gradio"
) -> None:
    """Upload `out` so the Space mirrors it exactly, and apply the visibility setting."""
    if api is None:
        from huggingface_hub import HfApi

        api = HfApi()
    api.create_repo(space_id, repo_type="space", space_sdk=sdk, private=not public, exist_ok=True)
    set_visibility(api, space_id, public)
    api.upload_folder(
        folder_path=str(out), repo_id=space_id, repo_type="space", commit_message=message, delete_patterns=["*"]
    )
    print(f"pushed to https://huggingface.co/spaces/{space_id} ({'public' if public else 'private'})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "build" / "space"))
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--demo", action="store_true", help="build with clearly-bannered simulated runs instead")
    ap.add_argument("--push", metavar="REPO_ID", help="upload to this Space (needs HF_TOKEN with write access)")
    ap.add_argument("--public", action="store_true", help="make the Space public (default: private)")
    ap.add_argument("--static", action="store_true", help="a static results page instead of the Gradio app")
    args = ap.parse_args()
    sdk = "static" if args.static else "gradio"
    out = build(Path(args.out), Path(args.results), args.demo, sdk=sdk)
    if args.push:
        push(out, args.push, args.public, sdk=sdk)


if __name__ == "__main__":
    main()
