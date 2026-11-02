"""Publish Brink's release artifacts to the Hugging Face Hub, following the responsible-release policy.

    python scripts/publish_hub.py --repo user/brink-artifacts                    # dry run: lists what would go up
    python scripts/publish_hub.py --repo user/brink-artifacts --yes              # upload (needs HF_TOKEN)
    python scripts/publish_hub.py --repo ... --yes --adapters rollback_guard     # + a protected adapter

Published: fingerprints (refusal directions, layer, refusal / compliance tokens, baselines), and every
experiment's summaries, reports, plots and CSV logs.
Never published: model replies to harmful prompts (generations.jsonl), and the deliberately drifted adapters
(`none`, `pos`, `top`). Only adapters from Brink-protected conditions can be added, and only by name.
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROTECTED = {"rollback_guard", "guard_always", "guard", "rollback"}
ALLOWED = {".csv", ".json", ".md", ".png"}

CARD = """---
license: mit
tags: [ai-safety, alignment, fine-tuning, interpretability, aya]
---

# Brink artifacts

Fingerprints and experiment results from **Brink**, which detects and prevents LLM safety collapse during
fine-tuning (Lexsi AI Safety & Alignment Hackathon 2026). Code: https://github.com/andrew1234-arch/brink

- `fingerprints/`: per-model (and per-language) refusal directions, monitored layer, refusal / compliance
  first tokens and baseline statistics (`.pt` + `.json`).
- `results/`: every experiment's report, summary, plots and CSV logs.

Model replies to harmful prompts and deliberately drifted adapters are not published.
"""


def collect(results: Path, stage: Path, adapters: list[str]) -> list[str]:
    files = []
    for f in results.rglob("*"):
        if not f.is_file():
            continue
        rel = f.relative_to(results)
        parts = set(rel.parts)
        if f.name.startswith("fingerprint") and f.suffix in (".pt", ".json"):
            dst = stage / "fingerprints" / rel
        elif "adapter" in parts:
            cond = rel.parts[rel.parts.index("adapter") - 1]
            if cond not in adapters:
                continue
            dst = stage / "adapters" / rel
        elif f.suffix in ALLOWED:
            dst = stage / "results" / rel
        else:
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, dst)
        files.append(str(dst.relative_to(stage)))
    (stage / "README.md").write_text(CARD)
    return sorted(files)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--adapters", nargs="*", default=[], help=f"conditions whose adapter to include: {sorted(PROTECTED)}")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--yes", action="store_true", help="actually upload (default: dry run)")
    args = ap.parse_args()
    bad = set(args.adapters) - PROTECTED
    if bad:
        raise SystemExit(f"refusing to publish adapters from {sorted(bad)}: only {sorted(PROTECTED)} are allowed")
    stage = Path(tempfile.mkdtemp(prefix="brink-release-"))
    files = collect(Path(args.results), stage, args.adapters)
    print(f"{len(files)} files staged in {stage}:")
    for f in files[:40]:
        print("  ", f)
    if len(files) > 40:
        print(f"   ... and {len(files) - 40} more")
    if not args.yes:
        print("dry run: nothing uploaded (pass --yes)")
        return
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(args.repo, repo_type="model", private=args.private, exist_ok=True)
    api.upload_folder(folder_path=str(stage), repo_id=args.repo, repo_type="model")
    print(f"uploaded to https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
