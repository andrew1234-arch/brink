"""Publish Brink's release artifacts to the Hugging Face Hub, following the responsible-release policy.

    python scripts/publish_hub.py --repo user/brink-artifacts                    # dry run: lists what would go up
    python scripts/publish_hub.py --repo user/brink-artifacts --yes              # upload (needs HF_TOKEN)
    python scripts/publish_hub.py --repo ... --yes --adapters rollback_guard     # + a protected adapter
    python scripts/publish_hub.py --repo user/brink-artifacts --dataset-repo user/brink-results \
        --space user/brink --collection "Brink" --yes --adapters rollback_guard brink   # the full submission release

The hackathon requires every model, adapter, log and dataset to be public (un-gated) and gathered in one Hugging
Face collection. With --dataset-repo, the logs and results go to a dataset repo and the fingerprints and adapters
to the model repo; --collection then puts both (and the Space, with --space) in one public collection.

Published: fingerprints (refusal directions, layer, refusal / compliance tokens, baselines), and every
experiment's summaries, reports, plots and CSV logs.
Never published: model replies to harmful prompts (generations.jsonl), and the deliberately drifted adapters
(`none`, `naive`, `pos`, `top`). Only adapters from Brink-protected conditions can be added, and only by name.
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROTECTED = {"rollback_guard", "guard_always", "guard", "rollback", "brink"}
ALLOWED = {".csv", ".json", ".md", ".png"}

CARD = """---
license: cc-by-nc-4.0
base_model: CohereLabs/tiny-aya-global
library_name: peft
tags: [ai-safety, alignment, fine-tuning, interpretability, aya, lora]
---

# Brink artifacts

Fingerprints and Brink-protected adapters from **Brink**, which detects and prevents LLM safety collapse during
fine-tuning (Lexsi AI Safety & Alignment Hackathon 2026). Code: https://github.com/andrew1234-arch/brink
{headline}
- `adapters/<experiment>/<condition>/`: LoRA adapters for `CohereLabs/tiny-aya-global` from Brink-protected runs
  only. `adapters/pipeline/brink/` is the Hindi assistant fine-tuned on contaminated data with Brink on (data
  gate, monitor, rollback and erosion guard).
- `fingerprints/`: per-model (and per-language) refusal directions, monitored layer, refusal / compliance
  first tokens and baseline statistics (`.pt` + `.json`).
- Results, reports and logs: {results}

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("CohereLabs/tiny-aya-global")
base = AutoModelForCausalLM.from_pretrained("CohereLabs/tiny-aya-global", dtype="auto", device_map="auto")
model = PeftModel.from_pretrained(base, "{repo}", subfolder="adapters/pipeline/brink")
```

**Licence:** the adapters and fingerprints are derived from `CohereLabs/tiny-aya-global` and follow its licence,
CC-BY-NC 4.0 (non-commercial). Brink's code is MIT.

Model replies to harmful prompts and deliberately drifted adapters are not published.
"""


def pipeline_headline(results: Path) -> str:
    """One line with the Hindi use case's before / after attack success, from results/pipeline/summary.json."""
    import json

    f = results / "pipeline" / "summary.json"
    if not f.exists():
        return ""
    c = json.loads(f.read_text()).get("conditions", {})
    try:
        before, naive, brink = c["naive"]["ref"]["asr"], c["naive"]["final"]["asr"], c["brink"]["final"]["asr"]
    except KeyError:
        return ""
    return (
        f"\n**Headline:** fine-tuning Tiny Aya as a Hindi assistant on data with 10 % harmful samples raised English"
        f" attack success from {before:.0%} to {naive:.0%}; with Brink it ended at {brink:.0%}"
        f" (`pipeline/summary.json`).\n"
    )


def model_card(repo: str, results: Path, dataset_repo: str | None = None) -> str:
    where = f"https://huggingface.co/datasets/{dataset_repo}" if dataset_repo else "`results/` in this repo"
    return CARD.format(headline=pipeline_headline(results), results=where, repo=repo)


DATASET_CARD = """---
license: mit
tags: [ai-safety, alignment, fine-tuning, red-teaming, aya]
pretty_name: Brink experiment results
---

# Brink experiment results

Every experiment's logs from **Brink**, which detects and prevents LLM safety collapse during fine-tuning (Lexsi
AI Safety & Alignment Hackathon 2026), on Cohere Tiny Aya. Code: https://github.com/andrew1234-arch/brink

One folder per experiment (`go_nogo`, `gate`, `repair`, `multilingual`, `pipeline`, `live`): `REPORT.md`,
`summary.json`, plots, and the CSV logs behind them (probe signals per training step, evaluations, alarm
events, per-sample gate scores).

Only rates, scores and labels are published. Model replies to harmful prompts are never published.
"""


def split_release(stage: Path, card: str = "") -> tuple[Path, Path]:
    """Split a staged release into (model folder: fingerprints, adapters; dataset folder: results)."""
    model, data = stage / "_model", stage / "_dataset"
    model.mkdir(), data.mkdir()
    for name in ("fingerprints", "adapters"):
        if (stage / name).exists():
            shutil.move(str(stage / name), model / name)
    if (stage / "results").exists():
        shutil.move(str(stage / "results"), data / "results")
    (model / "README.md").write_text(card or (stage / "README.md").read_text())
    (data / "README.md").write_text(DATASET_CARD)
    return model, data


def collect(results: Path, stage: Path, adapters: list[str]) -> list[str]:
    files = []
    for f in results.rglob("*"):
        if not f.is_file():
            continue
        rel = f.relative_to(results)
        parts = set(rel.parts)
        if f.name.startswith("fingerprint") and f.suffix in (".pt", ".json"):
            dst = stage / "fingerprints" / rel
        elif "adapter" in parts or "adapters" in parts:
            cond = (
                rel.parts[rel.parts.index("adapter") - 1]
                if "adapter" in parts
                else rel.parts[rel.parts.index("adapters") + 1]
            )
            if cond not in adapters:
                continue
            run = [p for p in rel.parts if p != "_work"]
            dst = stage / "adapters" / run[0] / cond / rel.name
        elif "_work" in parts:
            continue
        elif f.suffix in ALLOWED:
            dst = stage / "results" / rel
        else:
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, dst)
        files.append(str(dst.relative_to(stage)))
    (stage / "README.md").write_text(model_card("<repo>", results))
    return sorted(files)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument(
        "--adapters", nargs="*", default=[], help=f"conditions whose adapter to include: {sorted(PROTECTED)}"
    )
    ap.add_argument("--dataset-repo", default=None, help="put results and logs in this dataset repo")
    ap.add_argument("--space", default=None, help="Space to add to the collection, e.g. user/brink")
    ap.add_argument(
        "--collection",
        default=None,
        help="title of the public collections: with --dataset-repo, '<title>: models' and '<title>: datasets'",
    )
    ap.add_argument("--private", action="store_true", help="NOT allowed for the hackathon submission")
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
    card = model_card(args.repo, Path(args.results), args.dataset_repo)
    (stage / "README.md").write_text(card)
    items = [(args.repo, "model")]
    if args.dataset_repo:
        model_dir, data_dir = split_release(stage, card)
        api.create_repo(args.dataset_repo, repo_type="dataset", private=args.private, exist_ok=True)
        api.upload_folder(folder_path=str(data_dir), repo_id=args.dataset_repo, repo_type="dataset")
        print(f"uploaded to https://huggingface.co/datasets/{args.dataset_repo}")
        items.append((args.dataset_repo, "dataset"))
    else:
        model_dir = stage
    api.create_repo(args.repo, repo_type="model", private=args.private, exist_ok=True)
    api.upload_folder(folder_path=str(model_dir), repo_id=args.repo, repo_type="model")
    print(f"uploaded to https://huggingface.co/{args.repo}")
    if args.collection:
        publish_collections(api, args.collection, args.repo, args.dataset_repo, args.space, args.private)


def collection_plan(title: str, repo: str, dataset_repo: str | None, space: str | None) -> dict[str, list]:
    """The hackathon asks for a model collection and a dataset collection; the Space goes in both."""
    extra = [(space, "space")] if space else []
    if not dataset_repo:
        return {title: [(repo, "model"), *extra]}
    return {f"{title}: models": [(repo, "model"), *extra], f"{title}: datasets": [(dataset_repo, "dataset"), *extra]}


def publish_collections(api, title: str, repo: str, dataset_repo: str | None, space: str | None, private: bool):
    namespace = repo.split("/")[0]
    for name, items in collection_plan(title, repo, dataset_repo, space).items():
        col = api.create_collection(name, namespace=namespace, private=private, exists_ok=True)
        for item_id, kind in items:
            api.add_collection_item(col.slug, item_id=item_id, item_type=kind, exists_ok=True)
        print(f"collection: https://huggingface.co/collections/{col.slug} ({len(items)} items)")


if __name__ == "__main__":
    main()
