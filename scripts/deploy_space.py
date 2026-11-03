"""Deploy the dashboard to a Hugging Face Space from GitHub Actions (.github/workflows/deploy-space.yml).

Configuration comes from the environment, never from interpolation into shell:
    HF_TOKEN          Hugging Face token with write access (required)
    HF_SPACE_ID       e.g. your-hf-username/brink (required)
    HF_SPACE_PUBLIC   "true" for a public Space; anything else keeps it private. Applied on every deploy.
    DEPLOY_EVENT      the GitHub event: workflow_run (after CI passed), schedule or workflow_dispatch
    GITHUB_TOKEN, GITHUB_REPOSITORY   used to confirm CI passed for this commit on schedule / manual runs

Decisions:
- Results come from the `results/all` branch. "Branch doesn't exist" means demo mode. Any other git failure
  aborts the deploy, so a network glitch can never replace real results with simulated ones.
- Real mode needs at least one finished experiment (a REPORT.md). Otherwise the Space shows the simulated demo
  runs only; partial real runs are never mixed with simulated ones.
- Only code whose CI run passed is deployed.
- A scheduled run that finds nothing new (same code, same results, same mode) does nothing.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.build_space import build, push

RESULTS_BRANCH = "results/all"
MARKER = "DEPLOYED.json"


class DeployError(RuntimeError):
    pass


def git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True)


def results_sha(repo: Path) -> str | None:
    """SHA of results/all on origin, None if the branch doesn't exist. Raises on any other failure."""
    r = git(repo, "ls-remote", "--exit-code", "origin", f"refs/heads/{RESULTS_BRANCH}")
    if r.returncode == 2:  # --exit-code: 2 = no matching ref
        return None
    if r.returncode != 0 or not r.stdout.strip():
        raise DeployError(f"could not query {RESULTS_BRANCH}: {(r.stderr or r.stdout).strip()}")
    return r.stdout.split()[0]


def fetch_results(repo: Path) -> Path | None:
    """Check out results/ from results/all into the working tree. None if the branch has no results/ path."""
    r = git(repo, "fetch", "--depth", "1", "origin", RESULTS_BRANCH)
    if r.returncode != 0:
        raise DeployError(f"could not fetch {RESULTS_BRANCH}: {r.stderr.strip()}")
    tree = git(repo, "ls-tree", "--name-only", "FETCH_HEAD", "results")
    if tree.returncode != 0:
        raise DeployError(f"could not read {RESULTS_BRANCH}: {tree.stderr.strip()}")
    if not tree.stdout.strip():
        return None
    r = git(repo, "checkout", "FETCH_HEAD", "--", "results")
    if r.returncode != 0:
        raise DeployError(f"could not check out results: {r.stderr.strip()}")
    return repo / "results"


def finished_stages(results: Path | None) -> list[str]:
    if results is None or not results.exists():
        return []
    return sorted(p.parent.name for p in results.glob("*/REPORT.md"))


def ci_passed(sha: str) -> bool:
    """True if the CI workflow passed for `sha` (GitHub API; needs GITHUB_TOKEN with actions: read)."""
    repo, token = os.environ.get("GITHUB_REPOSITORY"), os.environ.get("GITHUB_TOKEN")
    if not repo or not token:
        raise DeployError("GITHUB_REPOSITORY / GITHUB_TOKEN missing: can't confirm CI passed")
    req = urllib.request.Request(
        f"https://api.github.com/repos/{repo}/actions/runs?head_sha={sha}&per_page=50",
        headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        runs = json.load(resp).get("workflow_runs", [])
    return any(r.get("name") == "CI" and r.get("conclusion") == "success" for r in runs)


def deployed_marker(space_id: str, api) -> dict | None:
    try:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(space_id, MARKER, repo_type="space", token=getattr(api, "token", None))
        return json.loads(Path(path).read_text())
    except Exception:
        return None


def deploy(
    repo: Path,
    space_id: str,
    public: bool,
    event: str,
    out: Path,
    api=None,
    ci_check=ci_passed,
    marker_reader=deployed_marker,
) -> str:
    code = git(repo, "rev-parse", "HEAD").stdout.strip()
    if event != "workflow_run" and not ci_check(code):
        return f"skipped: CI has not passed for {code[:7]}"
    rsha = results_sha(repo)
    results = fetch_results(repo) if rsha else None
    stages = finished_stages(results)
    mode = "real" if stages else "demo"
    marker = {"code_sha": code, "results_sha": rsha, "mode": mode, "stages": stages}
    if event == "schedule" and marker_reader(space_id, api) == marker:
        if api is not None:  # visibility may have been changed in the repo variables
            api.update_repo_settings(space_id, private=not public, repo_type="space")
        return "skipped: nothing new since the last deploy"
    folder = build(out, results, demo=(mode == "demo"))
    (folder / MARKER).write_text(json.dumps(marker, indent=2))
    what = f"{len(stages)} finished experiment(s)" if stages else "SIMULATED demo runs"
    push(folder, space_id, public, api=api, message=f"Deploy {code[:7]} with {what}")
    return f"deployed {mode} ({what}) at {code[:7]}"


def main() -> int:
    space_id, token = os.environ.get("HF_SPACE_ID", "").strip(), os.environ.get("HF_TOKEN", "").strip()
    if not space_id or not token:
        print("::error::Set the HF_TOKEN secret and the HF_SPACE_ID variable to deploy the Space.")
        return 1
    public = os.environ.get("HF_SPACE_PUBLIC", "").strip().lower() == "true"
    event = os.environ.get("DEPLOY_EVENT", "workflow_dispatch")
    from huggingface_hub import HfApi

    try:
        status = deploy(ROOT, space_id, public, event, ROOT / "build" / "space", api=HfApi(token=token))
    except DeployError as e:
        print(f"::error::{e}")
        return 1
    print(status)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a") as f:
            f.write(f"{status}\n\nhttps://huggingface.co/spaces/{space_id}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
