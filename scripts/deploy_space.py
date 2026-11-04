"""Deploy the dashboard to a Hugging Face Space from GitHub Actions (.github/workflows/deploy-space.yml).

Configuration comes from the environment, never from interpolation into shell:
    HF_TOKEN          Hugging Face token with write access (without it the deploy is skipped with a notice)
    HF_SPACE_ID       e.g. your-hf-username/brink
    HF_SPACE_PUBLIC   "true" for a public Space; anything else keeps it private. Applied on every deploy.
    HF_SPACE_TOKEN    optional READ token, installed as the Space secret HF_TOKEN so live runs can load the gated
                      datasets (AdvBench, HarmBench). Use a read-only token: the Space runs code with it.
    DEPLOY_EVENT      the GitHub event: workflow_run (after CI passed), schedule or workflow_dispatch
    GITHUB_TOKEN, GITHUB_REPOSITORY   used to confirm CI passed for this commit on schedule / manual runs

Decisions:
- Results come ONLY from the `results/all` branch, extracted into a separate folder; the code checkout's own
  results/ is ignored. "Branch doesn't exist" means demo mode. Any other git failure aborts the deploy, so a
  network glitch can never replace real results with simulated ones.
- Only finished experiments (with a REPORT.md) are published. With none, the Space shows the simulated demo
  runs only; partial real runs are never published or mixed with simulated ones.
- Only code whose CI run passed is deployed, and after CI only if that commit is still the tip of main, so a
  slow CI run on an older commit can never roll the Space back.
- A scheduled run whose code and results are both unchanged since the last deploy does nothing beyond applying
  the visibility setting, and it decides that before fetching anything.
"""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from brink.io import finished_stages
from scripts.build_space import build, push, set_visibility

RESULTS_BRANCH = "results/all"
MARKER = "DEPLOYED.json"


class DeployError(RuntimeError):
    pass


def git(repo: Path, *args: str, text: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=text)


def remote_sha(repo: Path, branch: str) -> str | None:
    """SHA of `branch` on origin, None if the branch doesn't exist. Raises on any other failure."""
    r = git(repo, "ls-remote", "--exit-code", "origin", f"refs/heads/{branch}")
    if r.returncode == 2:  # --exit-code: 2 = no matching ref
        return None
    if r.returncode != 0 or not r.stdout.strip():
        raise DeployError(f"could not query {branch}: {(r.stderr or r.stdout).strip()}")
    return r.stdout.split()[0]


def fetch_results(repo: Path, dest: Path) -> tuple[str, Path | None]:
    """Fetch results/all and extract its results/ folder into `dest`, leaving the checkout untouched.

    Returns (the SHA actually fetched, the extracted results folder or None if the branch has no results/).
    The SHA is read from what was fetched, so the deploy marker always describes the deployed content even if
    the branch moved after it was first queried.
    """
    r = git(repo, "fetch", "--no-tags", "--depth", "1", "origin", f"refs/heads/{RESULTS_BRANCH}")
    if r.returncode != 0:
        raise DeployError(f"could not fetch {RESULTS_BRANCH}: {r.stderr.strip()}")
    sha = git(repo, "rev-parse", "FETCH_HEAD").stdout.strip()
    tree = git(repo, "ls-tree", "--name-only", sha, "results")
    if tree.returncode != 0 or not sha:
        raise DeployError(f"could not read {RESULTS_BRANCH}: {tree.stderr.strip()}")
    if not tree.stdout.strip():
        return sha, None
    archive = git(repo, "archive", "--format=tar", sha, "results", text=False)
    if archive.returncode != 0:
        raise DeployError(f"could not extract results: {archive.stderr.decode(errors='replace').strip()}")
    with tarfile.open(fileobj=io.BytesIO(archive.stdout)) as tar:
        if hasattr(tarfile, "data_filter"):
            tar.extractall(dest, filter="data")  # no absolute paths, links out of dest or device files
        else:  # pragma: no cover - Python < 3.10.12
            tar.extractall(dest)
    return sha, dest / "results"


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
    """The marker of the last deploy. None if the Space or the marker doesn't exist yet; on any other error a
    warning is logged and None is returned, which only costs a redundant deploy."""
    from huggingface_hub import errors as hf_errors
    from huggingface_hub import hf_hub_download

    not_found = tuple(
        getattr(hf_errors, n)
        for n in ("RemoteEntryNotFoundError", "RepositoryNotFoundError", "RevisionNotFoundError")
        if hasattr(hf_errors, n)
    )  # not EntryNotFoundError: its subclass LocalEntryNotFoundError means "network failed"
    token = getattr(api, "token", None) or os.environ.get("HF_TOKEN") or None
    try:
        path = hf_hub_download(space_id, MARKER, repo_type="space", token=token, force_download=True)
        return json.loads(Path(path).read_text())
    except not_found:
        return None
    except Exception as e:
        print(f"::warning::could not read the last deploy marker ({type(e).__name__}: {e}); deploying anyway")
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
    space_token: str | None = None,
) -> str:
    code = git(repo, "rev-parse", "HEAD").stdout.strip()
    if event == "workflow_run":  # CI passed for `code`; deploy it only if nothing newer reached main since
        tip = remote_sha(repo, "main")
        if tip != code:
            return f"skipped: {code[:7]} is no longer the tip of main ({(tip or 'missing')[:7]}); its own run deploys"
    elif not ci_check(code):
        return f"skipped: CI has not passed for {code[:7]}"

    rsha = remote_sha(repo, RESULTS_BRANCH)
    if event == "schedule":
        last = marker_reader(space_id, api) or {}
        if (last.get("code_sha"), last.get("results_sha")) == (code, rsha):
            set_visibility(api, space_id, public)  # the only setting that can change without a new commit
            return "skipped: nothing new since the last deploy"

    with tempfile.TemporaryDirectory(prefix="brink-results-") as tmp:
        rsha, results = fetch_results(repo, Path(tmp)) if rsha else (None, None)
        stages = finished_stages(results)
        mode = "real" if stages else "demo"
        folder = build(out, results, demo=(mode == "demo"))
    marker = {"code_sha": code, "results_sha": rsha, "mode": mode, "stages": stages}
    (folder / MARKER).write_text(json.dumps(marker, indent=2))
    what = f"{len(stages)} finished experiment(s)" if stages else "SIMULATED demo runs"
    push(folder, space_id, public, api=api, message=f"Deploy {code[:7]} with {what}")
    if space_token:
        api.add_space_secret(space_id, "HF_TOKEN", space_token, description="read token for gated datasets")
    return f"deployed {mode} ({what}) at {code[:7]}"


def main() -> int:
    space_id, token = os.environ.get("HF_SPACE_ID", "").strip(), os.environ.get("HF_TOKEN", "").strip()
    if not space_id or not token:
        print("::notice::Deploy skipped: set the HF_TOKEN secret and the HF_SPACE_ID variable to deploy the Space.")
        return 0
    public = os.environ.get("HF_SPACE_PUBLIC", "").strip().lower() == "true"
    event = os.environ.get("DEPLOY_EVENT", "workflow_dispatch")
    from huggingface_hub import HfApi

    try:
        status = deploy(
            ROOT,
            space_id,
            public,
            event,
            ROOT / "build" / "space",
            api=HfApi(token=token),
            space_token=os.environ.get("HF_SPACE_TOKEN", "").strip() or None,
        )
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
