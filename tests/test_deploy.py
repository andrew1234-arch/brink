"""Space build and deploy logic, against real local git repositories and a fake Hugging Face client."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.build_space import gradio_pin, push, space_readme
from scripts.deploy_space import DeployError, deploy

ID = ["-c", "user.name=t", "-c", "user.email=t@t"]


def _git(cwd, *args):
    subprocess.run(["git", *ID, *args], cwd=cwd, check=True, capture_output=True)


class FakeApi:
    def __init__(self):
        self.calls = []

    def create_repo(self, repo_id, **kw):
        self.calls.append(("create", repo_id, kw))

    def update_repo_settings(self, repo_id, **kw):
        self.calls.append(("settings", repo_id, kw))

    def upload_folder(self, **kw):
        self.calls.append(("upload", kw["repo_id"], kw))

    def kinds(self):
        return [c[0] for c in self.calls]


@pytest.fixture
def repo(tmp_path):
    """origin (bare) with a main branch; returns (clone, origin, helper to publish a results/all branch)."""
    origin = tmp_path / "origin.git"
    subprocess.run(["git", "init", "-q", "--bare", "-b", "main", str(origin)], check=True)
    clone = tmp_path / "clone"
    subprocess.run(["git", "clone", "-q", str(origin), str(clone)], check=True, capture_output=True)
    (clone / "README.md").write_text("x\n")
    _git(clone, "add", ".")
    _git(clone, "commit", "-q", "-m", "init")
    _git(clone, "push", "-q", "origin", "HEAD:main")

    def publish_results(files: dict[str, str]):
        r = tmp_path / "results_repo"
        r.mkdir(exist_ok=True)
        _git(r, "init", "-q")
        for rel, text in files.items():
            (r / rel).parent.mkdir(parents=True, exist_ok=True)
            (r / rel).write_text(text)
        _git(r, "add", "-A")
        _git(r, "commit", "-q", "-m", "results")
        _git(r, "push", "-q", "-f", str(origin), "HEAD:refs/heads/results/all")

    return clone, origin, publish_results


def _run(clone, tmp_path, api, event="workflow_run", public=False, ci=True, marker=None):
    return deploy(
        clone,
        "me/brink",
        public,
        event,
        tmp_path / "space",
        api=api,
        ci_check=lambda sha: ci,
        marker_reader=lambda sid, a: marker,
    )


def test_no_results_branch_deploys_demo_private(repo, tmp_path):
    clone, _, _ = repo
    api = FakeApi()
    status = _run(clone, tmp_path, api)
    assert status.startswith("deployed demo")
    assert (tmp_path / "space" / "results" / "simulated_go_nogo").exists()
    create = next(c for c in api.calls if c[0] == "create")
    settings = next(c for c in api.calls if c[0] == "settings")
    upload = next(c for c in api.calls if c[0] == "upload")
    assert create[2]["private"] is True and settings[2]["private"] is True
    assert upload[2]["delete_patterns"] == ["*"]  # Space mirrors the build exactly


def test_partial_real_runs_are_not_mixed_with_demo(repo, tmp_path):
    clone, _, publish = repo
    publish({"results/go_nogo/pos_s0/probe.csv": "step,z_S3\n0,0\n"})  # failed stage: no REPORT.md
    status = _run(clone, tmp_path, FakeApi())
    assert status.startswith("deployed demo")
    assert not (tmp_path / "space" / "results" / "go_nogo").exists()  # partial real run left out
    assert (tmp_path / "space" / "results" / "simulated_gate").exists()


def test_finished_results_deploy_real_only(repo, tmp_path):
    clone, _, publish = repo
    publish({"results/go_nogo/REPORT.md": "# GO", "results/go_nogo/pos_s0/probe.csv": "step\n0\n"})
    status = _run(clone, tmp_path, FakeApi())
    assert status.startswith("deployed real")
    space = tmp_path / "space" / "results"
    assert (space / "go_nogo" / "REPORT.md").exists()
    assert not any(p.name.startswith("simulated") for p in space.iterdir())
    assert '"stages": [\n    "go_nogo"\n  ]' in (tmp_path / "space" / "DEPLOYED.json").read_text()


def test_branch_without_results_path_falls_back_to_demo(repo, tmp_path):
    clone, _, publish = repo
    publish({".gitignore": "*.pt\n"})
    assert _run(clone, tmp_path, FakeApi()).startswith("deployed demo")


def test_git_failure_aborts_instead_of_deploying_demo(repo, tmp_path):
    clone, _, _ = repo
    _git(clone, "remote", "set-url", "origin", str(tmp_path / "does-not-exist.git"))
    api = FakeApi()
    with pytest.raises(DeployError):
        _run(clone, tmp_path, api)
    assert "upload" not in api.kinds()  # real results are never overwritten


def test_untested_code_is_not_deployed(repo, tmp_path):
    clone, _, _ = repo
    api = FakeApi()
    assert _run(clone, tmp_path, api, event="schedule", ci=False).startswith("skipped: CI")
    assert api.calls == []


def test_schedule_skips_when_nothing_changed_but_applies_visibility(repo, tmp_path):
    clone, _, _ = repo
    first = FakeApi()
    _run(clone, tmp_path, first)
    import json

    marker = json.loads((tmp_path / "space" / "DEPLOYED.json").read_text())
    api = FakeApi()
    assert _run(clone, tmp_path, api, event="schedule", public=True, marker=marker).startswith("skipped: nothing new")
    assert api.calls == [("settings", "me/brink", {"private": False, "repo_type": "space"})]
    forced = FakeApi()
    assert _run(clone, tmp_path, forced, event="workflow_dispatch", marker=marker).startswith("deployed")


def test_visibility_follows_flag_on_existing_space(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    for public in (True, False):
        api = FakeApi()
        push(out, "me/brink", public, api=api)
        settings = [c for c in api.calls if c[0] == "settings"]
        assert settings == [("settings", "me/brink", {"private": not public, "repo_type": "space"})]


def test_gradio_pin_is_single_source():
    pin = gradio_pin()
    assert f"sdk_version: {pin}" in space_readme()
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text()
    assert "-r app/requirements.txt" in ci  # CI tests the exact pinned version


def test_workflows_do_not_interpolate_into_shell():
    """Template expressions in `run:` blocks are a shell-injection risk; values must go through env:."""
    for wf in (ROOT / ".github" / "workflows").glob("*.yml"):
        doc = yaml.safe_load(wf.read_text())
        for job in doc["jobs"].values():
            for step in job.get("steps", []):
                assert "${{" not in step.get("run", ""), f"{wf.name}: {step.get('name') or step['run'][:40]}"
