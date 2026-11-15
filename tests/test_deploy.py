"""Space build and deploy logic, against real local git repositories and a fake Hugging Face client."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.deploy_space as deploy_space
from scripts.build_space import gradio_pin, push, space_readme
from scripts.deploy_space import DeployError, deploy, deployed_marker

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

    def add_space_secret(self, repo_id, key, value, **kw):
        self.calls.append(("secret", repo_id, {"key": key, "value": value}))

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
    assert upload[2]["delete_patterns"] == ["*"]


def test_partial_real_runs_are_not_mixed_with_demo(repo, tmp_path):
    clone, _, publish = repo
    publish({"results/go_nogo/pos_s0/probe.csv": "step,z_S3\n0,0\n"})
    status = _run(clone, tmp_path, FakeApi())
    assert status.startswith("deployed demo")
    assert not (tmp_path / "space" / "results" / "go_nogo").exists()
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
    assert "upload" not in api.kinds()


def test_untested_code_is_not_deployed(repo, tmp_path):
    clone, _, _ = repo
    api = FakeApi()
    assert _run(clone, tmp_path, api, event="schedule", ci=False).startswith("skipped: CI")
    assert api.calls == []


def test_schedule_skips_when_nothing_changed_but_applies_visibility(repo, tmp_path):
    clone, _, _ = repo
    first = FakeApi()
    _run(clone, tmp_path, first)
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
    assert "-r app/requirements.txt" in ci
    tomllib = pytest.importorskip("tomllib")
    extras = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["optional-dependencies"]
    assert f"gradio=={pin}" in extras["dashboard"]


def _commit(clone, name):
    (clone / name).write_text(name)
    _git(clone, "add", name)
    _git(clone, "commit", "-q", "-m", name)


def test_late_ci_on_an_older_commit_does_not_roll_back(repo, tmp_path):
    clone, _, _ = repo
    _commit(clone, "newer.txt")
    _git(clone, "push", "-q", "origin", "HEAD:main")
    _git(clone, "checkout", "-q", "HEAD~1")
    api = FakeApi()
    status = _run(clone, tmp_path, api)
    assert status.startswith("skipped:") and "no longer the tip of main" in status
    assert api.calls == []


def test_only_finished_stages_are_published(repo, tmp_path):
    clone, origin, publish = repo
    publish(
        {
            "results/gate/REPORT.md": "# gate",
            "results/gate/scores.csv": "id,score\n0,1\n",
            "results/go_nogo/pos_s0/probe.csv": "step\n0\n",
        }
    )
    assert _run(clone, tmp_path, FakeApi()).startswith("deployed real (1 finished")
    space = tmp_path / "space" / "results"
    assert (space / "gate" / "scores.csv").exists()
    assert not (space / "go_nogo").exists()
    marker = json.loads((tmp_path / "space" / "DEPLOYED.json").read_text())
    sha = subprocess.run(["git", "rev-parse", "results/all"], cwd=origin, capture_output=True, text=True)
    assert marker["stages"] == ["gate"] and marker["results_sha"] == sha.stdout.strip()


def test_results_come_only_from_the_results_branch(repo, tmp_path):
    clone, _, publish = repo
    local = clone / "results" / "local_run"
    local.mkdir(parents=True)
    (local / "REPORT.md").write_text("# not from results/all")
    assert _run(clone, tmp_path, FakeApi()).startswith("deployed demo")
    publish({"results/gate/REPORT.md": "# gate"})
    assert _run(clone, tmp_path, FakeApi()).startswith("deployed real")
    assert sorted(p.name for p in (tmp_path / "space" / "results").iterdir()) == ["gate"]
    assert sorted(p.name for p in (clone / "results").iterdir()) == ["local_run"]


def test_schedule_decides_before_fetching(repo, tmp_path, monkeypatch):
    clone, _, publish = repo
    publish({"results/gate/REPORT.md": "# gate"})
    _run(clone, tmp_path, FakeApi())
    marker = json.loads((tmp_path / "space" / "DEPLOYED.json").read_text())

    def no_fetch(*a, **k):
        raise AssertionError("fetched although nothing changed")

    monkeypatch.setattr(deploy_space, "fetch_results", no_fetch)
    assert _run(clone, tmp_path, FakeApi(), event="schedule", marker=marker).startswith("skipped: nothing new")
    monkeypatch.undo()
    publish({"results/gate/REPORT.md": "# gate v2"})
    assert _run(clone, tmp_path, FakeApi(), event="schedule", marker=marker).startswith("deployed real")


def test_marker_reader_distinguishes_missing_from_errors(monkeypatch, capsys):
    import huggingface_hub
    from huggingface_hub import errors

    seen = {}

    def fake_download(repo_id, filename, **kw):
        seen.update(kw)
        raise exc

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    monkeypatch.setenv("HF_TOKEN", "hf_env")
    for name in ("RemoteEntryNotFoundError", "RepositoryNotFoundError"):
        cls = getattr(errors, name)
        exc = cls.__new__(cls)
        assert deployed_marker("me/brink", None) is None
        assert seen["token"] == "hf_env"
        assert "::warning::" not in capsys.readouterr().out
    exc = errors.LocalEntryNotFoundError("network down")
    assert deployed_marker("me/brink", None) is None
    assert "::warning::" in capsys.readouterr().out


def test_missing_token_skips_quietly(monkeypatch, capsys):
    monkeypatch.setenv("HF_SPACE_ID", "me/brink")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    assert deploy_space.main() == 0
    assert "::notice::" in capsys.readouterr().out


def test_deploy_workflow_guards():
    doc = yaml.safe_load((ROOT / ".github" / "workflows" / "deploy-space.yml").read_text())
    cond = doc["jobs"]["deploy"]["if"]
    assert "workflow_run.event == 'push'" in cond
    assert "head_repository.full_name == github.repository" in cond
    assert "workflow_dispatch" in doc["concurrency"]["group"]


def test_workflows_do_not_interpolate_into_shell():
    """Template expressions in `run:` blocks are a shell-injection risk; values must go through env:."""
    for wf in (ROOT / ".github" / "workflows").glob("*.yml"):
        doc = yaml.safe_load(wf.read_text())
        for job in doc["jobs"].values():
            for step in job.get("steps", []):
                assert "${{" not in step.get("run", ""), f"{wf.name}: {step.get('name') or step['run'][:40]}"


def test_space_token_becomes_the_space_secret(repo, tmp_path):
    clone, _, _ = repo
    api = FakeApi()
    deploy(clone, "me/brink", False, "workflow_run", tmp_path / "space", api=api, ci_check=lambda s: True,
           marker_reader=lambda s, a: None, space_token="hf_read")  # fmt: skip
    assert ("secret", "me/brink", {"key": "HF_TOKEN", "value": "hf_read"}) in api.calls
    without = FakeApi()
    _run(clone, tmp_path, without)
    assert "secret" not in without.kinds()


def test_static_deploy_is_a_results_page_without_app_or_secret(repo, tmp_path):
    clone, _, publish = repo
    publish({"results/repair/REPORT.md": "# done", "results/repair/summary.json": "{}"})
    api = FakeApi()
    status = deploy(clone, "me/brink", True, "workflow_run", tmp_path / "space", api=api, ci_check=lambda s: True,
                    marker_reader=lambda s, a: None, space_token="hf_read", sdk="static")  # fmt: skip
    assert status.startswith("deployed real (1 finished experiment(s), static page)")
    assert api.calls[0][2]["space_sdk"] == "static"
    assert "secret" not in api.kinds()
    out = tmp_path / "space"
    assert (out / "index.html").exists() and not (out / "app.py").exists()
    assert json.loads((out / "DEPLOYED.json").read_text())["sdk"] == "static"


def test_changing_the_sdk_redeploys_on_schedule(repo, tmp_path):
    clone, _, _ = repo
    _run(clone, tmp_path, FakeApi())
    marker = json.loads((tmp_path / "space" / "DEPLOYED.json").read_text())
    status = deploy(clone, "me/brink", False, "schedule", tmp_path / "space", api=FakeApi(),
                    ci_check=lambda s: True, marker_reader=lambda s, a: marker, sdk="static")  # fmt: skip
    assert status.startswith("deployed")


def test_gradio_paywall_names_the_static_fallback(repo, tmp_path):
    class Paywalled(FakeApi):
        def create_repo(self, repo_id, **kw):
            err = RuntimeError("402 Payment Required")
            err.response = type("R", (), {"status_code": 402})()
            raise err

    clone, _, _ = repo
    with pytest.raises(DeployError, match="HF_SPACE_SDK"):
        _run(clone, tmp_path, Paywalled())


def test_unknown_sdk_is_refused(monkeypatch, capsys):
    monkeypatch.setenv("HF_SPACE_ID", "me/brink")
    monkeypatch.setenv("HF_TOKEN", "hf_x")
    monkeypatch.setenv("HF_SPACE_SDK", "docker")
    assert deploy_space.main() == 1
    assert "::error::" in capsys.readouterr().out


def test_auto_falls_back_to_static_on_the_paywall_and_stays_static(repo, tmp_path):
    class Paywalled(FakeApi):
        def create_repo(self, repo_id, **kw):
            if kw.get("space_sdk") == "gradio":
                err = RuntimeError("402 Payment Required")
                err.response = type("R", (), {"status_code": 402})()
                raise err
            super().create_repo(repo_id, **kw)

    clone, _, _ = repo
    api = Paywalled()
    status = deploy(clone, "me/brink", True, "workflow_run", tmp_path / "space", api=api, ci_check=lambda s: True,
                    marker_reader=lambda s, a: None, space_token="hf_read", sdk="auto")  # fmt: skip
    assert "static page" in status and "secret" not in api.kinds()
    marker = json.loads((tmp_path / "space" / "DEPLOYED.json").read_text())
    assert marker["sdk"] == "static" and (tmp_path / "space" / "index.html").exists()
    again = FakeApi()
    deploy(clone, "me/brink", True, "workflow_dispatch", tmp_path / "space", api=again, ci_check=lambda s: True,
           marker_reader=lambda s, a: marker, sdk="auto")  # fmt: skip
    assert again.calls[0][2]["space_sdk"] == "static"
    unreadable = FakeApi()
    unreadable.space_info = lambda sid: type("I", (), {"sdk": "static"})()
    deploy(clone, "me/brink", True, "workflow_dispatch", tmp_path / "space", api=unreadable, ci_check=lambda s: True,
           marker_reader=lambda s, a: None, sdk="auto")  # fmt: skip
    assert unreadable.calls[0][2]["space_sdk"] == "static"
    assert deploy(clone, "me/brink", True, "schedule", tmp_path / "space", api=FakeApi(), ci_check=lambda s: True,
                  marker_reader=lambda s, a: marker, sdk="auto").startswith("skipped: nothing new")  # fmt: skip
