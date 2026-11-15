"""brink.io helpers that the experiments rely on."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from brink.io import ERROR_FILE, record_failure


def test_a_crash_leaves_its_traceback_in_the_results(tmp_path):
    for msg in ("first", "second"):
        with pytest.raises(RuntimeError), record_failure(tmp_path / "stage"):
            raise RuntimeError(f"CUDA out of memory ({msg})")
    text = (tmp_path / "stage" / ERROR_FILE).read_text()
    assert "Traceback" in text and text.index("(first)") < text.index("(second)")
    with record_failure(tmp_path / "ok"):
        pass
    assert not (tmp_path / "ok").exists()
