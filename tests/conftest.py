import os
import sys
import tempfile
from pathlib import Path

# Personal data goes to a throwaway folder; must be set before mindreader.engine / silentdict.paths load.
_tmp = tempfile.mkdtemp(prefix="mindreader-test-")
os.environ["MINDREADER_DATA"] = _tmp
os.environ["SILENTDICT_DATA"] = _tmp
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest


@pytest.fixture
def fresh_data(tmp_path, monkeypatch):
    """An empty personal dataset for one test (the engine reads these constants at call time)."""
    from silentdict import paths
    monkeypatch.setattr(paths, "DATA", tmp_path)
    monkeypatch.setattr(paths, "PERSONAL", tmp_path / "personal")
    monkeypatch.setattr(paths, "CLIPS", tmp_path / "personal" / "clips")
    monkeypatch.setattr(paths, "MANIFEST", tmp_path / "personal" / "manifest.jsonl")
    monkeypatch.setattr(paths, "PERSONAL_CONTEXT", tmp_path / "personal" / "context.txt")
    monkeypatch.setattr(paths, "REPORTS", tmp_path / "personal" / "reports")
    monkeypatch.setattr(paths, "HISTORY", tmp_path / "history")
    return tmp_path
