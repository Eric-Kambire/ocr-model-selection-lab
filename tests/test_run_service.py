"""Tests de la conservation et de la suppression sûre des runs locaux."""

import json
import os
import time
from pathlib import Path

from ocr_benchmark.application.run_service import list_run_ids, load_run_results, purge_expired_runs


def test_run_service_lists_loads_and_only_purges_valid_run_directories(tmp_path: Path):
    """La rétention ne doit jamais supprimer un dossier qui n'est pas un run."""
    runs_root = tmp_path / "runs"
    expired = runs_root / "20260701-120000-deadbeef"
    expired.mkdir(parents=True)
    (expired / "results.json").write_text(json.dumps([{"model": "mock"}]), encoding="utf-8")
    preserved = runs_root / "notes-administrateur"
    preserved.mkdir()
    (preserved / "important.txt").write_text("ne pas supprimer", encoding="utf-8")

    old_timestamp = time.time() - 3 * 86_400
    os.utime(expired, (old_timestamp, old_timestamp))

    assert list_run_ids(runs_root) == [expired.name]
    assert load_run_results(runs_root, expired.name) == [{"model": "mock"}]
    assert purge_expired_runs(runs_root, retention_days=1) == [expired.name]
    assert not expired.exists()
    assert (preserved / "important.txt").is_file()
