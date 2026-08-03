"""Contrat JSON des artefacts CNI produits à partir d'OpenCV/NumPy."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ocr_benchmark.cni_ingestion import write_cni_json
from ocr_benchmark.cni_runner import _write_results_index
from ocr_benchmark.json_utils import to_json_compatible


def test_json_normalizer_converts_numpy_scalars_arrays_and_paths(tmp_path: Path):
    value = {
        "component_id": np.int32(7),
        "score": np.float32(0.75),
        "box": np.array([1, 2, 30, 40], dtype=np.int32),
        "path": tmp_path / "crop.png",
    }

    converted = to_json_compatible(value)

    assert converted == {
        "component_id": 7,
        "score": 0.75,
        "box": [1, 2, 30, 40],
        "path": str(tmp_path / "crop.png"),
    }


def test_write_cni_json_accepts_numpy_values(tmp_path: Path):
    output = tmp_path / "preparation.json"

    write_cni_json(
        output,
        {
            "candidate_count": np.int32(3),
            "coverage": np.float32(0.5),
        },
    )

    assert json.loads(output.read_text(encoding="utf-8")) == {
        "candidate_count": 3,
        "coverage": 0.5,
    }


def test_results_checkpoint_accepts_numpy_values(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    _write_results_index(
        run_dir,
        [{"status": "success", "input_tokens": np.int32(128)}],
    )

    assert json.loads(
        (run_dir / "cni_results.json").read_text(encoding="utf-8")
    )[0]["input_tokens"] == 128
