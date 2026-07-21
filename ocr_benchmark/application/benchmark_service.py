"""Cas d'usage du benchmark OCR générique.

Ce module est la frontière entre l'interface (Gradio/CLI) et le moteur de
benchmark. Les entrées/sorties sont des valeurs Python simples, ce qui permet
de le réutiliser sans importer Gradio.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import Any

import pandas as pd

from ..domain import BenchmarkCase
from ..registry import ModelRegistry, build_default_registry
from ..reporting import save_run
from ..runner import BenchmarkRunner, RunnerProgress, summarize_results

LOGGER = logging.getLogger(__name__)


def load_dataset_catalog(
    project_root: Path,
    catalog_path: Path,
    *,
    ensure_catalog: Callable[[], None] | None = None,
) -> list[dict[str, Any]]:
    """Charge et valide le catalogue OCR sans dépendre de l'interface.

    Args:
        project_root: racine contre laquelle les chemins relatifs sont résolus.
        catalog_path: fichier ``dataset.json`` à lire.
        ensure_catalog: générateur optionnel appelé uniquement si le catalogue
            n'existe pas encore.

    Returns:
        Les entrées validées, avec des séparateurs normalisés pour Windows et
        Linux.
    """
    if not catalog_path.exists():
        if ensure_catalog is None:
            raise FileNotFoundError(f"Catalogue dataset absent : {catalog_path}")
        ensure_catalog()
    with catalog_path.open("r", encoding="utf-8") as stream:
        data = json.load(stream)
    if not isinstance(data, list):
        raise ValueError("dataset.json doit contenir une liste JSON.")

    required = {"image_path", "ground_truth", "category"}
    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Entrée dataset invalide à l'index {index}.")
        missing = required - item.keys()
        if missing:
            raise ValueError(f"Entrée dataset {index} incomplète : {sorted(missing)}")
        item["image_path"] = str(item["image_path"]).replace("\\", "/")
        image_path = project_root / Path(item["image_path"])
        if not image_path.is_file():
            raise FileNotFoundError(f"Image dataset introuvable : {image_path}")
    return data


def list_ollama_models() -> list[str]:
    """Retourne les modèles Ollama visibles, sans faire échouer l'application."""
    try:
        import ollama

        response = ollama.list()
        models = response.get("models", []) if isinstance(response, dict) else response.models
        names = [
            model.get("model") or model.get("name") if isinstance(model, dict) else getattr(model, "model", None)
            for model in models
        ]
        installed = [str(name) for name in names if name]
        LOGGER.info("Ollama models detected | count=%d | models=%s", len(installed), installed)
        return installed
    except Exception as exc:
        LOGGER.warning("Unable to list Ollama models | error=%s", exc, exc_info=True)
        return []


def select_dataset_category(dataset: Iterable[dict[str, Any]], category: str) -> list[dict[str, Any]]:
    """Filtre un catalogue selon une catégorie, ou retourne toutes les entrées."""
    items = list(dataset)
    return items if category == "All" else [item for item in items if item.get("category") == category]


def run_benchmark(
    selected_models: list[str],
    dataset: Iterable[dict[str, Any]],
    runs_root: Path,
    *,
    eval_mode: str = "Standard",
    mock_noise: float = 0.05,
    cpu_threads: int | None = None,
    unload_after_task: bool = True,
    registry: ModelRegistry | None = None,
) -> tuple[pd.DataFrame, list[dict[str, Any]], str]:
    """Exécute et persiste un benchmark complet depuis un cas d'usage unique."""
    items = list(dataset)
    if not items:
        return pd.DataFrame(), [], ""
    cases = [BenchmarkCase.from_dict(item) for item in items]
    runner = BenchmarkRunner(registry or build_default_registry())
    run_id, results = runner.run(
        selected_models,
        cases,
        eval_mode=eval_mode,
        mock_noise=mock_noise,
        cpu_threads=cpu_threads,
        unload_after_task=unload_after_task,
    )
    summary = summarize_results(results)
    save_run(run_id, summary, results, runs_root)
    return summary, results, run_id


def iter_benchmark(
    selected_models: list[str],
    selected_records: Iterable[dict[str, Any]],
    *,
    eval_mode: str,
    mock_noise: float,
    timeout_seconds: float | None,
    max_errors: int | None,
    model_prompt: str | None,
    cpu_threads: int | None,
    unload_after_task: bool,
    trace: Callable[[dict[str, Any]], None] | None = None,
    registry: ModelRegistry | None = None,
) -> Iterator[RunnerProgress]:
    """Expose les événements de progression sans connaître la présentation UI."""
    cases = [BenchmarkCase.from_dict(item) for item in selected_records]
    runner = BenchmarkRunner(registry or build_default_registry())
    yield from runner.iter_run(
        selected_models,
        cases,
        eval_mode=eval_mode,
        mock_noise=mock_noise,
        timeout_seconds=timeout_seconds,
        max_errors=max_errors,
        model_prompt=model_prompt,
        cpu_threads=cpu_threads,
        unload_after_task=unload_after_task,
        trace=trace,
    )
