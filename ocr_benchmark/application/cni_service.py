"""Cas d'usage CNI réutilisables par UI, CLI et futur worker fraude."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

from ..cni_ingestion import import_cni_zip, materialize_cni_labels, scan_cni_clients
from ..cni_qlicker import QlickerImportConfig, import_qlicker_clients
from ..cni_runner import iter_cni_benchmark
from ..registry import ModelRegistry, build_default_registry


def scan_cni_documents(
    clients_root: Path,
    labels_root: Path | None,
    *,
    recto_suffix: str,
    verso_suffix: str,
) -> list[dict[str, Any]]:
    """Scanne les paires CNI et matérialise les labels JSON voisins des clients."""
    return materialize_cni_labels(
        scan_cni_clients(
            clients_root,
            labels_root,
            recto_suffix=recto_suffix,
            verso_suffix=verso_suffix,
        )
    )


def import_cni_archive(zip_path: Path, imports_root: Path) -> dict[str, Any]:
    """Importe une archive CNI en conservant les protections du module ingestion."""
    return import_cni_zip(zip_path, imports_root)


def import_cni_from_remote(
    client_ids: list[str],
    destination_root: Path,
    config: QlickerImportConfig,
) -> list[dict[str, Any]]:
    """Importe les CNI distantes via le contrat réseau explicitement configuré."""
    return import_qlicker_clients(client_ids, destination_root, config)


def iter_cni_extraction(
    model_specs: list[str],
    clients: list[dict[str, Any]],
    runs_root: Path,
    **options: Any,
) -> Iterator[dict[str, Any]]:
    """Délègue l'extraction CNI au runner avec un registre de modèles injecté."""
    registry: ModelRegistry = options.pop("registry", None) or build_default_registry()
    yield from iter_cni_benchmark(registry, model_specs, clients, runs_root, **options)
