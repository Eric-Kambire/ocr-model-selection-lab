"""Archivage anonymisé et nettoyage sûr des analyses CNI locales."""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from .run_service import list_run_ids


def anonymize_cni_results(results: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Réduit les résultats CNI aux mesures sans identité ni contenu OCR.

    La correspondance ``client réel → case-XXX`` n'est jamais écrite. Elle est
    créée uniquement le temps de fabriquer l'archive, ce qui conserve les
    comparaisons entre modèles sans pouvoir réidentifier un client.
    """
    aliases: dict[str, str] = {}
    anonymous: list[dict[str, Any]] = []
    for item in results:
        raw_id = str(item.get("folder_client_id") or "unknown")
        case_id = aliases.setdefault(raw_id, f"case-{len(aliases) + 1:03d}")
        anonymous.append(
            {
                "case_id": case_id,
                "archive_anonymized": True,
                # Le champ conserve ce nom afin que les tableaux et graphes
                # CNI existants puissent charger l'archive sans adaptation.
                "folder_client_id": case_id,
                "model": item.get("model"),
                "status": item.get("status"),
                "strategy": item.get("strategy"),
                "label_status": item.get("label_status"),
                "accuracy": item.get("accuracy"),
                "score_status": item.get("score_status"),
                "recto_status": item.get("recto_status"),
                "verso_status": item.get("verso_status"),
                "cin_coherent": item.get("cin_coherent"),
                "date_validite_coherente": item.get("date_validite_coherente"),
                "latency": item.get("latency"),
                "input_tokens": item.get("input_tokens"),
                "output_tokens": item.get("output_tokens"),
                "tokens_per_second": item.get("tokens_per_second"),
                # Les messages complets peuvent contenir un chemin ou une
                # valeur extraite : seule la présence d'une erreur est gardée.
                "has_error": bool(item.get("error")),
            }
        )
    return anonymous


def write_anonymized_cni_archive(
    results: Sequence[Mapping[str, Any]],
    archive_root: Path,
) -> Path:
    """Écrit une archive JSON atomique, chargable après suppression du run."""
    if not results:
        raise ValueError("Aucun résultat CNI à anonymiser.")
    run_ids = {str(item.get("run_id") or "") for item in results}
    if len(run_ids) != 1 or not next(iter(run_ids), ""):
        raise ValueError("Les résultats doivent provenir d'un seul run CNI.")
    run_id = next(iter(run_ids))
    archive_root.mkdir(parents=True, exist_ok=True)
    path = archive_root / f"{run_id}-anonymized.json"
    temporary = path.with_suffix(path.suffix + ".tmp")
    value = {
        "schema_version": 1,
        "archive_kind": "cni_metrics_anonymized",
        "source_run": run_id,
        "created_at_epoch": time.time(),
        "results": anonymize_cni_results(results),
    }
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)
    return path


def list_anonymized_cni_archives(archive_root: Path) -> list[tuple[str, str]]:
    """Liste les archives anonymisées les plus récentes en premier."""
    if not archive_root.is_dir():
        return []
    paths = sorted(archive_root.glob("cni-*-anonymized.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    return [(path.stem, str(path)) for path in paths]


def load_anonymized_cni_archive(path_value: str | Path, archive_root: Path) -> list[dict[str, Any]]:
    """Charge une archive sans autoriser un chemin hors du répertoire dédié."""
    root = archive_root.resolve()
    path = Path(path_value).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError("Archive anonymisée hors du répertoire autorisé.") from exc
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("archive_kind") != "cni_metrics_anonymized":
        raise ValueError("Archive anonymisée invalide.")
    results = value.get("results")
    if not isinstance(results, list) or not all(isinstance(item, dict) for item in results):
        raise ValueError("Résultats anonymisés absents ou invalides.")
    return results


def cleanup_cni_run(
    results: Sequence[Mapping[str, Any]],
    clients: Sequence[Mapping[str, Any]],
    *,
    runs_root: Path,
    archive_root: Path,
    imports_root: Path,
    keep_anonymized_archive: bool,
    delete_detailed_run: bool,
    delete_imported_sources: bool,
    clear_preview_cache: bool,
) -> dict[str, Any]:
    """Archive puis supprime, dans cet ordre, les artefacts sensibles choisis.

    Les dossiers client locaux externes ne sont jamais supprimés. Seuls les
    lots situés sous ``imports_root/qlickeer_api/batch-*`` sont éligibles.
    """
    if not results:
        raise ValueError("Aucun résultat CNI disponible pour le nettoyage.")
    run_ids = {str(item.get("run_id") or "") for item in results}
    if len(run_ids) != 1 or not next(iter(run_ids), ""):
        raise ValueError("Le nettoyage exige les résultats d'un seul run CNI terminé.")
    run_id = next(iter(run_ids))
    if run_id not in list_run_ids(runs_root):
        raise ValueError("Run CNI introuvable ou déjà supprimé.")

    archive_path: Path | None = None
    if keep_anonymized_archive:
        archive_path = write_anonymized_cni_archive(results, archive_root)
    if delete_detailed_run and archive_path is None:
        raise ValueError("Refus de supprimer le run détaillé sans archive anonymisée.")

    deleted_batches: list[str] = []
    if delete_imported_sources:
        for batch in _import_batches_from_clients(clients, imports_root):
            shutil.rmtree(batch)
            deleted_batches.append(str(batch))

    run_dir = runs_root / run_id
    if delete_detailed_run:
        shutil.rmtree(run_dir)

    # Les aperçus sont des copies d'images de documents. Ils ne sont jamais
    # nécessaires pour relire l'archive anonymisée : le cache est donc effacé
    # uniquement sur demande explicite.
    preview_cache_deleted = False
    preview_cache = runs_root / "cni_source_previews"
    if clear_preview_cache and preview_cache.is_dir():
        shutil.rmtree(preview_cache)
        preview_cache_deleted = True

    return {
        "run_id": run_id,
        "archive_path": str(archive_path) if archive_path else None,
        "detailed_run_deleted": delete_detailed_run,
        "deleted_import_batches": deleted_batches,
        "preview_cache_deleted": preview_cache_deleted,
    }


def _import_batches_from_clients(clients: Sequence[Mapping[str, Any]], imports_root: Path) -> list[Path]:
    """Déduit uniquement des lots QlickEER sûrs depuis les diagnostics CNI."""
    expected_root = (imports_root / "qlickeer_api").resolve()
    batches: set[Path] = set()
    for client in clients:
        directory_value = client.get("client_dir")
        if not directory_value:
            continue
        client_dir = Path(str(directory_value)).resolve()
        batch = client_dir.parent
        try:
            batch.relative_to(expected_root)
        except ValueError:
            continue
        if batch.parent == expected_root and batch.name.startswith("batch-") and batch.is_dir():
            batches.add(batch)
    return sorted(batches)
