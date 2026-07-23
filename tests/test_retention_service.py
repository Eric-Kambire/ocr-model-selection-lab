"""Tests des archives CNI anonymisées et du nettoyage local sûr."""

from ocr_benchmark.application.retention_service import (
    cleanup_cni_run,
    load_anonymized_cni_archive,
)


def _result(client_id: str) -> dict:
    return {
        "run_id": "cni-20260723-120000-deadbeef",
        "folder_client_id": client_id,
        "model": "vision-model",
        "status": "success",
        "strategy": "separate_calls",
        "label_status": "label_materialized",
        "accuracy": 0.9,
        "latency": 1.2,
        "cin_recto": "A0000000",
        "error": "path D:/sensitive/client/error",
    }


def test_cleanup_keeps_metrics_without_identifier_or_raw_values(tmp_path):
    """L'archive sert aux graphes sans conserver la CNI ou l'identité."""
    runs_root = tmp_path / "runs"
    archive_root = tmp_path / "analysis_archive"
    imports_root = tmp_path / "imports"
    run_id = "cni-20260723-120000-deadbeef"
    (runs_root / run_id).mkdir(parents=True)
    batch = imports_root / "qlickeer_api" / "batch-demo"
    client_dir = batch / "customer-actual"
    client_dir.mkdir(parents=True)
    preview_cache = runs_root / "cni_source_previews"
    preview_cache.mkdir()
    (preview_cache / "preview.png").write_bytes(b"temporary")

    report = cleanup_cni_run(
        [_result("customer-actual")],
        [{"client_dir": str(client_dir)}],
        runs_root=runs_root,
        archive_root=archive_root,
        imports_root=imports_root,
        keep_anonymized_archive=True,
        delete_detailed_run=True,
        delete_imported_sources=True,
        clear_preview_cache=True,
    )

    loaded = load_anonymized_cni_archive(report["archive_path"], archive_root)
    assert loaded[0]["folder_client_id"] == "case-001"
    assert "cin_recto" not in loaded[0]
    assert "error" not in loaded[0]
    assert not (runs_root / run_id).exists()
    assert not batch.exists()
    assert report["preview_cache_deleted"] is True
    assert not preview_cache.exists()


def test_cleanup_refuses_detailed_deletion_without_anonymized_archive(tmp_path):
    """Aucun artefact détaillé n'est perdu sans sauvegarde métrique sûre."""
    runs_root = tmp_path / "runs"
    run_id = "cni-20260723-120000-deadbeef"
    (runs_root / run_id).mkdir(parents=True)

    try:
        cleanup_cni_run(
            [_result("customer-actual")], [],
            runs_root=runs_root,
            archive_root=tmp_path / "archives",
            imports_root=tmp_path / "imports",
            keep_anonymized_archive=False,
            delete_detailed_run=True,
            delete_imported_sources=False,
            clear_preview_cache=False,
        )
    except ValueError as error:
        assert "sans archive anonymisée" in str(error)
    else:
        raise AssertionError("La suppression sans archive aurait dû être refusée.")
