"""Runner séquentiel d'extraction CNI, conçu pour une mémoire limitée.

Un adaptateur est créé pour un modèle, tous les clients passent par ce modèle,
puis il est libéré avant de charger le modèle suivant.
"""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from pathlib import Path
from typing import Any, Iterator

# Le runner orchestre les modules spécialisés ; il ne contient ni logique de
# scan de dossiers, ni règle de crop, ni définition du contrat JSON.
from .cni_images import build_vertical_cni_composite, crop_cni_from_a4
from .cni_comparison import compare_cni_extraction
from .cni_ingestion import write_cni_json
from .cni_preprocessing import prepare_cni_source, preprocess_cni_image
from .cni_schema import (
    build_cni_global_json,
    build_cni_prompt,
    build_combined_cni_prompt,
    parse_cni_json_response,
    parse_combined_cni_json_response,
)
from .domain import InferenceResult, InferenceStatus
from .registry import ModelRegistry
from .runner import BenchmarkRunner

LOGGER = logging.getLogger(__name__)


def iter_cni_benchmark(
    registry: ModelRegistry,
    model_specs: list[str],
    clients: list[dict[str, Any]],
    runs_root: Path,
    *,
    strategy: str = "separate_calls",
    dpi: int = 300,
    timeout_seconds: float | None = None,
    cpu_threads: int | None = None,
    unload_after_task: bool = True,
    fields: dict[str, list[dict[str, str]]] | None = None,
    prompt_instructions: str | None = None,
    system_prompt: str | None = None,
    preprocessing: dict[str, bool] | None = None,
) -> Iterator[dict[str, Any]]:
    """Émet des événements live et persiste un jeu d'artefacts par modèle/client.

    Paramètres :
        registry: registre qui construit les adaptateurs de modèles.
        model_specs: un ou plusieurs modèles, traités séquentiellement.
        clients: diagnostics issus de ``scan_cni_clients``.
        runs_root: répertoire racine des artefacts persistants.
        strategy: ``separate_calls`` ou ``combined_vertical``.
        dpi: résolution de rendu des PDF.
        timeout_seconds: durée maximale d'un appel modèle.

    Émet :
        des dictionnaires ``processing`` ou ``completed``. Un événement final
        contient la ligne synthèse et les chemins vers les artefacts recto,
        verso, global et sortie brute. Le label n'est pas comparé ici.

    Le modèle est libéré dans ``finally`` avant le suivant. C'est la garantie
    mémoire principale : cocher plusieurs modèles ne les garde jamais tous en
    mémoire simultanément.
    """
    if strategy not in {"separate_calls", "combined_vertical"}:
        raise ValueError("CNI strategy must be 'separate_calls' or 'combined_vertical'.")
    valid_clients = [client for client in clients if client.get("status") == "ready"]
    run_id = "cni-" + time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
    run_dir = runs_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    total = len(model_specs) * len(valid_clients)
    completed = 0
    results: list[dict[str, Any]] = []
    started_at = time.monotonic()

    LOGGER.info(
        "CNI benchmark starting | run=%s | models=%s | valid_clients=%d | strategy=%s | dpi=%d",
        run_id, model_specs, len(valid_clients), strategy, dpi,
    )
    if not valid_clients:
        yield {
            "stage": "completed",
            "run_id": run_id,
            "completed": 0,
            "total": 0,
            "result": None,
            "message": "Aucun dossier client CNI valide à traiter.",
        }
        return

    for model_spec in model_specs:
        model = None
        model_name = model_spec.split(":", 1)[-1]
        try:
            # Une erreur de chargement est produite pour chaque client afin que
            # l'interface conserve la matrice complète modèle × client.
            model = registry.create(
                model_spec,
                cpu_threads=cpu_threads,
                unload_after_task=unload_after_task,
                timeout_seconds=timeout_seconds,
            )
            model_name = model.model_name
        except Exception as exc:
            LOGGER.exception("CNI model initialization failed | spec=%s", model_spec)
            for client in valid_clients:
                completed += 1
                result = _failed_client_result(run_id, model_name, client, strategy, f"model_load_failed: {type(exc).__name__}: {exc}")
                results.append(result)
                _write_results_index(run_dir, results)
                yield _completed_event(run_id, completed, total, result, started_at)
            continue

        try:
            for client in valid_clients:
                client_dir = run_dir / _safe_name(model_name) / _safe_name(str(client["folder_client_id"]))
                pair_started_at = time.monotonic()
                preparation_started_at = time.monotonic()
                try:
                    prepared = prepare_cni_client_images(client, client_dir, dpi, preprocessing=preprocessing)
                    preparation_finished_at = time.monotonic()
                except Exception as exc:
                    completed += 1
                    result = _failed_client_result(run_id, model_name, client, strategy, f"prepare_failed: {type(exc).__name__}: {exc}")
                    result["preprocessing_seconds"] = (
                        time.monotonic() - preparation_started_at
                    )
                    result["end_to_end_seconds"] = (
                        time.monotonic() - pair_started_at
                    )
                    results.append(result)
                    _write_results_index(run_dir, results)
                    LOGGER.exception("CNI document preparation failed | client=%s", client["folder_client_id"])
                    yield _completed_event(run_id, completed, total, result, started_at)
                    continue

                # L'événement est émis avant l'appel modèle : l'interface peut
                # afficher l'image en cours pendant que l'inférence travaille.
                yield _processing_event(run_id, completed, total, model_name, client, "recto" if strategy == "separate_calls" else "recto_verso", prepared, started_at)
                result = _extract_one_cni_client(
                    model,
                    run_id,
                    model_name,
                    client,
                    prepared,
                    strategy=strategy,
                    timeout_seconds=timeout_seconds,
                    fields=fields,
                    prompt_instructions=prompt_instructions,
                    system_prompt=system_prompt,
                )
                result["preprocessing_seconds"] = (
                    preparation_finished_at - preparation_started_at
                )
                result["end_to_end_seconds"] = (
                    time.monotonic() - pair_started_at
                )
                completed += 1
                results.append(result)
                _write_results_index(run_dir, results)
                yield _completed_event(run_id, completed, total, result, started_at)
        finally:
            close = getattr(model, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    LOGGER.exception("CNI model cleanup failed | model=%s", model_name)
            LOGGER.info("CNI model released | model=%s", model_name)


def prepare_cni_client_images(
    client: dict[str, Any],
    artefacts_dir: Path,
    dpi: int,
    *,
    preprocessing: dict[str, bool] | None = None,
) -> dict[str, Any]:
    """Normalise, détecte la carte puis prétraite seulement un crop fiable.

    Le point important est l'ordre : la détection travaille sur la source
    normalisée. Si elle n'est pas certaine, le modèle reçoit cette source entière
    et non une page tournée, assombrie ou coupée par erreur.
    """
    # Tout est écrit dans le run, jamais à côté des PDF utilisateur : les
    # sources restent intactes et chaque benchmark peut être rejoué/analyse.
    recto_page = artefacts_dir / "recto_page.png"
    verso_page = artefacts_dir / "verso_page.png"
    options = {key: bool((preprocessing or {}).get(key, False)) for key in ("deskew", "perspective", "contrast", "denoise")}
    recto_source = Path(str(client.get("recto_source") or client["recto_pdf"]))
    verso_source = Path(str(client.get("verso_source") or client["verso_pdf"]))
    recto_render = prepare_cni_source(recto_source, recto_page, dpi)
    verso_render = prepare_cni_source(verso_source, verso_page, dpi)
    recto_crop = crop_cni_from_a4(recto_page, artefacts_dir / "crop_recto.png", debug_path=artefacts_dir / "crop_recto_debug.png")
    verso_crop = crop_cni_from_a4(verso_page, artefacts_dir / "crop_verso.png", debug_path=artefacts_dir / "crop_verso_debug.png")
    recto_preprocessed = _preprocess_only_reliable_crop(recto_crop, artefacts_dir / "recto_preprocessed.png", options)
    verso_preprocessed = _preprocess_only_reliable_crop(verso_crop, artefacts_dir / "verso_preprocessed.png", options)
    recto_model_image = recto_preprocessed["image_path"]
    verso_model_image = verso_preprocessed["image_path"]
    # Même en mode séparé, conserver le composite facilite une inspection
    # humaine ultérieure et un éventuel nouvel essai en mode combiné.
    combined_path = build_vertical_cni_composite(
        Path(recto_model_image), Path(verso_model_image), artefacts_dir / "recto_verso_composite.png"
    )
    prepared = {
        "recto_source": str(recto_source),
        "verso_source": str(verso_source),
        "recto_page": recto_render,
        "verso_page": verso_render,
        "preprocessing_options": options,
        "recto_preprocessed": recto_preprocessed,
        "verso_preprocessed": verso_preprocessed,
        "recto_crop": recto_crop,
        "verso_crop": verso_crop,
        "recto_model_image": recto_model_image,
        "verso_model_image": verso_model_image,
        "combined_image": combined_path,
    }
    write_cni_json(artefacts_dir / "preparation.json", prepared)
    LOGGER.info(
        "CNI client prepared | client=%s | recto_crop=%s | verso_crop=%s",
        client["folder_client_id"], recto_crop["crop_status"], verso_crop["crop_status"],
    )
    return prepared


def _preprocess_only_reliable_crop(
    crop: dict[str, Any],
    output_path: Path,
    options: dict[str, bool],
) -> dict[str, Any]:
    """Préserve strictement la source si le crop automatique est incertain."""
    if crop.get("source_sent_unchanged"):
        return {
            "status": "skipped_crop_uncertain_original_preserved",
            "image_path": crop["image_path"],
            "operations": [],
        }
    return preprocess_cni_image(Path(crop["image_path"]), output_path, **options)


def _extract_one_cni_client(
    model: Any,
    run_id: str,
    model_name: str,
    client: dict[str, Any],
    prepared: dict[str, Any],
    *,
    strategy: str,
    timeout_seconds: float | None,
    fields: dict[str, list[dict[str, str]]] | None,
    prompt_instructions: str | None,
    system_prompt: str | None,
) -> dict[str, Any]:
    """Exécute une stratégie et écrit les JSON recto, verso et global."""
    artefacts_dir = Path(prepared["recto_crop"]["image_path"]).parent
    if strategy == "combined_vertical":
        # Un seul appel reçoit le composite, mais le parsing produit toujours
        # deux dictionnaires afin de préserver le contrat des artefacts.
        inference = _perform_cni_call(
            model,
            Path(prepared["combined_image"]),
            build_combined_cni_prompt(fields, instructions=prompt_instructions),
            timeout_seconds,
            artefacts_dir,
            "combined",
            system_prompt,
        )
        recto, verso, parse_error = parse_combined_cni_json_response(inference.text, fields)
        recto_inference = verso_inference = inference
        recto_parse_error = verso_parse_error = parse_error
    else:
        # Deux appels indépendants sont le mode de diagnostic recommandé : on
        # sait immédiatement quelle face a posé problème.
        recto_inference = _perform_cni_call(
            model,
            Path(prepared["recto_model_image"]),
            build_cni_prompt("recto", fields, instructions=prompt_instructions),
            timeout_seconds,
            artefacts_dir,
            "recto",
            system_prompt,
        )
        verso_inference = _perform_cni_call(
            model,
            Path(prepared["verso_model_image"]),
            build_cni_prompt("verso", fields, instructions=prompt_instructions),
            timeout_seconds,
            artefacts_dir,
            "verso",
            system_prompt,
        )
        recto, recto_parse_error = parse_cni_json_response(recto_inference.text, "recto", fields)
        verso, verso_parse_error = parse_cni_json_response(verso_inference.text, "verso", fields)

    # Écrire d'abord chaque face : le JSON global peut ensuite signaler une
    # incohérence sans perdre les lectures originales.
    recto_payload = _side_payload("recto", recto, recto_inference, recto_parse_error, prepared["recto_crop"])
    verso_payload = _side_payload("verso", verso, verso_inference, verso_parse_error, prepared["verso_crop"])
    write_cni_json(artefacts_dir / "recto.extraction.json", recto_payload)
    write_cni_json(artefacts_dir / "verso.extraction.json", verso_payload)
    global_payload = build_cni_global_json(client, recto, verso)
    label_payload = _read_json_mapping(client.get("label_path"))
    comparison = compare_cni_extraction(label_payload, global_payload)
    global_payload.update(
        {
            "run_id": run_id,
            "model": model_name,
            "strategy": strategy,
            "recto_status": recto_payload["status"],
            "verso_status": verso_payload["status"],
            "comparison": comparison,
        }
    )
    write_cni_json(artefacts_dir / "global.extraction.json", global_payload)
    status = _overall_status(recto_payload["status"], verso_payload["status"], recto_parse_error, verso_parse_error)
    if strategy == "combined_vertical":
        # Le recto et le verso proviennent du même appel. Les additionner
        # doublerait artificiellement temps et tokens dans la vue de résultats.
        total_latency = recto_inference.latency_seconds
        output_tokens = recto_inference.output_tokens
        input_tokens = recto_inference.input_tokens
        call_metrics = [
            _inference_metric_row(
                "recto_verso",
                recto_payload["status"],
                recto_inference,
            )
        ]
    else:
        total_latency = (
            recto_inference.latency_seconds + verso_inference.latency_seconds
        )
        output_tokens = _sum_optional(
            recto_inference.output_tokens,
            verso_inference.output_tokens,
        )
        input_tokens = _sum_optional(
            recto_inference.input_tokens,
            verso_inference.input_tokens,
        )
        call_metrics = [
            _inference_metric_row("recto", recto_payload["status"], recto_inference),
            _inference_metric_row("verso", verso_payload["status"], verso_inference),
        ]
    return {
        "run_id": run_id,
        "model": model_name,
        "folder_client_id": client["folder_client_id"],
        "status": status,
        "strategy": strategy,
        "label_status": client.get("label_status"),
        "label_path": client.get("label_path"),
        "accuracy": comparison["accuracy"],
        "text_similarity": comparison["text_similarity"],
        "cer": comparison["cer"],
        "wer": comparison["wer"],
        "score_status": comparison["score_status"],
        "field_comparison": comparison,
        "recto_status": recto_payload["status"],
        "verso_status": verso_payload["status"],
        "cin_recto": global_payload["cin_recto"],
        "cin_verso": global_payload["cin_verso"],
        "cin_fusionne": global_payload["cin_fusionne"],
        "cin_coherent": global_payload["cin_coherent"],
        "date_validite_coherente": global_payload["date_validite_coherente"],
        # ``latency`` est conservé pour les anciens consommateurs. Le nouveau
        # nom explicite qu'il ne contient ni crop ni préparation de document.
        "latency": total_latency,
        "inference_seconds": total_latency,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "tokens_per_second": (output_tokens / total_latency if output_tokens is not None and total_latency else None),
        "call_metrics": call_metrics,
        "recto_json_path": str(artefacts_dir / "recto.extraction.json"),
        "verso_json_path": str(artefacts_dir / "verso.extraction.json"),
        "global_json_path": str(artefacts_dir / "global.extraction.json"),
        "recto_raw_output_path": str(artefacts_dir / "raw_recto_output.txt"),
        "verso_raw_output_path": str(artefacts_dir / "raw_verso_output.txt"),
        "combined_raw_output_path": str(artefacts_dir / "raw_combined_output.txt"),
        # Ces chemins sont ceux effectivement remis au modèle. En fallback ils
        # pointent vers recto_page/verso_page, donc la source complète intacte.
        "recto_image_path": prepared["recto_model_image"],
        "verso_image_path": prepared["verso_model_image"],
        "combined_image_path": prepared["combined_image"],
        "recto_prompt_path": str(artefacts_dir / "prompt_recto.txt"),
        "verso_prompt_path": str(artefacts_dir / "prompt_verso.txt"),
        "combined_prompt_path": str(artefacts_dir / "prompt_combined.txt"),
        "error": _join_errors(recto_payload.get("error"), verso_payload.get("error")),
    }


def _perform_cni_call(
    model: Any,
    image_path: Path,
    prompt: str,
    timeout_seconds: float | None,
    artefacts_dir: Path,
    side: str,
    system_prompt: str | None,
) -> InferenceResult:
    """Appelle une image et conserve sortie brute ou sortie tardive."""
    def save_late(raw: Any | None, error: str | None) -> None:
        # Une réponse arrivée après timeout est précieuse pour le débogage, mais
        # elle ne doit jamais transformer rétroactivement un timeout en succès.
        try:
            value = raw if isinstance(raw, dict) else {"raw": str(raw) if raw is not None else None}
            value.update({"timing": "late_after_timeout", "error": error})
            write_cni_json(artefacts_dir / f"late_{side}_output.json", value)
        except Exception:
            LOGGER.exception("Unable to persist late CNI response | side=%s", side)

    # Le runner générique centralise le timeout fournisseur et le mécanisme de
    # réponse tardive afin que CNI et benchmark général se comportent pareil.
    # Le prompt exact reste dans le run, même si l'appel échoue. C'est le
    # point de départ indispensable pour comprendre une réponse invalide.
    (artefacts_dir / f"prompt_{side}.txt").write_text(
        "--- SYSTEM ---\n" + (system_prompt or "") + "\n\n--- USER ---\n" + prompt,
        encoding="utf-8",
    )
    raw = BenchmarkRunner._perform_with_timeout(
        model,
        str(image_path),
        timeout_seconds,
        prompt=prompt,
        system_prompt=system_prompt,
        late_result=save_late,
    )
    inference = raw if isinstance(raw, InferenceResult) else InferenceResult.from_legacy_dict(raw)
    (artefacts_dir / f"raw_{side}_output.txt").write_text(
        inference.raw_response or inference.text or "", encoding="utf-8"
    )
    return inference


def _side_payload(side: str, fields: dict[str, str | None], inference: InferenceResult, parse_error: str | None, crop: dict[str, Any]) -> dict[str, Any]:
    """Construit le JSON d'une face avec données modèle et métadonnées de crop."""
    status = inference.status.value if parse_error is None else "invalid_json"
    return {
        "side": side,
        "status": status,
        "fields": fields,
        "parse_error": parse_error,
        "error": inference.error,
        # La réponse brute reste visible même lorsque le parsing JSON échoue ou
        # que le fournisseur retourne un statut failed/timeout.
        "text": inference.text,
        "raw_response": inference.raw_response,
        "reasoning": inference.reasoning,
        "latency": inference.latency_seconds,
        "input_tokens": inference.input_tokens,
        "output_tokens": inference.output_tokens,
        "tokens_per_second": inference.tokens_per_second,
        "crop": crop,
    }


def _inference_metric_row(
    scope: str,
    status: str,
    inference: InferenceResult,
) -> dict[str, Any]:
    """Sérialise les mesures d'un appel réel, jamais d'une face logique dupliquée."""
    return {
        "scope": scope,
        "status": status,
        "latency_seconds": inference.latency_seconds,
        "input_tokens": inference.input_tokens,
        "output_tokens": inference.output_tokens,
        "tokens_per_second": inference.tokens_per_second,
    }


def _overall_status(recto_status: str, verso_status: str, recto_parse_error: str | None, verso_parse_error: str | None) -> str:
    """Réduit les deux statuts en un statut global, timeout prioritaire."""
    if recto_status == "timeout" or verso_status == "timeout":
        return "timeout"
    if recto_status != "success" or verso_status != "success":
        return "failed"
    if recto_parse_error or verso_parse_error:
        return "invalid_json"
    return "success"


def _processing_event(run_id: str, completed: int, total: int, model_name: str, client: dict[str, Any], side: str, prepared: dict[str, Any], started_at: float) -> dict[str, Any]:
    """Construit l'événement léger consommé par la vue live Gradio."""
    image = prepared["combined_image"] if side == "recto_verso" else prepared[f"{side}_model_image"]
    return {
        "stage": "processing",
        "run_id": run_id,
        "completed": completed,
        "total": total,
        "model": model_name,
        "folder_client_id": client["folder_client_id"],
        "side": side,
        "image_path": image,
        "elapsed_seconds": time.monotonic() - started_at,
        "result": None,
    }


def _completed_event(run_id: str, completed: int, total: int, result: dict[str, Any], started_at: float) -> dict[str, Any]:
    """Construit l'événement de fin après le checkpoint du résultat."""
    return {
        "stage": "completed",
        "run_id": run_id,
        "completed": completed,
        "total": total,
        "model": result["model"],
        "folder_client_id": result["folder_client_id"],
        "elapsed_seconds": time.monotonic() - started_at,
        "result": result,
    }


def _failed_client_result(run_id: str, model_name: str, client: dict[str, Any], strategy: str, error: str) -> dict[str, Any]:
    """Retourne une ligne d'échec normale pour ne jamais arrêter tout le run."""
    return {
        "run_id": run_id,
        "model": model_name,
        "folder_client_id": client["folder_client_id"],
        "status": "failed",
        "strategy": strategy,
        "label_status": client.get("label_status"),
        "label_path": client.get("label_path"),
        "accuracy": None,
        "text_similarity": None,
        "cer": None,
        "wer": None,
        "score_status": "not_scored_label_mapping_pending",
        "latency": 0.0,
        "inference_seconds": 0.0,
        "preprocessing_seconds": None,
        "end_to_end_seconds": None,
        "input_tokens": None,
        "output_tokens": None,
        "tokens_per_second": None,
        "call_metrics": [],
        "error": error,
    }


def _read_json_mapping(path_value: Any) -> dict[str, Any] | None:
    """Lit un label local sans rendre l'extraction dépendante de son format."""
    if not path_value:
        return None
    try:
        value = json.loads(Path(str(path_value)).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _write_results_index(run_dir: Path, results: list[dict[str, Any]]) -> None:
    """Checkpoint atomiquement la liste qui grandit après chaque client."""
    temporary = run_dir / "cni_results.json.tmp"
    temporary.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(run_dir / "cni_results.json")


def _safe_name(value: str) -> str:
    """Transforme un nom modèle/client en segment de dossier portable."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._") or "unknown"


def _sum_optional(first: int | None, second: int | None) -> int | None:
    """Additionne les compteurs seulement si le fournisseur les expose."""
    values = [value for value in (first, second) if value is not None]
    return sum(values) if values else None


def _join_errors(*errors: str | None) -> str | None:
    """Garde les deux erreurs de face visibles dans une cellule compacte."""
    values = [error for error in errors if error]
    return " | ".join(values) if values else None
