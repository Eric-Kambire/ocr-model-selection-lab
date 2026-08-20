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
from typing import Any, Generator, Iterator, Mapping

# Le runner orchestre les modules spécialisés ; il ne contient ni logique de
# scan de dossiers, ni règle de crop, ni définition du contrat JSON.
from .cni_crop_service import (
    DEFAULT_SMART_CROP_MARGIN,
    DEFAULT_SMART_CROP_MIN_SCORE,
    SMART_CROP_V4,
    crop_cni_for_benchmark,
)
from .cni_images import build_vertical_cni_composite
from .cni_comparison import compare_cni_extraction
from .cni_ingestion import write_cni_json
from .cni_preprocessing import prepare_cni_source, preprocess_cni_image
from .cni_schema import (
    build_cni_global_json,
    build_cni_output_schema,
    build_cni_face_hint,
    build_cni_prompt,
    build_combined_cni_prompt,
    parse_cni_json_response,
    parse_combined_cni_json_response,
)
from .cni_two_stage import (
    build_llm_structuring_prompt,
    build_vlm_transcription_prompt,
    two_stage_schema,
)
from .domain import InferenceResult, InferenceStatus
from .json_utils import dumps_json
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
    ignore_environment_proxy: bool = False,
    fields: dict[str, list[dict[str, str]]] | None = None,
    prompt_instructions: str | None = None,
    prompt_scope_mode: str = "side_specific",
    prompt_delivery_mode: str = "application_prompt",
    ollama_thinking_mode: str = "disabled",
    system_prompt: str | None = None,
    output_format_mode: str = "schema",
    model_output_modes: Mapping[str, str] | None = None,
    preprocessing: dict[str, Any] | None = None,
    pipeline_mode: str = "direct_vlm",
    llm_model_spec: str | None = None,
    vlm_transcription_instructions: str | None = None,
    llm_system_prompt: str | None = None,
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
    if pipeline_mode not in {"direct_vlm", "vlm_llm"}:
        raise ValueError("CNI pipeline mode must be 'direct_vlm' or 'vlm_llm'.")
    if strategy not in {"separate_calls", "combined_vertical"}:
        raise ValueError("CNI strategy must be 'separate_calls' or 'combined_vertical'.")
    if prompt_scope_mode not in {"side_specific", "full_rules"}:
        raise ValueError(
            "CNI prompt scope must be 'side_specific' or 'full_rules'."
        )
    if prompt_delivery_mode not in {
        "application_prompt",
        "image_only",
        "image_with_side_hint",
    }:
        raise ValueError(
            "CNI prompt delivery must be 'application_prompt', 'image_only' "
            "or 'image_with_side_hint'."
        )
    if ollama_thinking_mode not in {"disabled", "automatic", "enabled"}:
        raise ValueError(
            "Ollama thinking mode must be 'disabled', 'automatic' or 'enabled'."
        )
    valid_clients = [client for client in clients if client.get("status") == "ready"]
    if pipeline_mode == "vlm_llm":
        if len(model_specs) != 1:
            raise ValueError(
                "Le pipeline VLM + LLM exige exactement un VLM sélectionné."
            )
        if not str(llm_model_spec or "").strip():
            raise ValueError("Le pipeline VLM + LLM exige un modèle LLM.")
        yield from _iter_cni_two_stage_benchmark(
            registry,
            model_specs[0],
            str(llm_model_spec),
            valid_clients,
            runs_root,
            strategy=strategy,
            dpi=dpi,
            timeout_seconds=timeout_seconds,
            cpu_threads=cpu_threads,
            unload_after_task=unload_after_task,
            ignore_environment_proxy=ignore_environment_proxy,
            fields=fields,
            prompt_instructions=prompt_instructions,
            ollama_thinking_mode=ollama_thinking_mode,
            preprocessing=preprocessing,
            vlm_transcription_instructions=vlm_transcription_instructions,
            llm_system_prompt=llm_system_prompt,
        )
        return
    run_id = "cni-" + time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
    run_dir = runs_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    total = len(model_specs) * len(valid_clients)
    completed = 0
    results: list[dict[str, Any]] = []
    started_at = time.monotonic()

    LOGGER.info(
        "CNI benchmark starting | run=%s | models=%s | valid_clients=%d | "
        "strategy=%s | prompt_scope=%s | prompt_delivery=%s | dpi=%d | "
        "ollama_trust_environment=%s | ollama_thinking=%s",
        run_id,
        model_specs,
        len(valid_clients),
        strategy,
        prompt_scope_mode,
        prompt_delivery_mode,
        dpi,
        not bool(ignore_environment_proxy),
        ollama_thinking_mode,
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
        effective_output_format = _model_output_format(
            model_spec,
            model_name,
            output_format_mode,
            model_output_modes,
        )
        try:
            # Une erreur de chargement est produite pour chaque client afin que
            # l'interface conserve la matrice complète modèle × client.
            model = registry.create(
                model_spec,
                cpu_threads=cpu_threads,
                unload_after_task=unload_after_task,
                timeout_seconds=timeout_seconds,
                ignore_environment_proxy=ignore_environment_proxy,
                thinking_mode=ollama_thinking_mode,
            )
            model_name = model.model_name
            effective_output_format = _model_output_format(
                model_spec,
                model_name,
                output_format_mode,
                model_output_modes,
            )
            LOGGER.info(
                "CNI model output contract | spec=%s | model=%s | format=%s",
                model_spec,
                model_name,
                effective_output_format,
            )
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
                try:
                    prepared = prepare_cni_client_images(client, client_dir, dpi, preprocessing=preprocessing)
                except Exception as exc:
                    completed += 1
                    result = _failed_client_result(run_id, model_name, client, strategy, f"prepare_failed: {type(exc).__name__}: {exc}")
                    results.append(result)
                    _write_results_index(run_dir, results)
                    LOGGER.exception("CNI document preparation failed | client=%s", client["folder_client_id"])
                    yield _completed_event(run_id, completed, total, result, started_at)
                    continue

                # Le générateur interne rend la main juste avant chaque appel.
                # L'UI affiche donc réellement Recto 1/2 puis Verso 2/2 sans
                # compter deux fois la même paire dans la progression globale.
                extraction = _iter_extract_one_cni_client(
                    model,
                    run_id,
                    model_name,
                    client,
                    prepared,
                    strategy=strategy,
                    timeout_seconds=timeout_seconds,
                    fields=fields,
                    prompt_instructions=prompt_instructions,
                    prompt_scope_mode=prompt_scope_mode,
                    prompt_delivery_mode=prompt_delivery_mode,
                    system_prompt=system_prompt,
                    output_format_mode=effective_output_format,
                )
                while True:
                    try:
                        step = next(extraction)
                    except StopIteration as finished:
                        result = finished.value
                        break
                    yield _processing_event(
                        run_id,
                        completed,
                        total,
                        model_name,
                        client,
                        str(step["side"]),
                        prepared,
                        started_at,
                        substep=int(step["substep"]),
                        substeps=int(step["substeps"]),
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


def _iter_cni_two_stage_benchmark(
    registry: ModelRegistry,
    vlm_model_spec: str,
    llm_model_spec: str,
    valid_clients: list[dict[str, Any]],
    runs_root: Path,
    *,
    strategy: str,
    dpi: int,
    timeout_seconds: float | None,
    cpu_threads: int | None,
    unload_after_task: bool,
    ignore_environment_proxy: bool,
    fields: dict[str, list[dict[str, str]]] | None,
    prompt_instructions: str | None,
    ollama_thinking_mode: str,
    preprocessing: dict[str, Any] | None,
    vlm_transcription_instructions: str | None,
    llm_system_prompt: str | None,
) -> Iterator[dict[str, Any]]:
    """Exécute deux lots strictement séparés pour ne jamais garder VLM et LLM.

    Le premier lot produit des transcriptions persistées. Le VLM est ensuite
    libéré avant la création du LLM. Le second lot transforme les textes en
    JSON avec le schéma de la face transmis à la fois dans le prompt et dans
    le paramètre ``format`` d'Ollama.
    """

    run_id = "cni-" + time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
    run_dir = runs_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    total = len(valid_clients)
    completed = 0
    results: list[dict[str, Any]] = []
    started_at = time.monotonic()
    pair_model_name = (
        f"{vlm_model_spec.split(':', 1)[-1]} → "
        f"{llm_model_spec.split(':', 1)[-1]}"
    )
    if not valid_clients:
        yield {
            "stage": "completed", "run_id": run_id, "completed": 0,
            "total": 0, "result": None,
            "message": "Aucun dossier client CNI valide à traiter.",
        }
        return

    LOGGER.info(
        "CNI two-stage pipeline starting | run=%s | vlm=%s | llm=%s | "
        "clients=%d | strategy=%s",
        run_id, vlm_model_spec, llm_model_spec, total, strategy,
    )
    cached: list[dict[str, Any]] = []
    vlm = None
    try:
        vlm = registry.create(
            vlm_model_spec,
            cpu_threads=cpu_threads,
            unload_after_task=True,
            timeout_seconds=timeout_seconds,
            ignore_environment_proxy=ignore_environment_proxy,
            thinking_mode="disabled",
        )
        for client in valid_clients:
            artefacts_dir = (
                run_dir / _safe_name(pair_model_name)
                / _safe_name(str(client["folder_client_id"]))
            )
            try:
                prepared = prepare_cni_client_images(
                    client, artefacts_dir, dpi, preprocessing=preprocessing
                )
            except Exception as exc:
                completed += 1
                result = _failed_client_result(
                    run_id, pair_model_name, client, strategy,
                    f"prepare_failed: {type(exc).__name__}: {exc}",
                )
                results.append(result)
                _write_results_index(run_dir, results)
                yield _completed_event(
                    run_id, completed, total, result, started_at
                )
                continue

            sides = ["combined"] if strategy == "combined_vertical" else ["recto", "verso"]
            transcriptions: dict[str, InferenceResult] = {}
            for index, side in enumerate(sides, start=1):
                image_path = Path(
                    prepared["combined_image"]
                    if side == "combined"
                    else prepared[f"{side}_model_image"]
                )
                yield _processing_event(
                    run_id, completed, total, pair_model_name, client,
                    "recto_verso" if side == "combined" else side,
                    prepared, started_at, substep=index,
                    substeps=len(sides) * 2,
                    pipeline_stage="vlm_transcription",
                )
                transcriptions[side] = _perform_vlm_transcription_call(
                    vlm,
                    image_path,
                    build_vlm_transcription_prompt(
                        side, vlm_transcription_instructions
                    ),
                    timeout_seconds,
                    artefacts_dir,
                    side,
                )
            cached.append(
                {
                    "client": client,
                    "prepared": prepared,
                    "transcriptions": transcriptions,
                }
            )
    finally:
        _close_model(vlm, "VLM", vlm_model_spec)

    # Le LLM n'est construit qu'après la libération explicite du VLM.
    llm = None
    try:
        llm = registry.create(
            llm_model_spec,
            cpu_threads=cpu_threads,
            unload_after_task=unload_after_task,
            timeout_seconds=timeout_seconds,
            ignore_environment_proxy=ignore_environment_proxy,
            thinking_mode=ollama_thinking_mode,
        )
        for cached_client in cached:
            client = cached_client["client"]
            prepared = cached_client["prepared"]
            transcriptions = cached_client["transcriptions"]
            artefacts_dir = Path(prepared["recto_crop"]["image_path"]).parent
            sides = ["combined"] if strategy == "combined_vertical" else ["recto", "verso"]
            structured: dict[str, InferenceResult] = {}
            for index, side in enumerate(sides, start=1):
                yield _processing_event(
                    run_id, completed, total, pair_model_name, client,
                    "recto_verso" if side == "combined" else side,
                    prepared, started_at, substep=len(sides) + index,
                    substeps=len(sides) * 2,
                    pipeline_stage="llm_structuring",
                )
                vlm_result = transcriptions[side]
                structured[side] = _perform_llm_structuring_call(
                    llm,
                    side,
                    vlm_result,
                    fields,
                    prompt_instructions,
                    llm_system_prompt,
                    timeout_seconds,
                    artefacts_dir,
                )

            if strategy == "combined_vertical":
                inference = structured["combined"]
                recto, verso, parse_error = parse_combined_cni_json_response(
                    inference.text, fields
                )
                recto_inference = verso_inference = inference
                recto_parse_error = verso_parse_error = parse_error
                vlm_latency = transcriptions["combined"].latency_seconds
            else:
                recto_inference = structured["recto"]
                verso_inference = structured["verso"]
                recto, recto_parse_error = parse_cni_json_response(
                    recto_inference.text, "recto", fields
                )
                verso, verso_parse_error = parse_cni_json_response(
                    verso_inference.text, "verso", fields
                )
                vlm_latency = sum(
                    item.latency_seconds for item in transcriptions.values()
                )
            metadata = {
                "pipeline": "vlm_llm",
                "vlm_model": vlm_model_spec,
                "llm_model": llm_model_spec,
                "vlm_latency": vlm_latency,
                "vlm_input_tokens": _sum_many_optional(
                    item.input_tokens for item in transcriptions.values()
                ),
                "vlm_output_tokens": _sum_many_optional(
                    item.output_tokens for item in transcriptions.values()
                ),
                "recto": _transcription_metadata(transcriptions.get("recto") or transcriptions.get("combined")),
                "verso": _transcription_metadata(transcriptions.get("verso") or transcriptions.get("combined")),
                "recto_vlm_output_path": str(
                    artefacts_dir / (
                        "raw_vlm_combined_output.txt"
                        if strategy == "combined_vertical"
                        else "raw_vlm_recto_output.txt"
                    )
                ),
                "verso_vlm_output_path": str(
                    artefacts_dir / (
                        "raw_vlm_combined_output.txt"
                        if strategy == "combined_vertical"
                        else "raw_vlm_verso_output.txt"
                    )
                ),
                "combined_vlm_output_path": str(artefacts_dir / "raw_vlm_combined_output.txt"),
            }
            result = _finalize_cni_result(
                run_id, pair_model_name, client, prepared,
                strategy=strategy,
                prompt_delivery_mode="vlm_transcription_then_llm_json",
                recto=recto,
                verso=verso,
                recto_inference=recto_inference,
                verso_inference=verso_inference,
                recto_parse_error=recto_parse_error,
                verso_parse_error=verso_parse_error,
                pipeline_metadata=metadata,
            )
            completed += 1
            results.append(result)
            _write_results_index(run_dir, results)
            yield _completed_event(run_id, completed, total, result, started_at)
    except Exception as exc:
        LOGGER.exception(
            "CNI LLM stage failed | model=%s", llm_model_spec
        )
        for cached_client in cached[completed:]:
            completed += 1
            result = _failed_client_result(
                run_id, pair_model_name, cached_client["client"], strategy,
                f"llm_stage_failed: {type(exc).__name__}: {exc}",
            )
            results.append(result)
            _write_results_index(run_dir, results)
            yield _completed_event(run_id, completed, total, result, started_at)
    finally:
        _close_model(llm, "LLM", llm_model_spec)


def _close_model(model: Any, stage: str, model_spec: str) -> None:
    """Libère un adaptateur sans masquer les résultats déjà persistés."""

    close = getattr(model, "close", None)
    if callable(close):
        try:
            close()
        except Exception:
            LOGGER.exception("CNI %s cleanup failed | model=%s", stage, model_spec)
    LOGGER.info("CNI %s released | model=%s", stage, model_spec)


def _transcription_metadata(inference: InferenceResult) -> dict[str, Any]:
    """Expose la lecture VLM dans le JSON de face sans la confondre avec le LLM."""

    return {
        "stage": "vlm_transcription",
        "status": inference.status.value,
        "text": inference.text,
        "raw_response": inference.raw_response,
        "error": inference.error,
        "latency": inference.latency_seconds,
        "input_tokens": inference.input_tokens,
        "output_tokens": inference.output_tokens,
    }


def prepare_cni_client_images(
    client: dict[str, Any],
    artefacts_dir: Path,
    dpi: int,
    *,
    preprocessing: dict[str, Any] | None = None,
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
    options: dict[str, Any] = dict(preprocessing or {})
    crop_method = str(options.get("crop_method") or SMART_CROP_V4)
    minimum_score = float(
        options.get("smart_crop_min_score") or DEFAULT_SMART_CROP_MIN_SCORE
    )
    margin_ratio = float(
        options.get("smart_crop_margin") or DEFAULT_SMART_CROP_MARGIN
    )
    LOGGER.info(
        "CNI preprocessing options | client=%s | dpi=%d | crop_method=%s | "
        "smart_crop_min_score=%.3f | smart_crop_margin=%.3f | "
        "rotation_pillow=%s | rotation_opencv=%s | perspective=%s | "
        "contrast=%s | denoise=%s",
        client["folder_client_id"],
        int(dpi),
        crop_method,
        minimum_score,
        margin_ratio,
        bool(options.get("rotation_pillow")),
        bool(options.get("rotation_opencv")),
        bool(options.get("perspective")),
        bool(options.get("contrast")),
        bool(options.get("denoise")),
    )
    recto_source = Path(str(client.get("recto_source") or client["recto_pdf"]))
    verso_source = Path(str(client.get("verso_source") or client["verso_pdf"]))
    recto_render = prepare_cni_source(recto_source, recto_page, dpi)
    verso_render = prepare_cni_source(verso_source, verso_page, dpi)
    recto_crop = crop_cni_for_benchmark(
        recto_page,
        artefacts_dir / "crop_recto_diagnostics",
        output_path=artefacts_dir / "crop_recto.png",
        method=crop_method,
        minimum_score=minimum_score,
        margin_ratio=margin_ratio,
    )
    verso_crop = crop_cni_for_benchmark(
        verso_page,
        artefacts_dir / "crop_verso_diagnostics",
        output_path=artefacts_dir / "crop_verso.png",
        method=crop_method,
        minimum_score=minimum_score,
        margin_ratio=margin_ratio,
    )
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
    options: dict[str, Any],
) -> dict[str, Any]:
    """Préserve strictement la source si le crop automatique est incertain."""
    if crop.get("source_sent_unchanged"):
        return {
            "status": "skipped_crop_uncertain_original_preserved",
            "image_path": crop["image_path"],
            "operations": [],
        }
    return preprocess_cni_image(Path(crop["image_path"]), output_path, options)


def _iter_extract_one_cni_client(
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
    prompt_scope_mode: str,
    prompt_delivery_mode: str,
    system_prompt: str | None,
    output_format_mode: str,
) -> Generator[dict[str, Any], None, dict[str, Any]]:
    """Exécute une paire et signale chaque face juste avant son inférence.

    La valeur retournée par le générateur est le résultat final de la paire.
    Les valeurs produites par ``yield`` sont seulement des étapes live ; elles
    ne modifient donc pas le compteur principal modèle × client.
    """
    artefacts_dir = Path(prepared["recto_crop"]["image_path"]).parent
    image_only = prompt_delivery_mode == "image_only"
    uses_modelfile_system = prompt_delivery_mode in {
        "image_only",
        "image_with_side_hint",
    }
    effective_system_prompt = None if uses_modelfile_system else system_prompt
    if strategy == "combined_vertical":
        # Un seul appel reçoit le composite, mais le parsing produit toujours
        # deux dictionnaires afin de préserver le contrat des artefacts.
        yield {"side": "recto_verso", "substep": 1, "substeps": 1}
        inference = _perform_cni_call(
            model,
            Path(prepared["combined_image"]),
            _build_model_prompt(
                "combined",
                fields,
                prompt_instructions,
                prompt_scope_mode,
                prompt_delivery_mode,
            ),
            timeout_seconds,
            artefacts_dir,
            "combined",
            effective_system_prompt,
            output_format_mode,
            build_cni_output_schema("combined", fields),
            image_only=image_only,
            prompt_delivery_mode=prompt_delivery_mode,
        )
        recto, verso, parse_error = parse_combined_cni_json_response(inference.text, fields)
        recto_inference = verso_inference = inference
        recto_parse_error = verso_parse_error = parse_error
    else:
        # Deux appels indépendants sont le mode de diagnostic recommandé : on
        # sait immédiatement quelle face a posé problème.
        yield {"side": "recto", "substep": 1, "substeps": 2}
        recto_inference = _perform_cni_call(
            model,
            Path(prepared["recto_model_image"]),
            _build_model_prompt(
                "recto",
                fields,
                prompt_instructions,
                prompt_scope_mode,
                prompt_delivery_mode,
            ),
            timeout_seconds,
            artefacts_dir,
            "recto",
            effective_system_prompt,
            output_format_mode,
            build_cni_output_schema("recto", fields),
            image_only=image_only,
            prompt_delivery_mode=prompt_delivery_mode,
        )
        yield {"side": "verso", "substep": 2, "substeps": 2}
        verso_inference = _perform_cni_call(
            model,
            Path(prepared["verso_model_image"]),
            _build_model_prompt(
                "verso",
                fields,
                prompt_instructions,
                prompt_scope_mode,
                prompt_delivery_mode,
            ),
            timeout_seconds,
            artefacts_dir,
            "verso",
            effective_system_prompt,
            output_format_mode,
            build_cni_output_schema("verso", fields),
            image_only=image_only,
            prompt_delivery_mode=prompt_delivery_mode,
        )
        recto, recto_parse_error = parse_cni_json_response(recto_inference.text, "recto", fields)
        verso, verso_parse_error = parse_cni_json_response(verso_inference.text, "verso", fields)

    return _finalize_cni_result(
        run_id,
        model_name,
        client,
        prepared,
        strategy=strategy,
        prompt_delivery_mode=prompt_delivery_mode,
        recto=recto,
        verso=verso,
        recto_inference=recto_inference,
        verso_inference=verso_inference,
        recto_parse_error=recto_parse_error,
        verso_parse_error=verso_parse_error,
    )


def _build_model_prompt(
    side: str,
    fields: dict[str, list[dict[str, str]]] | None,
    instructions: str | None,
    prompt_scope_mode: str,
    prompt_delivery_mode: str,
) -> str:
    """Retourne exactement le texte USER correspondant au mode opérateur."""
    if prompt_delivery_mode == "image_only":
        return ""
    if prompt_delivery_mode == "image_with_side_hint":
        return build_cni_face_hint(side)
    if side == "combined":
        return build_combined_cni_prompt(fields, instructions=instructions)
    return build_cni_prompt(
        side,
        fields,
        instructions=instructions,
        prompt_scope_mode=prompt_scope_mode,
    )


def _finalize_cni_result(
    run_id: str,
    model_name: str,
    client: dict[str, Any],
    prepared: dict[str, Any],
    *,
    strategy: str,
    prompt_delivery_mode: str,
    recto: dict[str, str | None],
    verso: dict[str, str | None],
    recto_inference: InferenceResult,
    verso_inference: InferenceResult,
    recto_parse_error: str | None,
    verso_parse_error: str | None,
    pipeline_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Persiste le contrat commun aux pipelines direct et VLM → LLM."""

    artefacts_dir = Path(prepared["recto_crop"]["image_path"]).parent
    metadata = dict(pipeline_metadata or {})
    recto_payload = _side_payload(
        "recto", recto, recto_inference, recto_parse_error,
        prepared["recto_crop"], metadata.get("recto"),
    )
    verso_payload = _side_payload(
        "verso", verso, verso_inference, verso_parse_error,
        prepared["verso_crop"], metadata.get("verso"),
    )
    write_cni_json(artefacts_dir / "recto.extraction.json", recto_payload)
    write_cni_json(artefacts_dir / "verso.extraction.json", verso_payload)
    global_payload = build_cni_global_json(client, recto, verso)
    comparison = compare_cni_extraction(
        _read_json_mapping(client.get("label_path")), global_payload
    )
    global_payload.update(
        {
            "run_id": run_id,
            "model": model_name,
            "strategy": strategy,
            "prompt_delivery_mode": prompt_delivery_mode,
            "pipeline": metadata.get("pipeline", "direct_vlm"),
            "recto_status": recto_payload["status"],
            "verso_status": verso_payload["status"],
            "comparison": comparison,
        }
    )
    write_cni_json(artefacts_dir / "global.extraction.json", global_payload)
    status = _overall_status(
        recto_payload["status"], verso_payload["status"],
        recto_parse_error, verso_parse_error,
    )
    if strategy == "combined_vertical":
        total_latency = recto_inference.latency_seconds
        llm_output_tokens = recto_inference.output_tokens
        llm_input_tokens = recto_inference.input_tokens
    else:
        total_latency = (
            recto_inference.latency_seconds + verso_inference.latency_seconds
        )
        llm_output_tokens = _sum_optional(
            recto_inference.output_tokens, verso_inference.output_tokens
        )
        llm_input_tokens = _sum_optional(
            recto_inference.input_tokens, verso_inference.input_tokens
        )
    vlm_latency = metadata.get("vlm_latency")
    if vlm_latency is not None:
        total_latency += float(vlm_latency)
    input_tokens = _sum_optional(
        llm_input_tokens, metadata.get("vlm_input_tokens")
    )
    output_tokens = _sum_optional(
        llm_output_tokens, metadata.get("vlm_output_tokens")
    )
    return {
        "run_id": run_id,
        "model": model_name,
        "vlm_model": metadata.get("vlm_model"),
        "llm_model": metadata.get("llm_model"),
        "pipeline": metadata.get("pipeline", "direct_vlm"),
        "folder_client_id": client["folder_client_id"],
        "status": status,
        "strategy": strategy,
        "prompt_delivery_mode": prompt_delivery_mode,
        "label_status": client.get("label_status"),
        "label_path": client.get("label_path"),
        "accuracy": comparison["accuracy"],
        "score_status": comparison["score_status"],
        "field_comparison": comparison,
        "recto_status": recto_payload["status"],
        "verso_status": verso_payload["status"],
        "cin_recto": global_payload["cin_recto"],
        "cin_verso": global_payload["cin_verso"],
        "cin_fusionne": global_payload["cin_fusionne"],
        "cin_coherent": global_payload["cin_coherent"],
        "date_validite_coherente": global_payload["date_validite_coherente"],
        "latency": total_latency,
        "vlm_latency": vlm_latency,
        "llm_latency": total_latency - float(vlm_latency or 0),
        "vlm_input_tokens": metadata.get("vlm_input_tokens"),
        "vlm_output_tokens": metadata.get("vlm_output_tokens"),
        "llm_input_tokens": llm_input_tokens,
        "llm_output_tokens": llm_output_tokens,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "tokens_per_second": (
            output_tokens / total_latency
            if output_tokens is not None and total_latency
            else None
        ),
        "recto_json_path": str(artefacts_dir / "recto.extraction.json"),
        "verso_json_path": str(artefacts_dir / "verso.extraction.json"),
        "global_json_path": str(artefacts_dir / "global.extraction.json"),
        "recto_raw_output_path": str(artefacts_dir / "raw_recto_output.txt"),
        "verso_raw_output_path": str(artefacts_dir / "raw_verso_output.txt"),
        "combined_raw_output_path": str(artefacts_dir / "raw_combined_output.txt"),
        "recto_vlm_output_path": metadata.get("recto_vlm_output_path"),
        "verso_vlm_output_path": metadata.get("verso_vlm_output_path"),
        "combined_vlm_output_path": metadata.get("combined_vlm_output_path"),
        "recto_image_path": prepared["recto_model_image"],
        "verso_image_path": prepared["verso_model_image"],
        "combined_image_path": prepared["combined_image"],
        "recto_prompt_path": str(artefacts_dir / "prompt_recto.txt"),
        "verso_prompt_path": str(artefacts_dir / "prompt_verso.txt"),
        "combined_prompt_path": str(artefacts_dir / "prompt_combined.txt"),
        "error": _join_errors(
            recto_payload.get("error"), verso_payload.get("error")
        ),
    }


def _perform_cni_call(
    model: Any,
    image_path: Path,
    prompt: str,
    timeout_seconds: float | None,
    artefacts_dir: Path,
    side: str,
    system_prompt: str | None,
    output_format_mode: str,
    output_schema: dict[str, Any],
    *,
    image_only: bool = False,
    prompt_delivery_mode: str = "application_prompt",
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
        f"--- DELIVERY MODE ---\n"
        f"{prompt_delivery_mode}\n\n"
        f"--- OUTPUT FORMAT ---\n{output_format_mode}\n\n"
        + "--- SYSTEM SENT BY APPLICATION ---\n"
        + (system_prompt or "")
        + "\n\n--- USER TEXT SENT BY APPLICATION ---\n"
        + prompt,
        encoding="utf-8",
    )
    raw = BenchmarkRunner._perform_with_timeout(
        model,
        str(image_path),
        timeout_seconds,
        prompt=prompt,
        system_prompt=system_prompt,
        output_format=output_format_mode,
        output_schema=output_schema,
        image_only=image_only,
        prompt_delivery_mode=prompt_delivery_mode,
        late_result=save_late,
    )
    inference = raw if isinstance(raw, InferenceResult) else InferenceResult.from_legacy_dict(raw)
    (artefacts_dir / f"raw_{side}_output.txt").write_text(
        inference.raw_response or inference.text or "", encoding="utf-8"
    )
    return inference


def _perform_vlm_transcription_call(
    model: Any,
    image_path: Path,
    prompt: str,
    timeout_seconds: float | None,
    artefacts_dir: Path,
    side: str,
) -> InferenceResult:
    """Demande uniquement une lecture fidèle et conserve son artefact propre."""

    (artefacts_dir / f"prompt_vlm_{side}.txt").write_text(
        prompt, encoding="utf-8"
    )
    raw = BenchmarkRunner._perform_with_timeout(
        model,
        str(image_path),
        timeout_seconds,
        prompt=prompt,
        output_format="prompt",
        output_schema=None,
        prompt_delivery_mode="application_prompt",
    )
    inference = (
        raw if isinstance(raw, InferenceResult)
        else InferenceResult.from_legacy_dict(raw)
    )
    (artefacts_dir / f"raw_vlm_{side}_output.txt").write_text(
        inference.raw_response or inference.text or "", encoding="utf-8"
    )
    return inference


def _perform_llm_structuring_call(
    model: Any,
    side: str,
    transcription: InferenceResult,
    fields: dict[str, list[dict[str, str]]] | None,
    instructions: str | None,
    system_prompt: str | None,
    timeout_seconds: float | None,
    artefacts_dir: Path,
) -> InferenceResult:
    """Transforme une transcription en JSON, sans jamais renvoyer l'image."""

    if transcription.status is not InferenceStatus.SUCCESS or not transcription.text.strip():
        return InferenceResult(
            text="",
            latency_seconds=0.0,
            status=transcription.status,
            error=(
                "Étape LLM non exécutée car la transcription VLM est vide ou "
                f"en erreur : {transcription.error or transcription.status.value}"
            ),
            raw_response=transcription.raw_response,
            device="ollama",
        )
    prompt = build_llm_structuring_prompt(
        side, transcription.text, fields, instructions
    )
    schema = two_stage_schema(side, fields)
    (artefacts_dir / f"prompt_{side}.txt").write_text(
        "--- PIPELINE ---\nVLM transcription -> LLM JSON\n\n"
        "--- SYSTEM LLM ---\n" + (system_prompt or "")
        + "\n\n--- USER LLM ---\n" + prompt,
        encoding="utf-8",
    )
    raw = BenchmarkRunner._perform_text_with_timeout(
        model,
        prompt,
        timeout_seconds,
        system_prompt=system_prompt,
        output_format="schema",
        output_schema=schema,
    )
    inference = (
        raw if isinstance(raw, InferenceResult)
        else InferenceResult.from_legacy_dict(raw)
    )
    (artefacts_dir / f"raw_{side}_output.txt").write_text(
        inference.raw_response or inference.text or "", encoding="utf-8"
    )
    return inference


def _side_payload(
    side: str,
    fields: dict[str, str | None],
    inference: InferenceResult,
    parse_error: str | None,
    crop: dict[str, Any],
    pipeline_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Construit le JSON d'une face avec données modèle et métadonnées de crop."""
    status = inference.status.value if parse_error is None else "invalid_json"
    payload = {
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
    if pipeline_metadata:
        payload["pipeline"] = pipeline_metadata
    return payload


def _overall_status(recto_status: str, verso_status: str, recto_parse_error: str | None, verso_parse_error: str | None) -> str:
    """Réduit les deux statuts en un statut global, timeout prioritaire."""
    if recto_status == "timeout" or verso_status == "timeout":
        return "timeout"
    if recto_status == "incompatible_model" or verso_status == "incompatible_model":
        return "incompatible_model"
    if recto_status != "success" or verso_status != "success":
        return "failed"
    if recto_parse_error or verso_parse_error:
        return "invalid_json"
    return "success"


def _processing_event(
    run_id: str,
    completed: int,
    total: int,
    model_name: str,
    client: dict[str, Any],
    side: str,
    prepared: dict[str, Any],
    started_at: float,
    *,
    substep: int,
    substeps: int,
    pipeline_stage: str = "direct_extraction",
) -> dict[str, Any]:
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
        "substep": substep,
        "substeps": substeps,
        "pipeline_stage": pipeline_stage,
        "image_path": image,
        "elapsed_seconds": time.monotonic() - started_at,
        "result": None,
    }


def _model_output_format(
    model_spec: str,
    model_name: str,
    default_mode: str,
    overrides: Mapping[str, str] | None,
) -> str:
    """Résout une exception par modèle sans modifier le réglage global.

    La clé canonique est le spec complet (par exemple ``ollama:lightonocr``).
    Le nom exposé par l'adaptateur est aussi accepté pour les configurations
    historiques ou les adaptateurs de test.
    """
    valid_modes = {"schema", "json", "prompt"}
    fallback = default_mode if default_mode in valid_modes else "schema"
    values = overrides if isinstance(overrides, Mapping) else {}
    requested = values.get(model_spec, values.get(model_name))
    return str(requested) if requested in valid_modes else fallback


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
        "score_status": "not_scored_label_mapping_pending",
        "latency": 0.0,
        "input_tokens": None,
        "output_tokens": None,
        "tokens_per_second": None,
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
    temporary.write_text(dumps_json(results), encoding="utf-8")
    temporary.replace(run_dir / "cni_results.json")


def _safe_name(value: str) -> str:
    """Transforme un nom modèle/client en segment de dossier portable."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._") or "unknown"


def _sum_optional(first: int | None, second: int | None) -> int | None:
    """Additionne les compteurs seulement si le fournisseur les expose."""
    values = [value for value in (first, second) if value is not None]
    return sum(values) if values else None


def _sum_many_optional(values: Iterator[int | None]) -> int | None:
    """Additionne un nombre variable de compteurs fournisseur disponibles."""

    known = [value for value in values if value is not None]
    return sum(known) if known else None


def _join_errors(*errors: str | None) -> str | None:
    """Garde les deux erreurs de face visibles dans une cellule compacte."""
    values = [error for error in errors if error]
    return " | ".join(values) if values else None
