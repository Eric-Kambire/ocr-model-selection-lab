import ast
import json
import re
import subprocess
import sys
import unicodedata
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "benchmark_colab.ipynb"


def load_notebook():
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def notebook_text(notebook):
    return "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])


def code_cell_containing(notebook, needle):
    matches = [
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code" and needle in "".join(cell.get("source", []))
    ]
    assert matches, f"No code cell contains {needle!r}"
    return matches[0]


def literal_assignment(source, variable_name):
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == variable_name
            for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError(f"No literal assignment found for {variable_name}")


def test_notebook_is_valid_json_and_all_code_cells_compile():
    notebook = load_notebook()
    assert notebook["nbformat"] == 4
    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") == "code":
            source = "".join(cell.get("source", []))
            try:
                ast.parse(source)
            except SyntaxError as exc:
                raise AssertionError(f"Invalid Python in notebook cell {index}: {exc}") from exc


def test_generated_notebook_is_in_sync_with_its_source_script():
    before = NOTEBOOK.read_bytes()
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "rebuild_colab_notebook.py")],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert NOTEBOOK.read_bytes() == before


def test_notebook_has_no_saved_outputs_or_literal_access_tokens():
    notebook = load_notebook()
    for cell in notebook["cells"]:
        if cell.get("cell_type") == "code":
            assert cell.get("outputs", []) == []
            assert cell.get("execution_count") is None

    payload = notebook_text(notebook) + (
        ROOT / "scripts" / "rebuild_colab_notebook.py"
    ).read_text(encoding="utf-8")
    for pattern in (
        r"\bhf_[A-Za-z0-9]{20,}\b",
        r"\bgh[pousr]_[A-Za-z0-9]{20,}\b",
        r"\bgithub_pat_[A-Za-z0-9_]{20,}\b",
        r"\bdckr_pat_[A-Za-z0-9_-]{20,}\b",
        r"\bKGAT_[A-Za-z0-9_-]{20,}\b",
    ):
        assert not re.search(pattern, payload)


def test_notebook_contains_all_requested_dataset_loaders():
    text = notebook_text(load_notebook())
    for identifier in (
        "TheFinAI/MultiFinBen-EnglishOCR",
        "arunchincheti/handwritten_and_cheques_dataset",
        "naderabdelghany/iam-handwritten-forms-dataset",
        "bernardadhitya/handwritten-form-ocr-ie-json-dataset",
    ):
        assert identifier in text
    assert "ALLOW_DERIVED_LABELS_IN_RANKING = False" in text
    assert "is_scorable" in text


def test_colab_secrets_support_huggingface_kaggle_and_github_without_printing_values():
    notebook = load_notebook()
    text = notebook_text(notebook)
    secrets_cell = code_cell_containing(notebook, "def colab_secret")

    assert "from google.colab import userdata" in text
    assert "userdata.get(name)" in secrets_cell
    for secret_name in (
        "HF_TOKEN",
        "HUGGINGFACE_TOKEN",
        "HF_HUB_TOKEN",
        "KAGGLE_API_TOKEN",
        "KAGGLE_JSON",
        "KAGGLE_USERNAME",
        "KAGGLE_KEY",
        "GITHUB_TOKEN",
        "GH_TOKEN",
    ):
        assert secret_name in secrets_cell

    tree = ast.parse(secrets_cell)
    printed_expressions = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "print"
        ):
            printed_expressions.extend(ast.unparse(argument) for argument in node.args)
    printed_text = " ".join(printed_expressions)
    assert "SECRET_STATUS" in printed_text
    assert not any(
        raw_secret in printed_text
        for raw_secret in ("HF_TOKEN", "KAGGLE_API_TOKEN", "GITHUB_TOKEN", "kaggle_json")
    )


def test_dataset_downloads_are_bounded_and_authenticated():
    notebook = load_notebook()
    config_cell = code_cell_containing(notebook, "DATASET_SOURCES =")
    loader_cell = code_cell_containing(notebook, "def load_kaggle_iam")
    sources = literal_assignment(config_cell, "DATASET_SOURCES")

    assert all(source["enabled"] for source in sources.values())
    assert all(1 <= int(source["max_samples"]) <= 5 for source in sources.values())
    assert 'DATA_VOLUME = "DEMO"' in config_cell
    assert 'if DATA_VOLUME == "DECISION_30"' in config_cell
    assert 'DATASET_SOURCES["hf_multifin"]["max_samples"] = 30' in config_cell
    assert 'DATASET_SOURCES["hf_cheques"]["max_samples"] = 30' in config_cell
    iam = sources["kaggle_iam"]
    assert iam["download_full_dataset"] is False
    assert len(iam["sample_files"]) >= iam["max_samples"] >= 3
    assert all(path.startswith("data/000/") and path.endswith(".png") for path in iam["sample_files"])

    loader_tree = ast.parse(loader_cell)
    kaggle_calls = [
        node
        for node in ast.walk(loader_tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "dataset_download"
    ]
    assert kaggle_calls
    assert all(any(keyword.arg == "path" for keyword in call.keywords) for call in kaggle_calls)
    assert 'cfg["sample_files"][: int(cfg["max_samples"])]' in loader_cell
    assert 'token=hf_token_argument()' in loader_cell
    assert 'headers["Authorization"] = f"Bearer {GITHUB_TOKEN}"' in loader_cell


def test_dataset_labels_and_user_zip_are_safe_and_auditable():
    text = notebook_text(load_notebook())
    assert "def safe_extract_zip" in text
    assert "def assert_inside" in text
    assert 'for field in ("image_path", "ground_truth", "category")' in text
    assert "label_provenance" in text
    assert '"is_scorable": False' in text
    assert "ALLOW_DERIVED_LABELS_IN_RANKING = False" in text
    assert "image_sha256" in text


def test_notebook_is_standalone_and_does_not_clone_the_app_repo():
    text = notebook_text(load_notebook()).lower()
    assert "git clone" not in text
    assert "repo_url" not in text


def test_legacy_profile_uses_a_python_312_compatible_numpy_constraint():
    install_cell = code_cell_containing(load_notebook(), "RUNTIME_PROFILE =")
    assert 'legacy_numpy = "numpy>=1.26,<2" if sys.version_info >= (3, 12)' in install_cell
    assert 'if not package.startswith(("numpy", "pillow"))' in install_cell
    assert '"--force-reinstall", "pillow==11.1.0"' in install_cell
    assert 'from PIL import Image, ImageDraw, ImageFont' in install_cell
    assert "Image.new(\"RGB\", (2, 2), \"white\")" in install_cell


def test_model_catalog_contains_requested_models_and_hardware_contracts():
    notebook = load_notebook()
    catalog_cell = code_cell_containing(notebook, "MODEL_CATALOG =")
    catalog = literal_assignment(catalog_cell, "MODEL_CATALOG")
    defaults = literal_assignment(catalog_cell, "DEFAULT_SELECTED_MODELS")

    expected_ids = {
        "pp_ocrv6": "PP-OCRv6_medium_det+rec",
        "paddleocr_vl_1_6": "PaddlePaddle/PaddleOCR-VL-1.6",
        "glm_ocr": "zai-org/GLM-OCR",
        "granite_docling_258m": "ibm-granite/granite-docling-258M",
        "qwen_ocr_0_8b": "loay/English-Document-OCR-Qwen3.5-0.8B",
        "minicpm_v_4_6": "openbmb/MiniCPM-V-4.6",
        "chandra_ocr_2": "datalab-to/chandra-ocr-2",
        "lightonocr_2_1b": "lightonai/LightOnOCR-2-1B",
        "dots_ocr": "rednote-hilab/dots.ocr",
        "unlimited_ocr": "baidu/Unlimited-OCR",
        "locateanything_3b": "nvidia/LocateAnything-3B",
    }
    assert {key: catalog[key]["model_id"] for key in expected_ids} == expected_ids
    assert {"easyocr", "pp_ocrv6", "glm_ocr", "granite_docling_258m"}.issubset(defaults)

    required_contract = {
        "display_name",
        "adapter_kind",
        "model_id",
        "profile",
        "supports_cpu",
        "t4_supported",
        "min_vram_gb",
        "weights_gb",
        "license",
        "ranking_task",
        "prompt",
        "download_strategy",
        "note",
    }
    assert all(required_contract.issubset(config) for config in catalog.values())
    assert catalog["locateanything_3b"]["ranking_task"] == "localization"
    assert catalog["locateanything_3b"]["min_vram_gb"] == 40
    assert catalog["locateanything_3b"]["max_new_tokens"] == 8192
    assert catalog["glm_ocr"]["max_new_tokens"] == 8192
    assert catalog["granite_docling_258m"]["max_new_tokens"] == 8192
    assert catalog["dots_ocr"]["max_new_tokens"] == 24000
    assert catalog["dots_ocr"]["download_strategy"] == "dots_local"
    assert "markdown-first format" in catalog["qwen_ocr_0_8b"]["prompt"]
    assert catalog["dots_ocr"]["adapter_kind"] == "dots_ocr"
    assert catalog["unlimited_ocr"]["adapter_kind"] == "unlimited_ocr"
    assert "def model_readiness" in catalog_cell
    assert "VRAM insuffisante" in catalog_cell
    assert "T4 non supporté" in catalog_cell


def test_selected_weights_are_downloaded_lazily_with_hf_authentication():
    notebook = load_notebook()
    download_cell = code_cell_containing(notebook, "def download_model_weights")
    runner_cell = code_cell_containing(notebook, "def benchmark_stream")

    assert "snapshot_download" in download_cell
    assert "model_info" in download_cell
    assert "hf_hub_download" in download_cell
    assert "token=HF_TOKEN or None" in download_cell
    assert 'MODEL_CATALOG[model_name]' in download_cell
    assert "for selected_name in SELECTED_MODELS" in download_cell
    assert 'local_dir = MODEL_DIR / "DotsOCR"' in download_cell
    assert 'MODEL_LOCAL_PATHS[model_name] = str(local_dir)' in download_cell
    assert "yield from stream_loaded_adapter(" in runner_cell
    assert "finally:" in runner_cell
    assert "adapter.close()" in runner_cell
    assert "gc.collect()" in runner_cell
    assert "torch.cuda.empty_cache()" in runner_cell


def test_core_requested_models_have_real_adapters_and_are_wired_to_the_factory():
    notebook = load_notebook()
    adapter_cell = code_cell_containing(notebook, "class PPOCRv6Adapter")

    for class_name in (
        "PPOCRv6Adapter",
        "GLMOCRAdapter",
        "GraniteDoclingAdapter",
        "PaddleOCRVLAdapter",
        "MiniCPMAdapter",
        "LightOnOCRAdapter",
        "QwenGGUFAdapter",
        "ChandraAdapter",
        "DotsOCRAdapter",
        "UnlimitedOCRAdapter",
        "LocateAnythingAdapter",
    ):
        assert f"class {class_name}" in adapter_cell
    for mapping in (
        '"pp_ocrv6": PPOCRv6Adapter',
        '"glm_ocr": GLMOCRAdapter',
        '"granite_docling": GraniteDoclingAdapter',
        '"paddleocr_vl": PaddleOCRVLAdapter',
        '"minicpm": MiniCPMAdapter',
        '"lighton": LightOnOCRAdapter',
        '"qwen_gguf": QwenGGUFAdapter',
        '"chandra": ChandraAdapter',
        '"dots_ocr": DotsOCRAdapter',
        '"unlimited_ocr": UnlimitedOCRAdapter',
        '"locate_anything": LocateAnythingAdapter',
    ):
        assert mapping in adapter_cell

    assert "PaddleOCR(" in adapter_cell and ".predict(" in adapter_cell
    assert 'device=self.engine_device' in adapter_cell
    assert "apply_chat_template" in adapter_cell
    assert 'images_kwargs={' in adapter_cell
    assert "DocTagsDocument" in adapter_cell
    assert 'DoclingDocument.load_from_doctags(' in adapter_cell
    assert 'export_to_text(traverse_pictures=True)' in adapter_cell
    assert "scoring_text=plain_text" in adapter_cell
    assert "max_time=float(max_seconds)" in adapter_cell
    assert "def parse_boxes" in adapter_cell
    assert 'payload.update({"annotated_image_path": preview_path, "detected_boxes": len(boxes)})' in adapter_cell
    assert "from dots_ocr.utils import dict_promptmode_to_prompt" in adapter_cell
    assert 'model_source = MODEL_LOCAL_PATHS.get(name) or config["model_id"]' in adapter_cell
    assert 'effective_prompt = self.default_prompt if not prompt or prompt == "prompt_ocr" else prompt' in adapter_cell
    assert 'payload["prompt_used"] = effective_prompt' in adapter_cell
    assert "save_results=False, eval_mode=True" in adapter_cell
    assert 'if "<image>" not in effective_prompt' in adapter_cell
    assert 'effective_prompt = "<image>\\n" + effective_prompt' in adapter_cell
    assert 'output_tokens_kind="estimated_tokenizer_tokens"' in adapter_cell
    assert 'generation_mode="hybrid", temperature=0.7, do_sample=True' in adapter_cell


def test_wer_operates_on_words_not_joined_characters():
    notebook = load_notebook()
    metrics_cell = next(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code" and "def sequence_edit_distance" in "".join(cell.get("source", []))
    )
    namespace = {"np": np, "unicodedata": unicodedata, "re": re, "json": json}
    exec(metrics_cell, namespace)
    assert namespace["wer"]("alpha beta", "alpha gamma") == 0.5
    assert namespace["wer"]("alpha beta", "alpha beta") == 0.0


def test_metrics_normalize_text_and_score_structured_fields():
    notebook = load_notebook()
    metrics_cell = code_cell_containing(notebook, "def sequence_edit_distance")
    namespace = {"np": np, "unicodedata": unicodedata, "re": re, "json": json}
    exec(metrics_cell, namespace)

    assert np.isclose(namespace["cer"]("École", "ecole"), 0.2)
    assert namespace["metric_details"]("École", "ecole")["normalized_exact_match"] == 1.0
    assert namespace["wer"]("un deux trois", "un trois") == 1 / 3
    structured = namespace["structured_field_scores"](
        '{"montant": "1250", "devise": "EUR"}',
        '{"montant": "1250", "devise": "USD"}',
    )
    assert structured == {"field_precision": 0.5, "field_recall": 0.5, "field_f1": 0.5}
    assert namespace["structured_field_scores"]('{"montant": "1250"}', "pas du json") == {
        "field_precision": 0.0,
        "field_recall": 0.0,
        "field_f1": 0.0,
    }


def test_metric_guide_explains_quality_speed_tokens_resources_and_decision_limits():
    text = notebook_text(load_notebook())
    for concept in (
        "Character Error Rate",
        "Word Error Rate",
        "P95",
        "Documents/minute",
        "Tokens de sortie",
        "tokens par seconde",
        "VRAM",
        "RAM",
        "Taux de réussite",
        "Score de décision",
    ):
        assert concept.casefold() in text.casefold()
    assert "tokenizers différents" in text
    assert "N/A" in text


def test_reporting_excludes_failures_from_quality_and_latency():
    text = notebook_text(load_notebook())
    assert "results_frame.status.isin(SUCCESS_STATUSES)" in text
    assert '"latency": np.nan' in text
    assert "corpus_cer" in text
    assert "category_summary_df" in text
    assert "dashboard.html" in text


def test_reporting_uses_corpus_metrics_and_requires_reliable_category_coverage():
    notebook = load_notebook()
    summary_cell = code_cell_containing(notebook, "def summarize_results")
    assert "scored.char_edits.sum() / ref_chars" in summary_cell
    assert "scored.word_edits.sum() / ref_words" in summary_cell
    assert "summary.success_rate >= MIN_TECHNICAL_SUCCESS_RATE" in summary_cell
    assert "summary.scored_documents >= MIN_SCORED_DOCUMENTS" in summary_cell
    assert "summary.category_coverage >= 1.0" in summary_cell
    assert 'summary.ranking_task.eq("transcription")' in summary_cell
    assert 'summary.loc[~summary.eligible, "decision_score"] = np.nan' in summary_cell
    assert 'transcription_attempts = group[group.task_type.eq("transcription")]' in summary_cell
    assert 'extraction_attempts = group[group.task_type.eq("key_value_extraction")]' in summary_cell
    assert "extraction_success_rate" in summary_cell


def test_runner_records_reproducibility_and_keeps_raw_reasoning_out_of_scores():
    notebook = load_notebook()
    runner_cell = code_cell_containing(notebook, "def benchmark_stream")
    adapter_cell = code_cell_containing(notebook, "def completed_reasoning_blocks")

    assert "run_metadata.json" in runner_cell
    assert "resolved_model_revisions" in runner_cell
    assert "package_versions" in runner_cell
    assert "gpu_name" in runner_cell and "cuda_version" in runner_cell
    assert "def selection_signature" in runner_cell
    assert '"selection_signature": selection_signature(selection_frame)' in runner_cell
    assert '"runtime_profile": RUNTIME_PROFILE, "hardware_name": GPU_NAME' in runner_cell
    assert 'QUALITY_OUTPUT_STATUSES = SUCCESS_STATUSES | {"empty_output"}' in runner_cell
    assert 'status = "empty_output"' in runner_cell
    assert "remove_completed_reasoning(scoring_source)" in runner_cell
    assert '"raw_text": model_output' in runner_cell
    assert r'<think>.*?</think>' in adapter_cell


def test_gradio_public_colab_tunnel_is_authenticated_and_zip_upload_is_bounded():
    notebook = load_notebook()
    text = notebook_text(notebook)
    loader_cell = code_cell_containing(notebook, "MAX_ZIP_FILES")
    gradio_cell = code_cell_containing(notebook, "with gr.Blocks(")

    assert "GRADIO_USERNAME" in text and "GRADIO_PASSWORD" in text
    assert "GRADIO_AUTH" in text
    assert "auth=GRADIO_AUTH" in gradio_cell
    assert 'max_file_size="1GB"' in gradio_cell
    assert "MAX_ZIP_UNCOMPRESSED_BYTES" in loader_cell
    assert "MAX_ZIP_MEMBER_BYTES" in loader_cell
    assert "MAX_ZIP_COMPRESSION_RATIO" in loader_cell


def test_timeout_preserves_partial_output_and_runner_keeps_it_in_successful_results():
    notebook = load_notebook()
    adapter_cell = code_cell_containing(notebook, "class QwenGGUFAdapter")
    runner_cell = code_cell_containing(notebook, "SUCCESS_STATUSES =")

    assert "except subprocess.TimeoutExpired as exc" in adapter_cell
    assert '"--single-turn"' in adapter_cell
    assert "stdin=subprocess.DEVNULL" in adapter_cell
    assert 'status="timeout_with_output" if partial.strip() else "timeout"' in adapter_cell
    assert "text=partial" in adapter_cell
    assert "raw_response=partial" in adapter_cell
    assert 'SUCCESS_STATUSES = {"success", "slow_success", "timeout_with_output"}' in runner_cell
    assert 'prediction.get("raw_response", "")' in runner_cell
    assert "status in SUCCESS_STATUSES" in runner_cell


def test_gradio_can_launch_live_benchmarks_and_select_quantity_or_all_documents():
    notebook = load_notebook()
    gradio_cell = code_cell_containing(notebook, "with gr.Blocks(")

    assert "def ui_run_benchmark" in gradio_cell
    assert "gr.Progress(track_tqdm=False)" in gradio_cell
    assert "for event in benchmark_stream(" in gradio_cell
    for phase in ("loading", "analyzing", "result"):
        assert f'phase == "{phase}"' in gradio_cell
    assert "model_selector = gr.CheckboxGroup" in gradio_cell
    assert "category_selector = gr.CheckboxGroup" in gradio_cell
    for selection_mode in ("Quantité globale", "Par catégorie", "Tout le dataset"):
        assert selection_mode in gradio_cell
    assert 'run_button = gr.Button("Valider et lancer"' in gradio_cell
    assert "live_image = gr.Image" in gradio_cell
    assert "live_metrics = gr.Dataframe" in gradio_cell
    assert "live_text = gr.Textbox" in gradio_cell
    assert 'last_text = str(result.get("text") or result.get("raw_text")' in gradio_cell or "Modèle non exécuté dans ce runtime" in gradio_cell
    assert "Profil/GPU requis" in gradio_cell
    assert "Chargement impossible" in gradio_cell
    assert "launch_status, live_progress, live_image, live_metrics, live_text" in gradio_cell
    assert "run_button.click(" in gradio_cell
    assert "fn=ui_run_benchmark" in gradio_cell
    assert "demo.queue(default_concurrency_limit=1" in gradio_cell
    assert "demo.launch(" in gradio_cell
    assert "share=IS_COLAB, inline=IS_COLAB" in gradio_cell
    assert "run_event.then(" in gradio_cell
    assert "fn=show_result, inputs=result_selector" in gradio_cell
    assert "show_error=True" in gradio_cell
    assert "css=APP_CSS" in gradio_cell
    assert "max(2, len(dataset_df))" in gradio_cell

    tree = ast.parse(gradio_cell)
    ui_runner = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "ui_run_benchmark"
    )
    yields = [node for node in ast.walk(ui_runner) if isinstance(node, ast.Yield)]
    assert yields
    assert all(isinstance(node.value, ast.Tuple) and len(node.value.elts) == 9 for node in yields)


def test_gradio_explorer_dataset_import_and_settings_match_the_workflow():
    notebook = load_notebook()
    gradio_cell = code_cell_containing(notebook, "with gr.Blocks(")

    for tab in (
        "1 · Lancer un benchmark",
        "2 · Graphiques",
        "3 · Explorer les résultats",
        "4 · Comprendre les métriques",
        "5 · Dataset",
        "6 · Paramètres",
    ):
        assert f'gr.Tab("{tab}")' in gradio_cell
    assert "result_selector = gr.Dropdown" in gradio_cell
    assert "previous_button" in gradio_cell and "next_button" in gradio_cell
    assert "expected_text = gr.Textbox" in gradio_cell
    assert "extracted_text = gr.Textbox" in gradio_cell
    assert 'gr.Tab("Sortie brute")' in gradio_cell
    assert 'gr.Tab("Markdown")' in gradio_cell
    assert 'gr.Tab("HTML source")' in gradio_cell
    assert 'gr.Tab("Diff attendu/extrait")' in gradio_cell
    assert 'gr.Tab("Prompt envoyé")' in gradio_cell
    assert "dataset_zip = gr.File" in gradio_cell
    assert "labels.csv" in gradio_cell
    assert "image_path,ground_truth,category" in gradio_cell
    assert "timeout_seconds = gr.Slider" in gradio_cell
    assert "prompt_override = gr.Textbox" in gradio_cell
    assert "def read_run_archive" in gradio_cell
    assert "def ui_import_run_archives" in gradio_cell
    assert "run_archives = gr.File" in gradio_cell
    assert "merge_runs_button" in gradio_cell
    assert "Les archives n'utilisent pas exactement les mêmes documents et labels" in gradio_cell
    assert "GPU/CPU différents entre les archives" in gradio_cell
    assert "drop_duplicates([\"model\", \"document_id\"]" in gradio_cell


def test_timeout_contract_distinguishes_hard_generation_and_soft_measurement_limits():
    notebook = load_notebook()
    text = notebook_text(notebook)
    settings_cell = code_cell_containing(notebook, "Temps cible, pas toujours un arrêt forcé")

    assert "budget souple" in text
    assert "arrêt dur du sous-processus" in settings_cell
    assert "`max_time` demande" in settings_cell
    assert "GLM‑OCR, PaddleOCR‑VL, Granite Docling, MiniCPM, LightOnOCR et dots.ocr" in settings_cell
    assert "EasyOCR, PP‑OCRv6, Chandra, Unlimited‑OCR et LocateAnything" in settings_cell
    assert "slow_success" in settings_cell


def test_locateanything_runs_as_localization_and_is_not_ranked_as_transcription():
    notebook = load_notebook()
    runner_cell = code_cell_containing(notebook, "def benchmark_stream")
    summary_cell = code_cell_containing(notebook, "def summarize_results")

    assert 'cfg["adapter_kind"] in {"glm_ocr", "paddleocr_vl", "locate_anything"}' in runner_cell
    assert '"preview_image_path": prediction.get("annotated_image_path") or document.image_path' in runner_cell
    assert '"detected_boxes": prediction.get("detected_boxes", np.nan)' in runner_cell
    assert 'summary.ranking_task.eq("transcription")' in summary_cell

    tree = ast.parse(runner_cell)
    benchmark_function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "benchmark_stream"
    )
    benchmark_source = ast.get_source_segment(runner_cell, benchmark_function)
    assert '"skipped_task"' not in benchmark_source
    assert "yield from stream_loaded_adapter(" in benchmark_source


def test_gradio_theme_has_readable_light_background_and_compact_scrollable_panels():
    gradio_cell = code_cell_containing(load_notebook(), "APP_CSS =")
    assert "--lab-bg: #F5F7FB" in gradio_cell
    assert "--lab-panel: #FFFFFF" in gradio_cell
    assert "--lab-ink: #172033" in gradio_cell
    assert "max-width: 1480px" in gradio_cell
    assert ".compact-table { max-height: 390px; overflow: auto; }" in gradio_cell
    assert "overflow: hidden" not in gradio_cell
