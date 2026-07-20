from __future__ import annotations

import argparse
import html
import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Any

import gradio as gr
import pandas as pd

import dataset_generator
from models.ollama_model import DEFAULT_OCR_PROMPT
from ocr_benchmark.cni import (
    DEFAULT_RECTO_SUFFIX,
    DEFAULT_VERSO_SUFFIX,
    build_cni_prompt,
    build_combined_cni_prompt,
    import_cni_zip,
    load_cni_field_config,
    materialize_cni_labels,
    render_single_page_pdf,
    scan_cni_clients,
)
from ocr_benchmark.cni_runner import iter_cni_benchmark
from ocr_benchmark.domain import BenchmarkCase
from ocr_benchmark.dataset_repository import DatasetRepository
from ocr_benchmark.registry import build_default_registry
from ocr_benchmark.reporting import RunCheckpoint, save_run
from ocr_benchmark.runner import BenchmarkRunner, summarize_results
from ocr_benchmark.visualization import (
    category_quality_chart,
    cni_accuracy_chart,
    cni_latency_chart,
    empty_figure,
    latency_chart,
    quality_speed_chart,
    reliability_chart,
)

ROOT_DIR = Path(__file__).resolve().parent
DATASET_DIR = ROOT_DIR / "dataset"
CATALOG_PATH = DATASET_DIR / "dataset.json"
RUNS_DIR = ROOT_DIR / "runs"
CNI_IMPORTS_DIR = ROOT_DIR / "cni_imports"

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
LOGGER = logging.getLogger(__name__)

METRICS_HELP = """
## Comment lire les résultats

Le but n’est pas de chercher un unique « meilleur modèle », mais le meilleur compromis pour votre usage.

| Mesure | Définition | Lecture |
|---|---|---|
| **Quality score** | En mode Standard : `max(0, 1 − CER)`. En mode Bankmark : moyenne pondérée des métriques bancaires réellement applicables. | Plus haut = meilleur. |
| **CER** | Nombre minimal de caractères à insérer, supprimer ou remplacer, divisé par le nombre de caractères attendus. | Plus bas = meilleur. Peut dépasser 100 % si le modèle hallucine beaucoup de texte. |
| **WER** | Même principe que CER, calculé sur les mots. | Plus bas = meilleur. |
| **Mean latency** | Temps moyen, en secondes, pour traiter un document. | Sensible aux cas très lents. |
| **Median latency** | La moitié des documents est traitée plus vite, l’autre moitié plus lentement. | Représente mieux le cas typique. |
| **P95 latency** | 95 % des documents terminent en moins de ce temps. | Indique les lenteurs importantes à prévoir en production. |
| **Documents/s** | `1 / latence moyenne` dans ce benchmark séquentiel. | Plus haut = meilleur. Ce n’est pas une mesure de charge concurrente. |
| **Tokens/s** | Nombre de tokens texte générés par seconde, uniquement si le fournisseur expose ce compteur. | Utile pour les modèles génératifs. Non applicable à EasyOCR. |
| **Success rate** | Exécutions techniquement réussies / exécutions tentées. | Un modèle rapide mais instable ne doit pas être retenu. |
| **IBAN exact match** | Proportion des IBAN attendus retrouvés exactement. | Important : un seul caractère faux invalide un paiement. |
| **Amount exact match** | Proportion des montants attendus retrouvés exactement. | À privilégier pour les documents financiers. |
| **Table/Math preservation** | Présence de la structure attendue dans la sortie. | Calculé seulement si la référence contient cette structure. |

### Choisir un modèle

1. Éliminez les modèles dont le **success rate** n’est pas acceptable.
2. Fixez votre qualité minimale avec **CER/WER** ou les correspondances bancaires.
3. Parmi les modèles restants, comparez **P95**, consommation matérielle et **tokens/s**.
4. Vérifiez les résultats par catégorie : une bonne moyenne peut masquer une faiblesse sur les tableaux ou l’écriture manuscrite.
5. Confirmez sur un jeu de documents réels séparé du dataset de développement.
"""

DATA_FORMAT_HELP = """
### Format attendu

- **Image obligatoire** : `.jpg`, `.jpeg`, `.png` ou `.webp`, maximum **15 Mio**.
- **Label obligatoire** : transcription exacte de tout le texte visible.
- Conservez les retours à la ligne lorsqu’ils ont un sens.
- Pour un tableau, utilisez du Markdown : `| Colonne 1 | Colonne 2 |`.
- Pour une formule, utilisez LaTeX : `$x^2 + y^2$`.
- N’ajoutez aucune explication qui n’apparaît pas dans l’image.
- **Catégorie** : utilisez un nom stable, par exemple `bank`, `handwritten_form`,
  `invoice`, `table` ou `handwritten`.
- **Description** : provenance et particularités utiles du document.

Exemple de label :

```text
Nom: Marie Dupont
Date: 09/07/2026
Montant: 1 250,00 EUR
Signature: Marie Dupont
```

Le fichier est copié dans `dataset/user_uploads/` avec un nom non prédictible,
puis ajouté atomiquement à `dataset/dataset.json`.
"""

DEFAULT_CNI_SYSTEM_PROMPT = "Extract Moroccan CNI fields exactly. Return only one valid JSON object matching the requested schema. Never guess; use null if unreadable. Ignore QR, barcode and MRZ."
DEFAULT_CNI_USER_INSTRUCTIONS = "Read Latin values only. 'Né le' = birth date; nearby 'à' = birth city; 'Valable jusqu’au' = expiry. Do not confuse holder, parents, CAN or civil-status number."

APP_CSS = """
.gradio-container {
    max-width: 1500px !important;
    min-height: 100vh !important;
    overflow: visible !important;
    background: var(--body-background-fill) !important;
    color: var(--body-text-color) !important;
}
#main-tabs {
    min-height: calc(100vh - 190px) !important;
    overflow: visible !important;
}
#main-tabs > .tabitem {
    min-height: calc(100vh - 240px) !important;
    overflow: visible !important;
    padding-bottom: 4px !important;
}
#benchmark-layout,
#explorer-layout,
#dataset-layout,
#cni-explorer-layout {
    height: 100% !important;
    align-items: stretch !important;
}
#models-list .wrap {
    max-height: 145px !important;
    overflow-y: auto !important;
    scrollbar-width: thin;
}
#summary-panel {
    min-height: 0 !important;
}
.dashboard-grid {
    height: 100% !important;
}
.dashboard-chart {
    height: 310px !important;
    min-height: 250px !important;
}
.dashboard-chart > div {
    height: 100% !important;
}
#live-image,
#live-metrics {
    min-height: 270px !important;
}
#live-metrics {
    max-height: 300px !important;
    overflow: auto !important;
}
#category-quantities .wrap {
    max-height: 185px !important;
    overflow-y: auto !important;
}
#benchmark-config {
    gap: 8px !important;
}
#benchmark-config .form {
    gap: 6px !important;
}
#benchmark-config .block {
    margin: 0 !important;
}
#live-section-title h3 {
    margin: 4px 0 0 !important;
}
#metrics-pane {
    max-height: 70vh !important;
    overflow-y: auto !important;
    padding-right: 12px;
}
#dataset-catalog {
    height: calc(100vh - 300px) !important;
    overflow: auto !important;
}
#cni-tabs {
    margin-top: 8px !important;
}
#cni-tabs > .tabitem {
    min-height: calc(100vh - 310px) !important;
    padding-top: 12px !important;
}
#cni-prep-grid {
    gap: 18px !important;
    padding: 4px 0 8px;
    border-bottom: 1px solid var(--block-border-color);
}
#cni-source {
    padding-right: 18px;
    border-right: 1px solid var(--block-border-color);
}
#cni-source,
#cni-control {
    gap: 8px !important;
}
.cni-header {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    gap: 16px;
    padding: 2px 0 10px;
    border-bottom: 1px solid var(--block-border-color);
}
.cni-header h2 { margin: 0; font-size: 22px; line-height: 1.2; }
.cni-header span { color: var(--body-text-color-subdued); font-size: 13px; }
.cni-section-title { display: flex; gap: 8px; align-items: baseline; margin: 0 0 2px; font-size: 15px; font-weight: 650; }
.cni-section-title span { color: var(--body-text-color-subdued); font-size: 12px; font-weight: 500; }
#cni-runbar { align-items: end !important; gap: 8px !important; padding-top: 2px; }
#cni-runbar > button { min-height: 38px !important; }
#cni-live-workspace {
    gap: 16px !important;
}
#cni-live-workspace > * {
    min-height: 340px !important;
}
#cni-results-table {
    min-height: 280px !important;
}
#cni-results-filterbar { align-items: end !important; gap: 8px !important; }
#cni-results-navigation { align-items: center !important; gap: 8px !important; }
#cni-result-position { text-align: center; padding-top: 8px; }
#cni-result-identity { min-height: 92px; }
.gradio-container .tabitem {
    background: var(--body-background-fill) !important;
}
.gradio-container .block,
.gradio-container .form {
    border-color: var(--block-border-color) !important;
}
.gradio-container table {
    background: var(--block-background-fill) !important;
    color: var(--body-text-color) !important;
}
.gradio-container th,
.gradio-container td {
    border-color: var(--border-color-primary) !important;
}
.hero { padding: 14px 20px; border-radius: 16px; color: white;
        background: linear-gradient(135deg, #312e81 0%, #4f46e5 50%, #0f766e 100%);
        box-shadow: 0 12px 28px rgba(49,46,129,.24); margin-bottom: 8px; }
.hero h1 { margin: 0 0 3px 0; font-size: 26px; line-height: 1.15; }
.hero p { margin: 0; opacity: .9; font-size: 14px; }
"""


def select_dataset_items(
    dataset: list[dict[str, Any]],
    mode: str,
    global_quantity: int | float | None,
    category_quantities,
    *,
    shuffle: bool,
    seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(int(seed))

    def take(items: list[dict[str, Any]], quantity: int) -> list[dict[str, Any]]:
        quantity = max(0, min(quantity, len(items)))
        if shuffle:
            return rng.sample(items, quantity)
        return items[:quantity]

    if mode == "Tout le dataset":
        selected = list(dataset)
        if shuffle:
            rng.shuffle(selected)
        return selected

    if mode == "Quantité globale":
        quantity = int(global_quantity or 0)
        if quantity <= 0:
            raise ValueError("La quantité globale doit être supérieure à zéro.")
        return take(dataset, quantity)

    if isinstance(category_quantities, pd.DataFrame):
        rows = category_quantities.values.tolist()
    else:
        rows = category_quantities or []

    selected: list[dict[str, Any]] = []
    for row in rows:
        if len(row) < 3:
            continue
        category = str(row[0])
        quantity = int(float(row[2] or 0))
        available = [item for item in dataset if item["category"] == category]
        selected.extend(take(available, quantity))
    if not selected:
        raise ValueError("Sélectionnez au moins un document dans une catégorie.")
    if shuffle:
        rng.shuffle(selected)
    return selected


def build_run_preview(
    selected_models: list[str],
    selected_items: list[dict[str, Any]],
    eval_mode: str,
    timeout_seconds: float,
    seed: int,
) -> str:
    counts: dict[str, int] = {}
    for item in selected_items:
        counts[item["category"]] = counts.get(item["category"], 0) + 1
    distribution = ", ".join(f"`{name}`={count}" for name, count in sorted(counts.items()))
    evaluations = len(selected_models) * len(selected_items)
    return (
        "### Plan d’exécution\n\n"
        f"- **Modèles :** {len(selected_models)}\n"
        f"- **Documents :** {len(selected_items)}\n"
        f"- **Évaluations prévues :** {evaluations}\n"
        f"- **Répartition :** {distribution}\n"
        f"- **Mode :** {eval_mode}\n"
        f"- **Timeout :** {timeout_seconds:.0f} s par image\n"
        f"- **Seed :** {seed}\n\n"
        "Vérifiez ce plan puis cliquez sur **Confirmer et lancer**."
    )


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "—"
    seconds = max(0, int(seconds))
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _metric_percent(value: Any) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value) * 100:.2f} %"


def _live_metrics_markdown(result: dict[str, Any], next_label: str) -> str:
    expected = html.escape(str(result.get("ground_truth", "")))
    extracted = html.escape(str(result.get("extracted_text", "")))
    token_speed = result.get("tokens_per_second")
    token_text = (
        f"{float(token_speed):.2f} tokens/s" if token_speed is not None else "N/A"
    )
    error = (
        f"\n- **Erreur :** {html.escape(str(result['error']))}"
        if result.get("error")
        else ""
    )
    return (
        "### Dernier résultat\n\n"
        f"- **Modèle :** `{result['model']}`\n"
        f"- **Statut :** `{result['status']}`\n"
        f"- **Score :** {_metric_percent(result.get('accuracy'))}\n"
        f"- **CER :** {_metric_percent(result.get('cer'))}\n"
        f"- **WER :** {_metric_percent(result.get('wer'))}\n"
        f"- **Latence :** {float(result.get('latency') or 0):.3f} s\n"
        f"- **Vitesse tokens :** {token_text}\n"
        f"- **Device :** `{result.get('device', 'unknown')}`\n"
        f"- **Prochaine évaluation :** {next_label}"
        f"{error}\n\n"
        "<details><summary>Texte attendu</summary>"
        f"<pre>{expected}</pre></details>"
        "<details><summary>Texte extrait</summary>"
        f"<pre>{extracted}</pre></details>"
    )


def _live_result_table(results: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for index, result in enumerate(results[-25:], start=max(1, len(results) - 24)):
        rows.append(
            {
                "#": index,
                "Modèle": result["model"],
                "Image": Path(result["image_path"]).name,
                "Catégorie": result["category"],
                "Statut": result["status"],
                "Score": _metric_percent(result.get("accuracy")),
                "CER": _metric_percent(result.get("cer")),
                "WER": _metric_percent(result.get("wer")),
                "Latence (s)": round(float(result.get("latency") or 0), 3),
            }
        )
    return pd.DataFrame(rows)


def _cni_scan_table(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Build the non-sensitive readiness table for CNI input folders."""
    return pd.DataFrame([
        {
            "Client dossier": item.get("folder_client_id"),
            "Recto": "OK" if item.get("recto_pdf") else "Manquant",
            "Verso": "OK" if item.get("verso_pdf") else "Manquant",
            "Label": item.get("label_status", "—"),
            "Statut": item.get("status", "—"),
            "Alertes": ", ".join(item.get("issues") or []) or "—",
        }
        for item in records
    ])


def _cni_source_choices(records: list[dict[str, Any]]) -> list[tuple[str, str]]:
    """Liste les sources recto/verso PDF, JPEG ou PNG détectées au scan."""
    choices: list[tuple[str, str]] = []
    for record in records:
        for side in ("recto", "verso"):
            path_value = record.get(f"{side}_source") or record.get(f"{side}_pdf")
            if path_value:
                kind = str(record.get(f"{side}_format") or Path(str(path_value)).suffix.lstrip(".") or "fichier").upper()
                choices.append((f"{record.get('folder_client_id', 'Client')} — {side.title()} ({kind})", str(path_value)))
    return choices


def _preview_cni_source(path_value: str | None) -> tuple[Any, str]:
    """Show the source preview only when a document is selected."""
    if not path_value:
        return gr.update(value=None, visible=False), "Sélectionnez un PDF détecté pour l’aperçu."
    source = Path(path_value)
    if not source.is_file():
        return gr.update(value=None, visible=False), "⚠️ Le fichier sélectionné n’est plus disponible sur le disque."
    if source.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
        return gr.update(value=str(source), visible=True), f"**Aperçu :** `{source.name}` · image locale"
    if source.suffix.lower() != ".pdf":
        return gr.update(value=None, visible=False), f"⚠️ Format non pris en charge : `{source.suffix}`"
    preview_path = RUNS_DIR / "cni_source_previews" / f"{source.stem}-{source.stat().st_mtime_ns}.png"
    try:
        render_single_page_pdf(source, preview_path, dpi=150)
    except Exception as exc:
        return gr.update(value=None, visible=False), f"⚠️ Aperçu PDF impossible : `{type(exc).__name__}: {exc}`"
    return gr.update(value=str(preview_path), visible=True), f"**Aperçu :** `{source.name}` · PDF rendu à 150 DPI"


def _cni_source_mode_visibility(mode: str) -> tuple[Any, Any]:
    """Keep the local-folder and ZIP entry paths visually separate."""
    return gr.update(visible=mode == "folder"), gr.update(visible=mode == "zip")


def _cni_prompt_preview(strategy: str, system_prompt: str | None, instructions: str | None) -> str:
    """Affiche exactement les prompts CNI qui seront envoyés au modèle."""
    fields = load_cni_field_config(ROOT_DIR / "config" / "cni_fields.json")
    if strategy == "combined_vertical":
        return f"--- SYSTEM ---\n{system_prompt or ''}\n\n--- USER ---\n" + build_combined_cni_prompt(fields, instructions=instructions)
    return (
        f"--- SYSTEM ---\n{system_prompt or ''}\n\n--- USER RECTO ---\n"
        + build_cni_prompt("recto", fields, instructions=instructions)
        + "\n\n--- PROMPT VERSO ---\n"
        + build_cni_prompt("verso", fields, instructions=instructions)
    )


def _cni_alert_html(level: str, message: str) -> str:
    """Affiche un état CNI avec une couleur et un symbole lisibles."""
    styles = {
        "ready": ("●", "#e8f1fb", "#2563a8"),
        "success": ("✓", "#e8f7ee", "#167c46"),
        "warning": ("⚠", "#fff5df", "#9a5b00"),
        "error": ("✕", "#fcebea", "#b42318"),
    }
    symbol, background, color = styles.get(level, styles["ready"])
    return f"<div style='padding:10px;border-radius:8px;background:{background};color:{color};font-weight:600'>{symbol} {html.escape(str(message))}</div>"


def _cni_result_table(results: list[dict[str, Any]]) -> pd.DataFrame:
    """Format CNI benchmark results; accuracy remains unscored until label mapping exists."""
    return pd.DataFrame([
        {
            "Client": item.get("folder_client_id"),
            "Modèle": item.get("model"),
            "Statut": item.get("status"),
            "Accuracy": "Non noté" if item.get("accuracy") is None else f"{float(item['accuracy']) * 100:.2f}%",
            "Label": item.get("label_status", "—"),
            "CIN recto": item.get("cin_recto") or "—",
            "CIN verso": item.get("cin_verso") or "—",
            "CIN cohérent": "Oui" if item.get("cin_coherent") is True else "Non" if item.get("cin_coherent") is False else "—",
            "Champs à revoir": ", ".join(key for key, state in _cni_field_comparisons(item).items() if state == "different") or "—",
            "Latence (s)": round(float(item.get("latency") or 0), 3),
        }
        for item in results
    ])


def _cni_boolean(value: Any) -> str:
    """Rend visible l'état de cohérence sans planter sur une valeur absente."""
    if value is True:
        return "Oui"
    if value is False:
        return "Non"
    return "—"


def _read_json_if_available(path_value: Any) -> Any:
    """Read one JSON artefact for Gradio, returning a visible status on failure."""
    if not path_value:
        return {"status": "not_available"}
    path = Path(str(path_value))
    if not path.is_file():
        return {"status": "not_found"}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"status": "read_failed", "error": f"{type(exc).__name__}: {exc}"}


def _cni_field_comparisons(result: dict[str, Any]) -> dict[str, str]:
    """Compare les champs canoniques au label, quand celui-ci existe."""
    label = _read_json_if_available(result.get("label_path"))
    extracted = _read_json_if_available(result.get("global_json_path"))
    fields = ("cin", "prenom", "nom", "date_naissance", "ville_naissance", "date_validite", "adresse")
    if not isinstance(label, dict) or "status" in label:
        return {field: "label_missing" for field in fields}
    if not isinstance(extracted, dict) or "status" in extracted:
        return {field: "missing_model" for field in fields}
    aliases = {"cin": "cin_fusionne", "date_validite": "date_validite_fusionnee"}
    def normal(value: Any) -> str:
        return re.sub(r"[^a-z0-9]", "", str(value or "").casefold())
    output: dict[str, str] = {}
    for field in fields:
        expected = label.get(field)
        if expected is None:
            expected = next((side.get(field) for side in (label.get("recto"), label.get("verso")) if isinstance(side, dict) and field in side), None)
        actual = extracted.get(aliases.get(field, field))
        output[field] = "label_missing" if expected in (None, "") else "missing_model" if actual in (None, "") else "correct" if normal(expected) == normal(actual) else "different"
    return output


def _cni_raw_output(path_value: Any) -> str:
    """Expose aussi une réponse reçue après le timeout de l'interface."""
    path = Path(str(path_value)) if path_value else None
    if path is not None:
        side = "recto" if "recto" in path.name else "verso" if "verso" in path.name else "combined"
        late_path = path.parent / f"late_{side}_output.json"
        if late_path.is_file():
            try:
                late = json.loads(late_path.read_text(encoding="utf-8"))
                return "[Réponse arrivée après timeout]\n" + json.dumps(late, ensure_ascii=False, indent=2)
            except (OSError, json.JSONDecodeError) as exc:
                return f"Lecture de la réponse tardive impossible : {type(exc).__name__}: {exc}"
    value = _read_json_if_available(path_value)
    if not isinstance(value, dict):
        return "Aucun retour brut disponible."
    return str(value.get("raw_response") or value.get("text") or value.get("error") or "(sortie vide)")


def load_dataset() -> list[dict[str, Any]]:
    """Load and validate the catalog, generating it only when it is absent."""
    if not CATALOG_PATH.exists():
        dataset_generator.main()
    with CATALOG_PATH.open("r", encoding="utf-8") as stream:
        data = json.load(stream)
    if not isinstance(data, list):
        raise ValueError("dataset.json must contain a JSON array.")
    required = {"image_path", "ground_truth", "category"}
    for index, item in enumerate(data):
        missing = required - item.keys()
        if missing:
            raise ValueError(f"Dataset item {index} is missing: {sorted(missing)}")
        # Catalogs generated on Windows may contain backslashes. Convert both
        # separator styles before any filesystem access so the same catalog
        # works inside Linux containers and Colab.
        item["image_path"] = item["image_path"].replace("\\", "/")
        image_path = ROOT_DIR / Path(item["image_path"])
        if not image_path.is_file():
            raise FileNotFoundError(f"Dataset image does not exist: {image_path}")
    return data


def get_installed_ollama_models() -> list[str]:
    try:
        import ollama

        response = ollama.list()
        models = response.get("models", []) if isinstance(response, dict) else response.models
        names = []
        for model in models:
            if isinstance(model, dict):
                names.append(model.get("model") or model.get("name"))
            else:
                names.append(getattr(model, "model", None))
        return [name for name in names if name]
    except Exception as exc:
        print(f"Could not list Ollama models: {exc}")
        return []


def run_benchmark(
    selected_models: list[str],
    selected_category: str = "All",
    mock_noise: float = 0.05,
    eval_mode: str = "Standard",
) -> tuple[pd.DataFrame, list[dict[str, Any]], str]:
    dataset = load_dataset()
    if selected_category != "All":
        dataset = [item for item in dataset if item["category"] == selected_category]
    if not dataset:
        return pd.DataFrame(), [], ""

    runner = BenchmarkRunner(build_default_registry())
    cases = [BenchmarkCase.from_dict(item) for item in dataset]
    run_id, results = runner.run(
        selected_models,
        cases,
        eval_mode=eval_mode,
        mock_noise=mock_noise,
    )
    summary = summarize_results(results)
    run_dir = save_run(run_id, summary, results, RUNS_DIR)
    print(f"Benchmark {run_id} saved to {run_dir}")
    return summary, results, run_id


def _display_summary(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return summary
    display = summary.copy()
    percent_columns = [
        "Success rate",
        "Quality score",
        "CER",
        "WER",
        "Table preservation",
        "Math preservation",
        "IBAN exact match",
        "Amount exact match",
    ]
    for column in percent_columns:
        if column in display:
            display[column] = display[column].map(
                lambda value: "" if pd.isna(value) else f"{value * 100:.2f}%"
            )
    numeric_columns = [
        "Mean latency (s)",
        "Median latency (s)",
        "P95 latency (s)",
        "Documents/s",
        "Tokens/s",
    ]
    for column in numeric_columns:
        if column in display:
            display[column] = display[column].map(
                lambda value: "" if pd.isna(value) else f"{value:.3f}"
            )
    return display


def explain_recommendation(summary: pd.DataFrame) -> str:
    if summary.empty:
        return "### Recommandation\n\nAucun résultat exploitable."
    candidates = summary[
        (summary["Success rate"] >= 0.95) & summary["Quality score"].notna()
    ].copy()
    if candidates.empty:
        return (
            "### Recommandation\n\n"
            "Aucun modèle n’atteint 95 % de réussite technique. "
            "Corrigez d’abord les erreurs fournisseur avant de comparer la qualité."
        )
    candidates = candidates.sort_values(
        ["Quality score", "P95 latency (s)"],
        ascending=[False, True],
        na_position="last",
    )
    winner = candidates.iloc[0]
    token_note = (
        f", **{winner['Tokens/s']:.2f} tokens/s**"
        if pd.notna(winner["Tokens/s"])
        else ", tokens/s non applicable ou non exposé"
    )
    return (
        "### Recommandation automatique\n\n"
        f"**{winner['Model']}** est le candidat principal sur ce run : "
        f"qualité **{winner['Quality score'] * 100:.2f} %**, "
        f"réussite technique **{winner['Success rate'] * 100:.2f} %**, "
        f"P95 **{winner['P95 latency (s)']:.3f} s**{token_note}.\n\n"
        "Règle utilisée : exclure les modèles sous 95 % de réussite, choisir la "
        "meilleure qualité, puis le meilleur P95 en cas d’égalité. Cette "
        "recommandation doit être confirmée sur vos documents réels."
    )


def _catalog_html(dataset: list[dict[str, Any]]) -> str:
    rows = []
    for index, item in enumerate(dataset):
        rows.append(
            "<tr>"
            f"<td>{index}</td>"
            f"<td><code>{html.escape(Path(item['image_path']).name)}</code></td>"
            f"<td>{html.escape(item['category'])}</td>"
            f"<td>{html.escape(item.get('description', ''))}</td>"
            "</tr>"
        )
    return (
        "<div style='max-height:450px;overflow:auto'><table>"
        "<thead><tr><th>#</th><th>Fichier</th><th>Catégorie</th><th>Description</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )


def build_ui() -> gr.Blocks:
    dataset = load_dataset()
    dataset_repository = DatasetRepository(ROOT_DIR)
    image_choices = [
        f"{index}: {Path(item['image_path']).name} [{item['category']}]"
        for index, item in enumerate(dataset)
    ]
    model_choices = ["mock:MockOCR-V1", "mock:MockOCR-V2"]
    try:
        from models.easyocr_model import EASYOCR_AVAILABLE

        if EASYOCR_AVAILABLE:
            model_choices.append("easyocr:EasyOCR-Local")
    except ImportError:
        pass
    model_choices.extend(f"ollama:{name}" for name in get_installed_ollama_models())
    category_choices = sorted(
        {
            "bank",
            "complex_layout",
            "handwritten",
            "handwritten_form",
            "invoice",
            "tables",
            *(item["category"] for item in dataset),
        }
    )
    category_rows = [
        [
            name,
            sum(1 for item in dataset if item["category"] == name),
            sum(1 for item in dataset if item["category"] == name),
        ]
        for name in category_choices
        if any(item["category"] == name for item in dataset)
    ]

    with gr.Blocks(title="OCR Model Selection Lab", fill_height=True) as app:
        gr.HTML(
            "<div class='hero'><h1>OCR Model Selection Lab</h1>"
            "<p>Comparez qualité, vitesse et fiabilité sur CPU ou GPU avec un protocole traçable.</p></div>"
        )
        run_state = gr.State([])
        selection_state = gr.State([])
        detail_index = gr.State(0)
        cni_detail_index = gr.State(0)
        result_model = gr.Dropdown([], visible=False)
        cni_clients_state = gr.State([])
        cni_results_state = gr.State([])

        with gr.Tabs(elem_id="main-tabs"):
            with gr.Tab("1. Benchmark"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=1, elem_id="benchmark-config"):
                        models = gr.CheckboxGroup(
                            model_choices,
                            value=["mock:MockOCR-V1"],
                            label="Modèles",
                            elem_id="models-list",
                        )
                        selection_mode = gr.Radio(
                            ["Tout le dataset", "Quantité globale", "Par catégorie"],
                            value="Quantité globale",
                            label="Sélection des documents",
                        )
                        global_quantity = gr.Number(
                            value=min(10, len(dataset)),
                            minimum=1,
                            maximum=len(dataset),
                            precision=0,
                            label=f"Quantité globale — {len(dataset)} disponibles",
                        )
                        with gr.Accordion(
                            "Quantités par catégorie — ouvrir pour personnaliser",
                            open=False,
                        ):
                            category_quantities = gr.Dataframe(
                                value=category_rows,
                                headers=["Catégorie", "Disponibles", "Quantité"],
                                datatype=["str", "number", "number"],
                                column_count=(3, "fixed"),
                                interactive=True,
                                label="Quantité par catégorie",
                                elem_id="category-quantities",
                            )
                        prepare_run = gr.Button("Préparer le benchmark")
                    with gr.Column(scale=2, elem_id="summary-panel"):
                        run_preview = gr.Markdown(
                            "### Plan d’exécution\n\nConfigurez puis cliquez sur **Préparer**."
                        )
                        with gr.Row():
                            launch = gr.Button("Confirmer et lancer", variant="primary")
                            stop = gr.Button("Annuler", variant="stop")
                        status = gr.Textbox("Prêt.", label="État", interactive=False)
                        progress_bar = gr.Slider(
                            0,
                            100,
                            value=0,
                            step=0.1,
                            interactive=False,
                            label="Progression générale (%)",
                        )
                        live_counters = gr.Markdown(
                            "**Traité :** 0 / 0 · **Succès :** 0 · "
                            "**Erreurs :** 0 · **ETA :** —"
                        )
                        gr.Markdown(
                            "### Analyse en direct", elem_id="live-section-title"
                        )
                        with gr.Row(elem_id="benchmark-layout"):
                            with gr.Column(scale=1):
                                live_image = gr.Image(
                                    label="Image analysée",
                                    type="filepath",
                                    elem_id="live-image",
                                    height=270,
                                )
                            with gr.Column(scale=1, elem_id="live-metrics"):
                                live_metrics = gr.Markdown(
                                    "Les mesures du document apparaîtront ici."
                                )
                with gr.Accordion("Résultats et recommandation", open=False):
                    live_table = gr.Dataframe(
                        headers=[
                            "#",
                            "Modèle",
                            "Image",
                            "Catégorie",
                            "Statut",
                            "Score",
                            "CER",
                            "WER",
                            "Latence (s)",
                        ],
                        interactive=False,
                        label="Derniers résultats",
                    )
                    summary_table = gr.Dataframe(
                        label="Synthèse comparative courante",
                        interactive=False,
                    )
                    recommendation = gr.Markdown(
                        "### Recommandation\n\nDisponible après les premiers résultats."
                    )

            with gr.Tab("2. Paramètres"):
                gr.Markdown(
                    "Paramètres appliqués au prochain benchmark. Un appel fournisseur "
                    "ayant dépassé le timeout peut terminer en arrière-plan."
                )
                with gr.Row():
                    with gr.Column():
                        eval_mode = gr.Radio(
                            ["Standard", "Bankmark"],
                            value="Standard",
                            label="Mode d’évaluation",
                        )
                        noise = gr.Slider(
                            0.0,
                            0.30,
                            value=0.05,
                            step=0.01,
                            label="Bruit du modèle simulé",
                        )
                        randomize = gr.Checkbox(
                            value=True,
                            label="Mélanger les documents",
                        )
                        seed = gr.Number(
                            value=42,
                            precision=0,
                            label="Seed reproductible",
                        )
                    with gr.Column():
                        timeout_seconds = gr.Number(
                            value=300,
                            minimum=1,
                            maximum=7200,
                            precision=0,
                            label="Temps maximum par image (secondes)",
                        )
                        max_errors = gr.Number(
                            value=0,
                            minimum=0,
                            precision=0,
                            label="Arrêter après N erreurs — 0 = illimité",
                        )
                        checkpoint_enabled = gr.Checkbox(
                            value=True,
                            label="Sauvegarder après chaque document",
                        )
                        live_charts_enabled = gr.Checkbox(
                            value=True,
                            label="Actualiser les graphiques en direct",
                        )
                gr.Markdown(
                    "- Les résultats partiels sont conservés après annulation.\n"
                    "- L’exécution reste séquentielle pour comparer les latences.\n"
                    "- La seed reproduit la même sélection aléatoire."
                )
                model_prompt = gr.Textbox(
                    value=DEFAULT_OCR_PROMPT,
                    label="Prompt envoyé aux modèles génératifs compatibles",
                    lines=8,
                    info=(
                        "Utilisé par Ollama. Les moteurs OCR classiques et les "
                        "modèles simulés n’utilisent pas de prompt."
                    ),
                )

            with gr.Tab("3. Graphiques"):
                gr.Markdown(
                    "Les bulles du premier graphique représentent les modèles. "
                    "Le coin supérieur droit correspond au meilleur compromis qualité/vitesse."
                )
                with gr.Column(elem_classes=["dashboard-grid"]):
                    with gr.Row():
                        quality_plot = gr.Plot(value=empty_figure(), elem_classes=["dashboard-chart"])
                        latency_plot = gr.Plot(value=empty_figure(), elem_classes=["dashboard-chart"])
                    with gr.Row():
                        reliability_plot = gr.Plot(value=empty_figure(), elem_classes=["dashboard-chart"])
                        category_plot = gr.Plot(value=empty_figure(), elem_classes=["dashboard-chart"])

            with gr.Tab("4. Résultats détaillés") as details_tab:
                result_selector = gr.Dropdown(
                    [],
                    label="Liste des éléments testés — cliquez pour sélectionner",
                    info="La liste contient uniquement les évaluations passées par le benchmark.",
                )
                with gr.Row():
                    previous_result = gr.Button("← Précédent")
                    result_position = gr.Markdown(
                        "**Aucune page testée pour le moment.**"
                    )
                    next_result = gr.Button("Suivant →")
                with gr.Row(elem_id="explorer-layout"):
                    with gr.Column():
                        source_image = gr.Image(
                            label="Document testé",
                            type="filepath",
                            height=430,
                        )
                        result_identity = gr.Markdown(
                            "Le document et le modèle apparaîtront ici."
                        )
                    with gr.Column(scale=2):
                        detail_metrics = gr.Markdown(
                            "### Mesures\n\nAucun résultat sélectionné."
                        )
                        with gr.Row():
                            with gr.Column():
                                ground_truth = gr.Textbox(
                                    label="Texte attendu",
                                    lines=14,
                                    interactive=False,
                                )
                            with gr.Column():
                                gr.Markdown("**Affichage de la sortie**")
                                with gr.Tabs():
                                    with gr.Tab("Texte extrait"):
                                        extracted = gr.Textbox(
                                            label="Transcription normalisée",
                                            lines=12,
                                            interactive=False,
                                        )
                                    with gr.Tab("Sortie brute"):
                                        raw_output = gr.Textbox(
                                            label="Réponse brute du fournisseur",
                                            lines=12,
                                            interactive=False,
                                        )
                                    with gr.Tab("Markdown rendu"):
                                        markdown_output = gr.Markdown()
                                    with gr.Tab("HTML source"):
                                        html_source = gr.Code(
                                            label=(
                                                "Source HTML — non exécutée "
                                                "pour votre sécurité"
                                            ),
                                            language="html",
                                        )
                        with gr.Accordion("Toutes les mesures techniques", open=False):
                            details = gr.JSON(label="Mesures de ce document")

            with gr.Tab("5. Ajouter des données"):
                with gr.Row():
                    with gr.Column(scale=1):
                        upload_image = gr.File(
                            label="Image du document",
                            file_types=["image"],
                            type="filepath",
                        )
                        upload_category = gr.Dropdown(
                            category_choices,
                            value="handwritten_form",
                            allow_custom_value=True,
                            label="Catégorie",
                        )
                        upload_description = gr.Textbox(
                            label="Description / provenance",
                            placeholder="Ex. formulaire réel anonymisé, rempli à la main",
                        )
                    with gr.Column(scale=2):
                        upload_label = gr.Textbox(
                            label="Label exact / ground truth",
                            lines=12,
                            placeholder="Recopiez exactement tout le texte attendu...",
                        )
                        add_data_button = gr.Button(
                            "Valider et ajouter au dataset",
                            variant="primary",
                        )
                        add_data_status = gr.Markdown()
                gr.Markdown(DATA_FORMAT_HELP)

            with gr.Tab("6. Comprendre les métriques"):
                gr.Markdown(METRICS_HELP, elem_id="metrics-pane")

            with gr.Tab("7. Dataset"):
                with gr.Row(elem_id="dataset-layout"):
                    with gr.Column(scale=1):
                        dataset_selector = gr.Dropdown(
                            image_choices,
                            value=image_choices[0] if image_choices else None,
                            label="Document",
                        )
                        dataset_image = gr.Image(label="Image", type="filepath", height=360)
                    with gr.Column(scale=1):
                        dataset_category = gr.Textbox(label="Catégorie", interactive=False)
                        dataset_description = gr.Textbox(label="Description", interactive=False)
                        dataset_truth = gr.Textbox(label="Ground truth", lines=16, interactive=False)
                    with gr.Column(scale=2, elem_id="dataset-catalog"):
                        catalog_component = gr.HTML(_catalog_html(dataset))

            # Les vues CNI restent montées pendant le générateur de benchmark :
            # l'utilisateur peut donc changer d'onglet sans interrompre le suivi live.
            with gr.Tab("8. Benchmark CNI", render_children=True):
                gr.HTML(
                    "<header class='cni-header'><h2>Benchmark CNI</h2>"
                    "<span>Extraction structurée · exécution séquentielle</span></header>"
                )
                with gr.Tabs(elem_id="cni-tabs"):
                    with gr.Tab("1. Préparer", render_children=True):
                        with gr.Row(elem_id="cni-prep-grid"):
                            with gr.Column(scale=1, elem_id="cni-source"):
                                gr.HTML("<div class='cni-section-title'>01 <span>Source des documents</span></div>")
                                cni_input_mode = gr.Radio(
                                    [("Dossier local", "folder"), ("Archive ZIP", "zip")],
                                    value="folder",
                                    label="Source",
                                )
                                with gr.Group(visible=True) as cni_folder_source:
                                    cni_clients_root = gr.Textbox(label="Dossier clients", placeholder=r"D:\data\clients")
                                    cni_labels_root = gr.Textbox(label="Labels JSONB (optionnel)", placeholder=r"D:\data\labels")
                                    cni_scan = gr.Button("Scanner les dossiers")
                                with gr.Group(visible=False) as cni_zip_source:
                                    cni_zip = gr.File(label="Archive ZIP de test", file_types=[".zip"], type="filepath")
                                    cni_import_zip = gr.Button("Importer le ZIP")
                                cni_scan_status = gr.Markdown("Indiquez un dossier clients, puis scannez-le.")
                                with gr.Accordion("Aperçu d’un document", open=False):
                                    with gr.Row():
                                        with gr.Column(scale=1):
                                            cni_source_selector = gr.Dropdown(
                                                choices=_cni_source_choices([]),
                                                label="Document",
                                                info="Les PDF, JPEG et PNG recto/verso détectés après un scan apparaissent ici.",
                                            )
                                            cni_source_preview_info = gr.Markdown("Sélectionnez un document.")
                                        with gr.Column(scale=1):
                                            cni_source_preview = gr.Image(label="Aperçu", type="filepath", height=220, visible=False)
                            with gr.Column(scale=2, elem_id="cni-control"):
                                gr.HTML("<div class='cni-section-title'>02 <span>Contrôle avant lancement</span></div>")
                                cni_models = gr.CheckboxGroup(
                                    [choice for choice in model_choices if choice.startswith("ollama:")],
                                    label="Modèles Ollama",
                                    info="Les modèles sont exécutés strictement un par un.",
                                )
                                cni_refresh_models = gr.Button("Actualiser les modèles", size="sm")
                                with gr.Accordion("Diagnostic des dossiers", open=False):
                                    cni_scan_report = gr.Dataframe(
                                        headers=["Client dossier", "Recto", "Verso", "Label", "Statut", "Alertes"],
                                        label="Rapport de scan CNI", interactive=False,
                                    )
                        with gr.Row(elem_id="cni-runbar"):
                            gr.Markdown("**03 · Lancement**\n\nLe suivi détaillé apparaît dans la vue suivante.")
                            cni_continue_without_label = gr.Checkbox(
                                value=False,
                                label="Continuer sans labels",
                                info="Extraction et mesures techniques uniquement ; aucun score de comparaison.",
                            )
                            cni_launch = gr.Button("Lancer", variant="primary")
                            cni_stop = gr.Button("Annuler", variant="stop", visible=False)
                        cni_launch_feedback = gr.HTML(
                            "<div style='padding:10px;border-radius:8px;background:#e8f1fb;color:#2563a8'>● Prêt : sélectionnez des modèles, scannez les dossiers puis lancez le benchmark.</div>"
                        )
                    with gr.Tab("2. Suivi en direct", render_children=True):
                        cni_run_status = gr.Textbox(label="État CNI", value="Prêt.", interactive=False)
                        cni_progress = gr.Slider(0, 100, value=0, step=0.1, label="Progression CNI (%)", interactive=False)
                        cni_live_counters = gr.Markdown("**Traité :** 0 / 0 · **Succès :** 0 · **Erreurs :** 0")
                        with gr.Row(elem_id="cni-live-workspace"):
                            cni_live_image = gr.Image(label="Face en cours", type="filepath", height=400)
                            cni_live_result = gr.Markdown("Les JSON et mesures apparaîtront après le premier appel.")
                        cni_live_table = gr.Dataframe(
                            headers=["Client", "Modèle", "Statut", "Accuracy", "Label", "CIN recto", "CIN verso", "CIN cohérent", "Champs à revoir", "Latence (s)"],
                            label="Résultats reçus pendant le run",
                            interactive=False,
                        )
                    with gr.Tab("3. Résultats", render_children=True):
                        # Même hiérarchie que « 4. Résultats détaillés » :
                        # filtres, liste, navigation, puis inspection complète.
                        gr.Markdown("### Résultats détaillés CNI\n\nFiltrez les évaluations puis inspectez une paire recto/verso.")
                        with gr.Row(elem_id="cni-results-filterbar"):
                            cni_accuracy_min = gr.Slider(0, 100, value=0, step=1, label="Accuracy minimale (%)")
                            cni_accuracy_max = gr.Slider(0, 100, value=100, step=1, label="Accuracy maximale (%)")
                            cni_include_unscored = gr.Checkbox(value=True, label="Inclure non notés")
                            cni_field_filter = gr.Dropdown([("Tous les champs", ""), ("CIN", "cin"), ("Prénom", "prenom"), ("Nom", "nom"), ("Date de naissance", "date_naissance"), ("Ville de naissance", "ville_naissance"), ("Date de validité", "date_validite"), ("Adresse", "adresse")], value="", label="Champ")
                            cni_field_state_filter = gr.Dropdown([("Tous les états", ""), ("Correct", "correct"), ("Différent", "different"), ("Valeur modèle absente", "missing_model"), ("Label absent", "label_missing")], value="", label="État")
                            cni_apply_filters = gr.Button("Appliquer les filtres")
                        cni_results_table = gr.Dataframe(headers=["Client", "Modèle", "Statut", "Accuracy", "Label", "CIN recto", "CIN verso", "CIN cohérent", "Champs à revoir", "Latence (s)"], label="Éléments passés par le benchmark", interactive=False, elem_id="cni-results-table")
                        with gr.Row():
                            cni_accuracy_plot = gr.Plot(value=cni_accuracy_chart([]))
                            cni_latency_plot = gr.Plot(value=cni_latency_chart([]))
                        cni_result_selector = gr.Dropdown(label="Liste des paires testées — cliquez pour sélectionner", info="La liste contient les paires client/modèle effectivement passées par le benchmark.", choices=[])
                        with gr.Row(elem_id="cni-results-navigation"):
                            cni_previous_result = gr.Button("← Précédent")
                            cni_result_position = gr.Markdown("**Aucune paire testée pour le moment.**", elem_id="cni-result-position")
                            cni_next_result = gr.Button("Suivant →")
                        with gr.Row(elem_id="cni-explorer-layout"):
                            with gr.Column():
                                cni_recto_preview = gr.Image(label="Recto traité", type="filepath", height=265)
                                cni_verso_preview = gr.Image(label="Verso traité", type="filepath", height=265)
                                cni_result_identity = gr.Markdown("Le client et le modèle apparaîtront ici.", elem_id="cni-result-identity")
                            with gr.Column(scale=2):
                                cni_detail_metrics = gr.Markdown("### Mesures\n\nAucun résultat sélectionné.")
                                with gr.Row():
                                    with gr.Column():
                                        cni_label_json = gr.JSON(label="Label attendu (JSON converti)")
                                    with gr.Column():
                                        gr.Markdown("**Sorties structurées du modèle**")
                                        with gr.Tabs():
                                            with gr.Tab("Extraction recto", render_children=True):
                                                cni_recto_json = gr.JSON(label="JSON recto")
                                            with gr.Tab("Extraction verso", render_children=True):
                                                cni_verso_json = gr.JSON(label="JSON verso")
                                        with gr.Tab("Fusion globale", render_children=True):
                                            cni_global_json = gr.JSON(label="JSON global")
                                        with gr.Tab("Retour brut et erreurs", render_children=True):
                                            cni_recto_raw = gr.Code(label="Recto : retour brut conservé", language=None, lines=7, interactive=False)
                                            cni_verso_raw = gr.Code(label="Verso : retour brut conservé", language=None, lines=7, interactive=False)
                    with gr.Tab("4. Paramètres", render_children=True):
                        gr.Markdown(
                            "### Paramètres CNI\n\n"
                            "Les réglages sont appliqués au prochain lancement. Le prompt complet est affiché avant l'appel modèle."
                        )
                        with gr.Row():
                            cni_strategy = gr.Radio(
                                [
                                    ("Deux appels : recto puis verso — recommandé", "separate_calls"),
                                    ("Une image : recto en haut, verso en bas", "combined_vertical"),
                                ],
                                value="separate_calls",
                                label="Stratégie d'envoi au modèle",
                            )
                            cni_dpi = gr.Slider(150, 450, value=300, step=25, label="Résolution PDF (DPI)")
                            cni_timeout = gr.Number(value=300, minimum=1, maximum=7200, precision=0, label="Temps maximum par appel (s)")
                        with gr.Row():
                            cni_cpu_threads = gr.Number(value=max(1, min(8, os.cpu_count() or 1)), minimum=1, maximum=max(1, os.cpu_count() or 1), precision=0, label="Threads CPU Ollama")
                            cni_unload = gr.Checkbox(value=True, label="Décharger le modèle après chaque appel")
                        cni_preprocessing = gr.CheckboxGroup(
                            [("Redresser une légère inclinaison", "deskew"), ("Améliorer le contraste", "contrast"), ("Réduire le bruit", "denoise")],
                            value=[], label="Prétraitement image (optionnel)",
                            info="Appliqué après conversion PDF/JPEG/PNG et avant crop. Les opérations sont enregistrées dans preparation.json.",
                        )
                        with gr.Row():
                            cni_recto_suffix = gr.Textbox(value=DEFAULT_RECTO_SUFFIX, label="Suffixe recto", info="Texte avant l’extension, par exemple _CIN_Recto. PDF/JPEG/PNG acceptés.")
                            cni_verso_suffix = gr.Textbox(value=DEFAULT_VERSO_SUFFIX, label="Suffixe verso", info="Texte avant l’extension, par exemple _CIN_Verso. PDF/JPEG/PNG acceptés.")
                        cni_system_prompt = gr.Textbox(
                            value=DEFAULT_CNI_SYSTEM_PROMPT,
                            label="Prompt système",
                            lines=5,
                            info="Règle de plus haute priorité. Trop long ou contradictoire réduit la stabilité des réponses.",
                        )
                        cni_prompt_instructions = gr.Textbox(
                            value=DEFAULT_CNI_USER_INSTRUCTIONS,
                            label="Prompt utilisateur / consignes d'extraction",
                            lines=4,
                            info="Demande appliquée à chaque image. Les clés JSON doivent rester stables pour comparer les modèles.",
                        )
                        cni_prompt_preview = gr.Code(
                            value=_cni_prompt_preview("separate_calls", DEFAULT_CNI_SYSTEM_PROMPT, DEFAULT_CNI_USER_INSTRUCTIONS),
                            label="Prompts réellement envoyés (système + utilisateur)",
                            lines=18,
                            interactive=False,
                        )
                        cni_refresh_prompt = gr.Button("Actualiser l’aperçu du prompt")

        def on_prepare(
            model_specs,
            mode,
            quantity,
            quantities_by_category,
            shuffle,
            selected_seed,
            selected_eval_mode,
            selected_timeout,
        ):
            if not model_specs:
                return "### Plan d’exécution\n\n❌ Sélectionnez un modèle.", []
            try:
                selected = select_dataset_items(
                    dataset,
                    mode,
                    quantity,
                    quantities_by_category,
                    shuffle=bool(shuffle),
                    seed=int(selected_seed or 0),
                )
                preview = build_run_preview(
                    model_specs,
                    selected,
                    selected_eval_mode,
                    float(selected_timeout or 0),
                    int(selected_seed or 0),
                )
                return preview, selected
            except Exception as exc:
                return f"### Plan d’exécution\n\n❌ {exc}", []

        def on_run(
            model_specs,
            selected_records,
            selected_noise,
            selected_eval_mode,
            selected_timeout,
            selected_max_errors,
            save_checkpoints,
            update_live_charts,
            selected_model_prompt,
        ):
            empty = empty_figure()
            if not model_specs or not selected_records:
                yield (
                    pd.DataFrame(),
                    "Préparez et validez d’abord le plan d’exécution.",
                    "### Recommandation\n\nAucune exécution.",
                    [],
                    gr.update(choices=[]),
                    empty,
                    empty,
                    empty,
                    empty,
                    None,
                    "Aucun document sélectionné.",
                    0,
                    "**Traité :** 0 / 0",
                    pd.DataFrame(),
                )
                return

            cases = [BenchmarkCase.from_dict(item) for item in selected_records]
            runner = BenchmarkRunner(build_default_registry())
            results: list[dict[str, Any]] = []
            checkpoint: RunCheckpoint | None = None
            summary = pd.DataFrame()
            quality_figure = latency_figure = reliability_figure = category_figure = empty
            latest_update = None

            yield (
                summary,
                "Initialisation des modèles…",
                "### Recommandation\n\nCalcul en cours.",
                results,
                gr.update(choices=[]),
                quality_figure,
                latency_figure,
                reliability_figure,
                category_figure,
                None,
                "Chargement du premier modèle…",
                0,
                f"**Traité :** 0 / {len(model_specs) * len(cases)} · **ETA :** calcul en cours",
                pd.DataFrame(),
            )

            try:
                updates = runner.iter_run(
                    model_specs,
                    cases,
                    eval_mode=selected_eval_mode,
                    mock_noise=float(selected_noise),
                    timeout_seconds=float(selected_timeout or 0),
                    max_errors=int(selected_max_errors or 0),
                    model_prompt=selected_model_prompt,
                    trace=lambda event: RunCheckpoint(
                        event["run_id"], RUNS_DIR
                    ).append_trace(event),
                )
                for update in updates:
                    latest_update = update
                    if checkpoint is None:
                        checkpoint = RunCheckpoint(update.run_id, RUNS_DIR)
                    image_path = str(ROOT_DIR / Path(update.case.image_path))
                    percentage = (
                        update.completed / update.total * 100 if update.total else 0
                    )
                    successes = update.completed - update.error_count

                    if update.stage == "processing":
                        next_text = (
                            "### Traitement en cours\n\n"
                            f"- **Modèle :** `{update.model_name}`\n"
                            f"- **Image :** `{Path(update.case.image_path).name}`\n"
                            f"- **Catégorie :** `{update.case.category}`\n"
                            f"- **Timeout :** {float(selected_timeout):.0f} s\n\n"
                            "Le résultat apparaîtra dès que le modèle aura terminé."
                        )
                        yield (
                            _display_summary(summary),
                            f"Analyse de {Path(update.case.image_path).name}…",
                            explain_recommendation(summary),
                            results,
                            gr.update(),
                            quality_figure,
                            latency_figure,
                            reliability_figure,
                            category_figure,
                            image_path,
                            next_text,
                            percentage,
                            (
                                f"**Traité :** {update.completed} / {update.total} · "
                                f"**Succès :** {successes} · **Erreurs :** {update.error_count} · "
                                f"**Écoulé :** {format_duration(update.elapsed_seconds)} · "
                                f"**ETA :** {format_duration(update.estimated_remaining_seconds)}"
                            ),
                            _live_result_table(results),
                        )
                        continue

                    if update.result is None:
                        continue
                    results.append(update.result)
                    if save_checkpoints and checkpoint:
                        checkpoint.write(results)
                    summary = summarize_results(results)
                    if update_live_charts:
                        quality_figure = quality_speed_chart(summary)
                        latency_figure = latency_chart(summary)
                        reliability_figure = reliability_chart(summary)
                        category_figure = category_quality_chart(results)
                    tested = list(dict.fromkeys(result["model"] for result in results))
                    result = update.result
                    next_position = update.completed
                    next_label = (
                        f"{next_position + 1}/{update.total}"
                        if next_position < update.total
                        else "terminé"
                    )
                    metrics_text = _live_metrics_markdown(result, next_label)
                    yield (
                        _display_summary(summary),
                        f"Résultat reçu : {Path(update.case.image_path).name}",
                        explain_recommendation(summary),
                        results,
                        gr.update(
                            choices=tested,
                            value=tested[0] if tested else None,
                        ),
                        quality_figure,
                        latency_figure,
                        reliability_figure,
                        category_figure,
                        image_path,
                        metrics_text,
                        update.completed / update.total * 100,
                        (
                            f"**Traité :** {update.completed} / {update.total} · "
                            f"**Succès :** {update.completed - update.error_count} · "
                            f"**Erreurs :** {update.error_count} · "
                            f"**Écoulé :** {format_duration(update.elapsed_seconds)} · "
                            f"**ETA :** {format_duration(update.estimated_remaining_seconds)}"
                        ),
                        _live_result_table(results),
                    )
            finally:
                if results:
                    if checkpoint is None:
                        checkpoint = RunCheckpoint(results[0]["run_id"], RUNS_DIR)
                    summary = summarize_results(results)
                    checkpoint.finalize(summary, results)

            if results and latest_update:
                if not update_live_charts:
                    quality_figure = quality_speed_chart(summary)
                    latency_figure = latency_chart(summary)
                    reliability_figure = reliability_chart(summary)
                    category_figure = category_quality_chart(results)
                stopped = latest_update.completed < latest_update.total
                final_status = (
                    f"Arrêt anticipé après {latest_update.error_count} erreur(s)."
                    if stopped
                    else f"Terminé : {len(results)} évaluations."
                )
                yield (
                    _display_summary(summary),
                    f"{final_status} Run ID : {results[0]['run_id']}",
                    explain_recommendation(summary),
                    results,
                    gr.update(),
                    quality_figure,
                    latency_figure,
                    reliability_figure,
                    category_figure,
                    str(ROOT_DIR / Path(latest_update.case.image_path)),
                    "### Benchmark terminé\n\nLes résultats et rapports ont été sauvegardés.",
                    latest_update.completed / latest_update.total * 100,
                    (
                        f"**Traité :** {latest_update.completed} / {latest_update.total} · "
                        f"**Succès :** {latest_update.completed - latest_update.error_count} · "
                        f"**Erreurs :** {latest_update.error_count} · "
                        f"**Durée :** {format_duration(latest_update.elapsed_seconds)}"
                    ),
                    _live_result_table(results),
                )

        def result_choices(results):
            return [
                (
                    f"{position + 1}. {Path(result['image_path']).name} · "
                    f"{result['model']} · {result['status']}",
                    position,
                )
                for position, result in enumerate(results or [])
            ]

        def detail_metric_summary(result):
            input_tokens = result.get("input_tokens")
            output_tokens = result.get("output_tokens")
            token_speed = result.get("tokens_per_second")
            token_speed_text = (
                f"{float(token_speed):.2f}" if token_speed is not None else "N/A"
            )
            return (
                "### Mesures principales\n\n"
                f"**Temps :** {float(result.get('latency') or 0):.3f} s · "
                f"**Score :** {_metric_percent(result.get('accuracy'))} · "
                f"**CER :** {_metric_percent(result.get('cer'))} · "
                f"**WER :** {_metric_percent(result.get('wer'))}\n\n"
                f"**Tokens entrée :** {input_tokens if input_tokens is not None else 'N/A'} · "
                f"**Tokens sortie :** {output_tokens if output_tokens is not None else 'N/A'} · "
                f"**Tokens/s :** {token_speed_text}"
            )

        def rendered_outputs(result):
            text = str(result.get("extracted_text") or "")
            raw = str(result.get("raw_response") or text)
            return text, raw, text, text

        def show_detail(index, results, offset=0):
            results = results or []
            if not results:
                return (
                    gr.update(choices=[], value=None),
                    0,
                    "**Aucune page testée pour le moment.**",
                    None,
                    "Lancez un benchmark pour alimenter cet onglet.",
                    "### Mesures\n\nAucun résultat sélectionné.",
                    "",
                    "",
                    "",
                    "",
                    "",
                    {},
                )
            position = max(0, min(int(index or 0) + offset, len(results) - 1))
            result = results[position]
            hidden = {
                "ground_truth",
                "extracted_text",
                "description",
                "image_path",
                "raw_response",
                "reasoning",
            }
            metrics = {key: value for key, value in result.items() if key not in hidden}
            if result.get("reasoning"):
                metrics["reasoning"] = result["reasoning"]
            identity = (
                f"### {Path(result['image_path']).name}\n\n"
                f"- **Modèle :** `{result['model']}`\n"
                f"- **Catégorie :** `{result['category']}`\n"
                f"- **Statut :** `{result['status']}`\n"
                f"- **Description :** {result.get('description') or '—'}"
            )
            rendered = rendered_outputs(result)
            return (
                gr.update(choices=result_choices(results), value=position),
                position,
                (
                    f"**Page testée {position + 1} / {len(results)}** · "
                    f"{len(results)} évaluation(s) disponible(s)"
                ),
                str(ROOT_DIR / Path(result["image_path"])),
                identity,
                detail_metric_summary(result),
                result.get("ground_truth", ""),
                *rendered,
                metrics,
            )

        def show_current_detail(index, results):
            return show_detail(index, results)

        def show_previous_detail(index, results):
            return show_detail(index, results, -1)

        def show_next_detail(index, results):
            return show_detail(index, results, 1)

        def select_detail(selection, results):
            return show_detail(int(selection or 0), results)

        def browse_dataset(selection):
            if not selection:
                return None, "", "", ""
            index = int(selection.split(":", 1)[0])
            item = dataset[index]
            return (
                str(ROOT_DIR / item["image_path"]),
                item["category"],
                item.get("description", ""),
                item["ground_truth"],
            )

        def add_labeled_data(image_path, label, selected_category, description):
            if not image_path:
                return (
                    "❌ Sélectionnez une image.",
                    gr.update(),
                    _catalog_html(dataset),
                )
            try:
                record = dataset_repository.add_labeled_image(
                    image_path,
                    label or "",
                    selected_category or "",
                    description or "",
                )
                dataset[:] = load_dataset()
                updated_choices = [
                    f"{index}: {Path(item['image_path']).name} [{item['category']}]"
                    for index, item in enumerate(dataset)
                ]
                selected = updated_choices[-1]
                return (
                    "✅ Donnée validée et ajoutée. Elle est immédiatement disponible.",
                    gr.update(choices=updated_choices, value=selected),
                    _catalog_html(dataset),
                )
            except Exception as exc:
                return (
                    f"❌ {type(exc).__name__}: {exc}",
                    gr.update(),
                    _catalog_html(dataset),
                )

        def import_cni_test_zip(zip_path):
            """Importe un ZIP de test et préremplit les chemins clients/labels."""
            if not zip_path:
                return gr.update(), gr.update(), "Import impossible : sélectionnez une archive ZIP."
            try:
                imported = import_cni_zip(Path(zip_path), CNI_IMPORTS_DIR)
                root = Path(imported["import_root"])
                clients_path = root / "clients" if (root / "clients").is_dir() else root
                labels_path = root / "labels"
                LOGGER.info("CNI ZIP imported | files=%s | root=%s", imported["files"], root)
                return (
                    gr.update(value=str(clients_path)),
                    gr.update(value=str(labels_path) if labels_path.is_dir() else ""),
                    f"ZIP importé : {imported['files']} fichier(s). Vérifiez les chemins puis scannez.",
                )
            except Exception as exc:
                LOGGER.exception("CNI ZIP import failed")
                return gr.update(), gr.update(), f"Import ZIP impossible : {type(exc).__name__}: {exc}"

        def scan_cni_input(clients_root_text, labels_root_text, recto_suffix, verso_suffix):
            """Scanne les dossiers et met à jour l'état CNI et les aperçus."""
            if not clients_root_text or not str(clients_root_text).strip():
                return [], pd.DataFrame(), "Scan impossible : indiquez le dossier clients.", gr.update(choices=_cni_source_choices([]), value=None)
            try:
                clients_root = Path(str(clients_root_text).strip()).expanduser()
                labels_root = Path(str(labels_root_text).strip()).expanduser() if labels_root_text and str(labels_root_text).strip() else None
                if labels_root is not None and not labels_root.is_dir():
                    return [], pd.DataFrame(), f"Scan impossible : dossier labels introuvable : `{labels_root}`", gr.update(choices=_cni_source_choices([]), value=None)
                # L'interface ne propose que les clients et PDF issus du scan :
                # le rapprochement reste donc fondé sur le dossier client.
                records = materialize_cni_labels(scan_cni_clients(
                    clients_root, labels_root,
                    recto_suffix=str(recto_suffix or "").strip(),
                    verso_suffix=str(verso_suffix or "").strip(),
                ))
                ready = sum(record["status"] == "ready" for record in records)
                labels = sum(record.get("label_status") == "label_materialized" for record in records)
                unlabeled = sum(record["status"] == "ready" and record.get("label_status") != "label_materialized" for record in records)
                LOGGER.info("CNI scan completed | clients=%d | ready=%d | labels=%d | unlabeled=%d", len(records), ready, labels, unlabeled)
                return (
                    records,
                    _cni_scan_table(records),
                    (
                        f"Scan terminé : {len(records)} client(s) détecté(s), {ready} prêt(s), {labels} label(s) converti(s)."
                        + (" Cochez **Continuer sans labels** pour lancer les PDF non notés." if unlabeled else "")
                    ),
                    gr.update(choices=_cni_source_choices(records), value=None),
                )
            except Exception as exc:
                LOGGER.exception("CNI scan failed")
                return [], pd.DataFrame(), f"Scan CNI impossible : {type(exc).__name__}: {exc}", gr.update(choices=_cni_source_choices([]), value=None)

        def refresh_cni_models(selected_models):
            """Actualise les modèles Ollama en conservant les choix encore valides."""
            choices = [f"ollama:{name}" for name in get_installed_ollama_models()]
            return gr.update(choices=choices, value=[name for name in (selected_models or []) if name in choices])

        def cni_choices(results):
            """Construit des libellés stables liés aux index des résultats."""
            return [
                (f"{result.get('folder_client_id')} · {result.get('model')} · {result.get('status')}", index)
                for index, result in enumerate(results or [])
            ]

        def filter_cni_results(results, minimum, maximum, include_unscored, field_name, field_state):
            """Filtre l'accuracy sans masquer par défaut les lignes non notées."""
            lower, upper = sorted((float(minimum or 0), float(maximum or 100)))
            selected = []
            for result in results or []:
                accuracy = result.get("accuracy")
                if accuracy is None:
                    if not include_unscored:
                        continue
                elif not lower <= float(accuracy) * 100 <= upper:
                    continue
                if field_name and field_state and _cni_field_comparisons(result).get(field_name) != field_state:
                    continue
                selected.append(result)
            return _cni_result_table(selected)

        def cni_detail_metric_summary(result):
            """Présente les mesures CNI dans le même format que l'explorateur général."""
            input_tokens = result.get("input_tokens")
            output_tokens = result.get("output_tokens")
            token_speed = result.get("tokens_per_second")
            token_speed_text = f"{float(token_speed):.2f}" if token_speed is not None else "N/A"
            return (
                "### Mesures principales\n\n"
                f"**Temps total :** {float(result.get('latency') or 0):.3f} s · "
                f"**Accuracy :** {_metric_percent(result.get('accuracy'))} · "
                f"**Statut :** `{result.get('status', '—')}`\n\n"
                f"**Tokens entrée :** {input_tokens if input_tokens is not None else 'N/A'} · "
                f"**Tokens sortie :** {output_tokens if output_tokens is not None else 'N/A'} · "
                f"**Tokens/s :** {token_speed_text}\n\n"
                f"**CIN recto/verso cohérent :** {_cni_boolean(result.get('cin_coherent'))} · "
                f"**Label :** `{result.get('label_status') or 'absent'}`"
            )

        def show_cni_detail(index, results, offset=0):
            """Charge une paire CNI, avec boutons précédent/suivant sans bloquer le run."""
            results = results or []
            empty_json = {"status": "not_selected"}
            if not results:
                return (
                    gr.update(choices=[], value=None), 0,
                    "**Aucune paire testée pour le moment.**", None, None,
                    "Lancez un benchmark pour alimenter cet onglet.",
                    "### Mesures\n\nAucun résultat sélectionné.",
                    empty_json, empty_json, empty_json, empty_json,
                    "Aucun retour brut disponible.", "Aucun retour brut disponible.",
                )
            position = max(0, min(int(index or 0) + offset, len(results) - 1))
            result = results[position]
            identity = (
                f"### Client `{result.get('folder_client_id', '—')}`\n\n"
                f"- **Modèle :** `{result.get('model', '—')}`\n"
                f"- **Stratégie :** `{result.get('strategy', '—')}`\n"
                f"- **Statut :** `{result.get('status', '—')}`\n"
                f"- **Erreur :** {result.get('error') or '—'}"
            )
            return (
                gr.update(choices=cni_choices(results), value=position),
                position,
                f"**Paire testée {position + 1} / {len(results)}** · {len(results)} évaluation(s) disponible(s)",
                result.get("recto_image_path"), result.get("verso_image_path"),
                identity, cni_detail_metric_summary(result),
                _read_json_if_available(result.get("label_path")),
                _read_json_if_available(result.get("recto_json_path")),
                _read_json_if_available(result.get("verso_json_path")),
                _read_json_if_available(result.get("global_json_path")),
                _cni_raw_output(result.get("recto_json_path")),
                _cni_raw_output(result.get("verso_json_path")),
            )

        def select_cni_detail(selection, results):
            """Sélectionne explicitement une ligne de la liste CNI."""
            return show_cni_detail(int(selection or 0), results)

        def show_previous_cni_detail(index, results):
            """Passe à la paire CNI précédente."""
            return show_cni_detail(index, results, -1)

        def show_next_cni_detail(index, results):
            """Passe à la paire CNI suivante."""
            return show_cni_detail(index, results, 1)

        def on_cni_run(model_specs, client_records, strategy, dpi, timeout, threads, unload, preprocessing, system_prompt, prompt_instructions, continue_without_label):
            """Valide le lancement puis diffuse l'avancement CNI document par document."""
            empty = empty_figure()
            results: list[dict[str, Any]] = []

            def counters(total: int) -> str:
                successes = sum(result.get("status") == "success" for result in results)
                failures = len(results) - successes
                return f"**Traité :** {len(results)} / {total} · **Succès :** {successes} · **Erreurs :** {failures}"

            def view(feedback: str, status: str, progress: float, image_path, live_text: str, total: int, *, running: bool = False, alert_level: str = "ready", select_last: bool = False):
                table = _cni_result_table(results)
                selector = gr.update(
                    choices=cni_choices(results),
                    value=(len(results) - 1 if select_last and results else None),
                )
                return (
                    _cni_alert_html(alert_level, feedback),
                    gr.update(visible=not running), gr.update(visible=running),
                    status, progress, image_path, live_text,
                    counters(total), table, results, table, selector,
                    cni_accuracy_chart(results), cni_latency_chart(results),
                )

            if not model_specs:
                message = "Pré-contrôle impossible : sélectionnez au moins un modèle Ollama."
                LOGGER.warning("CNI launch rejected | reason=no_model")
                yield view(message, message, 0, None, message, 0)
                return
            if not client_records:
                message = "Pré-contrôle impossible : scannez d'abord un dossier clients valide."
                LOGGER.warning("CNI launch rejected | reason=no_scan")
                yield view(message, message, 0, None, message, 0)
                return

            ready_records = [record for record in client_records if record.get("status") == "ready"]
            invalid_count = len(client_records) - len(ready_records)
            if not ready_records:
                message = f"Pré-contrôle impossible : aucune paire recto/verso prête ({invalid_count} dossier(s) à corriger dans le rapport de scan)."
                LOGGER.warning("CNI launch rejected | reason=no_ready_pair | clients=%d", len(client_records))
                yield view(message, message, 0, None, message, 0)
                return

            unlabeled = [record for record in ready_records if record.get("label_status") != "label_materialized"]
            if unlabeled and not continue_without_label:
                message = (
                    f"Pré-contrôle requis : {len(unlabeled)} paire(s) n'ont pas de label exploitable. "
                    "Cochez Continuer sans labels pour lancer l'extraction non notée."
                )
                LOGGER.warning("CNI launch paused | reason=missing_label | unlabeled=%d", len(unlabeled))
                yield view(message, message, 0, None, message, len(ready_records))
                return

            total_pairs = len(ready_records) * len(model_specs)
            start_message = (
                f"Lancement confirmé : {len(ready_records)} paire(s), {len(model_specs)} modèle(s), "
                f"{total_pairs} évaluation(s) séquentielle(s)."
            )
            LOGGER.info(
                "CNI launch accepted | pairs=%d | models=%d | strategy=%s | dpi=%s | timeout=%s | unlabeled=%d | invalid=%d",
                len(ready_records), len(model_specs), strategy, dpi, timeout, len(unlabeled), invalid_count,
            )
            yield view(start_message, "Initialisation des modèles en cours.", 0, None, start_message, total_pairs, running=True)

            fields = load_cni_field_config(ROOT_DIR / "config" / "cni_fields.json")
            try:
                for event in iter_cni_benchmark(
                    build_default_registry(), list(model_specs), ready_records, RUNS_DIR,
                    strategy=str(strategy), dpi=int(dpi), timeout_seconds=float(timeout or 0),
                    cpu_threads=int(threads or 1), unload_after_task=bool(unload),
                    fields=fields, prompt_instructions=prompt_instructions, system_prompt=system_prompt,
                    preprocessing={str(value): True for value in (preprocessing or [])},
                ):
                    total, completed = int(event.get("total", total_pairs)), int(event.get("completed", 0))
                    progress = completed / total * 100 if total else 0
                    client_id = event.get("folder_client_id", "—")
                    model = event.get("model", "—")
                    if event.get("stage") == "processing":
                        side = event.get("side", "document")
                        LOGGER.info("CNI processing | client=%s | model=%s | side=%s | completed=%d/%d", client_id, model, side, completed, total)
                        live_text = (
                            "### Analyse CNI en direct\n\n"
                            f"- **Client :** `{client_id}`\n- **Modèle :** `{model}`\n- **Face :** `{side}`\n"
                            "- La sortie brute et le JSON seront conservés dès la réponse."
                        )
                        yield view(
                            "Lancement actif : consultez l’onglet 2. Suivi en direct.",
                            f"Analyse en cours : {client_id} ({side})", progress, event.get("image_path"), live_text, total, running=True,
                        )
                        continue

                    result = event.get("result")
                    if result:
                        results.append(result)
                    status_value = (result or {}).get("status", "unknown")
                    LOGGER.info("CNI result | client=%s | model=%s | status=%s | completed=%d/%d", client_id, model, status_value, completed, total)
                    live_text = (
                        "### Dernier résultat\n\n"
                        f"- **Client :** `{client_id}`\n- **Modèle :** `{model}`\n"
                        f"- **Statut :** `{status_value}`\n- **Label :** `{(result or {}).get('label_status', '—')}`"
                    )
                    finished = completed >= total
                    yield view(
                        "Benchmark terminé." if finished else "Lancement actif : consultez l’onglet 2. Suivi en direct.",
                        f"Résultat reçu : {client_id} ({status_value})", progress,
                        (result or {}).get("recto_image_path"), live_text, total,
                        running=not finished, alert_level="success" if finished else "ready", select_last=True,
                    )
            except Exception as exc:
                LOGGER.exception("CNI benchmark interrupted")
                message = f"Benchmark CNI interrompu : {type(exc).__name__}: {exc}"
                yield view(message, message, 0, None, "Consultez le terminal : l'erreur complète y est enregistrée.", total_pairs, alert_level="error")

        prepare_run.click(
            on_prepare,
            [
                models,
                selection_mode,
                global_quantity,
                category_quantities,
                randomize,
                seed,
                eval_mode,
                timeout_seconds,
            ],
            [run_preview, selection_state],
        )
        run_event = launch.click(
            on_run,
            [
                models,
                selection_state,
                noise,
                eval_mode,
                timeout_seconds,
                max_errors,
                checkpoint_enabled,
                live_charts_enabled,
                model_prompt,
            ],
            [
                summary_table,
                status,
                recommendation,
                run_state,
                result_model,
                quality_plot,
                latency_plot,
                reliability_plot,
                category_plot,
                live_image,
                live_metrics,
                progress_bar,
                live_counters,
                live_table,
            ],
        )
        stop.click(
            fn=None,
            cancels=[run_event],
        )
        detail_outputs = [
            result_selector,
            detail_index,
            result_position,
            source_image,
            result_identity,
            detail_metrics,
            ground_truth,
            extracted,
            raw_output,
            markdown_output,
            html_source,
            details,
        ]
        details_tab.select(
            show_current_detail,
            [detail_index, run_state],
            detail_outputs,
        )
        previous_result.click(
            show_previous_detail,
            [detail_index, run_state],
            detail_outputs,
        )
        next_result.click(
            show_next_detail,
            [detail_index, run_state],
            detail_outputs,
        )
        run_state.change(
            show_current_detail,
            [detail_index, run_state],
            detail_outputs,
        )
        result_selector.input(
            select_detail,
            [result_selector, run_state],
            detail_outputs,
        )
        dataset_selector.change(
            browse_dataset,
            dataset_selector,
            [dataset_image, dataset_category, dataset_description, dataset_truth],
        )
        add_data_button.click(
            add_labeled_data,
            [upload_image, upload_label, upload_category, upload_description],
            [add_data_status, dataset_selector, catalog_component],
        )
        cni_import_zip.click(
            import_cni_test_zip,
            inputs=[cni_zip],
            outputs=[cni_clients_root, cni_labels_root, cni_scan_status],
            queue=False,
        )
        cni_input_mode.change(
            _cni_source_mode_visibility,
            inputs=[cni_input_mode],
            outputs=[cni_folder_source, cni_zip_source],
            queue=False,
        )
        cni_scan.click(
            scan_cni_input,
            inputs=[cni_clients_root, cni_labels_root, cni_recto_suffix, cni_verso_suffix],
            outputs=[cni_clients_state, cni_scan_report, cni_scan_status, cni_source_selector],
            queue=False,
        )
        cni_source_selector.change(
            _preview_cni_source,
            inputs=[cni_source_selector],
            outputs=[cni_source_preview, cni_source_preview_info],
            queue=False,
        )
        cni_refresh_models.click(refresh_cni_models, inputs=[cni_models], outputs=[cni_models], queue=False)
        cni_refresh_prompt.click(
            _cni_prompt_preview,
            inputs=[cni_strategy, cni_system_prompt, cni_prompt_instructions],
            outputs=[cni_prompt_preview],
            queue=False,
        )
        cni_strategy.change(
            _cni_prompt_preview,
            inputs=[cni_strategy, cni_system_prompt, cni_prompt_instructions],
            outputs=[cni_prompt_preview],
            queue=False,
        )
        cni_event = cni_launch.click(
            on_cni_run,
            inputs=[cni_models, cni_clients_state, cni_strategy, cni_dpi, cni_timeout, cni_cpu_threads, cni_unload, cni_preprocessing, cni_system_prompt, cni_prompt_instructions, cni_continue_without_label],
            outputs=[
                cni_launch_feedback,
                cni_launch, cni_stop,
                cni_run_status, cni_progress, cni_live_image, cni_live_result,
                cni_live_counters, cni_live_table,
                cni_results_state, cni_results_table, cni_result_selector,
                cni_accuracy_plot, cni_latency_plot,
            ],
            concurrency_limit=1,
            concurrency_id="cni-benchmark-run",
        )
        cni_stop.click(
            lambda: (
                gr.update(visible=True), gr.update(visible=False),
                _cni_alert_html("warning", "Annulation demandée : l'appel en cours est arrêté dès que le fournisseur rend la main."),
                "Annulation demandée.",
            ),
            outputs=[cni_launch, cni_stop, cni_launch_feedback, cni_run_status],
            queue=False,
            cancels=[cni_event],
        )
        cni_apply_filters.click(
            filter_cni_results,
            inputs=[cni_results_state, cni_accuracy_min, cni_accuracy_max, cni_include_unscored, cni_field_filter, cni_field_state_filter],
            outputs=[cni_results_table], queue=False,
        )
        # L'exploration détaillée reste indépendante du générateur : les
        # boutons restent réactifs pendant l'arrivée des nouveaux résultats.
        cni_detail_outputs = [
            cni_result_selector,
            cni_detail_index,
            cni_result_position,
            cni_recto_preview,
            cni_verso_preview,
            cni_result_identity,
            cni_detail_metrics,
            cni_label_json,
            cni_recto_json,
            cni_verso_json,
            cni_global_json,
            cni_recto_raw,
            cni_verso_raw,
        ]
        cni_previous_result.click(
            show_previous_cni_detail,
            inputs=[cni_detail_index, cni_results_state],
            outputs=cni_detail_outputs,
            queue=False,
        )
        cni_next_result.click(
            show_next_cni_detail,
            inputs=[cni_detail_index, cni_results_state],
            outputs=cni_detail_outputs,
            queue=False,
        )
        cni_result_selector.input(
            select_cni_detail,
            inputs=[cni_result_selector, cni_results_state],
            outputs=cni_detail_outputs,
            queue=False,
        )
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Extensible OCR model benchmark.")
    parser.add_argument("--cli", action="store_true")
    parser.add_argument("--models", nargs="+", default=["mock:MockOCR-V1"])
    parser.add_argument(
        "--category",
        default="All",
        help="Dataset category, or All.",
    )
    parser.add_argument("--noise", type=float, default=0.05)
    parser.add_argument(
        "--eval-mode", default="Standard", choices=["Standard", "Bankmark"]
    )
    parser.add_argument("--host", default=os.getenv("GRADIO_SERVER_NAME", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("GRADIO_SERVER_PORT", "7860")))
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    if args.cli:
        summary, _, run_id = run_benchmark(
            args.models, args.category, args.noise, args.eval_mode
        )
        print(summary.to_string(index=False))
        print(f"Run ID: {run_id}")
        return

    build_ui().launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        ssr_mode=False,
        css=APP_CSS,
    )


if __name__ == "__main__":
    main()
