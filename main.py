from __future__ import annotations

import argparse
import html
import json
import logging
import os
import random
import time
from uuid import uuid4
from pathlib import Path
from typing import Any

import gradio as gr
import pandas as pd

import dataset_generator
from models.ollama_model import DEFAULT_OCR_PROMPT
from ocr_benchmark.application.benchmark_service import (
    iter_benchmark,
    list_ollama_models,
    load_dataset_catalog,
    run_benchmark as execute_benchmark,
    select_dataset_category,
)
from ocr_benchmark.application.cni_service import (
    import_cni_archive,
    iter_cni_extraction,
    scan_cni_documents,
)
from ocr_benchmark.application.qlicker_api_service import (
    editable_rows_to_query_pairs,
    execute_qlicker_get,
    merge_query_params,
    parse_qlicker_url,
    parse_extra_query_params,
)
from ocr_benchmark.application.qlicker_cni_import_service import (
    build_qlicker_cni_routes,
    iter_prepare_qlicker_cni_clients,
)
from ocr_benchmark.application.run_service import list_run_ids, load_run_results, purge_expired_runs
from ocr_benchmark.cni import (
    DEFAULT_RECTO_SUFFIX,
    DEFAULT_VERSO_SUFFIX,
    build_cni_prompt,
    build_combined_cni_prompt,
    load_cni_field_config,
    render_single_page_pdf,
)
from ocr_benchmark.dataset_repository import DatasetRepository
from ocr_benchmark.reporting import RunCheckpoint
from ocr_benchmark.runner import summarize_results
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


def _read_retention_days() -> int | None:
    """Lit la rétention locale des runs ; une valeur négative la désactive."""
    raw_value = os.getenv("RUN_RETENTION_DAYS", "30").strip()
    try:
        return int(raw_value)
    except ValueError:
        logging.getLogger(__name__).warning("RUN_RETENTION_DAYS invalide (%s), rétention désactivée.", raw_value)
        return None


RUN_RETENTION_DAYS = _read_retention_days()

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
#cni-model-selector .wrap {
    min-height: 46px !important;
}
#cni-model-selector button {
    min-width: 34px !important;
}
#cni-model-selector [data-testid="block-info"] {
    margin-top: 4px !important;
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
#cni-api-workspace {
    gap: 10px !important;
}
#cni-api-workspace > .form,
#cni-api-workspace > .block {
    margin: 0 !important;
}
#cni-api-config,
#cni-api-response {
    border-color: var(--block-border-color) !important;
}
#cni-api-parser,
#cni-api-guided {
    padding-top: 2px !important;
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
    """Projette des dossiers locaux dans l'inventaire unique des candidats CNI."""
    return pd.DataFrame([
        {
            "Retenir": True,
            "Client": item.get("folder_client_id"),
            "Identité": "—",
            "Origine": "Dossier local",
            "Recto": "OK" if item.get("recto_pdf") else "Manquant",
            "Verso": "OK" if item.get("verso_pdf") else "Manquant",
            "Label": item.get("label_status", "—"),
            "État": item.get("status", "—"),
            "Détail": ", ".join(item.get("issues") or []) or "—",
        }
        for item in records
    ])


def _qlicker_cni_candidates(customers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Crée une file d'import indépendante de la réponse API brute."""
    return [
        {
            "customer": dict(customer),
            "client_id": str(customer.get("id") or ""),
            "status": "discovered",
            "message": "Client découvert ; documents non récupérés.",
            "issues": [],
        }
        for customer in customers
        if customer.get("id")
    ]


def _cni_api_status_label(status: Any) -> str:
    """Traduit les états internes de la file API pour l'opérateur."""
    labels = {
        "discovered": "Découvert",
        "documents_detected": "Documents détectés",
        "downloaded": "Téléchargé",
        "label_normalized": "Label normalisé",
        "ready": "Prêt",
        "ready_without_label": "Prêt sans label",
        "failed": "Erreur",
    }
    return labels.get(str(status or ""), str(status or "—"))


def _cni_api_document_label(item: dict[str, Any], side: str) -> str:
    """Explique l'avancement d'une face sans le vague « à récupérer »."""
    if item.get(f"{side}_source"):
        return "Téléchargé"
    status = str(item.get("status") or "")
    if status == "documents_detected":
        return "Détecté"
    if status == "failed":
        return "Non disponible"
    return "À vérifier"


def _cni_api_label_label(item: dict[str, Any]) -> str:
    """Distingue un label non demandé, normalisé ou indisponible."""
    if item.get("label_path"):
        return "Normalisé"
    status = str(item.get("status") or "")
    if status == "ready_without_label":
        return "Indisponible"
    if status == "failed":
        return "Non atteint"
    return "À demander"


def _cni_api_table(
    records: list[dict[str, Any]],
    select_all: bool = False,
    selected_client_ids: set[str] | None = None,
) -> pd.DataFrame:
    """Projette GetCustomers dans le même inventaire que les dossiers locaux.

    Une liste clients ne contient pas encore les fichiers CNI : les colonnes
    recto/verso/label restent donc explicitement à ``À récupérer``. Elles
    seront remplacées par les états réels lorsqu'une future étape téléchargera
    puis matérialisera les documents sélectionnés.
    """
    return pd.DataFrame([
        {
            "Retenir": select_all or str(item.get("client_id") or customer.get("id") or "") in (selected_client_ids or set()),
            "Client": str(item.get("client_id") or customer.get("id") or ""),
            "Identité": " ".join(
                part for part in (str(customer.get("last_name") or "").strip(), str(customer.get("first_name") or "").strip()) if part
            ) or "—",
            "Origine": "API QlickEER",
            "Recto": _cni_api_document_label(item, "recto"),
            "Verso": _cni_api_document_label(item, "verso"),
            "Label": _cni_api_label_label(item),
            "État": _cni_api_status_label(item.get("status") or "discovered"),
            "Détail": " · ".join(
                part for part in (
                    str(customer.get("agency_name") or customer.get("agency_code") or "").strip(),
                    str(item.get("message") or customer.get("document_id") or "").strip(),
                    "; ".join(str(issue) for issue in item.get("issues", []) if issue),
                ) if part
            ) or "—",
        }
        for item in records
        for customer in [item.get("customer") if isinstance(item.get("customer"), dict) else item]
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


def _cni_source_mode_visibility(mode: str) -> tuple[Any, Any, Any, Any]:
    """Affiche la source active et le diagnostic utile à cette source."""
    return (
        gr.update(visible=mode == "folder"),
        gr.update(visible=mode == "zip"),
        gr.update(visible=mode == "api"),
        # Les actions de sélection globale ne concernent que la liste issue
        # de l'API ; les dossiers locaux sont tous cochés après le scan.
        gr.update(visible=mode == "api"),
    )


def _qlicker_test_result(
    base_url: str,
    endpoint: str,
    explicit_params: dict[str, Any],
    extra_params_json: str,
    timeout_seconds: float,
    proxy_url: str,
    use_system_proxy: bool,
    verify_ssl: bool,
) -> tuple[str, str]:
    """Teste un GET QlickEER sans téléchargement ni écriture locale.

    Les paramètres guidés sont prioritaires afin qu'un JSON additionnel ne
    remplace pas accidentellement `page`, `customerID` ou `loadDocuments`.
    """
    try:
        extra = parse_extra_query_params(extra_params_json)
        payload = execute_qlicker_get(
            base_url,
            endpoint,
            merge_query_params(explicit_params, extra),
            timeout_seconds=float(timeout_seconds or 30),
            proxy_url=proxy_url,
            use_system_proxy=bool(use_system_proxy),
            verify_ssl=bool(verify_ssl),
        )
        code = int(payload["response"]["status_code"])
        level = "success" if 200 <= code < 300 else "warning"
        tls = "SSL vérifié" if verify_ssl else "SSL non vérifié"
        message = f"GET terminé : HTTP {code} · {tls}. Aucun document n'a été enregistré localement."
        return _cni_alert_html(level, message), _qlicker_trace_preview(payload)
    except Exception as exc:
        LOGGER.exception("QlickEER API test failed")
        return _cni_alert_html("error", f"Test API impossible : {type(exc).__name__}: {exc}"), ""


def _qlicker_trace_preview(payload: dict[str, Any], limit: int = 50_000) -> str:
    """Protège le navigateur contre l'affichage d'une réponse API trop volumineuse.

    L'appelant conserve ``payload`` complet pour ses traitements métier ; seul
    le panneau de diagnostic Gradio est tronqué de manière explicite.
    """
    rendered = json.dumps(payload, ensure_ascii=False, indent=2, default=str)
    if len(rendered) <= limit:
        return rendered
    return (
        rendered[:limit]
        + f"\n\n… aperçu tronqué à {limit:,} caractères pour préserver l'interface "
        f"(réponse complète : {len(rendered):,} caractères)."
    )


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


def _cni_confidence_summary(result: dict[str, Any]) -> str:
    """Présente la confiance QlickEER par champ à côté du verdict OCR."""
    label = _read_json_if_available(result.get("label_path"))
    confidence = label.get("field_confidence", {}) if isinstance(label, dict) else {}
    if not isinstance(confidence, dict) or not confidence:
        return ""
    comparisons = _cni_field_comparisons(result)
    rows = [
        f"| `{field}` | {comparisons.get(field, '—')} | {float(value):.1f} % |"
        for field, value in confidence.items()
        if isinstance(value, (int, float))
    ]
    if not rows:
        return ""
    return "\n\n**Confiance du label QlickEER et comparaison OCR**\n\n| Champ | Comparaison | Confiance label |\n|---|---|---|\n" + "\n".join(rows)


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
    """Façade de compatibilité UI vers le cas d'usage dataset réutilisable."""
    return load_dataset_catalog(ROOT_DIR, CATALOG_PATH, ensure_catalog=dataset_generator.main)


def get_installed_ollama_models() -> list[str]:
    """Façade UI vers le service d'inventaire Ollama."""
    return list_ollama_models()


def run_benchmark(
    selected_models: list[str],
    selected_category: str = "All",
    mock_noise: float = 0.05,
    eval_mode: str = "Standard",
    cpu_threads: int | None = None,
    unload_after_task: bool = True,
) -> tuple[pd.DataFrame, list[dict[str, Any]], str]:
    """Façade CLI vers le cas d'usage benchmark réutilisable.

    Args:
        selected_models: Identifiants des adaptateurs à exécuter.
        selected_category: Catégorie du dataset ou ``All``.
        mock_noise: Bruit réservé aux modèles simulés.
        eval_mode: Méthode de score du benchmark.
        cpu_threads: Limite CPU transmise aux adaptateurs compatibles.
        unload_after_task: Libère le modèle entre deux tâches si possible.
    """
    dataset = select_dataset_category(load_dataset(), selected_category)
    return execute_benchmark(
        selected_models,
        dataset,
        RUNS_DIR,
        eval_mode=eval_mode,
        mock_noise=mock_noise,
        cpu_threads=cpu_threads,
        unload_after_task=unload_after_task,
    )


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

    def available_run_choices() -> list[tuple[str, str]]:
        """Return persisted runs newest first without loading their payloads."""
        return [
            (run_id, run_id)
            for run_id in list_run_ids(RUNS_DIR)
            if (RUNS_DIR / run_id / "results.json").is_file()
        ]

    def startup_run_info():
        """Keep page startup light; payloads are loaded only on explicit open."""
        deleted = purge_expired_runs(RUNS_DIR, retention_days=RUN_RETENTION_DAYS)
        if deleted:
            LOGGER.info("Run retention applied | deleted=%d | retention_days=%s", len(deleted), RUN_RETENTION_DAYS)
        choices = available_run_choices()
        if not choices:
            return gr.update(choices=[], value=None), "Aucun run sauvegardé."
        return gr.update(choices=choices, value=choices[0][1]), f"Dernier run disponible : `{choices[0][1]}`. Cliquez sur **Ouvrir le run**."

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
        # La liste complète permet de recalculer une sélection locale sans
        # perdre les dossiers écartés temporairement dans l'interface.
        cni_all_clients_state = gr.State([])
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
                                    [("Dossier local", "folder"), ("Archive ZIP", "zip"), ("API QlickEER", "api")],
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
                                # Ancienne zone technique conservée temporairement hors écran : les
                                # composants seront retirés après migration complète des routes API.
                                with gr.Group(visible=False, elem_id="cni-api-legacy") as cni_api_legacy_config:
                                    gr.Markdown("**API QlickEER · lecture seule** — configurez la connexion, choisissez une méthode d'appel, puis consultez la réponse. Aucun document n'est enregistré.")
                                    with gr.Accordion("1 · Connexion", open=True, elem_id="cni-api-config"):
                                        with gr.Row():
                                            cni_api_base_url = gr.Textbox(label="Base URL commune", placeholder="http://serveur-interne/api", scale=3)
                                            cni_api_timeout = gr.Number(value=30, minimum=1, precision=0, label="Timeout (s)", scale=1)
                                            cni_api_proxy = gr.Textbox(label="Proxy explicite", type="password", placeholder="http://proxy.interne.local:8080", info="Optionnel ; utilisé pour tous les tests et jamais enregistré.", scale=3)
                                            cni_api_use_system_proxy = gr.Checkbox(label="Utiliser le proxy système", value=False, info="Lit le proxy configuré sur ce poste Windows si aucun proxy explicite n'est saisi.", scale=1)
                                            cni_api_verify_ssl = gr.Checkbox(label="Vérifier SSL", value=True, info="Décochez seulement pour un certificat interne connu.", scale=1)
                                    gr.Markdown("**2 · Construire l'appel**")
                                    cni_api_call_mode = gr.Radio(
                                        [("Coller une URL depuis Postman", "parser"), ("Utiliser une route guidée", "guided")],
                                        value="parser", label="Méthode",
                                    )
                                    with gr.Group(visible=True, elem_id="cni-api-parser") as cni_api_parser_group:
                                        gr.Markdown("Collez l’URL, parsez-la, puis modifiez uniquement les paramètres utiles.")
                                        with gr.Row():
                                            cni_api_raw_url = gr.Textbox(label="URL complète", placeholder="https://serveur-interne/api/get_signed_documents_list?customerID=123", scale=6)
                                            cni_api_parse_url = gr.Button("Parser", variant="secondary", scale=1)
                                        with gr.Row():
                                            cni_api_parsed_endpoint = gr.Textbox(label="Endpoint", placeholder="get_signed_documents_list", info="Prérempli par le parser ; modifiable avant le test.", scale=2)
                                            cni_api_test_parsed = gr.Button("Exécuter le GET", variant="primary", scale=1)
                                        cni_api_parsed_params = gr.Dataframe(headers=["Paramètre", "Valeur", "Envoyer"], datatype=["str", "str", "bool"], row_count=(1, "dynamic"), column_count=(3, "fixed"), type="array", interactive=True, label="Paramètres")
                                        gr.Markdown("Décochez **Envoyer** pour omettre un paramètre ; une valeur vide envoie `paramètre=`.")
                                    with gr.Group(visible=False, elem_id="cni-api-guided") as cni_api_guided_group:
                                        cni_api_guided_route = gr.Dropdown(
                                            [("Liste clients", "list"), ("Info client", "info"), ("Liste documents", "documents"), ("Voir fichier", "view")],
                                            value="list", label="Route guidée",
                                        )
                                        with gr.Group(visible=True) as cni_api_list_group:
                                            cni_api_list_endpoint = gr.Textbox(label="Segment endpoint / fonction", placeholder="Ex. GetCustomers")
                                            with gr.Row():
                                                cni_api_from_date = gr.Textbox(label="from_date", placeholder="YYYY-MM-DD")
                                                cni_api_to_date = gr.Textbox(label="to_date", placeholder="YYYY-MM-DD")
                                                cni_api_step = gr.Textbox(label="step", placeholder="HH:MM:SS")
                                                cni_api_page = gr.Number(value=1, minimum=1, precision=0, label="page")
                                                cni_api_page_size = gr.Number(value=20, minimum=1, precision=0, label="pageSize")
                                            cni_api_list_extra = gr.Textbox(label="Autres paramètres — JSON", lines=3, placeholder='Ex. {"sort": null, "status": ""}')
                                            cni_api_test_list = gr.Button("Exécuter GET liste clients", variant="primary")
                                            cni_api_customers_state = gr.State([])
                                            gr.Markdown("**3 · Sélection des clients**")
                                            with gr.Row():
                                                cni_api_select_all = gr.Button("Tout sélectionner", size="sm")
                                                cni_api_clear_selection = gr.Button("Tout désélectionner", size="sm")
                                                cni_api_selected_summary = gr.Markdown("Aucun client chargé.")
                                            cni_api_customers_table = gr.Dataframe(
                                                headers=["Sélectionner", "ID client", "Nom", "Prénom", "Agence", "Statut", "Création", "Document"],
                                                datatype=["bool", "str", "str", "str", "str", "str", "str", "str"],
                                                interactive=True,
                                                label="Clients trouvés",
                                                max_height=300,
                                            )
                                            gr.Markdown("Les documents seront ajoutés directement au diagnostic CNI après validation du contrat `get_signed_documents_list`.")
                                        with gr.Group(visible=False) as cni_api_info_group:
                                            cni_api_info_endpoint = gr.Textbox(label="Segment endpoint / fonction", placeholder="Ex. GetCustomerData")
                                            with gr.Row():
                                                cni_api_customer_id = gr.Textbox(label="customerID", placeholder="Identifiant client")
                                                cni_api_load_documents = gr.Radio([("0 — sans documents", 0), ("1 — charger les documents", 1)], value=0, label="loadDocuments")
                                            cni_api_info_extra = gr.Textbox(label="Autres paramètres — JSON", lines=3, placeholder='Ex. {"includeHistory": null}')
                                            cni_api_test_info = gr.Button("Exécuter GET info client", variant="primary")
                                        with gr.Group(visible=False) as cni_api_documents_group:
                                            cni_api_documents_endpoint = gr.Textbox(label="Segment endpoint / fonction", value="get_signed_documents_list")
                                            cni_api_documents_customer_id = gr.Textbox(label="customerID", placeholder="Identifiant client")
                                            cni_api_documents_extra = gr.Textbox(label="Autres paramètres — JSON", lines=3, placeholder='Ex. {"documentType": "CIN"}')
                                            cni_api_test_documents = gr.Button("Exécuter GET liste documents", variant="primary")
                                        with gr.Group(visible=False) as cni_api_view_group:
                                            cni_api_view_endpoint = gr.Textbox(label="Segment endpoint / fonction", value="view_file")
                                            with gr.Row():
                                                cni_api_view_customer_id = gr.Textbox(label="customerID", placeholder="Identifiant client")
                                                cni_api_view_page = gr.Number(value=1, minimum=1, precision=0, label="page")
                                                cni_api_view_file = gr.Textbox(label="file", placeholder="Nom ou identifiant de fichier")
                                            cni_api_view_extra = gr.Textbox(label="3 autres paramètres — JSON", lines=3, placeholder='Ex. {"param4": "", "param5": null, "param6": "valeur"}')
                                            cni_api_test_view = gr.Button("Exécuter GET voir fichier", variant="primary")
                                    with gr.Accordion("3 · Réponse du dernier GET", open=False, elem_id="cni-api-response"):
                                        cni_api_feedback = gr.HTML(_cni_alert_html("ready", "Prêt : configurez une méthode puis exécutez un GET."))
                                        cni_api_trace = gr.Code(label="Requête et réponse", language="json", lines=12, interactive=False)
                                with gr.Group(visible=False, elem_id="cni-api-source") as cni_api_source:
                                    gr.Markdown("**API QlickEER** · la connexion et les routes se configurent dans `4. Paramètres → API QlickEER`.")
                                    cni_api_connection_status = gr.Markdown("Configuration requise : renseignez d'abord la route **Clients** dans les paramètres.")
                                    with gr.Row():
                                        cni_api_source_from_date = gr.Textbox(label="from_date", placeholder="YYYY-MM-DD")
                                        cni_api_source_to_date = gr.Textbox(label="to_date", placeholder="YYYY-MM-DD")
                                        cni_api_source_step = gr.Textbox(label="step", placeholder="HH:MM:SS")
                                        cni_api_source_page = gr.Number(value=1, minimum=1, precision=0, label="page")
                                        cni_api_source_page_size = gr.Number(value=20, minimum=1, precision=0, label="pageSize")
                                    cni_api_load_customers = gr.Button("Rechercher les clients", variant="primary")
                                    cni_api_source_customers_state = gr.State([])
                                    gr.Markdown("Les résultats de recherche apparaissent dans **Diagnostic de la source**. Sélectionnez ensuite les clients à préparer.")
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
                                # Le bouton est rendu dans la barre du sélecteur : il ne prend donc
                                # pas une ligne entière pour une action ponctuelle d'actualisation.
                                cni_refresh_models = gr.Button(
                                    # Le caractère hérite naturellement de la couleur du thème,
                                    # contrairement à une icône SVG avec une couleur figée.
                                    value="↻",
                                    size="sm",
                                    elem_id="cni-refresh-models",
                                    render=False,
                                )
                                cni_models = gr.Dropdown(
                                    [choice for choice in model_choices if choice.startswith("ollama:")],
                                    value=[],
                                    multiselect=True,
                                    filterable=True,
                                    label="Modèles Ollama",
                                    info="Recherchez un modèle puis sélectionnez-en un ou plusieurs. Les tags sont supprimables avec × ; l’exécution reste strictement séquentielle.",
                                    buttons=[cni_refresh_models],
                                    elem_id="cni-model-selector",
                                )
                                with gr.Accordion("Diagnostic de la source", open=False):
                                    gr.Markdown("Un seul inventaire pour tous les candidats. L’origine indique si la paire vient d’un dossier local, d’un ZIP ou de l’API.")
                                    with gr.Group(visible=False) as cni_api_inventory_actions:
                                        with gr.Row():
                                            cni_api_source_select_all = gr.Button("Tout sélectionner", size="sm")
                                            cni_api_source_clear_selection = gr.Button("Tout désélectionner", size="sm")
                                            cni_api_prepare_selected = gr.Button("Préparer la sélection", variant="primary", size="sm")
                                        cni_api_import_progress = gr.Markdown("Aucun client API en préparation.")
                                    with gr.Row():
                                        cni_source_selection_summary = gr.Markdown("Aucune source chargée.")
                                    cni_source_inventory_table = gr.Dataframe(
                                        headers=["Retenir", "Client", "Identité", "Origine", "Recto", "Verso", "Label", "État", "Détail"],
                                        datatype=["bool", "str", "str", "str", "str", "str", "str", "str", "str"],
                                        interactive=True,
                                        label="Candidats et diagnostic CNI",
                                        max_height=340,
                                    )
                                    with gr.Accordion("Détail du dernier appel API", open=False):
                                        cni_api_source_feedback = gr.HTML(_cni_alert_html("ready", "Prêt : configurez la route Clients puis recherchez."))
                                        cni_api_source_trace = gr.Code(label="Réponse technique", language="json", lines=10, interactive=False)
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
                        gr.Markdown("### Paramètres CNI\n\nLes réglages sont appliqués au prochain lancement.")
                        with gr.Tabs(elem_id="cni-settings-tabs"):
                            with gr.Tab("Exécution"):
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
                                with gr.Row():
                                    cni_recto_suffix = gr.Textbox(value=DEFAULT_RECTO_SUFFIX, label="Suffixe recto")
                                    cni_verso_suffix = gr.Textbox(value=DEFAULT_VERSO_SUFFIX, label="Suffixe verso")
                            with gr.Tab("Prétraitement"):
                                with gr.Row():
                                    cni_rotation_method = gr.Radio(
                                [("Aucune rotation automatique", "none"), ("Pillow · recherche par ratio", "pillow"), ("OpenCV · rectangle orienté", "opencv")],
                                value="none", label="Rotation automatique",
                                info="Une seule méthode peut être activée. Pillow cherche l'angle, OpenCV utilise minAreaRect.",
                            )
                                    cni_perspective_correction = gr.Checkbox(
                                value=False, label="Corriger la perspective (OpenCV)",
                                info="Redresse la carte seulement si un quadrilatère crédible est détecté.",
                            )
                                cni_preprocessing = gr.CheckboxGroup(
                            [("Améliorer le contraste", "contrast"), ("Réduire le bruit", "denoise")],
                            value=[], label="Améliorations complémentaires",
                            info="Appliquées après rotation puis avant crop. Chaque opération est enregistrée dans preparation.json.",
                        )
                            with gr.Tab("Prompt et champs"):
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
                            with gr.Tab("API QlickEER"):
                                gr.Markdown("Configurez chaque route une fois avec son URL Postman complète. Les paramètres parsés sont conservés pour la session ; aucun proxy explicite n'est sauvegardé.")
                                with gr.Tabs():
                                    with gr.Tab("Connexion"):
                                        with gr.Row():
                                            cni_api_settings_base_url = gr.Textbox(label="Base URL commune", placeholder="https://serveur-interne")
                                            cni_api_settings_timeout = gr.Number(value=30, minimum=1, precision=0, label="Timeout (s)")
                                            cni_api_settings_use_system_proxy = gr.Checkbox(value=False, label="Utiliser le proxy système")
                                            cni_api_settings_verify_ssl = gr.Checkbox(value=True, label="Vérifier SSL")
                                        cni_api_settings_proxy = gr.Textbox(label="Proxy explicite", type="password", placeholder="http://ncproxy:8080", info="À renseigner seulement si le proxy système est désactivé.")
                                        cni_api_import_root = gr.Textbox(
                                            value=str(CNI_IMPORTS_DIR / "qlickeer_api"),
                                            label="Dossier d'import API",
                                            info="Un sous-dossier horodaté est créé par préparation de lot.",
                                        )
                                    with gr.Tab("Clients"):
                                        cni_api_list_raw_url = gr.Textbox(label="URL Postman · liste clients", placeholder="https://serveur/api/get_customer?...", lines=2)
                                        cni_api_list_parse = gr.Button("Parser et enregistrer la route Clients")
                                        cni_api_list_endpoint_setting = gr.Textbox(label="Endpoint", interactive=False)
                                        cni_api_list_params_setting = gr.Dataframe(headers=["Paramètre", "Valeur", "Envoyer"], datatype=["str", "str", "bool"], type="array", interactive=True, label="Paramètres parsés")
                                    with gr.Tab("Détail client"):
                                        cni_api_info_raw_url = gr.Textbox(label="URL Postman · get_customer_data", lines=2)
                                        cni_api_info_parse = gr.Button("Parser et enregistrer la route Détail client")
                                        cni_api_info_endpoint_setting = gr.Textbox(label="Endpoint", interactive=False)
                                        cni_api_info_params_setting = gr.Dataframe(headers=["Paramètre", "Valeur", "Envoyer"], datatype=["str", "str", "bool"], type="array", interactive=True, label="Paramètres parsés")
                                    with gr.Tab("Documents"):
                                        cni_api_documents_raw_url = gr.Textbox(label="URL Postman · get_signed_documents_list", lines=2)
                                        cni_api_documents_parse = gr.Button("Parser et enregistrer la route Documents")
                                        cni_api_documents_endpoint_setting = gr.Textbox(label="Endpoint", interactive=False)
                                        cni_api_documents_params_setting = gr.Dataframe(headers=["Paramètre", "Valeur", "Envoyer"], datatype=["str", "str", "bool"], type="array", interactive=True, label="Paramètres parsés")
                                    with gr.Tab("Fichier"):
                                        cni_api_view_raw_url = gr.Textbox(label="URL Postman · view_file", lines=2)
                                        cni_api_view_parse = gr.Button("Parser et enregistrer la route Fichier")
                                        cni_api_view_endpoint_setting = gr.Textbox(label="Endpoint", interactive=False)
                                        cni_api_view_params_setting = gr.Dataframe(headers=["Paramètre", "Valeur", "Envoyer"], datatype=["str", "str", "bool"], type="array", interactive=True, label="Paramètres parsés")

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

            case_count = len(selected_records)
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
                f"**Traité :** 0 / {len(model_specs) * case_count} · **ETA :** calcul en cours",
                pd.DataFrame(),
            )

            try:
                updates = iter_benchmark(
                    model_specs,
                    selected_records,
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

        def reload_persisted_runs():
            choices = available_run_choices()
            return gr.update(choices=choices, value=(choices[0][1] if choices else None))

        def open_persisted_run(run_id):
            """Load a run from disk and feed it through the normal detail renderer."""
            if not run_id:
                return [[], *show_detail(0, []), "Aucun run sélectionné."]
            try:
                safe_id = str(run_id)
                restored = load_run_results(RUNS_DIR, safe_id)
                return [restored, *show_detail(0, restored), f"✅ Run `{safe_id}` rechargé ({len(restored)} résultat(s))."]
            except Exception as exc:
                return [[], *show_detail(0, []), f"❌ Impossible de recharger `{safe_id}` : {type(exc).__name__}: {exc}"]

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
                imported = import_cni_archive(Path(zip_path), CNI_IMPORTS_DIR)
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

        def qlickeer_call_mode_visibility(mode: str):
            """N'affiche que la méthode d'appel API actuellement choisie."""
            return gr.update(visible=mode == "parser"), gr.update(visible=mode == "guided")

        def qlickeer_guided_route_visibility(route: str):
            """N'affiche que les champs de la route guidée active."""
            return tuple(gr.update(visible=route == name) for name in ("list", "info", "documents", "view"))

        def parse_qlickeer_route(raw_url: str):
            """Transforme une URL Postman complète en base, endpoint et paramètres éditables."""
            try:
                base_url, endpoint, rows = parse_qlicker_url(raw_url)
                return base_url, endpoint, rows
            except ValueError as exc:
                LOGGER.warning("QlickEER route parse rejected | error=%s", exc)
                return gr.update(), gr.update(), []

        def load_configured_customers(base_url, endpoint, route_rows, from_date, to_date, step, page, page_size, timeout, proxy_url, use_system_proxy, verify_ssl):
            """Charge les clients API dans l'inventaire commun de la source.

            La requête reste identique à celle du laboratoire API historique,
            mais son rendu est converti dans le tableau unique utilisé aussi
            par un scan de dossiers locaux.
            """
            static = dict(editable_rows_to_query_pairs(route_rows))
            feedback, trace, _legacy_table, records, _summary = test_qlicker_list(
                base_url, endpoint, from_date, to_date, step, page, page_size,
                json.dumps(static, ensure_ascii=False), timeout, proxy_url,
                use_system_proxy, verify_ssl,
            )
            candidates = _qlicker_cni_candidates(records)
            table = _cni_api_table(candidates)
            return (
                feedback,
                trace,
                table,
                candidates,
                api_inventory_summary(table, candidates),
            )

        def parse_qlicker_url_for_ui(raw_url):
            """Remplit l'espace de travail éditable à partir d'une URL collée.

            Cette action ne fait aucun appel réseau : elle découpe seulement
            l'URL afin que l'utilisateur contrôle chaque paramètre avant GET.
            """
            try:
                base_url, endpoint, rows = parse_qlicker_url(raw_url)
                return (
                    gr.update(value=base_url),
                    gr.update(value=endpoint),
                    rows,
                    _cni_alert_html("success", f"URL analysée : {len(rows)} paramètre(s) modifiable(s) avant l'appel."),
                )
            except ValueError as exc:
                return gr.update(), gr.update(), [], _cni_alert_html("error", str(exc))

        def test_qlicker_parsed(base_url, endpoint, rows, timeout, proxy_url, use_system_proxy, verify_ssl):
            """Teste les paramètres issus du parser, dans leur ordre édité."""
            try:
                pairs = editable_rows_to_query_pairs(rows)
                payload = execute_qlicker_get(
                    base_url,
                    endpoint,
                    pairs,
                    timeout_seconds=float(timeout or 30),
                    proxy_url=proxy_url,
                    use_system_proxy=bool(use_system_proxy),
                    verify_ssl=bool(verify_ssl),
                )
                code = int(payload["response"]["status_code"])
                level = "success" if 200 <= code < 300 else "warning"
                tls = "SSL vérifié" if verify_ssl else "SSL non vérifié"
                return (
                    _cni_alert_html(level, f"GET analysé terminé : HTTP {code} · {tls}. Aucun fichier n'a été enregistré."),
                    _qlicker_trace_preview(payload),
                )
            except Exception as exc:
                LOGGER.exception("QlickEER parsed URL test failed")
                return _cni_alert_html("error", f"Test API impossible : {type(exc).__name__}: {exc}"), ""

        def _customer_table(records: list[dict[str, Any]], select_all: bool = False) -> pd.DataFrame:
            """Projette la réponse GetCustomers vers les seules colonnes utiles à l'opérateur."""
            return pd.DataFrame([
                {
                    "Sélectionner": select_all,
                    "ID client": str(customer.get("id") or ""),
                    "Nom": str(customer.get("last_name") or ""),
                    "Prénom": str(customer.get("first_name") or ""),
                    "Agence": str(customer.get("agency_name") or customer.get("agency_code") or ""),
                    "Statut": str(customer.get("status") or ""),
                    "Création": str(customer.get("creation_date") or ""),
                    "Document": str(customer.get("document_id") or ""),
                }
                for customer in records
            ])

        def _customer_selection_summary(table: Any, total: int | list[dict[str, Any]]) -> str:
            """Compte les lignes cochées sans exposer le JSON complet des clients."""
            rows = table.values.tolist() if isinstance(table, pd.DataFrame) else (table or [])
            selected = sum(1 for row in rows if row and bool(row[0]))
            total_count = len(total) if isinstance(total, list) else total
            return f"**{selected} client(s) sélectionné(s) sur {total_count}.**"

        def api_inventory_summary(table: Any, candidates: list[dict[str, Any]]) -> str:
            """Affiche les volumes utiles de la file de préparation API."""
            rows = table.values.tolist() if isinstance(table, pd.DataFrame) else (table or [])
            selected = sum(1 for row in rows if row and bool(row[0]))
            counts: dict[str, int] = {}
            for candidate in candidates or []:
                status = str(candidate.get("status") or "discovered")
                counts[status] = counts.get(status, 0) + 1
            ready = counts.get("ready", 0) + counts.get("ready_without_label", 0)
            waiting = sum(
                counts.get(status, 0)
                for status in ("discovered", "documents_detected", "downloaded", "label_normalized")
            )
            return (
                f"**Total API :** {len(candidates or [])} · "
                f"**Sélection :** {selected} · "
                f"**Prêts :** {ready} · **En attente :** {waiting} · "
                f"**Erreurs :** {counts.get('failed', 0)}"
            )

        def test_qlicker_list(base_url, endpoint, from_date, to_date, step, page, page_size, extra_json, timeout, proxy_url, use_system_proxy, verify_ssl):
            """Charge les clients et rend leur sélection possible, sans importer de document."""
            try:
                params = merge_query_params(
                    {
                        "from_date": str(from_date or "").strip() or None,
                        "to_date": str(to_date or "").strip() or None,
                        "step": str(step or "").strip() or None,
                        "page": int(page or 1),
                        "pageSize": int(page_size or 20),
                    },
                    parse_extra_query_params(extra_json),
                )
                payload = execute_qlicker_get(
                    base_url, endpoint, params,
                    timeout_seconds=float(timeout or 30), proxy_url=proxy_url,
                    use_system_proxy=bool(use_system_proxy), verify_ssl=bool(verify_ssl),
                )
                body = payload.get("response", {}).get("body", {})
                response_data = body.get("response_data", {}) if isinstance(body, dict) else {}
                customers = response_data.get("customers_found", []) if isinstance(response_data, dict) else []
                if not isinstance(customers, list):
                    customers = []
                records = [item for item in customers if isinstance(item, dict) and item.get("id")]
                code = int(payload["response"]["status_code"])
                level = "success" if 200 <= code < 300 else "warning"
                declared_total = response_data.get("total_customer", len(records)) if isinstance(response_data, dict) else len(records)
                message = f"Liste reçue : HTTP {code} · {len(records)} client(s) affiché(s) · total API : {declared_total}."
                return (
                    _cni_alert_html(level, message), _qlicker_trace_preview(payload),
                    _customer_table(records), records,
                    _customer_selection_summary([], len(records)),
                )
            except Exception as exc:
                LOGGER.exception("QlickEER customer list failed")
                return (
                    _cni_alert_html("error", f"Liste clients impossible : {type(exc).__name__}: {exc}"), "",
                    pd.DataFrame(), [], "Aucun client chargé.",
                )

        def select_all_customers(records: list[dict[str, Any]]):
            table = _customer_table(records or [], select_all=True)
            return table, _customer_selection_summary(table, len(records or []))

        def clear_customer_selection(records: list[dict[str, Any]]):
            table = _customer_table(records or [], select_all=False)
            return table, _customer_selection_summary(table, len(records or []))

        def select_all_api_source(records: list[dict[str, Any]]):
            """Coche tous les candidats provenant de la dernière liste API."""
            table = _cni_api_table(records or [], select_all=True)
            return table, api_inventory_summary(table, records or [])

        def clear_api_source_selection(records: list[dict[str, Any]]):
            """Décoche les candidats API sans effacer la réponse de l'appel."""
            table = _cni_api_table(records or [], select_all=False)
            return table, api_inventory_summary(table, records or [])

        def update_cni_source_selection(mode: str, table: Any, local_records: list[dict[str, Any]], api_records: list[dict[str, Any]]):
            """Met à jour la sélection locale ou compte celle de la liste API.

            Les clients API ne sont pas encore des paires prêtes : le tableau
            conserve donc leur sélection pour l'étape de récupération future,
            sans les injecter prématurément dans le runner CNI local.
            """
            rows = table.values.tolist() if isinstance(table, pd.DataFrame) else (table or [])
            if mode == "api":
                return gr.update(), api_inventory_summary(rows, api_records or [])
            selected_records = [
                record for index, record in enumerate(local_records or [])
                if index < len(rows) and rows[index] and bool(rows[index][0])
            ]
            return selected_records, f"**{len(selected_records)} dossier(s) retenu(s) sur {len(local_records or [])}.**"

        def prepare_selected_qlicker_clients(
            table: Any,
            candidates: list[dict[str, Any]],
            import_root_text: str,
            base_url: str,
            customer_endpoint: str,
            customer_rows: Any,
            documents_endpoint: str,
            documents_rows: Any,
            file_endpoint: str,
            file_rows: Any,
            timeout: float,
            proxy_url: str,
            use_system_proxy: bool,
            verify_ssl: bool,
            recto_suffix: str,
            verso_suffix: str,
        ):
            """Prépare séquentiellement les clients API cochés dans l'inventaire.

            Chaque ``yield`` met à jour une seule ligne logique dans la file.
            Les fichiers et labels sont ensuite rescannés avec le même contrat
            que les dossiers locaux, afin que le runner CNI n'ait aucun cas API
            spécial à connaître.
            """
            rows = table.values.tolist() if isinstance(table, pd.DataFrame) else (table or [])
            selected = [
                candidate for index, candidate in enumerate(candidates or [])
                if index < len(rows) and rows[index] and bool(rows[index][0])
            ]
            if not selected:
                yield (
                    gr.update(), gr.update(), "**Aucun client API sélectionné.**",
                    "Aucune préparation lancée.", gr.update(), gr.update(), gr.update(),
                    _cni_alert_html("warning", "Sélectionnez au moins un client API."),
                )
                return
            try:
                routes = build_qlicker_cni_routes(
                    customer_endpoint, customer_rows,
                    documents_endpoint, documents_rows,
                    file_endpoint, file_rows,
                )
                raw_root = str(import_root_text or "").strip()
                if not raw_root:
                    raise ValueError("Le dossier d'import API est obligatoire.")
                base_root = Path(raw_root).expanduser()
                # L'horodatage rend le dossier lisible ; le suffixe évite toute
                # collision si deux lots sont lancés dans la même seconde.
                batch_root = base_root / f"batch-{time.strftime('%Y%m%d-%H%M%S')}-{uuid4().hex[:8]}"
                batch_root.mkdir(parents=True, exist_ok=False)
            except Exception as exc:
                message = f"Configuration API incomplète : {type(exc).__name__}: {exc}"
                yield (
                    gr.update(), gr.update(), f"**{message}**", message,
                    gr.update(), gr.update(), gr.update(), _cni_alert_html("error", message),
                )
                return

            working = [dict(candidate) for candidate in (candidates or [])]
            selected_client_ids = {
                str(candidate.get("client_id") or candidate.get("customer", {}).get("id") or "")
                for candidate in selected
            }
            completed = 0
            final_statuses = {"ready", "ready_without_label", "failed"}
            try:
                for event in iter_prepare_qlicker_cni_clients(
                    selected,
                    batch_root,
                    base_url=base_url,
                    routes=routes,
                    timeout_seconds=float(timeout or 30),
                    proxy_url=proxy_url,
                    use_system_proxy=bool(use_system_proxy),
                    verify_ssl=bool(verify_ssl),
                    recto_suffix=str(recto_suffix or DEFAULT_RECTO_SUFFIX),
                    verso_suffix=str(verso_suffix or DEFAULT_VERSO_SUFFIX),
                ):
                    client_id = str(event.get("client_id") or "")
                    for position, candidate in enumerate(working):
                        if str(candidate.get("client_id") or candidate.get("customer", {}).get("id") or "") == client_id:
                            working[position] = {**candidate, **event}
                            break
                    if event.get("status") in final_statuses:
                        completed += 1
                    progress = (
                        f"**Préparation API :** {completed} / {len(selected)} terminé(s) · "
                        f"client `{client_id}` : `{event.get('status')}`."
                    )
                    yield (
                        _cni_api_table(working, selected_client_ids=selected_client_ids), working, progress,
                        api_inventory_summary(_cni_api_table(working, selected_client_ids=selected_client_ids), working),
                        gr.update(), gr.update(), gr.update(),
                        _cni_alert_html("ready", str(event.get("message") or "Préparation API en cours.")),
                    )
            except Exception as exc:
                LOGGER.exception("QlickEER batch preparation failed")
                message = f"Préparation API interrompue : {type(exc).__name__}: {exc}"
                yield (
                    _cni_api_table(working, selected_client_ids=selected_client_ids), working, f"**{message}**",
                    api_inventory_summary(_cni_api_table(working, selected_client_ids=selected_client_ids), working), gr.update(), gr.update(), gr.update(),
                    _cni_alert_html("error", message),
                )
                return

            # Le scanner local est réutilisé : les documents importés deviennent
            # des entrées CNI ordinaires et le benchmark reste découplé de l'API.
            records = scan_cni_documents(
                batch_root,
                None,
                recto_suffix=str(recto_suffix or DEFAULT_RECTO_SUFFIX),
                verso_suffix=str(verso_suffix or DEFAULT_VERSO_SUFFIX),
            )
            ready = sum(record.get("status") == "ready" for record in records)
            labels = sum(record.get("label_status") == "label_materialized" for record in records)
            summary = (
                f"**Lot API terminé :** {len(records)} client(s) matérialisé(s), "
                f"{ready} paire(s) prête(s), {labels} label(s) normalisé(s)."
            )
            yield (
                _cni_api_table(working, selected_client_ids=selected_client_ids), working, summary,
                api_inventory_summary(_cni_api_table(working, selected_client_ids=selected_client_ids), working), records, records,
                gr.update(choices=_cni_source_choices(records), value=None),
                _cni_alert_html("success", f"{summary} Dossier : `{batch_root}`."),
            )

        def test_qlicker_info(base_url, endpoint, customer_id, load_documents, extra_json, timeout, proxy_url, use_system_proxy, verify_ssl):
            """Teste l'endpoint d'information client sans supposer sa réponse JSON."""
            return _qlicker_test_result(
                base_url,
                endpoint,
                {"customerID": str(customer_id or "").strip() or None, "loadDocuments": int(load_documents or 0)},
                extra_json,
                float(timeout or 30),
                proxy_url,
                bool(use_system_proxy),
                bool(verify_ssl),
            )

        def test_qlicker_documents(base_url, endpoint, customer_id, extra_json, timeout, proxy_url, use_system_proxy, verify_ssl):
            """Teste la liste distante des documents signés d'un client."""
            return _qlicker_test_result(
                base_url,
                endpoint,
                {"customerID": str(customer_id or "").strip() or None},
                extra_json,
                float(timeout or 30),
                proxy_url,
                bool(use_system_proxy),
                bool(verify_ssl),
            )

        def test_qlicker_view(base_url, endpoint, customer_id, page, file_name, extra_json, timeout, proxy_url, use_system_proxy, verify_ssl):
            """Teste le retour d'un fichier sans le persister dans le benchmark."""
            return _qlicker_test_result(
                base_url,
                endpoint,
                {
                    "customerID": str(customer_id or "").strip() or None,
                    "page": int(page or 1),
                    "file": str(file_name or "").strip() or None,
                },
                extra_json,
                float(timeout or 30),
                proxy_url,
                bool(use_system_proxy),
                bool(verify_ssl),
            )

        def scan_cni_input(clients_root_text, labels_root_text, recto_suffix, verso_suffix):
            """Scanne les dossiers et met à jour l'état CNI et les aperçus."""
            if not clients_root_text or not str(clients_root_text).strip():
                return [], [], pd.DataFrame(), "Scan impossible : indiquez le dossier clients.", gr.update(choices=_cni_source_choices([]), value=None), "Aucune source chargée."
            try:
                clients_root = Path(str(clients_root_text).strip()).expanduser()
                labels_root = Path(str(labels_root_text).strip()).expanduser() if labels_root_text and str(labels_root_text).strip() else None
                if labels_root is not None and not labels_root.is_dir():
                    LOGGER.warning("CNI scan rejected | labels_root_not_found=%s", labels_root)
                    return [], [], pd.DataFrame(), f"Dossier labels introuvable : `{labels_root}`", gr.update(choices=_cni_source_choices([]), value=None), "Aucune source chargée."
                # L'état retourné est l'unique source clients d'un run. La liste
                # d'aperçu est donc elle aussi limitée aux PDF détectés au scan.
                records = scan_cni_documents(
                    clients_root,
                    labels_root,
                    recto_suffix=str(recto_suffix or "").strip(),
                    verso_suffix=str(verso_suffix or "").strip(),
                )
                ready = sum(record["status"] == "ready" for record in records)
                labels = sum(record.get("label_status") == "label_materialized" for record in records)
                unlabeled = sum(record["status"] == "ready" and record.get("label_status") != "label_materialized" for record in records)
                LOGGER.info("CNI scan completed | clients=%d | ready=%d | labels=%d | unlabeled=%d", len(records), ready, labels, unlabeled)
                return (
                    records,
                    records,
                    _cni_scan_table(records),
                    (
                        f"Scan terminé : {len(records)} client(s) détecté(s), {ready} prêt(s), {labels} label(s) converti(s)."
                        + (" Cochez **Continuer sans labels** pour lancer les PDF non notés." if unlabeled else "")
                    ),
                    gr.update(choices=_cni_source_choices(records), value=None),
                    f"**{len(records)} dossier(s) retenu(s) sur {len(records)}.**",
                )
            except Exception as exc:
                LOGGER.exception("CNI scan failed")
                return [], [], pd.DataFrame(), f"Scan CNI impossible : {type(exc).__name__}: {exc}", gr.update(choices=_cni_source_choices([]), value=None), "Aucune source chargée."

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
                + _cni_confidence_summary(result)
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

        def on_cni_run(model_specs, client_records, strategy, dpi, timeout, threads, unload, rotation_method, perspective_correction, preprocessing, system_prompt, prompt_instructions, continue_without_label):
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
                for event in iter_cni_extraction(
                    list(model_specs), ready_records, RUNS_DIR,
                    strategy=str(strategy), dpi=int(dpi), timeout_seconds=float(timeout or 0),
                    cpu_threads=int(threads or 1), unload_after_task=bool(unload),
                    fields=fields, prompt_instructions=prompt_instructions, system_prompt=system_prompt,
                    preprocessing={
                        **{str(value): True for value in (preprocessing or [])},
                        "rotation_pillow": rotation_method == "pillow",
                        "rotation_opencv": rotation_method == "opencv",
                        "perspective": bool(perspective_correction),
                    },
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
            outputs=[
                cni_folder_source,
                cni_zip_source,
                cni_api_source,
                cni_api_inventory_actions,
            ],
            queue=False,
        )
        cni_api_call_mode.change(
            qlickeer_call_mode_visibility,
            inputs=[cni_api_call_mode],
            outputs=[cni_api_parser_group, cni_api_guided_group],
            queue=False,
        )
        cni_api_guided_route.change(
            qlickeer_guided_route_visibility,
            inputs=[cni_api_guided_route],
            outputs=[cni_api_list_group, cni_api_info_group, cni_api_documents_group, cni_api_view_group],
            queue=False,
        )
        cni_api_list_parse.click(
            parse_qlickeer_route, inputs=[cni_api_list_raw_url],
            outputs=[cni_api_settings_base_url, cni_api_list_endpoint_setting, cni_api_list_params_setting], queue=False,
        )
        cni_api_info_parse.click(
            parse_qlickeer_route, inputs=[cni_api_info_raw_url],
            outputs=[cni_api_settings_base_url, cni_api_info_endpoint_setting, cni_api_info_params_setting], queue=False,
        )
        cni_api_documents_parse.click(
            parse_qlickeer_route, inputs=[cni_api_documents_raw_url],
            outputs=[cni_api_settings_base_url, cni_api_documents_endpoint_setting, cni_api_documents_params_setting], queue=False,
        )
        cni_api_view_parse.click(
            parse_qlickeer_route, inputs=[cni_api_view_raw_url],
            outputs=[cni_api_settings_base_url, cni_api_view_endpoint_setting, cni_api_view_params_setting], queue=False,
        )
        cni_api_load_customers.click(
            load_configured_customers,
            inputs=[
                cni_api_settings_base_url, cni_api_list_endpoint_setting, cni_api_list_params_setting,
                cni_api_source_from_date, cni_api_source_to_date, cni_api_source_step,
                cni_api_source_page, cni_api_source_page_size, cni_api_settings_timeout,
                cni_api_settings_proxy, cni_api_settings_use_system_proxy, cni_api_settings_verify_ssl,
            ],
            outputs=[
                cni_api_source_feedback, cni_api_source_trace, cni_source_inventory_table,
                cni_api_source_customers_state, cni_source_selection_summary,
            ], queue=False,
        )
        cni_api_source_select_all.click(
            select_all_api_source, inputs=[cni_api_source_customers_state],
            outputs=[cni_source_inventory_table, cni_source_selection_summary], queue=False,
        )
        cni_api_source_clear_selection.click(
            clear_api_source_selection, inputs=[cni_api_source_customers_state],
            outputs=[cni_source_inventory_table, cni_source_selection_summary], queue=False,
        )
        cni_api_prepare_selected.click(
            prepare_selected_qlicker_clients,
            inputs=[
                cni_source_inventory_table,
                cni_api_source_customers_state,
                cni_api_import_root,
                cni_api_settings_base_url,
                cni_api_info_endpoint_setting,
                cni_api_info_params_setting,
                cni_api_documents_endpoint_setting,
                cni_api_documents_params_setting,
                cni_api_view_endpoint_setting,
                cni_api_view_params_setting,
                cni_api_settings_timeout,
                cni_api_settings_proxy,
                cni_api_settings_use_system_proxy,
                cni_api_settings_verify_ssl,
                cni_recto_suffix,
                cni_verso_suffix,
            ],
            outputs=[
                cni_source_inventory_table,
                cni_api_source_customers_state,
                cni_api_import_progress,
                cni_source_selection_summary,
                cni_all_clients_state,
                cni_clients_state,
                cni_source_selector,
                cni_api_source_feedback,
            ],
            concurrency_limit=1,
            concurrency_id="qlickeer-cni-import",
        )
        cni_source_inventory_table.change(
            update_cni_source_selection,
            inputs=[cni_input_mode, cni_source_inventory_table, cni_all_clients_state, cni_api_source_customers_state],
            outputs=[cni_clients_state, cni_source_selection_summary], queue=False,
        )
        cni_api_parse_url.click(
            parse_qlicker_url_for_ui,
            inputs=[cni_api_raw_url],
            outputs=[cni_api_base_url, cni_api_parsed_endpoint, cni_api_parsed_params, cni_api_feedback],
            queue=False,
        )
        cni_api_test_parsed.click(
            test_qlicker_parsed,
            inputs=[
                cni_api_base_url, cni_api_parsed_endpoint, cni_api_parsed_params,
                cni_api_timeout, cni_api_proxy, cni_api_use_system_proxy, cni_api_verify_ssl,
            ],
            outputs=[cni_api_feedback, cni_api_trace],
            queue=False,
        )
        cni_api_test_list.click(
            test_qlicker_list,
            inputs=[
                cni_api_base_url, cni_api_list_endpoint,
                cni_api_from_date, cni_api_to_date, cni_api_step,
                cni_api_page, cni_api_page_size, cni_api_list_extra, cni_api_timeout,
                cni_api_proxy, cni_api_use_system_proxy, cni_api_verify_ssl,
            ],
            outputs=[
                cni_api_feedback, cni_api_trace, cni_api_customers_table,
                cni_api_customers_state, cni_api_selected_summary,
            ],
            queue=False,
        )
        cni_api_select_all.click(
            select_all_customers,
            inputs=[cni_api_customers_state],
            outputs=[cni_api_customers_table, cni_api_selected_summary],
            queue=False,
        )
        cni_api_clear_selection.click(
            clear_customer_selection,
            inputs=[cni_api_customers_state],
            outputs=[cni_api_customers_table, cni_api_selected_summary],
            queue=False,
        )
        cni_api_customers_table.change(
            _customer_selection_summary,
            inputs=[cni_api_customers_table, cni_api_customers_state],
            outputs=[cni_api_selected_summary],
            queue=False,
        )
        cni_api_test_info.click(
            test_qlicker_info,
            inputs=[
                cni_api_base_url, cni_api_info_endpoint, cni_api_customer_id,
                cni_api_load_documents, cni_api_info_extra, cni_api_timeout,
                cni_api_proxy, cni_api_use_system_proxy, cni_api_verify_ssl,
            ],
            outputs=[cni_api_feedback, cni_api_trace],
            queue=False,
        )
        cni_api_test_documents.click(
            test_qlicker_documents,
            inputs=[
                cni_api_base_url, cni_api_documents_endpoint, cni_api_documents_customer_id,
                cni_api_documents_extra, cni_api_timeout,
                cni_api_proxy, cni_api_use_system_proxy, cni_api_verify_ssl,
            ],
            outputs=[cni_api_feedback, cni_api_trace],
            queue=False,
        )
        cni_api_test_view.click(
            test_qlicker_view,
            inputs=[
                cni_api_base_url, cni_api_view_endpoint, cni_api_view_customer_id,
                cni_api_view_page, cni_api_view_file, cni_api_view_extra, cni_api_timeout,
                cni_api_proxy, cni_api_use_system_proxy, cni_api_verify_ssl,
            ],
            outputs=[cni_api_feedback, cni_api_trace],
            queue=False,
        )
        cni_scan.click(
            scan_cni_input,
            inputs=[cni_clients_root, cni_labels_root, cni_recto_suffix, cni_verso_suffix],
            outputs=[
                cni_all_clients_state,
                cni_clients_state,
                cni_source_inventory_table,
                cni_scan_status,
                cni_source_selector,
                cni_source_selection_summary,
            ],
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
            inputs=[cni_models, cni_clients_state, cni_strategy, cni_dpi, cni_timeout, cni_cpu_threads, cni_unload, cni_rotation_method, cni_perspective_correction, cni_preprocessing, cni_system_prompt, cni_prompt_instructions, cni_continue_without_label],
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
