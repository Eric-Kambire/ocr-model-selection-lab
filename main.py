from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path
from typing import Any

import gradio as gr
import pandas as pd

import dataset_generator
from ocr_benchmark.domain import BenchmarkCase
from ocr_benchmark.dataset_repository import DatasetRepository
from ocr_benchmark.registry import build_default_registry
from ocr_benchmark.reporting import save_run
from ocr_benchmark.runner import BenchmarkRunner, summarize_results
from ocr_benchmark.visualization import (
    category_quality_chart,
    empty_figure,
    latency_chart,
    quality_speed_chart,
    reliability_chart,
)

ROOT_DIR = Path(__file__).resolve().parent
DATASET_DIR = ROOT_DIR / "dataset"
CATALOG_PATH = DATASET_DIR / "dataset.json"
RUNS_DIR = ROOT_DIR / "runs"

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
#dataset-layout {
    height: 100% !important;
    align-items: stretch !important;
}
#models-list .wrap {
    max-height: 220px !important;
    overflow-y: auto !important;
    scrollbar-width: thin;
}
#summary-panel {
    height: 100% !important;
    overflow: auto !important;
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
#metrics-pane {
    max-height: 70vh !important;
    overflow-y: auto !important;
    padding-right: 12px;
}
#dataset-catalog {
    height: calc(100vh - 300px) !important;
    overflow: auto !important;
}
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
.hero { padding: 24px; border-radius: 18px; color: white;
        background: linear-gradient(135deg, #312e81 0%, #4f46e5 50%, #0f766e 100%);
        box-shadow: 0 18px 40px rgba(49,46,129,.28); margin-bottom: 14px; }
.hero h1 { margin: 0 0 8px 0; font-size: 32px; }
.hero p { margin: 0; opacity: .9; }
"""


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

    with gr.Blocks(title="OCR Model Selection Lab", fill_height=True) as app:
        gr.HTML(
            "<div class='hero'><h1>OCR Model Selection Lab</h1>"
            "<p>Comparez qualité, vitesse et fiabilité sur CPU ou GPU avec un protocole traçable.</p></div>"
        )
        run_state = gr.State([])

        with gr.Tabs(elem_id="main-tabs"):
            with gr.Tab("1. Benchmark"):
                with gr.Row(elem_id="benchmark-layout"):
                    with gr.Column(scale=1):
                        models = gr.CheckboxGroup(
                            model_choices,
                            value=["mock:MockOCR-V1"],
                            label="Modèles",
                            elem_id="models-list",
                        )
                        category = gr.Dropdown(
                            ["All", *category_choices],
                            value="All",
                            label="Catégorie",
                        )
                        eval_mode = gr.Radio(
                            ["Standard", "Bankmark"],
                            value="Standard",
                            label="Mode d’évaluation",
                        )
                        noise = gr.Slider(
                            0.0, 0.30, value=0.05, step=0.01, label="Bruit du modèle simulé"
                        )
                        launch = gr.Button("Lancer le benchmark", variant="primary")
                        status = gr.Textbox("Prêt.", label="État", interactive=False)
                    with gr.Column(scale=3, elem_id="summary-panel"):
                        summary_table = gr.Dataframe(
                            label="Synthèse comparative", interactive=False
                        )
                        recommendation = gr.Markdown(
                            "### Recommandation\n\nLancez un benchmark pour comparer les modèles."
                        )

            with gr.Tab("2. Graphiques"):
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

            with gr.Tab("3. Résultats détaillés"):
                with gr.Row(elem_id="explorer-layout"):
                    with gr.Column():
                        result_image = gr.Dropdown(
                            image_choices,
                            value=image_choices[0] if image_choices else None,
                            label="Document",
                        )
                        result_model = gr.Dropdown([], label="Modèle")
                        source_image = gr.Image(label="Document source", type="filepath")
                        result_description = gr.Textbox(label="Description", interactive=False)
                    with gr.Column(scale=2):
                        with gr.Row():
                            ground_truth = gr.Textbox(label="Texte attendu", lines=12)
                            extracted = gr.Textbox(label="Texte extrait", lines=12)
                        details = gr.JSON(label="Mesures de ce document")

            with gr.Tab("4. Ajouter des données"):
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

            with gr.Tab("5. Comprendre les métriques"):
                gr.Markdown(METRICS_HELP, elem_id="metrics-pane")

            with gr.Tab("6. Dataset"):
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

        def on_run(model_specs, selected_category, selected_noise, selected_mode):
            if not model_specs:
                return (
                    pd.DataFrame(),
                    "Sélectionnez au moins un modèle.",
                    "### Recommandation\n\nAucun modèle sélectionné.",
                    [],
                    gr.update(choices=[]),
                    empty_figure(),
                    empty_figure(),
                    empty_figure(),
                    empty_figure(),
                )
            try:
                summary, results, run_id = run_benchmark(
                    model_specs, selected_category, selected_noise, selected_mode
                )
                tested = list(dict.fromkeys(result["model"] for result in results))
                return (
                    _display_summary(summary),
                    f"Terminé : {len(results)} évaluations. Run ID : {run_id}",
                    explain_recommendation(summary),
                    results,
                    gr.update(choices=tested, value=tested[0] if tested else None),
                    quality_speed_chart(summary),
                    latency_chart(summary),
                    reliability_chart(summary),
                    category_quality_chart(results),
                )
            except Exception as exc:
                return (
                    pd.DataFrame(),
                    f"Échec : {type(exc).__name__}: {exc}",
                    "### Recommandation\n\nLe benchmark a échoué.",
                    [],
                    gr.update(choices=[]),
                    empty_figure("Benchmark failed."),
                    empty_figure("Benchmark failed."),
                    empty_figure("Benchmark failed."),
                    empty_figure("Benchmark failed."),
                )

        def explore(selection, model_name, results):
            if not selection:
                return None, "", "", "", {}
            index = int(selection.split(":", 1)[0])
            item = dataset[index]
            match = next(
                (
                    result
                    for result in (results or [])
                    if result["model"] == model_name
                    and os.path.normpath(result["image_path"])
                    == os.path.normpath(item["image_path"])
                ),
                None,
            )
            metrics = {}
            text = "Aucun résultat pour ce modèle et ce document."
            if match:
                text = match["extracted_text"]
                hidden = {"ground_truth", "extracted_text", "description", "image_path"}
                metrics = {key: value for key, value in match.items() if key not in hidden}
            return (
                str(ROOT_DIR / item["image_path"]),
                item.get("description", ""),
                item["ground_truth"],
                text,
                metrics,
            )

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
                    gr.update(choices=updated_choices, value=selected),
                    _catalog_html(dataset),
                )
            except Exception as exc:
                return (
                    f"❌ {type(exc).__name__}: {exc}",
                    gr.update(),
                    gr.update(),
                    _catalog_html(dataset),
                )

        launch.click(
            on_run,
            [models, category, noise, eval_mode],
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
            ],
        )
        result_image.change(
            explore,
            [result_image, result_model, run_state],
            [source_image, result_description, ground_truth, extracted, details],
        )
        result_model.change(
            explore,
            [result_image, result_model, run_state],
            [source_image, result_description, ground_truth, extracted, details],
        )
        dataset_selector.change(
            browse_dataset,
            dataset_selector,
            [dataset_image, dataset_category, dataset_description, dataset_truth],
        )
        add_data_button.click(
            add_labeled_data,
            [upload_image, upload_label, upload_category, upload_description],
            [add_data_status, result_image, dataset_selector, catalog_component],
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
