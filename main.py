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
LOGGER.info("Application initialisation | root=%s | log_level=%s", ROOT_DIR, LOG_LEVEL)

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

# Ces consignes restent courtes et complètent le contrat JSON centralisé. Elles
# sont modifiables dans l'onglet Paramètres CNI avant chaque lancement.
DEFAULT_CNI_OPERATOR_INSTRUCTIONS = (
    "The card may use an old or new Moroccan CNI layout. Prioritize the visible "
    "French/Arabic field labels and their alignment. Copy the printed Latin value "
    "exactly; use null rather than guessing an ambiguous character."
)

APP_CSS = """
.gradio-container {
    width: min(100% - 40px, 1600px) !important;
    max-width: 1600px !important;
    margin: 0 auto !important;
    padding: 18px 0 30px !important;
    min-height: 100vh !important;
    overflow: visible !important;
    background: var(--body-background-fill) !important;
    color: var(--body-text-color) !important;
}
#workspace-switcher-title {
    margin: 4px 0 -4px !important;
}
#workspace-switcher-title h2 {
    margin: 0 !important;
    font-size: 17px !important;
}
#workspace-switcher-title p {
    margin: 3px 0 0 !important;
    color: var(--body-text-color-subdued) !important;
    font-size: 13px !important;
}
#app-navigation {
    margin: 0 !important;
    padding: 8px !important;
    border: 1px solid var(--block-border-color) !important;
    border-radius: 14px !important;
    background: var(--body-background-fill) !important;
}
#app-navigation > [data-testid="block-info"] {
    display: none !important;
}
#app-navigation .wrap {
    display: grid !important;
    grid-template-columns: repeat(2, minmax(0, 1fr)) !important;
    gap: 8px !important;
}
#app-navigation label {
    margin: 0 !important;
    min-height: 58px !important;
    padding: 12px 16px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    gap: 9px !important;
    border: 1px solid var(--block-border-color) !important;
    border-radius: 10px !important;
    background: var(--background-fill-secondary) !important;
    cursor: pointer !important;
    user-select: none !important;
    transition: background-color .14s ease, border-color .14s ease, transform .14s ease !important;
}
#app-navigation label.selected {
    border-color: rgba(245, 137, 43, .78) !important;
    background: rgba(245, 137, 43, .15) !important;
    font-weight: 700 !important;
}
#app-navigation label:hover {
    border-color: rgba(245, 137, 43, .52) !important;
    transform: translateY(-1px) !important;
}
#app-navigation label span {
    font-size: 14px !important;
    font-weight: 650 !important;
}
#app-navigation label input {
    width: 13px !important;
    height: 13px !important;
    margin: 0 !important;
    accent-color: #f5892b !important;
}
#app-navigation label:focus-visible {
    outline: 2px solid #f5892b !important;
    outline-offset: 2px !important;
}
#page-shell {
    min-height: calc(100vh - 190px) !important;
    overflow: visible !important;
    gap: 14px !important;
    background: var(--body-background-fill) !important;
}
#page-navigation {
    position: sticky !important;
    top: 10px !important;
    z-index: 20 !important;
    margin: 0 !important;
    padding: 7px !important;
    border: 1px solid var(--block-border-color) !important;
    border-radius: 12px !important;
    background: var(--body-background-fill) !important;
    box-shadow: 0 6px 18px rgba(24, 24, 27, .06) !important;
}
#page-navigation > [data-testid="block-info"] {
    display: none !important;
}
#page-navigation .wrap {
    display: grid !important;
    grid-template-columns: repeat(7, minmax(0, 1fr)) !important;
    gap: 6px !important;
}
#page-navigation label {
    margin: 0 !important;
    min-height: 44px !important;
    padding: 9px 12px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    gap: 7px !important;
    border: 1px solid var(--block-border-color) !important;
    border-radius: 9px !important;
    background: var(--background-fill-secondary) !important;
    cursor: pointer !important;
    user-select: none !important;
    transition: background-color .14s ease, border-color .14s ease, transform .14s ease !important;
}
#page-navigation label.selected {
    border-color: rgba(245, 137, 43, .72) !important;
    background: rgba(245, 137, 43, .14) !important;
    font-weight: 600 !important;
}
#page-navigation label:hover {
    border-color: rgba(245, 137, 43, .48) !important;
    transform: translateY(-1px) !important;
}
#page-navigation label span {
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
    font-size: 12px !important;
    font-weight: 600 !important;
}
#page-navigation label input,
#cni-navigation label input {
    width: 11px !important;
    height: 11px !important;
    margin: 0 !important;
    accent-color: #f5892b !important;
    flex: 0 0 auto !important;
}
#page-navigation label:focus-visible,
#cni-navigation label:focus-visible {
    outline: 2px solid #f5892b !important;
    outline-offset: 2px !important;
}
#page-settings,
#page-charts,
#page-details,
#page-add-data,
#page-metrics,
#page-dataset {
    display: none !important;
}
#page-cni {
    display: none !important;
}
#benchmark-layout,
#explorer-layout,
#dataset-layout,
#cni-explorer-layout {
    height: 100% !important;
    align-items: stretch !important;
}
#page-benchmark {
    gap: 14px !important;
}
#page-benchmark > .row {
    gap: 18px !important;
    align-items: flex-start !important;
}
#benchmark-config {
    flex: 0 1 390px !important;
    max-width: 410px !important;
    padding-right: 18px !important;
    border-right: 1px solid var(--block-border-color) !important;
}
#models-list .wrap {
    max-height: 145px !important;
    overflow-y: auto !important;
    scrollbar-width: thin;
}
#summary-panel {
    min-height: 0 !important;
    gap: 10px !important;
}
#summary-panel > .row {
    gap: 8px !important;
}
#summary-panel > .row > button {
    min-height: 40px !important;
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
#cni-workspace {
    gap: 14px !important;
}
#page-cni {
    gap: 12px !important;
    /* Même fond que la branche principale : aucune zone blanche ne doit
       apparaître lorsque l'espace CNI est affiché ou qu'une vue change. */
    background: var(--body-background-fill) !important;
}
.cni-header {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    gap: 16px;
    padding: 2px 0 10px;
    border-bottom: 1px solid var(--block-border-color);
}
.cni-header h2 {
    margin: 0;
    font-size: 22px;
    line-height: 1.2;
}
.cni-header span {
    margin: 0;
    color: var(--body-text-color-subdued);
    font-size: 13px;
}
#cni-prep-grid {
    align-items: stretch !important;
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
.cni-section-title {
    display: flex;
    gap: 8px;
    align-items: baseline;
    margin: 0 0 2px;
    font-size: 15px;
    font-weight: 650;
}
.cni-section-title span {
    color: var(--body-text-color-subdued);
    font-size: 12px;
    font-weight: 500;
}
#cni-models .wrap {
    max-height: 118px !important;
    overflow-y: auto !important;
    scrollbar-width: thin;
}
#cni-runbar {
    align-items: end !important;
    gap: 8px !important;
    padding-top: 2px;
}
#cni-runbar > button {
    min-height: 38px !important;
}
#cni-navigation {
    margin: 0 !important;
}
#cni-navigation .wrap {
    display: grid !important;
    grid-template-columns: repeat(4, minmax(0, 180px)) !important;
    gap: 7px !important;
}
#cni-navigation label {
    min-height: 44px !important;
    padding: 9px 14px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    gap: 7px !important;
    border: 1px solid var(--block-border-color) !important;
    border-radius: 9px !important;
    background: var(--background-fill-secondary) !important;
    cursor: pointer !important;
    user-select: none !important;
    transition: background-color .14s ease, border-color .14s ease, transform .14s ease !important;
}
#cni-navigation label.selected {
    border-color: rgba(245, 137, 43, .72) !important;
    background: rgba(245, 137, 43, .14) !important;
    font-weight: 600;
}
#cni-navigation label:hover {
    border-color: rgba(245, 137, 43, .48) !important;
    transform: translateY(-1px) !important;
}
#cni-navigation label span {
    font-size: 13px !important;
    font-weight: 600 !important;
}
#cni-step-live,
#cni-step-results {
    display: none !important;
}
#cni-run-status textarea {
    font-weight: 600 !important;
}
#cni-live-image,
#cni-results-table {
    min-height: 280px !important;
}
#cni-results-filterbar {
    align-items: end !important;
    gap: 8px !important;
}
#cni-results-navigation {
    align-items: center !important;
    gap: 8px !important;
}
#cni-result-position {
    text-align: center;
    padding-top: 8px;
}
#cni-result-identity {
    min-height: 92px;
}
#cni-settings {
    margin-top: 2px !important;
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
    margin: 6px 0 0 !important;
}
#benchmark-layout {
    gap: 14px !important;
    padding-top: 10px !important;
    border-top: 1px solid var(--block-border-color) !important;
}
#live-image,
#live-metrics {
    border-radius: 10px !important;
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
.hero { padding: 14px 20px; border-radius: 16px; color: white;
        background: linear-gradient(135deg, #312e81 0%, #4f46e5 50%, #0f766e 100%);
        box-shadow: 0 12px 28px rgba(49,46,129,.24); margin-bottom: 0; }
.hero h1 { margin: 0 0 3px 0; color: white !important; font-size: 26px; line-height: 1.15; }
.hero p { margin: 0; color: white !important; opacity: .9; font-size: 14px; }
@media (max-width: 900px) {
    .gradio-container {
        width: min(100% - 24px, 1600px) !important;
        padding-top: 12px !important;
    }
    #page-navigation {
        position: static !important;
    }
    #page-navigation .wrap {
        grid-template-columns: repeat(2, minmax(0, 1fr)) !important;
    }
    #app-navigation .wrap {
        grid-template-columns: 1fr !important;
    }
    #cni-navigation .wrap {
        grid-template-columns: 1fr !important;
    }
    #page-navigation label,
    #cni-navigation label {
        justify-content: flex-start !important;
    }
    #benchmark-config {
        max-width: none !important;
        padding-right: 0 !important;
        border-right: 0 !important;
        border-bottom: 1px solid var(--block-border-color) !important;
        padding-bottom: 12px !important;
    }
    #page-benchmark > .row {
        gap: 14px !important;
    }
    .cni-header {
        align-items: flex-start;
        flex-direction: column;
        gap: 3px;
    }
    #cni-source {
        padding-right: 0;
        padding-bottom: 14px;
        border-right: 0;
        border-bottom: 1px solid var(--block-border-color);
    }
}
"""

# Le routage navigateur évite le montage paresseux des Tabs Gradio. Chaque page
# est rendue une fois ; la navigation ne fait que masquer/afficher la page et
# reste donc utilisable pendant qu'un générateur de benchmark tourne.
APP_JS = r"""
() => {
  const pageIds = [
    "page-benchmark", "page-settings", "page-charts", "page-details",
    "page-add-data", "page-metrics", "page-dataset"
  ];
  const installRouter = (selector, targetIds, initialIndex = 0) => {
    const navigation = document.querySelector(selector);
    if (!navigation || navigation.dataset.clientRouterInstalled) {
      return Boolean(navigation);
    }
    navigation.dataset.clientRouterInstalled = "true";
    const choices = () => Array.from(navigation.querySelectorAll("label"));
    const activate = (index, notifyGradio = false) => {
      const labels = choices();
      const selectedIndex = Math.max(0, Math.min(index, labels.length - 1));
      targetIds.forEach((id, position) => {
        const page = document.getElementById(id);
        if (page) page.style.setProperty("display", position === selectedIndex ? "flex" : "none", "important");
      });
      labels.forEach((label, position) => {
        const input = label.querySelector("input");
        const selected = position === selectedIndex;
        label.classList.toggle("selected", selected);
        label.setAttribute("role", "tab");
        label.setAttribute("aria-selected", String(selected));
        label.tabIndex = selected ? 0 : -1;
        if (input) {
          const changed = input.checked !== selected;
          input.checked = selected;
          input.setAttribute("aria-checked", String(selected));
          // La valeur Gradio est synchronisée, sans faire dépendre le routage
          // d'un rendu serveur : les onglets restent donc cliquables en live.
          if (notifyGradio && selected && changed) {
            input.dispatchEvent(new Event("input", { bubbles: true }));
            input.dispatchEvent(new Event("change", { bubbles: true }));
          }
        }
      });
      return selectedIndex;
    };
    const selectLabel = (label) => {
      const index = choices().indexOf(label);
      if (index < 0) return;
      const selectedIndex = activate(index, true);
      choices()[selectedIndex]?.focus();
    };
    navigation.setAttribute("role", "tablist");
    navigation.addEventListener("click", (event) => {
      const label = event.target.closest("label");
      if (!label || !navigation.contains(label)) return;
      event.preventDefault();
      event.stopImmediatePropagation();
      selectLabel(label);
    }, true);
    navigation.addEventListener("keydown", (event) => {
      const label = event.target.closest("label");
      if (!label || !navigation.contains(label)) return;
      const labels = choices();
      const current = labels.indexOf(label);
      if (current < 0) return;
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        selectLabel(label);
      } else if (["ArrowRight", "ArrowDown", "ArrowLeft", "ArrowUp", "Home", "End"].includes(event.key)) {
        event.preventDefault();
        const next = event.key === "Home" ? 0 : event.key === "End" ? labels.length - 1 :
          (current + (event.key === "ArrowRight" || event.key === "ArrowDown" ? 1 : -1) + labels.length) % labels.length;
        const selectedIndex = activate(next, true);
        labels[selectedIndex]?.focus();
      }
    });
    activate(initialIndex);
    return true;
  };
  const installWorkspaceSelector = () => {
    const navigation = document.querySelector("#app-navigation");
    const classicNavigation = document.querySelector("#page-navigation");
    const cniPage = document.getElementById("page-cni");
    if (!navigation || !classicNavigation || !cniPage || navigation.dataset.clientRouterInstalled) {
      return Boolean(navigation && classicNavigation && cniPage);
    }
    navigation.dataset.clientRouterInstalled = "true";
    const choices = () => Array.from(navigation.querySelectorAll("label"));
    const showWorkspace = (index, notifyGradio = false) => {
      const labels = choices();
      const selectedIndex = Math.max(0, Math.min(index, labels.length - 1));
      const showCni = selectedIndex === 1;
      const selectedClassicIndex = Math.max(0, Array.from(classicNavigation.querySelectorAll("label")).findIndex((label) => label.classList.contains("selected")));
      classicNavigation.style.setProperty("display", showCni ? "none" : "block", "important");
      pageIds.forEach((id, position) => {
        const page = document.getElementById(id);
        if (page) page.style.setProperty("display", !showCni && position === selectedClassicIndex ? "flex" : "none", "important");
      });
      cniPage.style.setProperty("display", showCni ? "flex" : "none", "important");
      labels.forEach((label, position) => {
        const input = label.querySelector("input");
        const selected = position === selectedIndex;
        label.classList.toggle("selected", selected);
        label.setAttribute("role", "tab");
        label.setAttribute("aria-selected", String(selected));
        label.tabIndex = selected ? 0 : -1;
        if (input) {
          const changed = input.checked !== selected;
          input.checked = selected;
          input.setAttribute("aria-checked", String(selected));
          if (notifyGradio && selected && changed) {
            input.dispatchEvent(new Event("input", { bubbles: true }));
            input.dispatchEvent(new Event("change", { bubbles: true }));
          }
        }
      });
      return selectedIndex;
    };
    const selectLabel = (label) => {
      const index = choices().indexOf(label);
      if (index < 0) return;
      const selectedIndex = showWorkspace(index, true);
      choices()[selectedIndex]?.focus();
    };
    navigation.setAttribute("role", "tablist");
    navigation.addEventListener("click", (event) => {
      const label = event.target.closest("label");
      if (!label || !navigation.contains(label)) return;
      event.preventDefault();
      event.stopImmediatePropagation();
      selectLabel(label);
    }, true);
    navigation.addEventListener("keydown", (event) => {
      const label = event.target.closest("label");
      if (!label || !navigation.contains(label)) return;
      const labels = choices();
      const current = labels.indexOf(label);
      if (current < 0) return;
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        selectLabel(label);
      } else if (["ArrowRight", "ArrowDown", "ArrowLeft", "ArrowUp", "Home", "End"].includes(event.key)) {
        event.preventDefault();
        const next = event.key === "Home" ? 0 : event.key === "End" ? labels.length - 1 :
          (current + (event.key === "ArrowRight" || event.key === "ArrowDown" ? 1 : -1) + labels.length) % labels.length;
        const selectedIndex = showWorkspace(next, true);
        labels[selectedIndex]?.focus();
      }
    });
    showWorkspace(0);
    return true;
  };
  const install = () => {
    const cniPageIds = ["cni-step-setup", "cni-step-live", "cni-step-results", "cni-step-settings"];
    const mainReady = installRouter("#page-navigation", pageIds);
    installRouter("#cni-navigation", cniPageIds);
    const workspaceReady = installWorkspaceSelector();
    return mainReady && workspaceReady;
  };
  if (!install()) {
    const observer = new MutationObserver(() => { if (install()) observer.disconnect(); });
    observer.observe(document.documentElement, { childList: true, subtree: true });
  }
}
"""
APP_HEAD = "<script>\n(" + APP_JS + ")();\n</script>"


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


def _late_trace_for(result: dict[str, Any]) -> dict[str, Any] | None:
    """Find a provider response that arrived after this result timed out."""
    if result.get("status") != "timeout" or not result.get("run_id"):
        return None
    trace_path = RUNS_DIR / str(result["run_id"]) / "traces.jsonl"
    if not trace_path.is_file():
        return None
    try:
        with trace_path.open("r", encoding="utf-8") as stream:
            events = [json.loads(line) for line in stream if line.strip()]
    except (OSError, json.JSONDecodeError):
        return None
    for event in reversed(events):
        if (
            event.get("timing") == "late_after_timeout"
            and event.get("model") == result.get("model")
            and event.get("image_path") == result.get("image_path")
        ):
            return event
    return None


def _live_metrics_markdown(result: dict[str, Any], next_label: str) -> str:
    expected = html.escape(str(result.get("ground_truth", "")))
    late = _late_trace_for(result)
    extracted_value = late.get("text", "") if late else result.get("extracted_text", "")
    extracted = html.escape(str(extracted_value))
    token_speed = result.get("tokens_per_second")
    token_text = (
        f"{float(token_speed):.2f} tokens/s" if token_speed is not None else "N/A"
    )
    error = (
        f"\n- **Erreur :** {html.escape(str(result['error']))}"
        if result.get("error")
        else ""
    )
    late_note = (
        "\n- **Sortie tardive conservée :** oui — disponible dans `traces.jsonl`."
        if late else ""
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
        f"{error}{late_note}\n\n"
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
    """Render the CNI input audit without exposing file-system internals."""
    return pd.DataFrame(
        [
            {
                "Client dossier": item.get("folder_client_id"),
                "Recto": "OK" if item.get("recto_pdf") else "Manquant",
                "Verso": "OK" if item.get("verso_pdf") else "Manquant",
                "Label": item.get("label_status", "—"),
                "Statut": item.get("status", "—"),
                "Alertes": ", ".join(item.get("issues") or []) or "—",
            }
            for item in records
        ]
    )


def _cni_source_choices(records: list[dict[str, Any]]) -> list[tuple[str, str]]:
    """Return readable recto/verso PDF choices after a client-folder scan."""
    choices: list[tuple[str, str]] = []
    for record in records:
        client_id = str(record.get("folder_client_id") or "Client inconnu")
        for side in ("recto", "verso"):
            path_value = record.get(f"{side}_pdf")
            if path_value:
                choices.append((f"{client_id} — {side.title()} (PDF)", str(path_value)))
    return choices


def _preview_cni_source(path_value: str | None) -> tuple[Any, str]:
    """Show an image preview only after the user selects a source document."""
    if not path_value:
        return gr.update(value=None, visible=False), "Sélectionnez un PDF détecté pour l’aperçu."
    source = Path(path_value)
    if not source.is_file():
        return gr.update(value=None, visible=False), "⚠️ Le fichier sélectionné n’est plus disponible sur le disque."
    if source.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
        return gr.update(value=str(source), visible=True), f"**Aperçu :** `{source.name}` · image locale"
    if source.suffix.lower() != ".pdf":
        return gr.update(value=None, visible=False), f"⚠️ Format non pris en charge pour l’aperçu : `{source.suffix}`"
    preview_path = RUNS_DIR / "cni_source_previews" / f"{source.stem}-{source.stat().st_mtime_ns}.png"
    try:
        render_single_page_pdf(source, preview_path, dpi=150)
    except Exception as exc:
        LOGGER.exception("CNI source preview failed | source=%s", source)
        return gr.update(value=None, visible=False), f"⚠️ Aperçu PDF impossible : `{type(exc).__name__}: {exc}`"
    return gr.update(value=str(preview_path), visible=True), f"**Aperçu :** `{source.name}` · PDF rendu à 150 DPI"


def _cni_source_mode_visibility(mode: str) -> tuple[Any, Any]:
    """Switch between local-folder fields and the ZIP upload, without mixing them."""
    return gr.update(visible=mode == "folder"), gr.update(visible=mode == "zip")


def _cni_prompt_preview(strategy: str, instructions: str | None) -> str:
    """Affiche exactement les prompts CNI qui seront envoyés au modèle."""
    fields = load_cni_field_config(ROOT_DIR / "config" / "cni_fields.json")
    if strategy == "combined_vertical":
        return build_combined_cni_prompt(fields, instructions=instructions)
    return (
        "--- PROMPT RECTO ---\n"
        + build_cni_prompt("recto", fields, instructions=instructions)
        + "\n\n--- PROMPT VERSO ---\n"
        + build_cni_prompt("verso", fields, instructions=instructions)
    )


def _cni_result_table(results: list[dict[str, Any]]) -> pd.DataFrame:
    """Present CNI outcomes while labels are not yet mapped to an accuracy score."""
    rows = []
    for item in results:
        accuracy = item.get("accuracy")
        rows.append(
            {
                "Client": item.get("folder_client_id"),
                "Modèle": item.get("model"),
                "Statut": item.get("status"),
                "Accuracy": "Non noté" if accuracy is None else f"{float(accuracy) * 100:.2f}%",
                "Label": item.get("label_status", "—"),
                "CIN recto": item.get("cin_recto") or "—",
                "CIN verso": item.get("cin_verso") or "—",
                "CIN cohérent": _cni_boolean(item.get("cin_coherent")),
                "Latence (s)": round(float(item.get("latency") or 0), 3),
            }
        )
    return pd.DataFrame(rows)


def _cni_boolean(value: Any) -> str:
    if value is True:
        return "Oui"
    if value is False:
        return "Non"
    return "—"


def _read_json_if_available(path_value: Any) -> Any:
    """Read an artefact for display and return a readable state on failure."""
    if not path_value:
        return {"status": "not_available"}
    path = Path(str(path_value))
    if not path.is_file():
        return {"status": "not_found", "path": str(path)}
    try:
        with path.open("r", encoding="utf-8") as stream:
            return json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        return {"status": "read_failed", "error": f"{type(exc).__name__}: {exc}"}


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
        installed = [name for name in names if name]
        LOGGER.info("Ollama models detected | count=%d | models=%s", len(installed), installed)
        return installed
    except Exception as exc:
        LOGGER.warning("Unable to list Ollama models | error=%s", exc, exc_info=True)
        return []


def run_benchmark(
    selected_models: list[str],
    selected_category: str = "All",
    mock_noise: float = 0.05,
    eval_mode: str = "Standard",
    cpu_threads: int | None = None,
    unload_after_task: bool = True,
) -> tuple[pd.DataFrame, list[dict[str, Any]], str]:
    dataset = load_dataset()
    if selected_category != "All":
        dataset = [item for item in dataset if item["category"] == selected_category]
    if not dataset:
        LOGGER.warning("Benchmark skipped: no dataset cases | category=%s", selected_category)
        return pd.DataFrame(), [], ""

    LOGGER.info(
        "Benchmark starting | models=%s | category=%s | cases=%d | eval_mode=%s",
        selected_models, selected_category, len(dataset), eval_mode,
    )
    runner = BenchmarkRunner(build_default_registry())
    cases = [BenchmarkCase.from_dict(item) for item in dataset]
    run_id, results = runner.run(
        selected_models,
        cases,
        eval_mode=eval_mode,
        cpu_threads=cpu_threads,
        unload_after_task=unload_after_task,
        mock_noise=mock_noise,
    )
    summary = summarize_results(results)
    run_dir = save_run(run_id, summary, results, RUNS_DIR)
    LOGGER.info("Benchmark completed | run_id=%s | results=%d | saved_to=%s", run_id, len(results), run_dir)
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

    def available_run_choices() -> list[tuple[str, str]]:
        """Return persisted runs newest first without loading their payloads."""
        if not RUNS_DIR.exists():
            return []
        choices = []
        for run_dir in sorted((p for p in RUNS_DIR.iterdir() if p.is_dir()), reverse=True):
            if (run_dir / "results.json").exists():
                choices.append((run_dir.name, run_dir.name))
        return choices

    def startup_run_info():
        """Keep page startup light; payloads are loaded only on explicit open."""
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
        cni_clients_state = gr.State([])
        cni_results_state = gr.State([])

        # Deux espaces séparés : le benchmark classique et le flux CNI n'ont
        # pas les mêmes données, vues ni paramètres. Cela évite de coupler les
        # layouts et rend le débogage du module CNI plus direct.
        gr.HTML(
            "<div id='workspace-switcher-title'><h2>Choisir un espace de travail</h2>"
            "<p>Sélectionnez le benchmark adapté à votre type de document.</p></div>"
        )
        workspace_navigation = gr.Radio(
            ["1. Benchmark classique", "2. Benchmark CNI"],
            value="1. Benchmark classique",
            label="Espace de travail",
            elem_id="app-navigation",
        )

        # Les pages restent montées ; le routeur navigateur change uniquement
        # leur visibilité. Ainsi, un benchmark actif ne fige pas la navigation.
        with gr.Group(elem_id="page-shell"):
            page_navigation = gr.Radio(
                [
                    "1. Benchmark", "2. Paramètres", "3. Graphiques",
                    "4. Résultats détaillés", "5. Ajouter des données",
                    "6. Comprendre les métriques", "7. Dataset",
                ],
                value="1. Benchmark",
                label="Navigation",
                elem_id="page-navigation",
            )
            with gr.Column(visible=True, elem_id="page-benchmark") as benchmark_page:
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

            with gr.Column(visible=True, elem_id="page-settings") as settings_page:
                gr.Markdown(
                    "Paramètres appliqués au prochain benchmark. Un appel fournisseur "
                    "ayant dépassé le timeout est marqué immédiatement ; le fournisseur "
                    "peut toutefois conserver une requête réseau jusqu’à sa propre annulation."
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
                        cpu_threads = gr.Number(
                            value=max(1, min(8, os.cpu_count() or 1)),
                            minimum=1,
                            maximum=max(1, os.cpu_count() or 1),
                            precision=0,
                            label="Threads CPU par modèle",
                            info="Le benchmark reste strictement séquentiel : un seul modèle et une seule image à la fois.",
                        )
                        unload_after_task = gr.Checkbox(
                            value=True,
                            label="Décharger le modèle après chaque image",
                            info="Réduit la mémoire persistante, mais peut ralentir le run Ollama.",
                        )
                        max_errors = gr.Number(
                            value=0,
                            minimum=0,
                            precision=0,
                            label="Arrêter après N erreurs — 0 = illimité",
                        )
                        checkpoint_enabled = gr.Checkbox(
                            value=True,
                            label="Checkpoint permanent après chaque document",
                            interactive=False,
                            info="Toujours actif pour permettre la restauration après actualisation.",
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

            with gr.Column(visible=True, elem_id="page-charts") as charts_page:
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

            with gr.Column(visible=True, elem_id="page-details") as details_page:
                with gr.Row():
                    persisted_runs = gr.Dropdown(
                        choices=available_run_choices(),
                        label="Runs sauvegardés",
                        info="Recharge un benchmark après actualisation de la page.",
                        scale=3,
                    )
                    refresh_runs = gr.Button("Actualiser la liste")
                    open_run = gr.Button("Ouvrir le run", variant="secondary")
                persisted_run_status = gr.Markdown("Les runs terminés sont conservés dans `runs/`.")
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
                                    with gr.Tab("Texte extrait", render_children=True):
                                        extracted = gr.Textbox(
                                            label="Transcription normalisée",
                                            lines=12,
                                            interactive=False,
                                        )
                                    with gr.Tab("Sortie brute", render_children=True):
                                        raw_output = gr.Textbox(
                                            label="Réponse brute du fournisseur",
                                            lines=12,
                                            interactive=False,
                                        )
                                    with gr.Tab("Markdown rendu", render_children=True):
                                        markdown_output = gr.Markdown()
                                    with gr.Tab("HTML source", render_children=True):
                                        html_source = gr.Code(
                                            label=(
                                                "Source HTML — non exécutée "
                                                "pour votre sécurité"
                                            ),
                                            language="html",
                                        )
                        with gr.Accordion("Toutes les mesures techniques", open=False):
                            details = gr.JSON(label="Mesures de ce document")

            with gr.Column(visible=True, elem_id="page-add-data") as add_data_page:
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

            with gr.Column(visible=True, elem_id="page-metrics") as metrics_page:
                gr.Markdown(METRICS_HELP, elem_id="metrics-pane")

            with gr.Column(visible=True, elem_id="page-dataset") as dataset_page:
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

            with gr.Column(visible=True, elem_id="page-cni") as cni_page:
                gr.HTML(
                    "<header class='cni-header'><h2>Benchmark CNI</h2>"
                    "<span>Extraction structurée · exécution séquentielle</span></header>"
                )
                cni_navigation = gr.Radio(
                    ["1. Préparer", "2. Suivi en direct", "3. Résultats", "4. Paramètres"],
                    value="1. Préparer",
                    label="Espace CNI",
                    elem_id="cni-navigation",
                )
                with gr.Column(elem_id="cni-step-setup"):
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
                                cni_scan = gr.Button("Scanner les dossiers", variant="secondary")
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
                                            info="Les PDF recto/verso détectés après un scan apparaissent ici.",
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
                                cni_scan_table = gr.Dataframe(
                                    headers=["Client dossier", "Recto", "Verso", "Label", "Statut", "Alertes"],
                                    label="Rapport de scan CNI",
                                    interactive=False,
                                )
                    with gr.Row(elem_id="cni-runbar"):
                        gr.Markdown("**03 · Lancement**\n\nLe suivi détaillé apparaît dans la vue suivante.")
                        cni_continue_without_label = gr.Checkbox(
                            value=False,
                            label="Continuer sans labels",
                            info="Extraction et mesures techniques uniquement ; aucun score de comparaison.",
                        )
                        cni_launch = gr.Button("Lancer", variant="primary")
                        cni_stop = gr.Button("Annuler", variant="stop")
                    cni_launch_feedback = gr.Markdown(
                        "Prêt : sélectionnez des modèles, scannez les dossiers puis lancez le benchmark."
                    )
                with gr.Column(elem_id="cni-step-live"):
                    cni_run_status = gr.Textbox(label="État de l'exécution", value="Prêt.", interactive=False, elem_id="cni-run-status")
                    cni_progress = gr.Slider(0, 100, value=0, step=0.1, label="Progression CNI (%)", interactive=False)
                    cni_live_counters = gr.Markdown("**Traité :** 0 / 0 · **Succès :** 0 · **Erreurs :** 0")
                    with gr.Row(elem_id="cni-workspace"):
                        cni_live_image = gr.Image(label="Face en cours", type="filepath", height=430, elem_id="cni-live-image")
                        cni_live_result = gr.Markdown("Les JSON et mesures apparaîtront après le premier appel.")
                    cni_live_table = gr.Dataframe(
                        headers=["Client", "Modèle", "Statut", "Accuracy", "Label", "CIN recto", "CIN verso", "CIN cohérent", "Latence (s)"],
                        label="Résultats reçus pendant le run",
                        interactive=False,
                    )
                with gr.Column(elem_id="cni-step-results"):
                    # La structure reprend l'explorateur de « 4. Résultats
                    # détaillés » : filtre, liste, navigation puis inspecteur.
                    gr.Markdown("### Résultats détaillés CNI\n\nFiltrez les évaluations puis inspectez une paire recto/verso.")
                    with gr.Row(elem_id="cni-results-filterbar"):
                        cni_accuracy_min = gr.Slider(0, 100, value=0, step=1, label="Accuracy minimale (%)")
                        cni_accuracy_max = gr.Slider(0, 100, value=100, step=1, label="Accuracy maximale (%)")
                        cni_include_unscored = gr.Checkbox(value=True, label="Inclure les résultats non notés")
                        cni_apply_filters = gr.Button("Appliquer les filtres")
                    cni_results_table = gr.Dataframe(
                        headers=["Client", "Modèle", "Statut", "Accuracy", "Label", "CIN recto", "CIN verso", "CIN cohérent", "Latence (s)"],
                        label="Éléments passés par le benchmark",
                        interactive=False,
                        elem_id="cni-results-table",
                    )
                    with gr.Row():
                        cni_accuracy_plot = gr.Plot(value=cni_accuracy_chart([]), elem_classes=["dashboard-chart"])
                        cni_latency_plot = gr.Plot(value=cni_latency_chart([]), elem_classes=["dashboard-chart"])
                    cni_result_selector = gr.Dropdown(
                        label="Liste des paires testées — cliquez pour sélectionner",
                        info="La liste contient les paires client/modèle effectivement passées par le benchmark.",
                        choices=[],
                    )
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
                with gr.Column(elem_id="cni-step-settings"):
                    gr.Markdown(
                        "### Paramètres CNI\n\n"
                        "Les réglages sont appliqués au prochain lancement. Le prompt complet ci-dessous est celui envoyé au modèle."
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
                    with gr.Row():
                        cni_timeout = gr.Number(value=300, minimum=1, maximum=7200, precision=0, label="Temps maximum par appel (s)")
                        cni_cpu_threads = gr.Number(value=max(1, min(8, os.cpu_count() or 1)), minimum=1, maximum=max(1, os.cpu_count() or 1), precision=0, label="Threads CPU Ollama")
                        cni_unload = gr.Checkbox(value=True, label="Décharger le modèle après chaque appel")
                    cni_prompt_instructions = gr.Textbox(
                        value=DEFAULT_CNI_OPERATOR_INSTRUCTIONS,
                        label="Consignes additionnelles de prompt engineering",
                        lines=5,
                        info="Ajoutées après le contrat CNI. Ne modifiez pas les clés JSON demandées ; elles restent imposées par le système.",
                    )
                    cni_prompt_preview = gr.Code(
                        value=_cni_prompt_preview("separate_calls", DEFAULT_CNI_OPERATOR_INSTRUCTIONS),
                        label="Prompt complet envoyé au modèle",
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
            selected_cpu_threads,
            selected_unload_after_task,
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
                    cpu_threads=int(selected_cpu_threads or 1),
                    unload_after_task=bool(selected_unload_after_task),
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
                    # Checkpoints are durable by design. The legacy checkbox
                    # remains for API compatibility, but a completed document
                    # must never depend on a UI toggle to survive refresh/crash.
                    if checkpoint:
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
            late = _late_trace_for(result)
            if late:
                text = str(late.get("text") or text)
                raw = str(late.get("raw_response") or text)
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
            late = _late_trace_for(result)
            if late:
                metrics["late_output"] = "Réponse reçue après timeout; conservée dans traces.jsonl"
                metrics["late_latency"] = late.get("latency")
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
            safe_id = Path(str(run_id)).name
            results_path = RUNS_DIR / safe_id / "results.json"
            try:
                with results_path.open("r", encoding="utf-8") as stream:
                    restored = json.load(stream)
                if not isinstance(restored, list):
                    raise ValueError("results.json doit contenir une liste")
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

        def refresh_cni_models(selected_models):
            """Actualise les modèles Ollama sans modifier le benchmark général."""
            choices = [f"ollama:{name}" for name in get_installed_ollama_models()]
            kept = [name for name in (selected_models or []) if name in choices]
            return gr.update(choices=choices, value=kept)

        def import_cni_test_zip(zip_path):
            """Importe un ZIP de test et préremplit les chemins clients/labels."""
            if not zip_path:
                return gr.update(), gr.update(), "❌ Sélectionnez une archive ZIP."
            try:
                imported = import_cni_zip(Path(zip_path), CNI_IMPORTS_DIR)
                root = Path(imported["import_root"])
                clients_path = root / "clients" if (root / "clients").is_dir() else root
                labels_path = root / "labels"
                message = (
                    f"ZIP importé : {imported['files']} fichier(s). "
                    "Vérifiez les chemins puis cliquez sur **Scanner les dossiers**."
                )
                LOGGER.info("CNI ZIP imported | files=%d | root=%s", imported["files"], root)
                return (
                    gr.update(value=str(clients_path)),
                    gr.update(value=str(labels_path) if labels_path.is_dir() else ""),
                    message,
                )
            except Exception as exc:
                LOGGER.exception("CNI ZIP import failed")
                return gr.update(), gr.update(), f"Import ZIP impossible : {type(exc).__name__}: {exc}"

        def scan_cni_input(clients_root_text, labels_root_text):
            """Scanne les dossiers et copie les JSONB valides près des clients."""
            if not clients_root_text or not str(clients_root_text).strip():
                return [], pd.DataFrame(), "Indiquez le dossier clients.", gr.update(choices=_cni_source_choices([]), value=None)
            try:
                clients_root = Path(str(clients_root_text).strip()).expanduser()
                labels_root = Path(str(labels_root_text).strip()).expanduser() if labels_root_text and str(labels_root_text).strip() else None
                if labels_root is not None and not labels_root.is_dir():
                    LOGGER.warning("CNI scan rejected | labels_root_not_found=%s", labels_root)
                    return [], pd.DataFrame(), f"Dossier labels introuvable : `{labels_root}`", gr.update(choices=_cni_source_choices([]), value=None)
                # L'état retourné est l'unique source clients d'un run. La liste
                # d'aperçu est donc elle aussi limitée aux PDF détectés au scan.
                records = materialize_cni_labels(scan_cni_clients(clients_root, labels_root))
                ready = sum(record["status"] == "ready" for record in records)
                labels = sum(record.get("label_status") == "label_materialized" for record in records)
                unlabeled = sum(record["status"] == "ready" and record.get("label_status") != "label_materialized" for record in records)
                LOGGER.info("CNI input scanned | clients=%d | ready=%d | labels=%d", len(records), ready, labels)
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
                LOGGER.exception("CNI input scan failed")
                return [], pd.DataFrame(), f"Scan CNI impossible : {type(exc).__name__}: {exc}", gr.update(choices=_cni_source_choices([]), value=None)

        def cni_result_choices(results):
            """Crée des libellés de liste liés aux index stables des résultats."""
            return [
                (
                    f"{result.get('folder_client_id')} · {result.get('model')} · {result.get('status')}",
                    index,
                )
                for index, result in enumerate(results or [])
            ]

        def filter_cni_results(results, minimum, maximum, include_unscored):
            """Filtre l'intervalle d'accuracy sans cacher les lignes non notées."""
            lower, upper = sorted((float(minimum or 0), float(maximum or 100)))
            filtered = []
            for result in results or []:
                accuracy = result.get("accuracy")
                if accuracy is None:
                    if include_unscored:
                        filtered.append(result)
                    continue
                if lower <= float(accuracy) * 100 <= upper:
                    filtered.append(result)
            return _cni_result_table(filtered)

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
                gr.update(choices=cni_result_choices(results), value=position),
                position,
                f"**Paire testée {position + 1} / {len(results)}** · {len(results)} évaluation(s) disponible(s)",
                result.get("recto_image_path"),
                result.get("verso_image_path"),
                identity,
                cni_detail_metric_summary(result),
                _read_json_if_available(result.get("label_path")),
                _read_json_if_available(result.get("recto_json_path")),
                _read_json_if_available(result.get("verso_json_path")),
                _read_json_if_available(result.get("global_json_path")),
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

        def on_cni_run(model_specs, client_records, strategy, dpi, timeout, threads, unload, prompt_instructions, continue_without_label):
            """Valide le lancement puis diffuse l'avancement CNI document par document."""
            results: list[dict[str, Any]] = []

            def counters(total: int) -> str:
                successes = sum(result.get("status") == "success" for result in results)
                failures = len(results) - successes
                return f"**Traité :** {len(results)} / {total} · **Succès :** {successes} · **Erreurs :** {failures}"

            def view(feedback: str, status: str, progress: float, image_path, live_text: str, total: int, *, select_last: bool = False):
                table = _cni_result_table(results)
                selector = gr.update(
                    choices=cni_result_choices(results),
                    value=(len(results) - 1 if select_last and results else None),
                )
                return (
                    feedback, status, progress, image_path, live_text,
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
                message = f"Pré-contrôle impossible : aucune paire PDF prête ({invalid_count} dossier(s) à corriger dans le rapport de scan)."
                LOGGER.warning("CNI launch rejected | reason=no_ready_pair | clients=%d", len(client_records))
                yield view(message, message, 0, None, message, 0)
                return

            unlabeled = [record for record in ready_records if record.get("label_status") != "label_materialized"]
            if unlabeled and not continue_without_label:
                message = f"Pré-contrôle requis : {len(unlabeled)} paire(s) n'ont pas de label exploitable. Cochez Continuer sans labels pour lancer l'extraction non notée."
                LOGGER.warning("CNI launch paused | reason=missing_label | unlabeled=%d", len(unlabeled))
                yield view(message, message, 0, None, message, len(ready_records))
                return

            total_pairs = len(ready_records) * len(model_specs)
            start_message = f"Lancement confirmé : {len(ready_records)} paire(s), {len(model_specs)} modèle(s), {total_pairs} évaluation(s) séquentielle(s)."
            LOGGER.info(
                "CNI launch accepted | pairs=%d | models=%d | strategy=%s | dpi=%s | timeout=%s | cpu_threads=%s | unload=%s | unlabeled=%d | invalid=%d",
                len(ready_records), len(model_specs), strategy, dpi, timeout, threads, unload, len(unlabeled), invalid_count,
            )
            yield view(start_message, "Initialisation des modèles en cours.", 0, None, start_message, total_pairs)

            cni_fields = load_cni_field_config(ROOT_DIR / "config" / "cni_fields.json")
            try:
                events = iter_cni_benchmark(
                    build_default_registry(), list(model_specs), ready_records, RUNS_DIR,
                    strategy=str(strategy), dpi=int(dpi), timeout_seconds=float(timeout or 0),
                    cpu_threads=int(threads or 1), unload_after_task=bool(unload),
                    fields=cni_fields, prompt_instructions=prompt_instructions,
                )
                for event in events:
                    total, completed = int(event.get("total", total_pairs)), int(event.get("completed", 0))
                    progress = completed / total * 100 if total else 0
                    client_id = event.get("folder_client_id", "—")
                    model = event.get("model", "—")
                    if event.get("stage") == "processing":
                        side = event.get("side", "document")
                        LOGGER.info("CNI processing | client=%s | model=%s | side=%s | completed=%d/%d", client_id, model, side, completed, total)
                        live = (
                            "### Analyse CNI en direct\n\n"
                            f"- **Client dossier :** `{client_id}`\n- **Modèle :** `{model}`\n"
                            f"- **Étape :** `{side}`\n- La sortie brute et le JSON seront conservés dès la réponse."
                        )
                        yield view("Lancement actif : consultez l’onglet 2. Suivi en direct.", f"Analyse en cours : {client_id} ({side})", progress, event.get("image_path"), live, total)
                        continue

                    result = event.get("result")
                    if result:
                        results.append(result)
                    status_value = (result or {}).get("status", "unknown")
                    LOGGER.info("CNI result | client=%s | model=%s | status=%s | completed=%d/%d", client_id, model, status_value, completed, total)
                    live = (
                        "### Dernier résultat\n\n"
                        f"- **Client :** `{client_id}`\n- **Modèle :** `{model}`\n"
                        f"- **Statut :** `{status_value}`\n- **Label :** `{(result or {}).get('label_status', '—')}`\n"
                        f"- **CIN recto/verso cohérent :** {_cni_boolean((result or {}).get('cin_coherent'))}"
                    )
                    yield view("Lancement actif : consultez l’onglet 2. Suivi en direct.", f"Résultat reçu : {client_id} ({status_value})", progress, (result or {}).get("recto_image_path"), live, total, select_last=True)
            except Exception as exc:
                LOGGER.exception("CNI benchmark interrupted")
                message = f"Benchmark CNI interrompu : {type(exc).__name__}: {exc}"
                yield view(message, message, 0, None, "Consultez le terminal : l'erreur complète y est enregistrée.", total_pairs)

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
                cpu_threads,
                unload_after_task,
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
            # Keep one benchmark execution at a time, but isolate it in its
            # own concurrency group. Navigation, dataset browsing and detail
            # buttons must continue to be served while this generator yields.
            concurrency_limit=1,
            concurrency_id="benchmark-run",
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
        # Do not attach a ``Tab.select`` handler here. In Gradio 6 a select
        # listener on a nested tab tree may be dispatched while *any* top-level
        # page is changed, serialising the full ``run_state`` in the browser and
        # freezing navigation. The explicit selector and previous/next buttons
        # below still load the detail on demand without coupling it to routing.
        previous_result.click(
            show_previous_detail,
            [detail_index, run_state],
            detail_outputs,
            queue=False,
        )
        next_result.click(
            show_next_detail,
            [detail_index, run_state],
            detail_outputs,
            queue=False,
        )
        # Do not bind ``run_state.change`` here. The benchmark generator emits
        # a state update for every image; recalculating all detail components on
        # every update caused the browser to serialize the growing result list
        # repeatedly and made the completed run appear frozen. The tab selector
        # and explicit previous/next controls are sufficient.
        result_selector.input(
            select_detail,
            [result_selector, run_state],
            detail_outputs,
            queue=False,
        )
        refresh_runs.click(
            reload_persisted_runs,
            outputs=[persisted_runs],
            queue=False,
        )
        open_run.click(
            open_persisted_run,
            inputs=[persisted_runs],
            outputs=[run_state, *detail_outputs, persisted_run_status],
            queue=False,
        )
        dataset_selector.change(
            browse_dataset,
            dataset_selector,
            [dataset_image, dataset_category, dataset_description, dataset_truth],
            queue=False,
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
        # Changer de mode ne modifie que la visibilité : un chemin saisi reste
        # mémorisé si l'utilisateur revient ensuite au mode dossier.
        cni_input_mode.change(
            _cni_source_mode_visibility,
            inputs=[cni_input_mode],
            outputs=[cni_folder_source, cni_zip_source],
            queue=False,
        )
        cni_scan.click(
            scan_cni_input,
            inputs=[cni_clients_root, cni_labels_root],
            outputs=[cni_clients_state, cni_scan_table, cni_scan_status, cni_source_selector],
            queue=False,
        )
        # L'aperçu est en lecture seule et provient uniquement du scan courant.
        cni_source_selector.change(
            _preview_cni_source,
            inputs=[cni_source_selector],
            outputs=[cni_source_preview, cni_source_preview_info],
            queue=False,
        )
        cni_refresh_models.click(
            refresh_cni_models,
            inputs=[cni_models],
            outputs=[cni_models],
            queue=False,
        )
        cni_refresh_prompt.click(
            _cni_prompt_preview,
            inputs=[cni_strategy, cni_prompt_instructions],
            outputs=[cni_prompt_preview],
            queue=False,
        )
        cni_strategy.change(
            _cni_prompt_preview,
            inputs=[cni_strategy, cni_prompt_instructions],
            outputs=[cni_prompt_preview],
            queue=False,
        )
        cni_event = cni_launch.click(
            on_cni_run,
            inputs=[
                cni_models,
                cni_clients_state,
                cni_strategy,
                cni_dpi,
                cni_timeout,
                cni_cpu_threads,
                cni_unload,
                cni_prompt_instructions,
                cni_continue_without_label,
            ],
            outputs=[
                cni_launch_feedback,
                cni_run_status,
                cni_progress,
                cni_live_image,
                cni_live_result,
                cni_live_counters,
                cni_live_table,
                cni_results_state,
                cni_results_table,
                cni_result_selector,
                cni_accuracy_plot,
                cni_latency_plot,
            ],
            concurrency_limit=1,
            concurrency_id="cni-benchmark-run",
        )
        cni_stop.click(fn=None, cancels=[cni_event])
        cni_apply_filters.click(
            filter_cni_results,
            inputs=[cni_results_state, cni_accuracy_min, cni_accuracy_max, cni_include_unscored],
            outputs=[cni_results_table],
            queue=False,
        )
        # Comme sur la page de résultats générale, l'exploration détaillée est
        # volontairement indépendante du générateur de benchmark : l'UI reste
        # réactive pendant l'arrivée progressive des nouveaux résultats.
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
        # Only refresh the run selector during startup. Loading the full result
        # payload is explicit, preventing a large raw response from blocking
        # the initial Gradio page and its tabs.
        app.load(
            startup_run_info,
            outputs=[persisted_runs, persisted_run_status],
            queue=False,
        )
    # Gradio's default queue limit can serialize every event behind a long OCR
    # request. A small global pool keeps lightweight UI actions responsive while
    # the benchmark group above remains strictly single-model/single-run.
    app.queue(default_concurrency_limit=4)
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
    parser.add_argument("--cpu-threads", type=int, default=None)
    parser.add_argument(
        "--unload-after-task", action=argparse.BooleanOptionalAction, default=True,
        help="Décharger le modèle Ollama après chaque image (défaut: activé).",
    )
    parser.add_argument("--host", default=os.getenv("GRADIO_SERVER_NAME", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("GRADIO_SERVER_PORT", "7860")))
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    if args.cli:
        summary, _, run_id = run_benchmark(
            args.models, args.category, args.noise, args.eval_mode,
            cpu_threads=args.cpu_threads,
            unload_after_task=args.unload_after_task,
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
        js=APP_JS,
        head=APP_HEAD,
    )


if __name__ == "__main__":
    main()
