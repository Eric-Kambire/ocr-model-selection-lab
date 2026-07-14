# Notebooks Colab par paires

Ces notebooks sont une alternative autonome à l'application locale. Chaque fichier
contient au maximum deux modèles et n'importe pas le dépôt GitHub de l'application.
Le générateur est `scripts/build_pair_notebooks.py`.

| Notebook | Modèles | Intention |
|---|---|---|
| `01_classic_ocr.ipynb` | EasyOCR, PP-OCRv6 | référence légère CPU/GPU |
| `02_transformers_documents.ipynb` | GLM-OCR, Granite Docling 258M | documents et sortie structurée |
| `03_paddle_qwen.ipynb` | PaddleOCR-VL 1.6, Qwen3.5 OCR 0.8B | vision/document et GGUF |
| `04_compact_vlm.ipynb` | MiniCPM-V 4.6, LightOnOCR-2 1B | VLM compacts |
| `05_specialized_gpu.ipynb` | Chandra OCR 2, dots.ocr | runtimes GPU spécialisés |
| `06_legacy_localization.ipynb` | Unlimited-OCR, LocateAnything-3B | modèles lourds / localisation |

## Utilisation

1. Ouvrir un notebook dans Colab et sélectionner un GPU.
2. Exécuter les cellules dans l'ordre, dans un runtime frais après une installation.
3. Renseigner `HF_TOKEN` dans les Secrets Colab seulement si le dépôt le demande.
4. Lire la table `SMOKE` avant de lancer le benchmark. `success` signifie téléchargement,
   chargement et inférence réussis; `failed_load` conserve l'erreur technique complète.
5. Les sorties sont enregistrées dans `/content/ocr_pair_benchmark/artifacts`, notamment
   `raw_outputs.jsonl` (une ligne par document, y compris timeout), `results.csv` et les graphiques.

Les modèles lourds ou spécialisés sont volontairement signalés comme incompatibles si la
mémoire/runtime Colab est insuffisante. Cela évite de boucler ou de présenter un faux succès.
