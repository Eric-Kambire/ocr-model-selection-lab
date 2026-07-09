# OCR Model Selection Lab

Plateforme extensible pour comparer des modèles OCR sur la qualité, la vitesse et
la fiabilité. Elle fonctionne en interface Gradio ou en CLI, sur CPU local, dans
Docker, et sur CPU/GPU dans Google Colab.

## Démarrage local CPU

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-ocr.txt
python main.py
```

L’interface est disponible sur `http://127.0.0.1:7860`.

Pour utiliser Ollama, lancez Ollama localement et installez au moins un modèle
vision. Les modèles détectés apparaissent automatiquement dans l’interface.

## Docker CPU

```bash
docker compose up --build
```

Ouvrez `http://localhost:7860`. Le conteneur contacte Ollama sur la machine hôte
via `host.docker.internal:11434`. Les résultats sont conservés dans le volume
Docker `benchmark-runs`.

Le Dockerfile ne contient aucun secret. Copiez `.env.example` vers `.env`
uniquement si une configuration locale est nécessaire.

## Docker GPU

Prérequis : GPU NVIDIA, pilote compatible, NVIDIA Container Toolkit et support
Compose de `gpus: all`.

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

EasyOCR détecte CUDA automatiquement. Sur une machine CPU, utilisez uniquement
le fichier Compose principal.

## Google Colab

Ouvrez `benchmark_colab.ipynb`, choisissez `CPU` ou `T4 GPU` dans
**Runtime > Change runtime type**, puis exécutez les cellules dans l’ordre. Le
notebook vérifie le matériel réellement disponible et lance une URL Gradio
partageable.

## CLI

```bash
python main.py --cli --models mock:MockOCR-V1 --category tables
python main.py --cli --models ollama:llama3.2-vision --eval-mode Bankmark
```

Chaque exécution produit un répertoire `runs/<run_id>/` contenant :

- `results.json` : résultats complets et typés ;
- `summary.csv` : comparaison par modèle ;
- `details.csv` : résultat par document ;
- `report.md` : synthèse et définitions.

## Ajouter des données

L’onglet **Ajouter des données** accepte une image JPG, JPEG, PNG ou WEBP de
15 Mio maximum, un label obligatoire, une catégorie et une description.

Le label doit être la transcription exacte du document. Conservez les retours à
la ligne, utilisez un tableau Markdown lorsque nécessaire et n’ajoutez aucun
texte absent de l’image. Les ajouts sont copiés dans `dataset/user_uploads/` et
le catalogue est remplacé atomiquement.

## Import Kaggle reproductible

Les 30 formulaires FUNSD annotés présents dans `dataset/kaggle_forms/` peuvent
être réimportés avec :

```bash
pip install -r requirements-data.txt
python scripts/import_kaggle_forms.py --count 30
```

Source : `senju14/ocr-dataset-of-multi-type-documents`, licence MIT déclarée
sur Kaggle. Le script apparie images et annotations par identifiant, car certains
couples sont répartis dans des dossiers de split différents.

## Ajouter un modèle

Un adaptateur doit exposer `model_name` et `perform_ocr(image_path)`. Enregistrez
ensuite une factory :

```python
from ocr_benchmark.registry import build_default_registry

registry = build_default_registry()
registry.register(
    "my_provider",
    lambda model_name, **options: MyOCRAdapter(model_name),
)
```

Le modèle devient adressable sous la forme `my_provider:model-name`. Le résultat
standard contient `text`, `latency`, `status`, `error`, `device` et, lorsque le
fournisseur les expose, `input_tokens`, `output_tokens`, `tokens_per_second`.

Ne fabriquez pas une mesure de tokens pour les moteurs OCR classiques : elle
n’est pas comparable aux tokens d’un modèle génératif.

## Protocole de sélection

1. Vérifier le taux de réussite technique.
2. Fixer un seuil de qualité par catégorie.
3. Comparer médiane et P95, pas uniquement la moyenne.
4. Examiner les métriques critiques du métier, comme les IBAN et les montants.
5. Valider le finaliste sur un corpus réel tenu à l’écart du développement.

Les définitions détaillées sont disponibles dans l’onglet
**Comprendre les métriques**.

## Tests

```bash
pip install -r requirements-dev.txt
pytest -q
```
