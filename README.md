# OCR Model Selection Lab

Plateforme extensible pour comparer des modèles OCR sur la qualité, la vitesse et
la fiabilité. Elle fonctionne en interface Gradio ou en CLI, sur CPU local, dans
Docker, et sur CPU/GPU dans Google Colab.

## Architecture locale

Le projet est un **monolithe modulaire avec une architecture hexagonale légère** :
Gradio, le CLI et une future API sont les interfaces ; les services applicatifs
portent les cas d'usage ; le domaine calcule et évalue ; les adaptateurs parlent
à Ollama, au disque, aux ZIP ou à une API externe. Ainsi, la logique n'est pas
liée à Gradio et peut être réutilisée plus tard dans un outil de détection de
fraude ou un worker serveur.

```text
Interface (Gradio / CLI / future API)
                  ↓
Services applicatifs
                  ↓
Domaine benchmark et CNI
                  ↓
Adaptateurs (Ollama, fichiers, ZIP, Qlicker, Docker)
```

Les fichiers ont des responsabilités explicites :

```text
main.py                         # démarrage Gradio ou CLI, configuration et flux UI
ocr_benchmark/application/benchmark_service.py # cas d'usage benchmark classique
ocr_benchmark/application/cni_service.py       # scan/import/exécution CNI
ocr_benchmark/application/run_service.py       # restauration et rétention des runs
ocr_benchmark/domain.py         # objets métier : cas, inférence, résultat, statuts
ocr_benchmark/runner.py         # orchestration séquentielle, timeout et progression
ocr_benchmark/registry.py       # frontière entre un nom de modèle et son adaptateur
ocr_benchmark/dataset_repository.py # lecture/écriture contrôlée du catalogue
ocr_benchmark/reporting.py      # exports JSON/CSV/Markdown et traces JSONL
models/                         # adaptateurs EasyOCR, Ollama et Mock
dataset/                        # catalogue et images, jamais de secret
runs/<run_id>/                  # artefacts d'une exécution, ignorés par Git
```

### Sous-système CNI marocaines

Le flux CNI n’est pas concentré dans un seul gros fichier :

```text
ocr_benchmark/cni_ingestion.py  # scan client, import ZIP, JSONB externe → JSON local
ocr_benchmark/cni_images.py     # rendu PDF/image et opérations d'image réutilisables
ocr_benchmark/cni_preprocessing.py # source unique : rotation, contour et crop CNI
ocr_benchmark/cni_schema.py     # champs configurables, prompt, parsing et fusion JSON
ocr_benchmark/cni_runner.py     # exécution séquentielle, live events, artefacts de run
ocr_benchmark/cni.py            # façade d'import compatible pour le reste de l'application
config/cni_fields.json          # champs d'extraction modifiables sans changer le code
docs/CNI_BENCHMARK_IMPLEMENTATION_PLAN.md # contrat de données et décisions métier
```

Chaque module a une seule responsabilité. La préparation CNI n'est notamment
implémentée qu'une fois dans `cni_preprocessing.py`; `cni_images.py` l'utilise
au lieu de dupliquer la logique. Un problème de fichier, crop, réponse JSON ou
exécution peut donc être retrouvé dans le bon module, sans modifier l'interface.

La description détaillée des frontières, de la réutilisation future et des
choix de stockage est disponible dans `docs/ARCHITECTURE_INTERNE.md`.

Le flux d'une image est : `dataset → registry → adapter → runner → evaluator →
reporting/UI`. Le runner charge un seul modèle à la fois, traite les documents,
appelle `close()` dans un bloc `finally`, puis passe au modèle suivant. Cette
contrainte est volontaire : elle évite de garder plusieurs modèles en mémoire
sur un poste CPU ou une petite carte GPU.

## Timeout et sorties tardives

Le timeout du runner protège l'interface, mais Python ne peut pas tuer sans risque
un thread fournisseur déjà engagé. L'appel est donc marqué `timeout` au moment
limite et exclu des scores. Si le fournisseur finit plus tard, sa réponse brute
est enregistrée dans `traces.jsonl` avec `timing: "late_after_timeout"`; elle ne
réapparaît jamais comme un succès. Pour Ollama, configurez aussi le timeout HTTP
dans les paramètres afin d'arrêter la requête réseau réelle.

## Débogage local

Activez les logs détaillés avant de démarrer :

```powershell
$env:LOG_LEVEL = "DEBUG"
python main.py
```

Les événements importants sont écrits dans le terminal : création du modèle,
début/fin de chaque inférence, statut, latence, tokens, timeout et libération de
la mémoire. Pour une exécution reproductible, utilisez le CLI :

```powershell
python main.py --cli --models mock:MockOCR-V1 --category cheques
```

Si une image échoue, vérifiez d'abord `runs/<run_id>/traces.jsonl`, puis
`details.csv`. Une erreur d'adaptateur est isolée au document concerné et ne doit
pas empêcher les autres modèles de produire leurs résultats.

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

## Utiliser l’interface

Dans **Benchmark**, choisissez les modèles puis la quantité de documents :

- tout le dataset ;
- une quantité globale, répartie entre les catégories ;
- une quantité différente pour chaque catégorie.

Cliquez sur **Préparer le benchmark** pour vérifier le plan, puis sur
**Confirmer et lancer**. Pendant l’exécution, l’image courante, le résultat OCR,
la qualité, CER, WER, latence, compteurs, progression et ETA sont actualisés.
**Annuler** interrompt la file et conserve les résultats déjà produits.

Dans **Résultats détaillés**, la liste permet d’ouvrir directement une
évaluation ou de naviguer avec **Précédent/Suivant**. Le temps, la qualité et
les compteurs de tokens sont affichés au-dessus du comparatif. La sortie peut
être consultée comme transcription, réponse fournisseur brute, Markdown rendu
ou source HTML non exécutée.

L’onglet **Paramètres** permet notamment de fixer le temps maximal par image,
le nombre maximal d’erreurs, la seed de sélection, le mélange des documents et
la sauvegarde après chaque résultat. Il expose aussi le prompt envoyé aux
modèles génératifs compatibles. Un appel fournisseur qui dépasse le timeout
peut finir en arrière-plan, mais son résultat est ignoré par le benchmark.

## Docker CPU

```bash
docker compose up --build
```

Ouvrez `http://localhost:7860`. Le conteneur contacte Ollama sur la machine hôte
via `host.docker.internal:11434`. Les résultats sont conservés dans le volume
Docker `benchmark-runs`. Le service `init-storage` initialise les volumes
`benchmark-runs` et `benchmark-dataset` avec les permissions du compte applicatif
non-root. Les données CNI, imports et logs sont exclus du contexte de build via
`.dockerignore`.

Le Dockerfile ne contient aucun secret. Copiez `.env.example` vers `.env`
uniquement si une configuration locale est nécessaire.

## Données sensibles, volumes et rétention

Les scans CNI, labels, sorties brutes et runs ne doivent pas être committés ni
placés dans l'image Docker. Montez des volumes dédiés et protégez le disque de
l'hôte (BitLocker sous Windows, LUKS sous Linux) ; les permissions Docker ne
remplacent pas le chiffrement du disque.

`RUN_RETENTION_DAYS=30` est la valeur par défaut : les dossiers de runs générés
plus anciens sont supprimés au démarrage de l'interface. Mettez une valeur
négative pour désactiver la purge temporairement. Les détails de déploiement et
les limites actuelles (authentification, Qlicker et stockage central) figurent
dans `DEPLOYMENT.md` et `docs/ARCHITECTURE_INTERNE.md`.

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
**Runtime > Change runtime type**, puis exécutez les cellules dans l’ordre.

Le notebook Colab est autonome : il ne clone pas le repo et ne dépend pas du
code local pour exécuter le benchmark. La logique minimale nécessaire est écrite
directement dans les cellules : dataset, adaptateurs, métriques, graphiques,
exports et interface Gradio légère.

Le notebook utilise par défaut un mini dataset synthétique. Pour tester vos
propres documents dans Colab, activez `IMPORT_CUSTOM_DATASET_ZIP = True` puis
uploadez un ZIP contenant soit `labels.csv`, soit `dataset.json`.

Format CSV minimal :

```csv
image_path,ground_truth,category,description
images/cheque_001.png,"texte attendu exact",bank,"chèque manuscrit"
images/form_001.jpg,"texte attendu exact",handwritten_form,"formulaire rempli"
```

Les chemins `image_path` sont relatifs au ZIP. Les images sont copiées dans
le workspace temporaire Colab et ajoutées au dataset de la session.

Le notebook contient aussi des cellules prêtes pour préparer plusieurs familles
de modèles : EasyOCR, PaddleOCR, Qwen OCR 0.8B, MiniCPM-V 4.6, Chandra OCR,
LightOnOCR, dots.ocr, PaddleOCR-VL, LocateAnything et Unlimited OCR. Certains
modèles sont téléchargés seulement : ils nécessitent un adaptateur Python
spécifique avant d’être comparables.

## CLI

```bash
python main.py --cli --models mock:MockOCR-V1 --category tables
python main.py --cli --models ollama:llama3.2-vision --eval-mode Bankmark
```

Chaque exécution produit un répertoire `runs/<run_id>/` contenant :

- `results.json` : résultats complets et typés ;
- `summary.csv` : comparaison par modèle ;
- `details.csv` : résultat par document ;
- `traces.jsonl` : sorties fournisseur brutes, texte et raisonnement exposé, y
  compris les réponses arrivées après un timeout ;
- `report.md` : synthèse et définitions.

Une réponse reçue après le timeout reste exclue des scores et conserve le statut
`timeout`. Elle est néanmoins ajoutée à `traces.jsonl` avec
`timing: "late_after_timeout"` afin de permettre l’audit et un nettoyage
ultérieur.

## Convertir des images brutes en PDF A4

Le script [images_to_a4_corner_pdf.py](scripts/images_to_a4_corner_pdf.py)
crée **un PDF A4 par image**, avec l'image conservée dans un coin. Il est utile
pour préparer des scans de CNI avant le scan du dossier client.

Mode interactif : le script liste les images du dossier, puis demande `all`
pour tout convertir ou une sélection telle que `1,3-5`.

```powershell
python scripts/images_to_a4_corner_pdf.py "D:\data\clients\client-001"
```

Par défaut, les PDF sont créés dans le même dossier que les images afin de
conserver un nom comme `client_CIN_Recto.pdf`. L'image est placée en **haut à
gauche**, dans une zone standard de **120 × 90 mm**, sans recadrage ni
déformation. Vous pouvez choisir un autre coin et automatiser la sélection :

```powershell
python scripts/images_to_a4_corner_pdf.py "D:\data\clients\client-001" --select all --corner bottom-left
```

Options utiles : `--recursive` pour les sous-dossiers, `--output-dir` pour
écrire ailleurs et `--overwrite` pour remplacer un PDF existant.

## Regrouper les PDF Recto/Verso en dossiers clients

Le script [group_cni_pdf_pairs.py](scripts/group_cni_pdf_pairs.py) prend un
dossier plat de PDF et repère les paires ayant le même préfixe :
`123_CIN_Recto.pdf` avec `123_CIN_Verso.pdf`, par exemple. Il crée ensuite un
dossier aléatoire par paire (`client-a1b2c3d4e5f6`) dans
`clients_generated/`, en conservant les noms de PDF.

```powershell
python scripts/group_cni_pdf_pairs.py "D:\data\pdf_plats"
```

Par sécurité, le script **copie** les PDF ; les originaux restent intacts.
Utilisez `--move` seulement lorsque vous voulez réellement les déplacer. Avant
toute écriture, vous pouvez vérifier les paires et les PDF orphelins :

```powershell
python scripts/group_cni_pdf_pairs.py "D:\data\pdf_plats" --dry-run
```

Le fichier `clients_generated/client_mapping.json` relie chaque identifiant
aléatoire au préfixe de la paire source. Les PDF sans recto/verso associé et
les doublons sont explicitement affichés dans le terminal.

## Coller un recto et un verso en une image

Le script [combine_cni_image_pairs.py](scripts/combine_cni_image_pairs.py)
trouve les paires image ayant le même préfixe et produit une image verticale :
recto en haut, verso en bas. Par défaut, il crée du **PNG sans perte** : les
images ne sont ni recadrées ni redimensionnées.

```powershell
python scripts/combine_cni_image_pairs.py "D:\data\images"
```

Les collages sont créés dans `combined_images/`. Pour réduire la taille des
fichiers avec JPEG, choisissez une qualité explicite :

```powershell
python scripts/combine_cni_image_pairs.py "D:\data\images" --format jpeg --jpeg-quality 85
```

`--png-compress-level 0..9` règle uniquement la compression PNG sans perte ;
il ne réduit pas la qualité visuelle.

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
