# OCR Model Selection Lab — exécution locale

Application locale pour comparer des modèles OCR et analyser des CNI marocaines.
Cette branche contient uniquement le code d'exécution local : **ni Docker, ni
notebook Colab, ni workflow de publication de conteneur**.

L'interface Gradio sert à choisir les documents, les modèles Ollama, les
paramètres d'exécution et à explorer les résultats. Le traitement est
séquentiel : un modèle et une tâche à la fois, afin de limiter la mémoire CPU ou
GPU.

## Démarrage rapide

Prérequis : Python 3.10 à 3.14 et `pip`. Installez Ollama séparément seulement
si vous voulez tester ses modèles.

### Windows — PowerShell

```powershell
git clone <URL_DU_REPO>
cd Benchmark
git switch --track origin/codex/clean-runtime
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python main.py
```

Si PowerShell refuse l'activation :

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### macOS / Linux — Terminal

```bash
git clone <URL_DU_REPO>
cd Benchmark
git switch --track origin/codex/clean-runtime
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python main.py
```

Ouvrez ensuite <http://127.0.0.1:7860>. Pour un autre port :

```bash
python main.py --port 7861
```

## Dépendances optionnelles

```bash
# EasyOCR local
python -m pip install -r requirements-ocr.txt

# Laboratoire FastAPI QlickEER (le serveur Gradio n'en a pas besoin)
python -m pip install -r requirements-gateway.txt

# Scripts d'import Kaggle
python -m pip install -r requirements-data.txt

# Tests et contrôle de style
python -m pip install -r requirements-dev.txt
python -m pytest -q
python -m ruff check .
```

`requirements.txt` ne mélange pas les dépendances du serveur FastAPI avec
l'application. Les versions restent volontairement compatibles avec les Python
récents ; les anciens correctifs Colab `numpy<2` ne s'appliquent pas à cette
branche locale.

## Utiliser Ollama

Ollama est une application séparée qui expose par défaut
`http://127.0.0.1:11434`. Après l'avoir installé et démarré :

```bash
ollama pull <modele-vision>
ollama list
```

Dans l'interface, utilisez la liste multi-sélection **Modèles Ollama**, puis le
bouton `↻` pour rafraîchir les modèles disponibles. L'application appelle
l'API locale Ollama ; elle ne télécharge pas automatiquement de poids.

## Architecture

Le projet est un **monolithe modulaire à architecture hexagonale légère** :

```text
Gradio / CLI
      │
      ▼
services applicatifs ── benchmark, CNI, QlickEER, runs, rétention
      │
      ▼
domaine et orchestration ── évaluation, statut, métriques, séquence
      │
      ▼
adaptateurs ── Ollama, fichiers, PDF/images, ZIP, API HTTP
```

```text
main.py                                      # composition Gradio / CLI
ocr_benchmark/application/benchmark_service.py # orchestration benchmark classique
ocr_benchmark/application/cni_service.py       # scan et préparation CNI
ocr_benchmark/application/qlicker_api_service.py # HTTP, proxy, SSL, fichiers binaires
ocr_benchmark/application/qlicker_cni_import_service.py # import API multi-clients
ocr_benchmark/application/retention_service.py # anonymisation et nettoyage sûr
ocr_benchmark/cni_ingestion.py                # dossiers, PDF/images, labels JSONB
ocr_benchmark/cni_preprocessing.py            # rotation, crop, perspective
ocr_benchmark/cni_runner.py                   # analyse CNI séquentielle et artefacts
ocr_benchmark/cni_schema.py                   # champs, prompts et fusion JSON
models/                                       # adaptateurs OCR
config/cni_fields.json                        # champs CNI modifiables sans code
scripts/                                      # outils indépendants et laboratoires
tests/                                        # tests automatisés
```

La séparation permet de réutiliser le traitement documentaire dans une future
chaîne de détection de fraude sans réécrire la logique dans Gradio.

## CNI et QlickEER

La partie API vit uniquement dans **8. Benchmark CNI**. Son objectif est de
ramener des dossiers clients depuis QlickEER vers un lot local contrôlé, puis de
faire l'OCR sans redemander les mêmes documents au serveur.

### 1. Configurer une fois les routes

Ouvrez `8. Benchmark CNI → 4. Paramètres → API QlickEER`. Dans l'onglet
**Connexion**, indiquez :

- la **Base URL** commune, par exemple `https://serveur-interne` ;
- le **timeout HTTP** en secondes ;
- soit **Utiliser le proxy système**, soit un **proxy explicite** ;
- **Vérifier SSL**. Ne le désactivez que pour une API interne dont le certificat
  est connu et approuvé par votre organisation.

Les quatre onglets suivants correspondent aux quatre appels disponibles. Collez
dans chacun l'URL complète testée dans Postman, puis cliquez sur **Parser et
enregistrer**.

| Onglet | Fonction QlickEER | Entrées injectées par l'application | Résultat utilisé |
|---|---|---|---|
| Clients | `get_customer` | filtres, dates, `step`, `page`, `pageSize` et autres paramètres | liste paginée des clients trouvés |
| Détail client | `get_customer_data` | `customerID`, `loadDocuments` et paramètres restants | données client et `customer_data` utilisé comme label candidat |
| Documents | `get_signed_documents_list` | `customerID` | noms des fichiers disponibles, dont les CNI recto/verso |
| Fichier | `view_file` | `customerID`, `page`, `file` | octets du PDF ou de l'image |

Le parseur sépare une URL en **endpoint** et en tableau de paramètres. Après le
parsing, vous pouvez modifier la valeur de chaque paramètre, décider s'il doit
être envoyé, ou ajouter une ligne. Il n'exécute aucune requête tant que vous ne
lancez pas une action de l'interface.

### 2. Rechercher et sélectionner plusieurs clients

Retournez dans `8. Benchmark CNI → 1. Préparer`, choisissez la source
**API QlickEER**, puis lancez la recherche. La réponse `get_customer` est
affichée dans le diagnostic sous forme de candidats. Chaque ligne est
sélectionnable ; **Tout sélectionner** est seulement un raccourci. Le compteur
résume total trouvé, sélection, prêts, en attente et erreurs.

Cliquez sur **Préparer la sélection**. Cette action ne lance pas l'OCR : elle
traite les clients sélectionnés un par un, afin de limiter mémoire et charge
réseau.

```text
client découvert
  → liste des documents
  → recto/verso CNI détectés
  → fichiers téléchargés localement
  → get_customer_data normalisé comme label
  → prêt (ou prêt sans label / erreur explicite)
```

Le diagnostic affiche l'état de chaque client. Un client sans label peut être
analysé en cochant **Continuer sans labels** ; il produira des métriques de
vitesse, mais pas d'accuracy de comparaison.

### 3. Où sont écrits les fichiers et quand l'API est rappelée

Chaque préparation crée un lot :

```text
cni_imports/qlickeer_api/batch-AAAAmmJJ-HHMMSS-<id>/
  <client-id>/
    <client-id>_CIN_Recto.pdf | .jpg | .png
    <client-id>_CIN_Verso.pdf | .jpg | .png
    <client-id>.json
```

Une fois le lot prêt, les vues `2. Suivi en direct` et `3. Résultats` lisent
ces fichiers locaux et les artefacts du run : elles ne rappellent pas
QlickEER. Un nouveau clic sur **Préparer la sélection** crée en revanche un
nouveau lot et rappelle l'API. Après un rechargement de page, rescanner le
dossier de lot permet de reprendre la préparation locale ; l'état visuel de la
session n'est pas une base de données.

### 4. Lancer et interpréter l'analyse

Dans `4. Paramètres`, choisissez les modèles Ollama, le prompt, la stratégie
recto/verso et les options de prétraitement. Le lancement effectue ensuite
l'extraction **séquentiellement**, un client/modèle à la fois. `2. Suivi en
direct` montre l'image et le résultat courant ; `3. Résultats` contient les
filtres, graphiques, JSON attendu, JSON extrait et retour brut conservé en cas
d'erreur ou de timeout.

`view_file` peut répondre avec des octets binaires. Le téléchargement préserve
le format retourné : `application/pdf` devient `.pdf`, `image/jpeg` devient
`.jpg` et `image/png` devient `.png`. Il n'y a **pas de conversion HTTP**. Lors
du prétraitement OCR, un PDF est rendu en image PNG à la résolution choisie ;
une image JPEG/PNG est ouverte directement. Cette image de travail est ensuite
éventuellement tournée, redressée, recadrée et envoyée au modèle.

## Données, archive anonymisée et nettoyage

Les résultats détaillés contiennent potentiellement des images CNI, du texte
OCR, des JSON, des chemins et des identifiants. Ils sont écrits localement sous
`runs/cni-.../`. Les lots téléchargés via QlickEER sont placés sous
`cni_imports/qlickeer_api/batch-.../`. Ces répertoires sont ignorés par Git.

Après une analyse terminée, ouvrez `4. Paramètres → Nettoyage` :

- **archive anonymisée** : conserve seulement le modèle, le statut, les
  scores, les temps, tokens et un alias temporaire `case-001` ; aucune table
  de correspondance avec le client réel n'est écrite ;
- **suppression du run détaillé** : enlève les images, les PDF rendus, les
  JSON, les sorties brutes et les mesures identifiantes ;
- **suppression des imports QlickEER** : enlève seulement les lots
  `cni_imports/qlickeer_api/batch-*`, jamais un dossier local saisi par
  l'utilisateur ;
- **aperçus temporaires** : efface `runs/cni_source_previews/`.

La suppression d'un run détaillé est refusée si l'archive anonymisée n'est pas
créée. Une analyse active ne peut pas être nettoyée. Les archives sont stockées
dans `analysis_archive/` et peuvent être rechargées dans `3. Résultats` pour
consulter graphiques et métriques, sans pouvoir afficher un document ou une
identité.

Pour un usage interne, protégez le disque de l'hôte (BitLocker, FileVault ou
LUKS), limitez les permissions du dossier de travail au groupe applicatif, et
définissez une durée de rétention adaptée à votre politique.

## Commandes utiles

```bash
# Interface Gradio
python main.py

# Benchmark classique sans interface
python main.py --cli --models mock:MockOCR-V1 --category tables

# Logs détaillés
# PowerShell
$env:LOG_LEVEL = "DEBUG"; python main.py
# macOS / Linux
LOG_LEVEL=DEBUG python main.py
```

Les logs terminal donnent les étapes d'import, d'inférence et de nettoyage. Les
résultats techniques détaillés ne doivent pas être copiés dans un ticket ou un
canal de discussion sans anonymisation préalable.
