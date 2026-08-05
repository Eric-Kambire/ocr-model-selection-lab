# Diagnostic pas à pas du runtime Ollama

Ces scripts isolent chaque couche du Benchmark CNI. Ils ne modifient aucun
modèle et enregistrent leurs rapports dans `runs/runtime_debug/`, dossier ignoré
par Git.

L'objectif précis est de trouver quelle couche impose la coupure observée vers
60 secondes :

```text
Modèle / moteur Ollama
        ↓
Serveur Ollama
        ↓
HTTP, proxy et TLS
        ↓
SDK Python ollama / HTTPX
        ↓
OllamaOCRModel et BenchmarkRunner
        ↓
Interface Gradio
```

Exécutez les commandes depuis la racine du dépôt, avec le même environnement
Python que l’application.

## Télécharger le dépôt la première fois

Depuis le dossier `RUNTIME` :

```bash
git clone --branch codex/cni-ui-workflow --single-branch \
  https://github.com/Eric-Kambire/ocr-model-selection-lab.git

cd ocr-model-selection-lab
```

La commande `--single-branch` télécharge seulement l'historique nécessaire à
`codex/cni-ui-workflow`.

## Mettre à jour le dépôt déjà téléchargé

Si le terminal se trouve dans le dossier parent `RUNTIME`, cette commande met à
jour le dépôt sans devoir d'abord entrer dedans :

```bash
git -C ocr-model-selection-lab pull --ff-only origin codex/cni-ui-workflow
```

Puis entrez dans le dépôt :

```bash
cd ocr-model-selection-lab
```

Si le terminal se trouve déjà dans `ocr-model-selection-lab` :

```bash
git switch codex/cni-ui-workflow
git pull --ff-only origin codex/cni-ui-workflow
```

Avant la mise à jour, vous pouvez contrôler votre situation avec :

```bash
git status
git branch --show-current
```

`--ff-only` est volontaire : Git refuse de fabriquer automatiquement un commit
de fusion si le dossier contient un historique divergent. En présence de
modifications locales importantes, sauvegardez-les avant le `pull`.

## Préparer l'environnement Python sur macOS

Pour créer un environnement la première fois :

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Pour les exécutions suivantes :

```bash
cd /chemin/vers/RUNTIME/ocr-model-selection-lab
source .venv/bin/activate
```

Vérifiez que vous utilisez le bon Python :

```bash
which python
python --version
```

## Différence entre les scripts

| Script | Couche testée | Charge un modèle ? | Envoie l'image ? | Ce qu'il permet d'exclure |
|---|---|---:|---:|---|
| `00_environment.py` | Python, variables shell/launchctl et versions | Non | Non | Mauvais environnement, mauvais `OLLAMA_HOST`, proxy caché |
| `01_network_health.py` | HTTPX vers `/api/version` et `/api/tags` | Non | Non | DNS, TCP, TLS, proxy et serveur indisponible |
| `02_model_show.py` | SDK Ollama, `list`, `show`, capacité vision | Non | Non | Mauvais nom de modèle ou modèle sans vision |
| `03_prepare_image.py` | Conversion PDF/JPEG/PNG de l'application | Non | Non | Image invalide ou problème de conversion |
| `04_direct_ollama_image.py` | SDK Python Ollama direct | Oui | Oui | Gradio et BenchmarkRunner |
| `05_application_runner_image.py` | Adaptateur réel et BenchmarkRunner | Oui | Oui | Permet de comparer SDK direct et application |
| `06_raw_http_image.py` | HTTPX brut vers `POST /api/chat`, sans SDK | Oui | Oui | SDK Python Ollama |
| `07_mac_ollama_logs.py` | Journaux du serveur et de l'application macOS | Observe | Non | Montre si le serveur continue ou s'arrête réellement |

## 0. Environnement

```bash
python scripts/runtime_debug/00_environment.py \
  --output runs/runtime_debug/00_environment.json
```

Vérifiez notamment `OLLAMA_HOST`, `OLLAMA_LOAD_TIMEOUT`, les proxies,
`NO_PROXY`, l’exécutable Python et les versions `ollama/httpx`.

Sur macOS, une application démarrée graphiquement peut recevoir des variables
`launchctl` différentes de celles visibles dans le terminal. Le rapport affiche
les deux sources.

## 1. Santé réseau sans modèle

Avec l’environnement normal :

```bash
python scripts/runtime_debug/01_network_health.py \
  --timeout 30 \
  --output runs/runtime_debug/01_network_normal.json
```

Puis sans les proxies hérités par HTTPX :

```bash
python scripts/runtime_debug/01_network_health.py \
  --timeout 30 \
  --ignore-env-proxy \
  --output runs/runtime_debug/01_network_no_proxy.json
```

Si le premier échoue et le second réussit, le proxy système intercepte Ollama.

## 2. Liste, show et capacité vision

```bash
python scripts/runtime_debug/02_model_show.py \
  --model "NOM_EXACT_DU_MODELE" \
  --timeout 30 \
  --output runs/runtime_debug/02_model_show.json
```

Le rapport contient la réponse complète de `ollama show`, les paramètres du
modèle, son Modelfile lorsqu’Ollama le fournit et la preuve de capacité vision.

## 3. Conversion du document

PDF :

```bash
python scripts/runtime_debug/03_prepare_image.py \
  "/chemin/document.pdf" \
  --dpi 300
```

JPEG ou PNG :

```bash
python scripts/runtime_debug/03_prepare_image.py \
  "/chemin/document.jpg"
```

Le script appelle la même fonction `prepare_cni_source` que le workflow CNI.
Il affiche dimensions, taille, SHA-256 et chemin du PNG produit.

## 4. Appel Ollama direct

Commencez sans streaming, comme l’application :

```bash
python scripts/runtime_debug/04_direct_ollama_image.py \
  "runs/runtime_debug/prepared/document.png" \
  --model "NOM_EXACT_DU_MODELE" \
  --timeout 300 \
  --num-ctx 8192 \
  --num-predict 4096
```

Testez ensuite le streaming pour voir les fragments en direct :

```bash
python scripts/runtime_debug/04_direct_ollama_image.py \
  "runs/runtime_debug/prepared/document.png" \
  --model "NOM_EXACT_DU_MODELE" \
  --timeout 300 \
  --stream \
  --think true
```

En streaming, le rapport conserve aussi les fragments déjà reçus si Ollama
coupe la connexion. Les champs `first_chunk_seconds` et
`last_chunk_seconds` indiquent quand le premier et le dernier fragment sont
arrivés.

La différence est essentielle :

- sans streaming, aucun contenu n'est nécessairement reçu avant la fin ;
- avec streaming, Ollama envoie progressivement des lignes contenant
  `thinking` ou `content` ;
- un proxy avec une limite d'inactivité peut couper le premier cas et laisser
  fonctionner le second.

Pour ignorer les proxies :

```bash
python scripts/runtime_debug/04_direct_ollama_image.py \
  "runs/runtime_debug/prepared/document.png" \
  --model "NOM_EXACT_DU_MODELE" \
  --timeout 300 \
  --ignore-env-proxy
```

Pour un modèle dont le `SYSTEM` est embarqué dans son Modelfile :

```bash
python scripts/runtime_debug/04_direct_ollama_image.py \
  "runs/runtime_debug/prepared/document.png" \
  --model "NOM_DU_MODELE_DERIVE" \
  --timeout 300 \
  --image-only
```

Pour utiliser un long prompt système stocké dans un fichier :

```bash
python scripts/runtime_debug/04_direct_ollama_image.py \
  "runs/runtime_debug/prepared/document.png" \
  --model "NOM_EXACT_DU_MODELE" \
  --system-file "/chemin/system_prompt.txt" \
  --prompt-file "/chemin/user_prompt.txt" \
  --timeout 300
```

## 5. Même appel via l’application

Utilisez volontairement deux limites différentes. Ainsi, une erreur à 60 s
alors que les deux valeurs valent 300 ne peut pas venir de ces paramètres :

```bash
python scripts/runtime_debug/05_application_runner_image.py \
  "runs/runtime_debug/prepared/document.png" \
  --model "NOM_EXACT_DU_MODELE" \
  --http-timeout 300 \
  --runner-timeout 300
```

## 6. Appel HTTP brut sans le SDK Ollama

Ce script appelle directement `POST /api/chat` avec HTTPX. Il sépare le SDK
`ollama` de la couche HTTP située en dessous.

Les quatre limites sont indépendantes :

- `connect-timeout` : connexion DNS, TCP et TLS ;
- `read-timeout` : attente entre deux lectures de la réponse ;
- `write-timeout` : envoi du JSON et de l'image encodée ;
- `pool-timeout` : attente d'une connexion HTTPX disponible.

Sans streaming :

```bash
python scripts/runtime_debug/06_raw_http_image.py \
  "runs/runtime_debug/prepared/document.png" \
  --model "NOM_EXACT_DU_MODELE" \
  --connect-timeout 30 \
  --read-timeout 300 \
  --write-timeout 300 \
  --output runs/runtime_debug/raw_non_stream.json
```

Avec streaming :

```bash
python scripts/runtime_debug/06_raw_http_image.py \
  "runs/runtime_debug/prepared/document.png" \
  --model "NOM_EXACT_DU_MODELE" \
  --connect-timeout 30 \
  --read-timeout 300 \
  --write-timeout 300 \
  --stream \
  --output runs/runtime_debug/raw_stream.json
```

## 7. Logs Ollama sur macOS

Dans un premier terminal, vous pouvez observer directement le serveur :

```bash
tail -f ~/.ollama/logs/server.log
```

Dans un deuxième terminal, lancez le test. Exécutez ensuite cette commande
immédiatement après la coupure :

```bash
python scripts/runtime_debug/07_mac_ollama_logs.py \
  --lines 500 \
  --output runs/runtime_debug/mac_logs_after_timeout.json
```

Le script lit `~/.ollama/logs/server.log` et
`~/.ollama/logs/app.log`, puis conserve le journal complet et les lignes
contenant notamment `timeout`, `cancel`, `runner`, `memory` ou `error`.

## Hypothèses et lecture du diagnostic

| Observation | Hypothèse principale | Vérification suivante |
|---|---|---|
| Étape 1 échoue normalement mais réussit avec `--ignore-env-proxy` | Le proxy intercepte l'appel local | Vérifier `HTTP_PROXY`, `HTTPS_PROXY`, `ALL_PROXY` et `NO_PROXY` |
| Étape 2 indique `vision=false` | Le modèle est un LLM texte, pas un VLM | Choisir un modèle déclarant la capacité `vision` |
| Étape 3 échoue | Le problème précède Ollama | Vérifier format, chemin, Pillow et PyMuPDF |
| SDK sans streaming coupe à 60 s, SDK streaming réussit | Limite d'inactivité réseau probable | Comparer les deux modes HTTP bruts de l'étape 6 |
| HTTP brut streaming réussit, HTTP brut non-streaming coupe | Proxy, passerelle ou TLS coupe une connexion silencieuse | Tester `--ignore-env-proxy` et vérifier `NO_PROXY` |
| HTTP brut réussit mais SDK direct échoue | SDK Ollama ou configuration HTTPX du SDK | Comparer les rapports 04 et 06 |
| SDK direct réussit mais runner échoue | Timeout ou coordination dans l'application | Comparer `http_timeout_seconds` et `runner_timeout_seconds` |
| Le client échoue mais `server.log` continue la génération | Le serveur n'a pas imposé le timeout | Chercher client, proxy, TLS ou runner |
| Le serveur écrit une erreur et arrête le runner | Ollama, mémoire, Metal ou modèle | Lire les lignes `runner`, `memory`, `metal`, `panic` |
| `done_reason=length` | Limite de tokens atteinte | Augmenter `num_predict` si le modèle le permet |
| `done_reason=stop` | Arrêt normal décidé par le modèle | Ce n'est pas un timeout |
| Exception `ReadTimeout` vers 60 s | Aucun octet reçu pendant la limite de lecture | Comparer streaming et valeur `read-timeout` affichée |
| Exception `ConnectTimeout` | DNS, TCP, TLS ou proxy n'a pas établi la connexion | Vérifier l'hôte et le proxy |
| Exception `RemoteProtocolError` | La connexion a été fermée par l'autre côté | Comparer avec les logs serveur à la même heure |

Les exceptions sont enregistrées avec leur type exact (`ConnectTimeout`,
`ReadTimeout`, `RemoteProtocolError`, etc.) et leur traceback.

## Paramètres qui ne représentent pas le timeout total

- `OLLAMA_LOAD_TIMEOUT` limite un chargement de modèle qui reste bloqué ;
- `OLLAMA_KEEP_ALIVE` règle combien de temps le modèle reste en mémoire après
  une requête ;
- `num_ctx` règle la fenêtre de contexte ;
- `num_predict` limite le nombre de tokens générés ;
- ces paramètres ne sont pas une limite générale de 60 secondes.

## Ordre conseillé pour le problème exact de 60 secondes

1. environnement ;
2. santé HTTP ;
3. informations du modèle ;
4. préparation de l'image ;
5. SDK direct sans puis avec streaming ;
6. HTTP brut sans puis avec streaming ;
7. runner applicatif avec les deux timeouts à 300 ;
8. capture des logs macOS juste après l'échec.

Ne changez qu'une seule variable entre deux essais. Utilisez la même image, le
même modèle, le même prompt et les mêmes valeurs `num_ctx/num_predict`. Sinon,
les durées ne seront pas comparables.

## Fichiers à transmettre après le test

Pour analyser l'origine de la coupure, conservez au minimum :

```text
runs/runtime_debug/00_environment.json
runs/runtime_debug/01_network_normal.json
runs/runtime_debug/01_network_no_proxy.json
runs/runtime_debug/02_model_show.json
runs/runtime_debug/sdk_non_stream.json
runs/runtime_debug/sdk_stream.json
runs/runtime_debug/raw_non_stream.json
runs/runtime_debug/raw_stream.json
runs/runtime_debug/application.json
runs/runtime_debug/mac_logs_after_timeout.json
```

Les champs les plus importants sont :

- `elapsed_seconds` ;
- `exception_type` et `exception` ;
- `first_chunk_seconds` et `last_chunk_seconds` ;
- `chunks_received` ;
- `thinking` et `content` partiels ;
- les lignes intéressantes de `server.log`.
