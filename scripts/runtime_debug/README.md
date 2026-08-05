# Diagnostic pas à pas du runtime Ollama

Ces scripts isolent chaque couche du Benchmark CNI. Ils ne modifient aucun
modèle et enregistrent leurs rapports dans `runs/runtime_debug/`, dossier ignoré
par Git.

Exécutez les commandes depuis la racine de `codex/cni-ui-workflow`, avec le même
environnement Python que l’application.

## 0. Environnement

```bash
python scripts/runtime_debug/00_environment.py \
  --output runs/runtime_debug/00_environment.json
```

Vérifiez notamment `OLLAMA_HOST`, `OLLAMA_LOAD_TIMEOUT`, les proxies,
`NO_PROXY`, l’exécutable Python et les versions `ollama/httpx`.

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

## Lecture du diagnostic

- Étape 1 échoue : réseau, proxy ou serveur Ollama.
- Étape 2 échoue : modèle absent, serveur ou métadonnées incompatibles.
- Étape 3 échoue : PDF/image ou dépendance de conversion.
- Étape 4 coupe vers 60 s : client HTTP, proxy ou serveur intermédiaire.
- Étape 4 réussit mais étape 5 échoue : adaptateur ou garde-fou applicatif.
- Les deux étapes réussissent : le problème vient du câblage ou de l’état UI.

Les exceptions sont enregistrées avec leur type exact (`ConnectTimeout`,
`ReadTimeout`, `RemoteProtocolError`, etc.) et leur traceback.
