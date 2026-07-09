# Build cloud et publication GHCR

Le workflow `.github/workflows/publish-container.yml` construit l’image sur un
runner GitHub, la publie dans GitHub Container Registry, puis teste l’image
publiée par son digest.

Il publie également la même image publique dans Docker Hub :

```text
erickambire/ocr-model-selection-lab
```

## Prérequis

1. placer ce projet dans un dépôt GitHub ;
2. activer GitHub Actions ;
3. autoriser les workflows en lecture/écriture :
   **Settings > Actions > General > Workflow permissions > Read and write**.

Le workflow utilise uniquement le `GITHUB_TOKEN` automatique. Aucun mot de passe
Docker ou token personnel n’est nécessaire pour publier dans GHCR. Docker Hub
requiert deux secrets Actions : `DOCKERHUB_USERNAME` et `DOCKERHUB_TOKEN`.

## Premier build manuel

Dans GitHub :

1. ouvrir **Actions** ;
2. sélectionner **Build, test and publish container** ;
3. choisir **Run workflow**.

Le workflow publie au minimum un tag `sha-...`.

## Publier une version

```bash
git tag v1.0.0
git push origin v1.0.0
```

Cela publie notamment :

```text
ghcr.io/<owner>/<repository>:1.0.0
ghcr.io/<owner>/<repository>:1.0
ghcr.io/<owner>/<repository>:sha-<commit>
erickambire/ocr-model-selection-lab:1.0.0
erickambire/ocr-model-selection-lab:1.0
```

Le digest affiché par le workflow est la référence la plus reproductible.

## Rendre l’image publique

La première image GHCR est privée par défaut :

1. ouvrir le profil ou l’organisation GitHub ;
2. ouvrir **Packages** puis le package ;
3. **Package settings > Change visibility > Public**.

Pour une image privée, le serveur destinataire doit faire un `docker login
ghcr.io` avec un token autorisé à lire les packages.

## Déployer

Copier `.env.deploy.example` vers `.env`, puis remplacer :

```dotenv
OCR_BENCHMARK_IMAGE=ghcr.io/<owner>/<repository>:1.0.0
```

Lancer :

```bash
docker compose -f docker-compose.distribution.yml pull
docker compose -f docker-compose.distribution.yml up -d
```

## Mettre à jour

Publier un nouveau tag, par exemple `v1.1.0`, modifier `.env`, puis :

```bash
docker compose -f docker-compose.distribution.yml pull
docker compose -f docker-compose.distribution.yml up -d
```

La mise à jour reste volontairement explicite. Elle ne remplace pas un
conteneur en cours de benchmark sans décision de l’opérateur.
