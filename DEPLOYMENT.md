# Déploiement Docker

## Principe

Une image Docker est un paquet immuable contenant Python, les dépendances,
l’application et le dataset livré. Un conteneur est une instance de cette image.
Les résultats et images ajoutées sont placés dans des volumes persistants.

Le conteneur accède à Ollama installé sur la machine hôte via
`host.docker.internal:11434`.

L’image CPU portable n’embarque pas PyTorch/EasyOCR afin de rester partageable.
Elle exécute les modèles Ollama sur le CPU de la machine hôte. L’image GPU
`Dockerfile.gpu` embarque EasyOCR et CUDA.

## Construire localement

```bash
docker compose build
docker compose up -d
docker compose ps
```

Interface : `http://localhost:7860`.

Vérification :

```bash
docker inspect --format "{{.State.Health.Status}}" ocr-benchmark-suite
docker compose logs --tail 100
```

## Envoyer le projet source

Envoyer le dossier en excluant `.git`, `runs`, les caches Python et les fichiers
temporaires. Le destinataire exécute :

```bash
docker compose up -d --build
```

Cette méthode est simple mais reconstruit l’image sur sa machine et nécessite
Internet pour télécharger les dépendances.

## Envoyer l’image sans registre

Sur la machine de construction :

```powershell
.\scripts\export_docker_image.ps1
```

Transmettre :

- `dist/ocr-benchmark-suite-1.0.0.tar`
- `docker-compose.distribution.yml`
- `.env.deploy.example`

Sur la machine destinataire :

```bash
docker load --input ocr-benchmark-suite-1.0.0.tar
cp .env.deploy.example .env
docker compose -f docker-compose.distribution.yml up -d
```

Cette méthode fonctionne hors ligne, mais le TAR peut faire plusieurs
gigaoctets à cause de PyTorch et EasyOCR.

## Envoyer via un registre

Le projet fournit un workflow GitHub Actions prêt à publier dans GHCR sans
utiliser le disque local. Voir `CLOUD_BUILD.md`.

```bash
docker tag ocr-benchmark-suite:1.0.0 registry.example.com/team/ocr-benchmark-suite:1.0.0
docker login registry.example.com
docker push registry.example.com/team/ocr-benchmark-suite:1.0.0
```

Configurer ensuite dans `.env` :

```dotenv
OCR_BENCHMARK_IMAGE=registry.example.com/team/ocr-benchmark-suite:1.0.0
```

Puis :

```bash
docker compose -f docker-compose.distribution.yml pull
docker compose -f docker-compose.distribution.yml up -d
```

## Mise à jour

Un conteneur ne se met pas à jour tout seul. C’est volontaire : un benchmark
doit rester reproductible et associé à une version précise.

Procédure recommandée :

1. construire et publier une nouvelle version, par exemple `1.1.0` ;
2. modifier `OCR_BENCHMARK_IMAGE` dans `.env` ;
3. exécuter `docker compose pull` puis `docker compose up -d` ;
4. vérifier le healthcheck ;
5. conserver temporairement l’ancienne image pour un retour arrière.

Éviter le tag `latest` en production. Les outils de mise à jour automatique
comme Watchtower sont possibles, mais déconseillés ici car ils peuvent changer
le moteur au milieu d’une campagne de benchmark.

## Données persistantes

Les volumes suivants survivent au remplacement du conteneur :

- `benchmark-runs` : rapports et résultats ;
- `benchmark-dataset` : catalogue, dataset livré et images ajoutées depuis
  l’interface.

Attention : lors d’une mise à jour d’image, Docker conserve le volume dataset.
Les nouvelles données intégrées dans une version plus récente de l’image ne sont
donc pas fusionnées automatiquement dans un volume existant. Cette opération
doit être réalisée par une migration explicite pour ne jamais écraser les labels
ajoutés par l’utilisateur.
