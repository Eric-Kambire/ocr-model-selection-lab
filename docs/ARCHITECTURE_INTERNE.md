# Architecture interne : monolithe modulaire

## But

L'application reste une seule application locale et un seul dépôt. Elle n'est pas
découpée en microservices. En revanche, chaque responsabilité a une frontière
explicite afin qu'un futur composant fraude ou un autre type de document puisse
être ajouté sans modifier l'interface Gradio ni les adaptateurs existants.

```text
Interface (Gradio, CLI, future API)
        ↓ appelle
Application (cas d'usage)
        ↓ orchestre
Domaine (contrats BenchmarkCase, InferenceResult, règles de score)
        ↓ utilise
Adaptateurs / infrastructure (Ollama, fichiers, ZIP, QlickEER)
```

## Répertoires et responsabilités

| Répertoire ou fichier | Responsabilité | Ne doit pas contenir |
|---|---|---|
| `main.py` | Construction Gradio, formatage des vues, branchement des boutons. | OCR, lecture/écriture métier, appels réseau. |
| `ocr_benchmark/application/benchmark_service.py` | Catalogue, exécution générique, événements de benchmark. | Composants Gradio. |
| `ocr_benchmark/application/cni_service.py` | Cas d'usage CNI : scan, import, extraction. | HTML ou état de navigateur. |
| `ocr_benchmark/application/run_service.py` | Consultation et rétention des runs locaux. | Règles OCR/CNI. |
| `ocr_benchmark/application/retention_service.py` | Archive anonymisée et nettoyage des artefacts CNI. | Interface Gradio, contenu OCR en archive. |
| `ocr_benchmark/domain.py` | Contrats stables d'entrée/sortie. | Chemins Gradio, appels Ollama. |
| `ocr_benchmark/runner.py` | Orchestration modèle par modèle. | Détails de layout CNI. |
| `ocr_benchmark/cni_*.py` | Composants CNI spécialisés. | Code de navigation UI. |
| `models/` | Adaptateurs de fournisseurs OCR. | Calcul des métriques métier. |

## Pourquoi ce n'est pas du MVC strict

MVC est adapté aux applications web avec routes, contrôleurs HTTP et vues de
pages. Ici, Gradio est une interface événementielle. Imposer des contrôleurs
MVC ajouterait des fichiers sans réduire le couplage.

La séparation ci-dessus est plus utile : l'interface appelle un service ; le
service retourne des données ; l'interface décide seulement comment les
afficher. Une future API REST ou un worker interne appellera exactement les
mêmes services.

## Contrat pour les futurs composants documentaires

Chaque nouveau document (chèque, facture, CNI, justificatif) doit suivre le
même enchaînement :

```text
source → normalisation page/image → préparation → extraction → validation → artefacts
```

Le composant doit retourner au minimum :

- l'identifiant de run et l'identifiant du document ;
- les artefacts produits et leurs chemins contrôlés ;
- la sortie brute du modèle ;
- le résultat structuré ;
- les erreurs et timings ;
- la configuration, le modèle et le prompt utilisés.

Les signaux de fraude viendront après la validation sous la forme d'un module
distinct : ils consommeront le résultat structuré et les artefacts, sans
modifier le moteur OCR.

## Données temporaires et rétention

- Les runs sont stockés dans `runs/` et supprimés au démarrage selon
  `RUN_RETENTION_DAYS` (30 jours par défaut, valeur négative = désactivé).
- Les artefacts doivent être téléchargés avant expiration si une conservation
  longue est nécessaire.
- Les imports CNI, labels et uploads sont ignorés par Git et restent sur le
  poste local qui les traite.
- Le chiffrement est une responsabilité du serveur : BitLocker sur Windows ou
  LUKS sur Linux, plus ACL du dossier applicatif. Le code Python ne peut pas rendre
  sûr un disque non chiffré.
