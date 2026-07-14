# Plan d’implémentation — Benchmark des CNI marocaines

## Objectif

Comparer des modèles OCR/VLM sur l’extraction structurée des informations d’une
CNI marocaine scannée sur une feuille A4. Le but n’est **pas** de reconnaître
une ancienne ou une nouvelle mise en page : chaque client doit produire les
mêmes données métier, quelle que soit la présentation de la carte.

Les libellés sont bilingues arabe/français. Les valeurs évaluées dans cette
première version sont les valeurs latines visibles sur la carte ; aucune
traduction ni translittération n’est demandée au modèle.

## Répertoires et identifiants

```text
CLIENTS_ROOT/
  <folder_client_id>/                         # identifiant canonique
    <document_id>_CIN_Recto.pdf               # document_id peut être différent
    <document_id>_CIN_Verso.pdf

LABELS_ROOT/
  <folder_client_id>.jsonb                    # fichier externe à importer
```

`folder_client_id` est l’unique clé de rapprochement entre un dossier client et
son label. Le préfixe du nom de PDF est conservé dans les métadonnées, mais ne
doit jamais être comparé à l’identifiant du dossier ni empêcher l’exécution.

Le premier import suppose qu’un fichier `.jsonb` contient du JSON texte UTF-8.
Le module le parse et crée, sans modifier son contenu métier :

```text
CLIENTS_ROOT/<folder_client_id>/<folder_client_id>.json
```

Si le fichier n’est pas du JSON texte, le client reçoit le statut explicite
`label_parse_failed`. Un adaptateur de dump ou de base PostgreSQL sera ajouté
lorsqu’un exemple anonymisé du format réel sera disponible.

## Champs de la première version

| Côté | Champs |
|---|---|
| Recto | `cin`, `nom`, `prenom`, `date_naissance`, `ville_naissance`, `date_validite` |
| Verso | `cin`, `date_validite`, `adresse` |

Les champs sont déclarés dans `config/cni_fields.json`. Le prompt est construit
à partir de ce fichier : ajouter un champ ne nécessite donc pas de réécrire un
long prompt manuel. Les fonctions de traitement restent simples, typées et
documentées ; le module n’introduit pas de hiérarchie de classes.

## Préparation d’une paire de documents

1. Scanner les sous-dossiers de `CLIENTS_ROOT` et valider un recto et un verso,
   chacun sur une seule page PDF.
2. Rendre les pages en PNG haute résolution.
3. Chercher automatiquement la carte dans l’image A4, la recadrer et conserver
   l’original et le recadrage. Le ratio attendu est celui d’une carte ID-1.
4. En cas de crop incertain ou impossible, conserver l’A4, produire un statut
   et un log ; aucun document n’est ignoré silencieusement.
5. Matérialiser le label externe s’il est disponible.

Les résultats ne sont jamais écrits à côté des PDFs source. Ils restent dans un
run isolé pour éviter tout écrasement :

```text
runs/cni-<run_id>/<folder_client_id>/
  recto.extraction.json
  verso.extraction.json
  global.extraction.json
  raw_recto_output.txt
  raw_verso_output.txt
  crop_recto.png
  crop_verso.png
```

## Stratégies d’inférence, réglables dans ⚙ Paramètres CNI

### `separate_calls` — défaut et méthode de diagnostic

1. Appel modèle sur le recto avec un prompt recto strict.
2. Appel modèle sur le verso avec un prompt verso strict.
3. Fusion déterministe par le code Python ; aucun troisième appel IA.

Cette méthode donne un `recto.json`, un `verso.json`, puis un `global.json`.
Elle est la meilleure pour isoler une erreur de lecture à une face précise.

### `combined_vertical` — alternative VLM

1. Recadrer recto et verso.
2. Construire une seule image avec recto en haut, verso en bas, et un séparateur
   lisible.
3. Faire un unique appel modèle demandant deux objets JSON.
4. Le code écrit malgré tout les trois artefacts JSON.

Cette option peut être plus rapide pour les VLM capables de raisonner sur une
paire d’images, mais elle complique le diagnostic et ne remplace pas le mode
séparé pour une première évaluation.

## JSON produits avant toute comparaison de label

`recto.extraction.json` et `verso.extraction.json` contiennent seulement les
champs lus sur leur face, le statut, les mesures techniques et le texte brut.

`global.extraction.json` conserve les valeurs des deux faces et la fusion :

```json
{
  "folder_client_id": "CLIENT_001",
  "cin_recto": "BM42518",
  "cin_verso": "BM42518",
  "cin_fusionne": "BM42518",
  "cin_coherent": true,
  "date_validite_recto": "2029-03-21",
  "date_validite_verso": "2029-03-21",
  "date_validite_fusionnee": "2029-03-21",
  "date_validite_coherente": true
}
```

Cette cohérence recto/verso est une métrique interne. Elle n’est pas encore une
comparaison avec le label JSONB.

## Exécution et mémoire

La liste des modèles est obtenue par `ollama.list()` avec un bouton
« Actualiser les modèles ». Plusieurs modèles peuvent être cochés, mais la file
reste volontairement séquentielle : modèle A, tous les clients, déchargement ;
puis modèle B. Un seul modèle et une seule image/appel sont actifs à la fois.

Les timeout, sorties tardives, réponses brutes et logs terminal suivent les
mêmes règles que le benchmark courant. Un label absent, un PDF manquant, un
PDF multi-page ou un JSON invalide devient une ligne de résultat visible.

## Interface et résultats

La page « Benchmark CNI » comportera :

- chemin local `CLIENTS_ROOT`, chemin `LABELS_ROOT`, ou import ZIP ;
- aperçu et rapport de scan avant lancement ;
- sélection de plusieurs modèles Ollama ;
- bouton `⚙ Paramètres CNI` pour la stratégie, le crop, DPI, timeout et
  déchargement ;
- image en cours, face en cours, JSON recto/verso/global et logs live ;
- explorateur détaillé avec PDF/image, label, sortie brute et champ en erreur ;
- graphiques par modèle, face et champ.

Lorsque le mapping du label sera validé, le résultat détaillé exposera une
accuracy globale et une accuracy par champ. Les filtres accepteront tout
intervalle, notamment `100–100`, `90–99.99`, ou une plage personnalisée. Les
lignes sans label restent visibles sous `non_noté` et ne doivent pas être
présentées comme des erreurs de modèle.

## Import de données de test

Deux entrées sont prévues :

1. **Chemin local** pour les essais sur le poste qui exécute Gradio.
2. **Archive ZIP** pour une importation portable ou Docker.

L’import ZIP rejette les chemins sortant du dossier cible, ne modifie jamais le
repo Git et extrait les fichiers dans un espace de données ignoré par Git. Les
CNI, labels, crops et résultats bruts ne doivent être ni commités ni publiés
sur GitHub ou Docker Hub.

## Déploiement sur les branches

Le module CNI est livré sur `codex/standalone-colab-benchmark` et sur `main`.
Les commits propres à la navigation/layout de la branche Codex ne sont pas
fusionnés dans `main` : seules les modifications CNI compatibles y sont
reportées et testées séparément.
