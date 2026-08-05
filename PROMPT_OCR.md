# Prompt OCR — CNI marocaines

Ce document conserve les décisions de prompt afin de faire évoluer les essais
sans modifier les résultats historiques.

## Objectif actuel

Extraire les champs configurés d'une CNI marocaine, ancienne ou nouvelle, dans
un JSON stable. La phase actuelle lit uniquement les valeurs imprimées en
caractères latins : elle ne traduit ni ne translittère l'arabe.

## Stratégie

Le mode recommandé effectue deux appels au même modèle : recto puis verso. Le
mode alternatif envoie une image composite, recto en haut et verso en bas. Les
deux JSON de face et le JSON global restent toujours conservés.

## Modes disponibles dans l'interface

Les réglages sont dans `8. Benchmark CNI → 4. Paramètres → Prompt et sortie`.

| Source des instructions | Routage | Texte réellement envoyé |
| --- | --- | --- |
| Prompt de l'application | Règles spécifiques | SYSTEM éditable + prompt USER propre au recto ou au verso. |
| Prompt de l'application | Règles complètes | SYSTEM éditable + mêmes règles métier, avec une indication de face et un schéma propres à l'image. |
| Image seule | Sans objet | Aucun texte applicatif ; le SYSTEM vient du Modelfile. |
| Image + face | Sans objet | SYSTEM du Modelfile + indication USER minimale RECTO, VERSO ou image combinée. |

Le mode « règles complètes » n'envoie volontairement pas des messages
strictement identiques octet par octet : l'indication de face et le schéma
restent adaptés afin de ne pas demander des champs verso à une image recto.

Le réglage `think` est indépendant du prompt :

- `disabled` envoie `think=false` et convient au JSON déterministe ;
- `automatic` laisse Ollama appliquer le défaut du modèle ;
- `enabled` envoie `think=true` et peut augmenter fortement la latence.

## Rôles

| Partie | But | Risque |
| --- | --- | --- |
| Prompt système | Règles prioritaires : JSON valide, aucune invention. | Des règles trop longues ou contradictoires réduisent la stabilité. |
| Prompt utilisateur | Champs et consignes propres à l'image. | Modifier les clés JSON empêche une comparaison fiable. |
| Prompt envoyé | Copie persistée dans chaque run. | Sert à reproduire, pas à modifier un score après coup. |

## Hypothèses ouvertes

1. Les valeurs de référence de nom, prénom et adresse utilisent l'alphabet
   latin : lettres A-Z, accents français, chiffres et ponctuation. Le mot
   « latin » ne désigne pas ici la langue latine.
2. Les suffixes par défaut sont `_CIN_Recto` et `_CIN_Verso`, mais restent
   éditables avant le scan.
3. `null` signifie « illisible ou ambigu », et non « absent du document ».
4. Les futurs champs métier seront ajoutés dans `config/cni_fields.json`.

## À décider ensuite

- Normalisation des dates avant comparaison.
- Comparaison tolérante ou stricte de l'adresse.
- Évaluation éventuelle d'un OCR complet séparé de l'extraction JSON.
- Correspondance exacte entre les clés du JSONB et celles du contrat CNI.

Chaque appel conserve son image, ses prompts, son retour brut et ses JSON,
y compris en cas de réponse invalide ou de timeout.
