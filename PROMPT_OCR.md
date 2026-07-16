# Prompt OCR — CNI marocaines

Ce document fixe les décisions de prompt actuellement retenues. Il est
volontairement séparé du code : les essais de formulation pourront être
discutés ici sans modifier la pipeline ou les résultats historiques.

## Objectif de la phase actuelle

Extraire seulement les champs configurés d'une CNI marocaine, ancienne ou
nouvelle, dans un JSON stable. La comparaison avec le label et l'évaluation de
l'arabe ne font pas encore partie de cette phase.

Les valeurs attendues sont les valeurs imprimées en caractères latins. Le
modèle ne doit ni traduire, ni translittérer, ni deviner une information
illisible.

## Stratégie retenue

Par défaut, deux appels sont effectués au même modèle : un pour le recto, puis
un pour le verso. Cela permet de savoir précisément quelle face a échoué. Une
stratégie alternative compose les deux faces dans une seule image, recto en
haut et verso en bas ; elle reste utile pour comparer les modèles, mais rend le
diagnostic moins précis.

Le JSON final conserve toujours `recto`, `verso`, `cin_recto`, `cin_verso` et
la valeur fusionnée. Aucune valeur d'une face ne doit écraser silencieusement
l'autre.

## Rôles des prompts

| Partie | Rôle | Conséquence si elle est mal réglée |
| --- | --- | --- |
| Prompt système | Règles stables et prioritaires : JSON valide, pas d'invention, uniquement les clés demandées. | Un système trop long ou contradictoire peut réduire la stabilité du JSON. |
| Prompt utilisateur | Consigne propre à l'image : face traitée, champs et schéma JSON, plus les consignes d'essai. | Modifier les clés ou demander du texte libre rend les scores impossibles à comparer. |
| Prompt envoyé | Copie exacte système + utilisateur persistée dans chaque run. | Sert à reproduire une sortie, jamais à modifier a posteriori son score. |

## Prompt de base actuel

Le système demande une extraction JSON stricte, sans valeur inventée.

Le message utilisateur demande :

- de lire seulement les valeurs latines visibles ;
- de produire `null` lorsqu'une valeur est illisible ;
- de conserver orthographe, ponctuation et accents ;
- de normaliser seulement une date clairement lisible en `YYYY-MM-DD` ;
- de retourner uniquement l'objet JSON attendu.

Le prompt ne s'appuie pas, pour l'instant, sur la lecture ni sur la traduction
de caractères arabes.

## Hypothèses à confirmer

1. Les labels externes contiendront les mêmes noms de clés que le contrat
   actuel, ou une table de correspondance sera fournie.
2. Les valeurs de nom, prénom et adresse de référence seront latines.
3. Les fichiers suivent par défaut `<document_id>_CIN_Recto.pdf` et
   `<document_id>_CIN_Verso.pdf`, mais les deux suffixes sont éditables dans
   l'interface avant le scan.
4. Un `null` signifie « non lu ou ambigu » et non une valeur absente du
   document.
5. Les champs supplémentaires seront ajoutés dans `config/cni_fields.json`
   après décision métier ; le prompt les récupère automatiquement.

## Questions ouvertes pour le prochain échange

- Faut-il accepter des dates non normalisées dans le label, par exemple
  `21.03.2029`, et les comparer après normalisation ?
- Pour l'adresse, doit-on comparer strictement les espaces, abréviations et
  numéros, ou utiliser une comparaison tolérante ?
- Souhaites-tu une sortie distincte « texte OCR complet » avant le JSON, afin
  d'évaluer la lecture générale indépendamment de l'extraction structurée ?
- Quelles clés exactes du futur JSONB doivent devenir des champs obligatoires ?

## Règle de reproductibilité

Chaque appel conserve son image de travail, le prompt système/utilisateur
envoyé, le retour brut, le JSON de la face et le JSON global. Même un timeout
ou une réponse JSON invalide reste inspectable dans l'interface et sur disque.
