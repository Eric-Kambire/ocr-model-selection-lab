# Contrat `cni_fields.json`

`cni_fields.json` est du JSON strict : il ne peut donc pas contenir de
commentaires. Ce document décrit son rôle.

- `recto` : champs demandés lors de l’appel sur la face recto ;
- `verso` : champs demandés lors de l’appel sur la face verso ;
- `key` : clé JSON stable, utilisée par le prompt, le parseur et les artefacts ;
- `type` : indication métier (`text` ou `date`) pour documenter l’attendu.

Pour ajouter un champ, ajoutez le même objet dans la face appropriée, par
exemple :

```json
{"key": "sexe", "type": "text"}
```

Le code construit automatiquement le prompt et remplit `null` si un modèle ne
répond pas pour ce champ. N’ajoutez pas de commentaire dans le fichier JSON :
un commentaire le rendrait invalide.
