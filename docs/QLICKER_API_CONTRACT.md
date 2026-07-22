# Connecteur QlickEER — contrat à renseigner

Le dépôt ne contient pas de documentation QlickEER vérifiable. Le connecteur est donc volontairement générique : il ne devine aucune route ni structure privée.

## Ce qu'il attend

Pour chaque `client_id`, il faut trois GET configurables : recto, verso, label.

- Les documents peuvent répondre directement avec `application/pdf`, `image/jpeg` ou `image/png`.
- Si l'API répond d'abord avec un objet JSON contenant une URL temporaire, renseigner `document_url_key` avec le nom de cette clé, par exemple `download_url`.
- Le label doit répondre par un objet JSON. Il est écrit dans `clients/<client_id>/<client_id>.json`.

Les modèles de chemin utilisent `{client_id}`. Exemple à adapter seulement après confirmation de QlickEER :

```python
QlickerImportConfig(
    base_url="https://api.exemple.tld/v1",
    recto_path_template="documents/{client_id}/recto",
    verso_path_template="documents/{client_id}/verso",
    label_path_template="labels/{client_id}",
    document_url_key="download_url",  # laisser vide si le GET retourne le binaire
)
```

## Secret et sécurité

Définir le token avant l'import, sans le mettre dans le code ou les fichiers de run :

```powershell
$env:QLICKER_API_TOKEN = "..."
```

Le connecteur transmet `Authorization: Bearer <token>` par défaut. La clé et le préfixe sont configurables si QlickEER utilise autre chose.

## Validation avant intégration UI

1. Confirmer les trois routes, le format du label, le mécanisme de pagination et le type d'authentification avec l'équipe QlickEER.
2. Tester un seul `client_id` sur un environnement de recette.
3. Vérifier que les noms produits suivent `id_CIN_Recto.(pdf|jpg|png)` et `id_CIN_Verso.(pdf|jpg|png)`.
4. Lancer ensuite le scan CNI local : le reste de la pipeline ne dépend plus de QlickEER.
