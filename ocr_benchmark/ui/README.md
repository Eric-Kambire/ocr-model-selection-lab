# Organisation de l'interface

`main.py` assemble l'application et relie les événements. Les composants d'un
espace métier vivent dans `ui/<espace>/`, tandis que les lectures, écritures et
calculs restent dans `application/` et dans le noyau métier.

Pour la CNI :

- `settings_view.py` assemble les sous-vues ;
- `execution_settings.py` contrôle stratégie et ressources ;
- `preprocessing_settings.py` contrôle crop, rotation et amélioration d'image ;
- `prompt_settings.py` contrôle prompts et contrat JSON ;
- `handlers.py` traduit les événements Gradio vers les services applicatifs.

Cette limite évite que Gradio devienne une dépendance des algorithmes de crop,
du runner, de QlickEER ou des adaptateurs de modèles.
