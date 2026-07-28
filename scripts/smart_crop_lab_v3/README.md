# Smart Crop & Degradation Lab — V3

Cette version réunit dans une seule interface Gradio :

1. la détection et le redressement d'une carte d'identité ;
2. une galerie de toutes les étapes du pipeline ;
3. une lecture animée avec pause réglable ;
4. un laboratoire de génération de cas dégradés ;
5. l'annotation manuelle des quatre coins ;
6. l'export de l'image, du PDF, du masque et du JSON.

## Lancement rapide sous Windows

1. Décompresser le ZIP.
2. Double-cliquer sur `run_windows.bat`.
3. Attendre l'installation initiale.
4. L'interface s'ouvre sur `http://127.0.0.1:7860`.

## Lancement manuel

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py
```

## Onglet Smart Crop

- charge une image ou un PDF ;
- lance la détection ;
- consulte la galerie des étapes ;
- règle la pause puis clique sur `Lire les étapes` ;
- télécharge le ZIP des résultats.

## Onglet Laboratoire

- charge une image ou une page PDF ;
- utilise le pinceau/gomme de l'éditeur ;
- clique sur l'aperçu pour placer un bruit localisé ;
- active l'annotation et clique les quatre coins dans l'ordre :
  haut-gauche, haut-droit, bas-droit, bas-gauche ;
- applique rotation, perspective, lumière, ombre, reflet, flou, compression et bruit ;
- exporte : original, image dégradée, PDF, JSON et masque.

## Algorithme

L'algorithme est une combinaison CPU de :

- Sobel ;
- Canny ;
- morphologie ;
- contours ;
- Hough et LSD ;
- densité/variance locale ;
- espace couleur LAB ;
- score multi-critères ;
- homographie.

Aucune hypothèse A4 et aucune orientation EXIF.

## Intégration dans ce dépôt

Le moteur n'est pas dupliqué dans ce dossier. `smart_crop.py` est un adaptateur
vers `ocr_benchmark/cni_smart_crop_v3.py`, également utilisé par :

- `scripts/cni_crop_methods_lab.py` pour comparer plusieurs méthodes ;
- `scripts/cni_crop_stepper_app.py` pour le parcours pédagogique en six étapes ;
- les futurs services de traitement, sans dépendance à Gradio.

Ainsi, une correction du détecteur bénéficie à toutes les interfaces. Si aucun
quadrilatère n'atteint le score minimal, le fichier normalisé complet est
conservé : aucun mauvais crop n'est forcé.
