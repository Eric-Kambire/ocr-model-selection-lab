"""Adaptateur du laboratoire autonome vers le moteur partagé du projet."""

from __future__ import annotations

import sys
from pathlib import Path

# ``app.py`` est lancé depuis ce dossier. On ajoute donc la racine du dépôt au
# chemin d'import avant de charger le même moteur que le reste du projet.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ocr_benchmark.cni_smart_crop import *  # noqa: F401,F403
from ocr_benchmark.cni_smart_crop import main


if __name__ == "__main__":
    main()
