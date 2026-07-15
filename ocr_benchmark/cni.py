"""Façade publique des outils de benchmark des CNI marocaines.

Ce fichier conserve les anciens imports stables. La logique est séparée dans
des modules spécialisés afin de pouvoir déboguer données, images, contrat JSON
et exécution indépendamment.
"""

# Façade de compatibilité : l'interface peut importer ``cni`` sans dépendre du
# rangement interne des modules spécialisés.
from .cni_images import build_vertical_cni_composite, crop_cni_from_a4, render_single_page_pdf
from .cni_ingestion import import_cni_zip, materialize_cni_labels, scan_cni_clients, write_cni_json
from .cni_schema import (
    DEFAULT_CNI_FIELD_CONFIG,
    RECTO_FIELDS,
    VERSO_FIELDS,
    build_cni_global_json,
    build_cni_prompt,
    build_combined_cni_prompt,
    load_cni_field_config,
    parse_cni_json_response,
    parse_combined_cni_json_response,
)

__all__ = [
    "DEFAULT_CNI_FIELD_CONFIG", "RECTO_FIELDS", "VERSO_FIELDS",
    "build_cni_global_json", "build_cni_prompt", "build_combined_cni_prompt",
    "build_vertical_cni_composite", "crop_cni_from_a4", "import_cni_zip",
    "load_cni_field_config", "materialize_cni_labels", "parse_cni_json_response",
    "parse_combined_cni_json_response", "render_single_page_pdf", "scan_cni_clients",
    "write_cni_json",
]
