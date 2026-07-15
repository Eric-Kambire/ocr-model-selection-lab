"""Public compatibility facade for the Moroccan CNI benchmark helpers.

Responsibilities live in focused modules:

* :mod:`ocr_benchmark.cni_ingestion` — folders, external JSONB labels and ZIPs;
* :mod:`ocr_benchmark.cni_images` — one-page PDF rendering and card crops;
* :mod:`ocr_benchmark.cni_schema` — fields, prompts, response parsing and merge.

Existing callers can keep importing from ``ocr_benchmark.cni``. New code may
import the focused module directly when that makes its dependency clearer.
"""

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
