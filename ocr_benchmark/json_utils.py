"""Conversion explicite des objets scientifiques vers le contrat JSON.

OpenCV, NumPy et certains SDK renvoient des scalaires comme ``numpy.int32``
ou ``numpy.float32``. Ils ressemblent à des nombres Python mais le module
standard :mod:`json` ne sait pas les sérialiser directement.
"""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Any, Mapping


def to_json_compatible(value: Any) -> Any:
    """Convertit récursivement une valeur en types JSON natifs."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Enum):
        return to_json_compatible(value.value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {
            str(key): to_json_compatible(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple, set, frozenset)):
        return [to_json_compatible(item) for item in value]

    to_list = getattr(value, "tolist", None)
    if callable(to_list):
        return to_json_compatible(to_list())
    to_item = getattr(value, "item", None)
    if callable(to_item):
        converted = to_item()
        if converted is not value:
            return to_json_compatible(converted)

    raise TypeError(
        f"Type non sérialisable en JSON : {type(value).__module__}."
        f"{type(value).__name__}"
    )


def dumps_json(value: Any, *, indent: int | None = 2) -> str:
    """Sérialise une valeur après normalisation scientifique contrôlée."""

    return json.dumps(
        to_json_compatible(value),
        ensure_ascii=False,
        indent=indent,
    )
