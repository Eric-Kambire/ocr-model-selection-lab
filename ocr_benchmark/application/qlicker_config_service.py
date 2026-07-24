"""Persistance locale et réinitialisable de la configuration QlickEER.

Le fichier est volontairement séparé des résultats d'analyse : il contient les
routes et réglages de connexion réutilisables, pas les réponses API ni les
documents clients. Son chemin est ignoré par Git.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence


ROUTE_NAMES = ("list", "info", "documents", "view")


def default_qlicker_config(import_root: str) -> dict[str, Any]:
    """Retourne une configuration vide, sérialisable et sûre par défaut."""
    return {
        "schema_version": 1,
        "base_url": "",
        "timeout_seconds": 30,
        "use_system_proxy": False,
        "verify_ssl": True,
        "proxy_url": "",
        "import_root": import_root,
        "routes": {
            name: {"raw_url": "", "endpoint": "", "params": []}
            for name in ROUTE_NAMES
        },
    }


def load_qlicker_config(path: Path, *, import_root: str) -> dict[str, Any]:
    """Lit la configuration locale, ou retourne des valeurs par défaut.

    Une erreur de lecture ne bloque pas l'interface : le fichier est ignoré et
    une configuration vide est utilisée. Les valeurs sont normalisées avant
    d'être envoyées à Gradio.
    """
    defaults = default_qlicker_config(import_root)
    if not path.is_file():
        return defaults
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return defaults
    if not isinstance(raw, Mapping) or raw.get("schema_version") != 1:
        return defaults
    return _normalise_config(raw, defaults)


def save_qlicker_config(path: Path, value: Mapping[str, Any], *, import_root: str) -> dict[str, Any]:
    """Normalise puis écrit le JSON local de façon atomique."""
    normalised = _normalise_config(value, default_qlicker_config(import_root))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(normalised, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)
    return normalised


def reset_qlicker_config(path: Path, *, import_root: str) -> dict[str, Any]:
    """Supprime seulement le fichier de configuration local et retourne le défaut."""
    if path.exists():
        path.unlink()
    return default_qlicker_config(import_root)


def qlicker_config_from_ui(
    *,
    base_url: Any,
    timeout_seconds: Any,
    use_system_proxy: Any,
    verify_ssl: Any,
    proxy_url: Any,
    import_root: Any,
    routes: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Construit une configuration depuis les composants Gradio sans pandas."""
    return {
        "schema_version": 1,
        "base_url": str(base_url or "").strip(),
        "timeout_seconds": _positive_int(timeout_seconds, 30),
        "use_system_proxy": bool(use_system_proxy),
        "verify_ssl": bool(verify_ssl),
        "proxy_url": str(proxy_url or "").strip(),
        "import_root": str(import_root or "").strip(),
        "routes": {
            name: {
                "raw_url": str((routes.get(name) or {}).get("raw_url") or "").strip(),
                "endpoint": str((routes.get(name) or {}).get("endpoint") or "").strip(),
                "params": _normalise_rows((routes.get(name) or {}).get("params")),
            }
            for name in ROUTE_NAMES
        },
    }


def _normalise_config(raw: Mapping[str, Any], defaults: Mapping[str, Any]) -> dict[str, Any]:
    """Conserve seulement les clés UI reconnues et des types JSON simples."""
    raw_routes = raw.get("routes") if isinstance(raw.get("routes"), Mapping) else {}
    return qlicker_config_from_ui(
        base_url=raw.get("base_url"),
        timeout_seconds=raw.get("timeout_seconds"),
        use_system_proxy=raw.get("use_system_proxy"),
        verify_ssl=raw.get("verify_ssl", True),
        proxy_url=raw.get("proxy_url"),
        import_root=raw.get("import_root") or defaults["import_root"],
        routes={
            name: raw_routes.get(name) if isinstance(raw_routes.get(name), Mapping) else {}
            for name in ROUTE_NAMES
        },
    )


def _normalise_rows(rows: Any) -> list[list[Any]]:
    """Convertit les tableaux Gradio/DataFrame en lignes JSON stables."""
    if hasattr(rows, "values"):
        rows = rows.values.tolist()
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        return []
    normalised: list[list[Any]] = []
    for row in rows:
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes)) or not row:
            continue
        name = str(row[0] or "").strip()
        if not name:
            continue
        value = "" if len(row) < 2 or row[1] is None else str(row[1])
        enabled = bool(row[2]) if len(row) >= 3 else True
        normalised.append([name, value, enabled])
    return normalised


def _positive_int(value: Any, fallback: int) -> int:
    """Évite une valeur vide, nulle ou négative dans le timeout HTTP."""
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return fallback
