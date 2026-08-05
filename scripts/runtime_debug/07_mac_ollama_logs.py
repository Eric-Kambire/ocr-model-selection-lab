"""Capture les dernières lignes des logs Ollama macOS autour d'un timeout."""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path


## Emplacements officiels des journaux de l'application Ollama sur macOS.
DEFAULT_LOGS = [
    Path.home() / ".ollama" / "logs" / "server.log",
    Path.home() / ".ollama" / "logs" / "app.log",
]

## Termes signalés séparément pour accélérer la première lecture.
ERROR_MARKERS = (
    "error",
    "timeout",
    "timed out",
    "cancel",
    "broken pipe",
    "connection reset",
    "eof",
    "panic",
    "runner",
    "load",
    "memory",
    "metal",
)


def read_last_lines(path: Path, line_count: int) -> list[str]:
    """Retourne les dernières lignes d'un fichier texte."""

    ## errors=replace empêche un caractère invalide de casser le diagnostic.
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return lines[-line_count:]


def collect_logs(line_count: int) -> dict:
    """Construit un rapport contenant les journaux bruts et les lignes notables."""

    files: list[dict] = []
    for path in DEFAULT_LOGS:
        if not path.is_file():
            files.append(
                {
                    "path": str(path),
                    "status": "not_found",
                    "lines": [],
                    "interesting_lines": [],
                }
            )
            continue

        ## Toutes les lignes restent présentes ; le filtre est seulement une aide.
        lines = read_last_lines(path, line_count)
        interesting = [
            line
            for line in lines
            if any(marker in line.lower() for marker in ERROR_MARKERS)
        ]
        files.append(
            {
                "path": str(path),
                "status": "success",
                "bytes": path.stat().st_size,
                "modified_at_epoch": path.stat().st_mtime,
                "lines": lines,
                "interesting_lines": interesting,
            }
        )

    return {
        "platform": platform.platform(),
        "captured_at_epoch": time.time(),
        "requested_last_lines": line_count,
        "files": files,
    }


def main() -> None:
    ## Exécuter idéalement le script juste avant et juste après la coupure.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lines",
        type=int,
        default=300,
        help="Nombre de lignes conservées pour chaque journal.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/runtime_debug/mac_ollama_logs.json"),
        help="Chemin du rapport JSON.",
    )
    args = parser.parse_args()

    ## Une valeur au moins égale à 1 garantit un rapport non vide.
    report = collect_logs(max(1, int(args.lines)))
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)

    ## La sauvegarde permet de comparer l'état avant et après l'essai de 60 s.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(f"\nRapport écrit dans : {args.output}")


## Point d'entrée du script autonome.
if __name__ == "__main__":
    main()
