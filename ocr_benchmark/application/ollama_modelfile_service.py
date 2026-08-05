"""Création contrôlée de modèles Ollama dérivés depuis un ``Modelfile``.

Le service ne dépend pas de Gradio. Il peut donc être appelé depuis le petit
laboratoire UI, un script CLI ou, plus tard, un service interne.
"""

from __future__ import annotations

import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path


# Ollama accepte notamment ``nom``, ``nom:tag`` et les espaces de noms avec
# ``/``. La validation reste volontairement stricte pour éviter les noms
# ambigus et empêcher qu'une entrée UI soit interprétée comme une option CLI.
MODEL_NAME_PATTERN = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._/-]*(?::[A-Za-z0-9][A-Za-z0-9._-]*)?$"
)

DEFAULT_CNI_MODEL_SYSTEM_PROMPT = """You are a vision extraction engine for Moroccan identity cards.
An input contains one CNI side or one vertical composite with recto above verso.
Read only visible Latin-script values. Ignore Arabic text, QR codes, barcodes and MRZ.
Never invent a value. Use null when a field is absent or unreadable.
Do not confuse the holder with the parents.
Do not confuse the visible CNI number, CAN and civil-status number.
Normalize dates to YYYY-MM-DD when the complete date is visible.
Return only valid JSON, without Markdown fences, commentary or reasoning.

For a recto image, use these keys:
{"cin": null, "nom": null, "prenom": null, "date_naissance": null,
 "ville_naissance": null, "date_validite": null}

For a verso image, use these keys:
{"cin": null, "date_validite": null, "adresse": null}

For a combined image, return:
{"recto": {...recto keys...}, "verso": {...verso keys...}}"""


@dataclass(frozen=True)
class OllamaModelCreationResult:
    """Résultat sérialisable de ``ollama create``."""

    success: bool
    model_name: str
    modelfile: str
    stdout: str
    stderr: str
    return_code: int | None
    error: str | None = None


def build_modelfile_template(
    base_model: str,
    *,
    system_prompt: str = DEFAULT_CNI_MODEL_SYSTEM_PROMPT,
    num_ctx: int = 8192,
    num_predict: int = 4096,
    temperature: float = 0.0,
) -> str:
    """Construit un template lisible avec un prompt système multiligne.

    ``SYSTEM triple-quotes`` est la syntaxe officielle Ollama adaptée aux
    instructions longues. Un prompt contenant lui-même trois guillemets
    consécutifs est refusé afin de ne jamais produire un fichier ambigu.
    """

    validated_base = validate_model_name(base_model, field_name="modèle source")
    if '"""' in system_prompt:
        raise ValueError(
            "Le prompt système contient trois guillemets consécutifs, ce qui "
            "fermerait prématurément le bloc SYSTEM du Modelfile."
        )
    return (
        f"FROM {validated_base}\n\n"
        "# Paramètres de génération modifiables.\n"
        f"PARAMETER temperature {float(temperature):g}\n"
        f"PARAMETER num_ctx {max(256, int(num_ctx))}\n"
        f"PARAMETER num_predict {max(1, int(num_predict))}\n\n"
        "# Le texte entre triples guillemets peut être très long.\n"
        'SYSTEM """\n'
        f"{system_prompt.strip()}\n"
        '"""\n'
    )


def bind_selected_base_model(modelfile: str, base_model: str) -> str:
    """Force l'unique instruction ``FROM`` à utiliser le modèle sélectionné."""

    selected = validate_model_name(base_model, field_name="modèle source")
    text = (modelfile or "").strip()
    if not text:
        raise ValueError("Le Modelfile est vide.")

    lines = text.splitlines()
    from_indexes = [
        index
        for index, line in enumerate(lines)
        if re.match(r"^\s*FROM(?:\s+|$)", line, flags=re.IGNORECASE)
    ]
    if len(from_indexes) > 1:
        raise ValueError("Le Modelfile doit contenir une seule instruction FROM.")
    if from_indexes:
        lines[from_indexes[0]] = f"FROM {selected}"
    else:
        lines.insert(0, f"FROM {selected}")
        lines.insert(1, "")
    return "\n".join(lines).rstrip() + "\n"


def create_ollama_model(
    *,
    base_model: str,
    new_model_name: str,
    modelfile: str,
    timeout_seconds: float = 1800,
    ollama_executable: str = "ollama",
) -> OllamaModelCreationResult:
    """Crée un modèle dérivé avec la CLI Ollama, sans interpolation shell.

    Le timeout concerne uniquement la construction du modèle. Il est distinct
    du timeout d'inférence configuré dans le Benchmark CNI.
    """

    base = validate_model_name(base_model, field_name="modèle source")
    target = validate_model_name(new_model_name, field_name="nouveau modèle")
    if target.casefold() == base.casefold():
        raise ValueError(
            "Le nouveau nom doit être différent du modèle source afin de ne pas "
            "écraser sa configuration."
        )
    effective_modelfile = bind_selected_base_model(modelfile, base)
    timeout = max(1.0, float(timeout_seconds))

    with tempfile.TemporaryDirectory(prefix="ollama-modelfile-") as directory:
        path = Path(directory) / "Modelfile"
        path.write_text(effective_modelfile, encoding="utf-8")
        try:
            completed = subprocess.run(
                [ollama_executable, "create", target, "-f", str(path)],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
                check=False,
            )
        except FileNotFoundError:
            return OllamaModelCreationResult(
                False,
                target,
                effective_modelfile,
                "",
                "",
                None,
                "La commande 'ollama' est introuvable dans le PATH.",
            )
        except subprocess.TimeoutExpired as exc:
            return OllamaModelCreationResult(
                False,
                target,
                effective_modelfile,
                _process_output(exc.stdout),
                _process_output(exc.stderr),
                None,
                f"Création interrompue après {timeout:g} secondes.",
            )

    error = None
    if completed.returncode != 0:
        error = (
            completed.stderr.strip()
            or completed.stdout.strip()
            or f"ollama create a retourné le code {completed.returncode}."
        )
    return OllamaModelCreationResult(
        completed.returncode == 0,
        target,
        effective_modelfile,
        completed.stdout.strip(),
        completed.stderr.strip(),
        completed.returncode,
        error,
    )


def save_modelfile_artifact(
    output_root: Path,
    model_name: str,
    modelfile: str,
) -> Path:
    """Conserve le blueprint réellement utilisé dans un emplacement ignoré Git."""

    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", model_name).strip("._") or "model"
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / f"{safe_name}.Modelfile"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(modelfile, encoding="utf-8")
    temporary.replace(path)
    return path


def validate_model_name(value: str, *, field_name: str) -> str:
    """Valide un identifiant Ollama avant tout appel de processus."""

    name = str(value or "").strip()
    if not name:
        raise ValueError(f"Le champ {field_name} est obligatoire.")
    if not MODEL_NAME_PATTERN.fullmatch(name) or name.startswith("-"):
        raise ValueError(
            f"Nom invalide pour {field_name}: {name!r}. Utilisez lettres, "
            "chiffres, '.', '_', '-', '/', et éventuellement ':tag'."
        )
    return name


def _process_output(value: str | bytes | None) -> str:
    """Normalise la sortie attachée à une exception ``TimeoutExpired``."""

    if value is None:
        return ""
    return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value
