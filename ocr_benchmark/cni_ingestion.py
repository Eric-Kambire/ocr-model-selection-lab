"""Découverte des entrées CNI, import des labels JSONB et ZIP sécurisé.

Le nom du dossier est l'identifiant client canonique. Le préfixe du PDF reste
une métadonnée et ne sert jamais à rechercher un label.
"""

# ─── POURQUOI CES IMPORTS ? ───────────────────────────────────────────────────

# 'annotations' permet d'écrire les types (ex: Path | None) sans erreur sur
# les vieilles versions de Python (< 3.10). Toujours en première ligne.
from __future__ import annotations

# 'json' : lit et écrit des fichiers au format JSON (les labels, les artefacts).
import json

# 'logging' : système de journalisation Python. Permet d'écrire des messages
# INFO, WARNING, ERROR dans le terminal sans utiliser print().
import logging

# 're' : expressions régulières. Permet de chercher si un nom de fichier
# correspond à un patron précis, ex: "12345_cin_recto.pdf".
import re

# 'shutil' : utilitaires de haut niveau sur les fichiers.
# On l'utilise pour copier le contenu d'un fichier ZIP et pour supprimer
# un dossier entier en cas d'erreur (shutil.rmtree).
import shutil

# 'time' : donne accès à l'heure actuelle. On s'en sert pour créer un nom
# de dossier d'import unique basé sur la date et l'heure (ex: 20260716-144500).
import time

# 'zipfile' : ouvre et lit les archives .zip. Utilisé pour importer un jeu
# de CNI livré dans une archive ZIP.
import zipfile

# 'Path' : manipulation moderne des chemins de fichiers (Windows/Linux/Mac).
# 'PurePosixPath' : version "lecture seule" d'un chemin au format POSIX (/)
# pour analyser les chemins internes d'une archive ZIP (toujours en slash).
from pathlib import Path, PurePosixPath

# 'Any' : type générique utilisé dans les annotations pour dire qu'une valeur
# peut être de n'importe quel type (dict, list, str...).
from typing import Any

# ─── CONFIGURATION DU MODULE ──────────────────────────────────────────────────

# Crée un "logger" propre à ce module. Les messages apparaîtront avec le nom
# du module dans les logs, ce qui facilite le débogage.
LOGGER = logging.getLogger(__name__)

# Suffixes par défaut. Ils restent insensibles aux majuscules/minuscules et
# peuvent être remplacés dans l'interface CNI : par exemple ``_ID_R`` ou
# ``-recto``. L'extension ``.pdf`` n'est volontairement pas saisie par l'user.
DEFAULT_RECTO_SUFFIX = "_CIN_Recto"
DEFAULT_VERSO_SUFFIX = "_CIN_Verso"


def _build_side_filename_patterns(recto_suffix: str, verso_suffix: str) -> dict[str, re.Pattern[str]]:
    """Construit les patrons PDF depuis les suffixes configurés par l'utilisateur.

    Entrées :
        recto_suffix, verso_suffix: texte final placé avant ``.pdf``.
    Sortie :
        un patron par face qui extrait le préfixe en ``document_id``.

    ``re.escape`` est indispensable : un suffixe contenant ``+`` ou ``.`` ne
    doit jamais devenir une expression régulière involontaire.
    """
    normalized = {
        "recto": str(recto_suffix or "").strip(),
        "verso": str(verso_suffix or "").strip(),
    }
    if not normalized["recto"] or not normalized["verso"]:
        raise ValueError("Les suffixes PDF recto et verso ne peuvent pas être vides.")
    return {
        side: re.compile(rf"^(?P<document_id>.+){re.escape(suffix)}\.pdf$", re.IGNORECASE)
        for side, suffix in normalized.items()
    }


# ─── FONCTION 1 : scan_cni_clients ────────────────────────────────────────────
# CE QU'ELLE FAIT : parcourt un dossier racine, explore chaque sous-dossier
# client, cherche les PDF recto et verso avec le bon format de nom, et construit
# une "fiche de diagnostic" par client sous forme de dictionnaire Python.
# CE QU'ON FAIT AVEC : on appelle cette fonction pour savoir quels clients sont
# prêts à être traités ("ready") et lesquels ont un problème ("invalid_input").

def scan_cni_clients(
    clients_root: Path,
    labels_root: Path | None = None,
    *,
    recto_suffix: str = DEFAULT_RECTO_SUFFIX,
    verso_suffix: str = DEFAULT_VERSO_SUFFIX,
) -> list[dict[str, Any]]:
    """Construit un diagnostic d'entrée pour chaque sous-dossier client.

    Les suffixes sont éditables sans modifier le code ; seul le nom du dossier
    continue de définir l'identifiant client canonique et le nom du label.
    """

    # Si le dossier racine des clients n'existe pas du tout, on lève une erreur
    # immédiatement : il n'y a rien à scanner.
    if not clients_root.is_dir():
        raise FileNotFoundError(f"Clients folder not found: {clients_root}")

    # Les patrons sont construits une seule fois pour l'ensemble du scan.
    side_filename = _build_side_filename_patterns(recto_suffix, verso_suffix)

    # Liste vide qui va accueillir toutes les fiches de diagnostic (une par client).
    records: list[dict[str, Any]] = []

    # On parcourt tous les éléments du dossier racine, triés par ordre alphabétique,
    # en gardant uniquement ceux qui sont des dossiers (pas les fichiers isolés).
    for folder in sorted(path for path in clients_root.iterdir() if path.is_dir()):

        # On crée la fiche de diagnostic initiale pour ce dossier client.
        # Elle contient toutes les informations connues dès le départ.
        record: dict[str, Any] = {
            # "folder_client_id" : le nom du dossier EST l'identifiant du client.
            "folder_client_id": folder.name,
            # "client_dir" : le chemin complet du dossier client (en texte).
            "client_dir": str(folder),
            # Les chemins des PDF commencent à None, ils seront remplis si trouvés.
            "recto_pdf": None,
            "verso_pdf": None,
            # Les IDs des documents (la partie avant "_cin_recto") commencent à None.
            "recto_document_id": None,
            "verso_document_id": None,
            # Chemin attendu du fichier JSONB source (le label venant de l'extérieur).
            # Si labels_root n'est pas fourni, on met None.
            "label_source": str(labels_root / f"{folder.name}.jsonb") if labels_root else None,
            # Chemin où le label sera copié DANS le dossier client (fichier local).
            "label_path": str(folder / f"{folder.name}.json"),
            # Statut initial du label : "non configuré" si pas de dossier de labels,
            # sinon "non trouvé" (on cherchera plus loin).
            "label_status": "label_root_not_set" if labels_root is None else "label_not_found",
            # Statut global du client : "ready" par défaut, devient "invalid_input"
            # si on détecte un problème.
            "status": "ready",
            # Liste des problèmes détectés (PDF manquant, doublon, IDs incohérents...).
            "issues": [],
        }

        # On parcourt tous les fichiers dans le dossier de ce client pour
        # trouver les PDF recto et verso.
        for candidate in folder.iterdir():
            # Si l'élément n'est pas un fichier (ex: c'est un sous-dossier), on passe.
            if not candidate.is_file():
                continue

            # Pour chaque fichier trouvé, on teste s'il correspond au patron recto
            # ou au patron verso définis dans _SIDE_FILENAME.
            for side, matcher in side_filename.items():
                # On applique l'expression régulière sur le NOM du fichier (pas le chemin entier).
                match = matcher.match(candidate.name)

                # Si le nom du fichier ne correspond pas à ce patron, on passe au suivant.
                if not match:
                    continue

                # Si on avait déjà trouvé un fichier recto (ou verso), c'est un doublon.
                # On enregistre le problème mais on ne plante pas.
                if record[f"{side}_pdf"] is not None:
                    record["issues"].append(f"duplicate_{side}_pdf")
                else:
                    # On enregistre le chemin complet du fichier PDF trouvé.
                    record[f"{side}_pdf"] = str(candidate)
                    # On extrait le "document_id" (la partie capturée par la regex).
                    record[f"{side}_document_id"] = match.group("document_id")

        # Après avoir parcouru tous les fichiers, on vérifie si une face manque.
        # On fait ça pour "recto" et "verso".
        for side in ("recto", "verso"):
            # Si le PDF n'a pas été trouvé, on ajoute un problème à la fiche.
            if record[f"{side}_pdf"] is None:
                record["issues"].append(f"missing_{side}_pdf")

        # Si les deux document_id existent mais sont différents, c'est suspect :
        # les deux PDF viennent peut-être de deux clients différents.
        if (record["recto_document_id"] and record["verso_document_id"]
                and record["recto_document_id"] != record["verso_document_id"]):
            record["issues"].append("document_id_differs_between_sides")

        # Si un dossier de labels a été fourni ET que le fichier JSONB correspondant
        # au client existe, on marque le label comme disponible.
        if labels_root and Path(record["label_source"]).is_file():
            record["label_status"] = "label_available"

        # S'il y a au moins un problème détecté, le statut global devient "invalid_input".
        # Cela signifie que ce client ne sera PAS traité par le runner.
        if record["issues"]:
            record["status"] = "invalid_input"

        # On ajoute la fiche complète de ce client à notre liste de résultats.
        records.append(record)

    # On écrit dans les logs combien de clients ont été trouvés.
    LOGGER.info("CNI client scan complete | root=%s | clients=%d", clients_root, len(records))

    # On retourne la liste complète des fiches de diagnostic.
    return records


# ─── FONCTION 2 : materialize_cni_labels ──────────────────────────────────────
# CE QU'ELLE FAIT : pour chaque client dont le scan a signalé un label disponible
# (fichier .jsonb), elle lit ce fichier, vérifie que c'est du JSON valide, et le
# copie dans le dossier du client sous un nom standard (.json).
# CE QU'ON FAIT AVEC : on appelle cette fonction après scan_cni_clients pour que
# les labels soient disponibles localement avant de lancer le benchmark.

def materialize_cni_labels(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Copie un JSONB texte UTF-8 valide dans le dossier client correspondant."""

    # Liste qui va contenir les fiches mises à jour (avec le statut du label).
    updated: list[dict[str, Any]] = []

    # On parcourt chaque fiche de diagnostic produite par scan_cni_clients.
    for original in records:
        # On crée une copie du dictionnaire pour ne pas modifier l'original.
        record = dict(original)

        # On récupère le chemin source du label (fichier .jsonb externe) et le
        # chemin cible où on va le copier (dans le dossier client).
        source_text, target_text = record.get("label_source"), record.get("label_path")

        # Si l'un ou l'autre chemin est absent (cas où labels_root n'était pas fourni),
        # on ajoute la fiche telle quelle et on passe au client suivant.
        if not source_text or not target_text:
            updated.append(record)
            continue

        # On convertit les textes en objets Path.
        source, target = Path(source_text), Path(target_text)

        # Si le fichier .jsonb source n'existe pas sur le disque, le label est absent.
        # On le note dans la fiche et on passe au client suivant. L'OCR pourra quand
        # même tourner, mais l'accuracy ne pourra pas être calculée.
        if not source.is_file():
            record["label_status"] = "label_not_found"
            updated.append(record)
            continue

        # On tente de lire et valider le fichier .jsonb.
        try:
            # On lit le contenu du fichier source et on le parse en JSON Python.
            value = json.loads(source.read_text(encoding="utf-8"))

            # Le label doit être soit un objet JSON (dict), soit un tableau (list).
            # Si c'est autre chose (ex: un simple nombre), c'est invalide.
            if not isinstance(value, (dict, list)):
                raise ValueError("JSON label must be an object or array")

            # On écrit le label de façon atomique dans le dossier client
            # (d'abord dans un fichier temporaire, puis on renomme).
            _atomic_write_json(target, value)

            # On marque la fiche : le label a bien été copié localement.
            record["label_status"] = "label_materialized"

            # On écrit un message INFO dans les logs pour confirmer la copie.
            LOGGER.info("CNI label materialized | client=%s | target=%s",
                        record["folder_client_id"], target)

        # On capture les erreurs possibles : problème de lecture (OSError),
        # encodage invalide (UnicodeDecodeError), JSON malformé (JSONDecodeError),
        # ou notre propre erreur de validation (ValueError).
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            # On marque que le parsing du label a échoué.
            record["label_status"] = "label_parse_failed"
            # On ajoute le détail de l'erreur dans la liste des problèmes.
            record["issues"] = [*record.get("issues", []), f"label_parse_failed:{type(exc).__name__}"]
            # On écrit un WARNING dans les logs (pas une erreur fatale).
            LOGGER.warning("CNI label parsing failed | client=%s | source=%s | error=%s",
                           record["folder_client_id"], source, exc)

        # Qu'il y ait eu une erreur ou non, on ajoute toujours la fiche à la liste.
        updated.append(record)

    # On retourne la liste mise à jour.
    return updated


# ─── FONCTION 3 : import_cni_zip ──────────────────────────────────────────────
# CE QU'ELLE FAIT : extrait le contenu d'une archive ZIP dans un dossier d'import
# horodaté, avec une protection contre les attaques "ZIP-slip" (un fichier ZIP
# malveillant qui tente d'écrire en dehors du dossier cible).
# CE QU'ON FAIT AVEC : quand un utilisateur uploade un ZIP de CNI via l'interface.

def import_cni_zip(zip_path: Path, imports_root: Path) -> dict[str, Any]:
    """Extrait une archive de test dans le répertoire local d'import."""

    # On vérifie que le fichier est bien un ZIP (existe et a l'extension .zip).
    if not zip_path.is_file() or zip_path.suffix.lower() != ".zip":
        raise ValueError("A .zip archive is required for CNI import.")

    # On crée le dossier racine d'import s'il n'existe pas (parents=True crée
    # aussi les dossiers parents manquants).
    imports_root.mkdir(parents=True, exist_ok=True)

    # On crée un nom de dossier unique basé sur la date et l'heure actuelles
    # pour ne jamais écraser un import précédent.
    destination = imports_root / f"cni-import-{time.strftime('%Y%m%d-%H%M%S')}"

    # On crée ce dossier de destination. exist_ok=False signifie qu'il NE doit
    # PAS déjà exister (ce qui est garanti par l'horodatage).
    destination.mkdir(parents=True, exist_ok=False)

    # Compteur du nombre de fichiers extraits avec succès.
    extracted = 0

    try:
        # On ouvre l'archive ZIP en mode lecture.
        with zipfile.ZipFile(zip_path) as archive:
            # On parcourt la liste de tous les fichiers/dossiers dans le ZIP.
            for member in archive.infolist():
                # On analyse le chemin interne du fichier ZIP avec PurePosixPath
                # car les chemins internes d'un ZIP utilisent toujours le slash /.
                relative = PurePosixPath(member.filename)

                # PROTECTION ZIP-SLIP : on rejette les chemins absolus (/etc/passwd)
                # ou qui remontent dans les dossiers parents (../../etc/passwd).
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError(f"Unsafe ZIP path: {member.filename}")

                # Si c'est un dossier dans le ZIP (et non un fichier), on le saute.
                # Les dossiers sont créés automatiquement lors de l'extraction des fichiers.
                if member.is_dir():
                    continue

                # On construit le chemin complet du fichier à créer sur le disque.
                # joinpath(*relative.parts) recompose le chemin correctement sur Windows.
                target = destination.joinpath(*relative.parts)

                # On crée les dossiers parents si nécessaire.
                target.parent.mkdir(parents=True, exist_ok=True)

                # On copie le contenu du fichier depuis le ZIP vers le disque.
                # shutil.copyfileobj copie par blocs (efficace pour les gros fichiers).
                with archive.open(member) as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output)

                # On incrémente le compteur de fichiers extraits.
                extracted += 1

    except Exception:
        # En cas d'erreur (ZIP corrompu, chemin unsafe...), on supprime le dossier
        # de destination ENTIER pour ne pas laisser un import partiel sur le disque.
        # ignore_errors=True évite une deuxième erreur si la suppression elle-même échoue.
        shutil.rmtree(destination, ignore_errors=True)
        # On propage l'exception originale vers l'appelant.
        raise

    # On écrit dans les logs combien de fichiers ont été extraits.
    LOGGER.info("CNI ZIP imported | archive=%s | destination=%s | files=%d",
                zip_path, destination, extracted)

    # On retourne un résumé : où les fichiers ont été extraits et combien.
    return {"import_root": str(destination), "files": extracted}


# ─── FONCTION 4 : write_cni_json ──────────────────────────────────────────────
# CE QU'ELLE FAIT : façade publique simple pour écrire un fichier JSON d'artefact.
# CE QU'ON FAIT AVEC : appelée par le runner CNI pour sauvegarder chaque résultat
# (recto.extraction.json, verso.extraction.json, global.extraction.json...).

def write_cni_json(path: Path, value: dict[str, Any]) -> None:
    """Écrit un artefact CNI en JSON UTF-8 lisible de façon atomique."""
    # On délègue directement à la fonction interne atomique.
    _atomic_write_json(path, value)


# ─── FONCTION INTERNE : _atomic_write_json ────────────────────────────────────
# CE QU'ELLE FAIT : écrit un fichier JSON de façon sûre.
# La technique "écriture atomique" consiste à écrire d'abord dans un fichier
# temporaire (.tmp), puis à le renommer en fichier final d'un seul coup.
# CE QU'ON FAIT AVEC : cela évite qu'un refresh de l'interface (Gradio) lise
# un fichier JSON à moitié écrit et provoque une erreur de parsing.

def _atomic_write_json(path: Path, value: Any) -> None:
    """Écrit d'abord un temporaire voisin, puis remplace l'artefact final."""

    # On crée les dossiers parents s'ils n'existent pas encore.
    path.parent.mkdir(parents=True, exist_ok=True)

    # On crée un chemin temporaire en ajoutant ".tmp" à l'extension du fichier.
    # Ex: "recto.json" → "recto.json.tmp"
    temporary = path.with_suffix(path.suffix + ".tmp")

    # On convertit la valeur Python en texte JSON et on l'écrit dans le temporaire.
    # ensure_ascii=False : conserve les accents et caractères arabes tels quels.
    # indent=2 : rend le JSON lisible avec une indentation de 2 espaces.
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")

    # On renomme le fichier temporaire en fichier final.
    # Sur la plupart des systèmes, cette opération est atomique : personne ne
    # peut lire un fichier à moitié écrit car le renommage est instantané.
    temporary.replace(path)
