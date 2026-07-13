"""Rebuild the standalone, documented Google Colab benchmark notebook.

The notebook is generated because editing large .ipynb JSON blobs by hand is
error-prone.  Keep this file in sync with benchmark_colab.ipynb.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "benchmark_colab.ipynb"


def _source(text: str) -> list[str]:
    normalized = textwrap.dedent(text).strip("\n") + "\n"
    return normalized.splitlines(keepends=True)


def markdown(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": _source(text)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"collapsed": False},
        "outputs": [],
        "source": _source(text),
    }


cells: list[dict] = []

cells.append(markdown(r"""
    # OCR Model Selection Lab — notebook Colab autonome

    Ce notebook est une alternative complète à l'application locale : il ne clone pas le dépôt GitHub de l'application et toute la logique de téléchargement, d'inférence, de métriques, de graphiques et d'interface Gradio est écrite dans les cellules.

    **Parcours recommandé**

    1. Activez un GPU Colab : **Exécution → Modifier le type d'exécution → T4 GPU**.
    2. Ajoutez les secrets `HF_TOKEN`, `KAGGLE_API_TOKEN` et, facultativement, `GITHUB_TOKEN` dans le panneau **Secrets** de Colab.
    3. Exécutez les cellules dans l'ordre. Les échantillons réels sont petits et les modèles sont chargés un par un.
    4. Dans Gradio, choisissez les modèles, le nombre de documents puis cliquez sur **Valider et lancer**.

    Le notebook conserve les sorties lentes, enregistre un checkpoint après chaque image et sépare clairement les erreurs techniques des scores OCR. Une recommandation basée sur moins de 30 documents scorables par catégorie est marquée **exploratoire**.
"""))

cells.append(markdown(r"""
    ## 1. Installer l'environnement principal

    Le profil `CORE` réunit les modèles compatibles avec Transformers 5.x : PP‑OCRv6, GLM‑OCR, Granite Docling, PaddleOCR‑VL, MiniCPM et LightOnOCR. Les modèles dots.ocr, Unlimited‑OCR, Chandra et le checkpoint Qwen GGUF disposent de profils alternatifs dans la cellule (`DOTS_456`, `LEGACY_457`, `CORE_CHANDRA`, `LLAMA_CPP`). Changez de profil seulement dans un **runtime Colab frais** : ne rétrogradez jamais Transformers après l'avoir importé. LocateAnything reste un benchmark de localisation séparé.

    L'installation n'utilise pas Ollama. Les poids viennent directement de Hugging Face ou du moteur OCR concerné.
"""))

cells.append(code(r"""
    # ÉTAPE 1 — Dépendances du profil CORE
    import os
    import shutil
    import subprocess
    import sys

    # Changez ce profil uniquement dans un runtime frais, puis redémarrez avant de revenir à CORE.
    # CORE | CORE_CHANDRA | DOTS_456 | LEGACY_457 | LLAMA_CPP
    RUNTIME_PROFILE = "CORE"

    def pip_install(*packages):
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "--upgrade", *packages],
            check=True,
        )

    common_packages = (
        "pandas>=2.2", "numpy>=1.26", "pillow>=10", "matplotlib>=3.8",
        "plotly>=5.24", "gradio>=6.0,<7", "tqdm>=4.66", "psutil>=5.9",
        "python-Levenshtein>=0.25", "datasets>=3.0", "huggingface_hub>=0.30",
        "kagglehub>=0.3.12", "accelerate>=1.0", "sentencepiece", "protobuf", "safetensors",
        "markdown>=3.8", "beautifulsoup4>=4.13", "av>=12",
    )
    if RUNTIME_PROFILE == "CORE":
        pip_install(
            *common_packages, "transformers>=5.8.0,<6", "easyocr>=1.7",
            "paddleocr>=3.7.0", "docling-core>=2.0",
        )
    elif RUNTIME_PROFILE == "CORE_CHANDRA":
        pip_install(
            *common_packages, "transformers>=5.8.0,<6", "docling-core>=2.0",
            "easyocr>=1.7", "paddleocr>=3.7.0", "chandra-ocr[hf]",
        )
    elif RUNTIME_PROFILE == "DOTS_456":
        if not shutil.which("nvidia-smi"):
            raise RuntimeError("Le profil DOTS_456 exige un GPU NVIDIA L4/A100 ou plus récent.")
        pip_install(
            *common_packages, "transformers==4.56.1", "qwen-vl-utils",
            "packaging", "ninja", "git+https://github.com/rednote-hilab/dots.ocr.git",
        )
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "--upgrade", "flash-attn==2.8.3", "--no-build-isolation"],
            check=True,
        )
    elif RUNTIME_PROFILE == "LEGACY_457":
        # LocateAnything publie NumPy 1.25/Pillow 11 avec Transformers 4.57.1.
        # NumPy 1.25 n'a toutefois pas de wheel Python 3.12 (runtime Colab actuel),
        # d'où le repli compatible 1.26.x. On évite aussi deux contraintes NumPy/Pillow dans le même pip.
        legacy_numpy = "numpy>=1.26,<2" if sys.version_info >= (3, 12) else "numpy==1.25.0"
        legacy_packages = tuple(
            package for package in common_packages
            if not package.startswith(("numpy", "pillow"))
        ) + (legacy_numpy, "pillow==11.1.0")
        pip_install(
            *legacy_packages, "transformers==4.57.1", "einops==0.8.2", "addict==2.4.0",
            "easydict==1.13", "pymupdf>=1.27", "opencv-python-headless==4.11.0.86",
            "peft", "decord==0.6.0", "lmdb==1.7.5",
        )
    elif RUNTIME_PROFILE == "LLAMA_CPP":
        pip_install(*common_packages)
        llama_root = "/content/llama.cpp"
        if not os.path.exists(llama_root):
            subprocess.run(["git", "clone", "--depth", "1", "https://github.com/ggml-org/llama.cpp", llama_root], check=True)
        cuda_build = "ON" if shutil.which("nvidia-smi") else "OFF"
        subprocess.run(["cmake", "-S", llama_root, "-B", f"{llama_root}/build", f"-DGGML_CUDA={cuda_build}"], check=True)
        subprocess.run(["cmake", "--build", f"{llama_root}/build", "--config", "Release", "-j", "2"], check=True)
        os.environ["PATH"] = f"{llama_root}/build/bin:" + os.environ.get("PATH", "")
    else:
        raise ValueError(f"Profil inconnu: {RUNTIME_PROFILE}")

    print("Profil installé:", RUNTIME_PROFILE)
    print("Aucun modèle n'est gardé en mémoire à ce stade.")
"""))

cells.append(markdown(r"""
    ## 2. Détecter le CPU/GPU et préparer l'espace de travail

    Cette cellule détecte automatiquement Colab. Sur votre PC elle peut fonctionner en CPU ; sur Colab, un T4 permet de tester les modèles génératifs sélectionnés. Les dossiers créés sous `/content` sont temporaires et disparaissent quand la session Colab s'arrête : exportez donc les résultats importants à la fin.
"""))

cells.append(code(r"""
    # ÉTAPE 2 — Runtime et répertoires
    import ast
    import base64
    import gc
    import hashlib
    import html
    import json
    import math
    import os
    import platform
    import random
    import re
    import secrets
    import shutil
    import subprocess
    import time
    import unicodedata
    import zipfile
    from io import BytesIO
    from pathlib import Path, PurePosixPath
    from urllib.error import HTTPError
    from urllib.parse import quote
    from urllib.request import Request, urlopen

    import numpy as np
    import pandas as pd
    from PIL import Image, ImageDraw, ImageFont

    try:
        import torch
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        torch = None
        DEVICE = "cpu"

    try:
        from google.colab import userdata
        colab_userdata = userdata
        IS_COLAB = True
    except ImportError:
        userdata = None
        colab_userdata = None
        IS_COLAB = False

    RUNTIME_PROFILE = globals().get("RUNTIME_PROFILE", "CORE")
    WORK_DIR = Path("/content/ocr_model_selection_lab") if IS_COLAB else Path.cwd() / "ocr_model_selection_lab"
    DATASET_DIR = WORK_DIR / "dataset"
    UPLOAD_DIR = DATASET_DIR / "user_uploads"
    RUNS_DIR = WORK_DIR / "runs"
    MODEL_DIR = WORK_DIR / "models"
    for folder in (DATASET_DIR, UPLOAD_DIR, RUNS_DIR, MODEL_DIR):
        folder.mkdir(parents=True, exist_ok=True)

    GPU_NAME = torch.cuda.get_device_name(0) if torch and DEVICE == "cuda" else "CPU"
    GPU_VRAM_GB = (
        torch.cuda.get_device_properties(0).total_memory / 1024**3
        if torch and DEVICE == "cuda" else 0.0
    )
    GPU_CAPABILITY = (
        torch.cuda.get_device_capability(0)
        if torch and DEVICE == "cuda" else (0, 0)
    )
    IS_T4 = "T4" in GPU_NAME.upper()

    print("Python:", platform.python_version())
    print("Calcul:", DEVICE.upper(), "|", GPU_NAME, f"| {GPU_VRAM_GB:.1f} Go VRAM" if GPU_VRAM_GB else "")
    print("Workspace:", WORK_DIR)
    if DEVICE == "cpu":
        print("Conseil: activez un GPU T4 dans Colab pour GLM-OCR et les autres VLM.")
"""))

cells.append(markdown(r"""
    ## 3. Lire les secrets Colab sans les exposer

    Dans Colab, ouvrez l'icône **clé** puis créez :

    - `HF_TOKEN` : token Hugging Face en lecture, utile pour les modèles gated et pour éviter les limites anonymes ;
    - `KAGGLE_API_TOKEN` : nouveau token Kaggle recommandé ;
    - `GITHUB_TOKEN` : facultatif, pour augmenter la limite de l'API GitHub publique ;
    - `GRADIO_USERNAME` + `GRADIO_PASSWORD` : facultatifs ; sinon Colab crée un mot de passe temporaire pour protéger le tunnel public ;
    - anciens comptes Kaggle seulement : `KAGGLE_USERNAME` + `KAGGLE_KEY`, ou `KAGGLE_JSON` contenant le JSON du fichier `kaggle.json`.

    Colab fournit les valeurs via `from google.colab import userdata` puis `userdata.get("HF_TOKEN")`. La cellule ci-dessous n'affiche **que des booléens**, jamais une valeur de token. Les secrets ne sont pas inclus quand vous partagez le notebook : chaque destinataire doit recréer ces mêmes noms et autoriser l'accès au notebook.
"""))

cells.append(code(r"""
    # ÉTAPE 3 — Secrets Colab / variables locales
    def colab_secret(name):
        if colab_userdata is None:
            return None
        try:
            value = userdata.get(name)
        except Exception:
            return None
        return value.strip() if isinstance(value, str) and value.strip() else None

    def secret_value(*names):
        for name in names:
            value = colab_secret(name) or os.getenv(name)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None

    HF_TOKEN = secret_value("HF_TOKEN", "HUGGINGFACE_TOKEN", "HF_HUB_TOKEN")
    KAGGLE_API_TOKEN = secret_value("KAGGLE_API_TOKEN")
    GITHUB_TOKEN = secret_value("GITHUB_TOKEN", "GH_TOKEN")
    GRADIO_USERNAME = secret_value("GRADIO_USERNAME")
    GRADIO_PASSWORD = secret_value("GRADIO_PASSWORD")

    if HF_TOKEN:
        os.environ["HF_TOKEN"] = HF_TOKEN
    if KAGGLE_API_TOKEN:
        os.environ["KAGGLE_API_TOKEN"] = KAGGLE_API_TOKEN
    if GITHUB_TOKEN:
        os.environ["GITHUB_TOKEN"] = GITHUB_TOKEN

    kaggle_json = secret_value("KAGGLE_JSON")
    if kaggle_json and not KAGGLE_API_TOKEN:
        try:
            credentials = json.loads(kaggle_json)
            os.environ["KAGGLE_USERNAME"] = str(credentials["username"])
            os.environ["KAGGLE_KEY"] = str(credentials["key"])
        except (KeyError, TypeError, json.JSONDecodeError):
            print("KAGGLE_JSON existe mais son format n'est pas {username, key}.")
    elif not KAGGLE_API_TOKEN:
        legacy_username = secret_value("KAGGLE_USERNAME")
        legacy_key = secret_value("KAGGLE_KEY")
        if legacy_username and legacy_key:
            os.environ["KAGGLE_USERNAME"] = legacy_username
            os.environ["KAGGLE_KEY"] = legacy_key

    SECRET_STATUS = {
        "Hugging Face": bool(HF_TOKEN),
        "Kaggle": bool(KAGGLE_API_TOKEN or (os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY"))),
        "GitHub (facultatif)": bool(GITHUB_TOKEN),
        "Gradio personnalisé (facultatif)": bool(GRADIO_USERNAME and GRADIO_PASSWORD),
    }
    print("Secrets disponibles:", SECRET_STATUS)

    if IS_COLAB:
        temporary_gradio_password = not (GRADIO_USERNAME and GRADIO_PASSWORD)
        GRADIO_USERNAME = GRADIO_USERNAME or "colab"
        GRADIO_PASSWORD = GRADIO_PASSWORD or secrets.token_urlsafe(12)
        GRADIO_AUTH = (GRADIO_USERNAME, GRADIO_PASSWORD)
        if temporary_gradio_password:
            print("Connexion Gradio temporaire — utilisateur:", GRADIO_USERNAME)
            print("Connexion Gradio temporaire — mot de passe:", GRADIO_PASSWORD)
            print("Ne partagez ni le lien Gradio ni ce mot de passe: les documents peuvent être privés.")
    else:
        GRADIO_AUTH = (GRADIO_USERNAME, GRADIO_PASSWORD) if GRADIO_USERNAME and GRADIO_PASSWORD else None
"""))

cells.append(markdown(r"""
    ## 4. Configurer les données de test

    Les quatre sources réelles ci-dessous sont activées avec de petits quotas. `DATA_VOLUME="DEMO"` est rapide. Passez à `DATA_VOLUME="DECISION_30"` pour demander 30 documents annotés dans chacune des deux catégories Hugging Face ; le run devient nettement plus long, mais peut dépasser le seuil minimal de 30 labels par catégorie. Pour IAM, le notebook télécharge **des fichiers précis** (environ quelques Mo) au lieu de l'archive Kaggle complète d'environ 4,6 Go. Les images IAM n'ont pas de transcription dans cette archive : elles sont visibles et exécutées, mais exclues des scores CER/WER.

    Le dépôt GitHub contient des sorties OCR dérivées, pas des vérités terrain relues par un humain. Elles restent donc exclues du classement par défaut. Les doublons entre Kaggle, GitHub et Hugging Face sont détectés par SHA‑256 et une seule copie est benchmarkée.

    Pour vos propres données, préparez un ZIP contenant `labels.csv` et les images. Colonnes obligatoires : `image_path,ground_truth,category`. Colonnes facultatives : `description,task_type,prompt,split,is_scorable,label_provenance`.
"""))

cells.append(code(r"""
    # ÉTAPE 4A — Sources et quotas modifiables
    USE_SYNTHETIC_FALLBACK = True
    EXCLUDE_DUPLICATE_IMAGES = True
    ALLOW_DERIVED_LABELS_IN_RANKING = False
    KAGGLE_INTERACTIVE_LOGIN = False
    DATA_VOLUME = "DEMO"  # DEMO | DECISION_30

    DATASET_SOURCES = {
        "hf_multifin": {
            "enabled": True,
            "repo_id": "TheFinAI/MultiFinBen-EnglishOCR",
            "revision": "08cbac5db10834b6cbce428364e0bd8c52eea6fb",
            "split": "train",
            "max_samples": 3,
        },
        "hf_cheques": {
            "enabled": True,
            "repo_id": "arunchincheti/handwritten_and_cheques_dataset",
            "revision": "4d81a7c9b1af2fcbb9abc7c1f85f1c7b789c01a2",
            "split": "test",
            "max_samples": 4,
        },
        "kaggle_iam": {
            "enabled": True,
            "handle": "naderabdelghany/iam-handwritten-forms-dataset/versions/1",
            "source_revision": "version-1",
            "max_samples": 3,
            "download_full_dataset": False,
            "sample_files": [
                "data/000/a01-000u.png",
                "data/000/a01-003u.png",
                "data/000/a01-007u.png",
                "data/000/a01-011u.png",
                "data/000/a01-014u.png",
            ],
        },
        "github_forms": {
            "enabled": True,
            "repo": "bernardadhitya/handwritten-form-ocr-ie-json-dataset",
            "revision": "6b9113e8e18973293cc003bc079c21e2f7f3d6e5",
            "max_samples": 3,
            "label_mode": "ocr_text",
        },
    }

    if DATA_VOLUME == "DECISION_30":
        DATASET_SOURCES["hf_multifin"]["max_samples"] = 30
        DATASET_SOURCES["hf_cheques"]["max_samples"] = 30
        DATASET_SOURCES["kaggle_iam"]["max_samples"] = len(DATASET_SOURCES["kaggle_iam"]["sample_files"])
        DATASET_SOURCES["github_forms"]["max_samples"] = 30
    elif DATA_VOLUME != "DEMO":
        raise ValueError("DATA_VOLUME doit valoir DEMO ou DECISION_30.")

    ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}
    print("Volume de données:", DATA_VOLUME)
    print("Sources actives:", [name for name, cfg in DATASET_SOURCES.items() if cfg["enabled"]])
"""))

cells.append(code(r"""
    # ÉTAPE 4B — Fonctions de chargement, upload et contrôle des données
    def make_sample_image(path, lines, size=(1100, 760)):
        path.parent.mkdir(parents=True, exist_ok=True)
        image = Image.new("RGB", size, "#fbfaf7")
        draw = ImageDraw.Draw(image)
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 28)
        except Exception:
            font = None
        y = 55
        for line in lines:
            draw.text((55, y), line, fill="#172033", font=font)
            y += 46
        image.save(path)

    def build_sample_dataset():
        specs = [
            ("bank", "cheque_sample.png", "BANQUE EXEMPLE\nMontant: 1 250,00 EUR\nBénéficiaire: Eric Kambire\nDate: 12/07/2026"),
            ("table", "table_sample.png", "Produit | Quantité | Prix\nCahier | 4 | 12.00\nStylo | 10 | 7.50"),
            ("form", "form_sample.png", "Nom: Eric Kambire\nAdresse: 10 rue Exemple\nTéléphone: 0600000000"),
        ]
        rows = []
        for category, filename, truth in specs:
            path = DATASET_DIR / "synthetic" / filename
            make_sample_image(path, truth.splitlines())
            rows.append({
                "id": f"synthetic-{filename}", "image_path": str(path), "ground_truth": truth,
                "category": category, "description": "Repli synthétique généré dans le notebook",
                "source": "synthetic", "source_revision": "notebook-v2", "split": "demo",
                "task_type": "transcription", "prompt": "", "label_provenance": "synthetic_exact",
                "is_scorable": True, "license": "generated_in_notebook",
            })
        return rows

    def assert_inside(child, parent):
        child, parent = Path(child).resolve(), Path(parent).resolve()
        if child != parent and parent not in child.parents:
            raise ValueError(f"Chemin ZIP non autorisé: {child}")
        return child

    MAX_ZIP_FILES = 1000
    MAX_ZIP_UNCOMPRESSED_BYTES = 1024**3
    MAX_ZIP_MEMBER_BYTES = 100 * 1024**2
    MAX_ZIP_COMPRESSION_RATIO = 200

    def safe_extract_zip(zip_path, extract_dir):
        extract_dir = Path(extract_dir)
        with zipfile.ZipFile(zip_path) as archive:
            members = archive.infolist()
            if len(members) > MAX_ZIP_FILES:
                raise ValueError(f"ZIP refusé: {len(members)} fichiers > limite {MAX_ZIP_FILES}.")
            total_size = sum(member.file_size for member in members)
            if total_size > MAX_ZIP_UNCOMPRESSED_BYTES:
                raise ValueError("ZIP refusé: taille décompressée supérieure à 1 Go.")
            for member in members:
                if member.flag_bits & 0x1:
                    raise ValueError("ZIP chiffré non supporté.")
                if member.file_size > MAX_ZIP_MEMBER_BYTES:
                    raise ValueError(f"Fichier ZIP trop grand: {member.filename}")
                ratio = member.file_size / max(1, member.compress_size)
                if ratio > MAX_ZIP_COMPRESSION_RATIO:
                    raise ValueError(f"Ratio de compression suspect: {member.filename}")
                target = assert_inside(extract_dir / member.filename, extract_dir)
                if member.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as source, target.open("wb") as destination:
                    shutil.copyfileobj(source, destination)

    def read_records_from_folder(folder):
        folder = Path(folder)
        labels_csv = next(folder.rglob("labels.csv"), None)
        dataset_json = next(folder.rglob("dataset.json"), None)
        if labels_csv:
            records = pd.read_csv(labels_csv).fillna("").to_dict("records")
            base_dir = labels_csv.parent
        elif dataset_json:
            records = json.loads(dataset_json.read_text(encoding="utf-8"))
            base_dir = dataset_json.parent
        else:
            raise ValueError("Le ZIP doit contenir labels.csv ou dataset.json.")
        def parse_bool(value, default=True):
            if isinstance(value, bool):
                return value
            if value is None or str(value).strip() == "":
                return default
            normalized = str(value).strip().lower()
            if normalized in {"1", "true", "yes", "oui"}:
                return True
            if normalized in {"0", "false", "no", "non"}:
                return False
            raise ValueError(f"Booléen invalide: {value}")

        rows = []
        for index, record in enumerate(records, start=1):
            for field in ("image_path", "ground_truth", "category"):
                if not str(record.get(field, "")).strip():
                    raise ValueError(f"Ligne {index}: champ obligatoire manquant: {field}")
            source = assert_inside(base_dir / str(record["image_path"]), base_dir)
            if not source.is_file():
                matches = list(base_dir.rglob(Path(str(record["image_path"])).name))
                if len(matches) != 1:
                    raise FileNotFoundError(record["image_path"])
                source = matches[0]
            if source.suffix.lower() not in ALLOWED_EXTENSIONS:
                raise ValueError(f"Format image non supporté: {source.suffix}")
            fingerprint = hashlib.sha1(source.read_bytes()).hexdigest()[:12]
            destination = UPLOAD_DIR / f"{fingerprint}{source.suffix.lower()}"
            shutil.copy2(source, destination)
            rows.append({
                "id": f"custom-{fingerprint}", "image_path": str(destination),
                "ground_truth": str(record["ground_truth"]),
                "category": str(record["category"]).strip().lower().replace(" ", "_"),
                "description": str(record.get("description", source.name)), "source": "custom_zip",
                "source_revision": fingerprint, "split": str(record.get("split", "custom")),
                "task_type": str(record.get("task_type", "transcription")),
                "prompt": str(record.get("prompt", "")),
                "label_provenance": str(record.get("label_provenance", "user_provided")),
                "is_scorable": parse_bool(record.get("is_scorable", True)),
                "license": str(record.get("license", "user_provided")),
            })
        return rows

    def save_any_image(value, destination):
        destination = Path(destination).with_suffix(".png")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(value, Image.Image):
            image = value
        elif isinstance(value, dict) and value.get("bytes") is not None:
            image = Image.open(BytesIO(value["bytes"]))
        elif isinstance(value, dict) and value.get("path"):
            image = Image.open(value["path"])
        elif isinstance(value, (bytes, bytearray)):
            image = Image.open(BytesIO(value))
        elif isinstance(value, str) and len(value) < 1000 and Path(value).is_file():
            image = Image.open(value)
        elif isinstance(value, str):
            payload = value.split(",", 1)[-1] if value.startswith("data:") else value
            image = Image.open(BytesIO(base64.b64decode(payload)))
        else:
            raise TypeError(f"Format image inconnu: {type(value)}")
        image.convert("RGB").save(destination, format="PNG")
        return destination

    def canonical_structured_answer(value):
        parsed = value
        if isinstance(value, str):
            try:
                parsed = ast.literal_eval(value)
            except Exception:
                return None
        if isinstance(parsed, list) and all(isinstance(item, dict) for item in parsed):
            merged = {}
            for item in parsed:
                merged.update(item)
            parsed = merged
        return json.dumps(parsed, ensure_ascii=False, sort_keys=True) if isinstance(parsed, dict) else None

    def hf_token_argument():
        return HF_TOKEN or None

    def load_hf_multifin(cfg):
        from datasets import load_dataset
        stream = load_dataset(
            cfg["repo_id"], split=cfg["split"], streaming=True,
            revision=cfg.get("revision"), token=hf_token_argument(),
        )
        rows = []
        for index, item in enumerate(stream):
            if index >= int(cfg["max_samples"]):
                break
            path = save_any_image(item["image"], DATASET_DIR / "hf_multifin" / f"{index:05d}.png")
            truth = str(item.get("text", ""))
            rows.append({
                "id": f"hf-multifin-{index}", "image_path": str(path), "ground_truth": truth,
                "category": "financial_report", "description": "Page de rapport financier",
                "source": "hf_multifin", "source_revision": cfg["revision"], "split": cfg["split"],
                "task_type": "transcription", "prompt": "", "label_provenance": "dataset_annotation",
                "is_scorable": bool(truth.strip()), "license": "apache-2.0",
            })
        return rows

    def load_hf_cheques(cfg):
        from datasets import load_dataset
        stream = load_dataset(
            cfg["repo_id"], split=cfg["split"], streaming=True,
            revision=cfg.get("revision"), token=hf_token_argument(),
        )
        rows = []
        for index, item in enumerate(stream):
            if index >= int(cfg["max_samples"]):
                break
            path = save_any_image(item["image"], DATASET_DIR / "hf_cheques" / f"{index:05d}.png")
            structured = canonical_structured_answer(item.get("answers"))
            truth = structured if structured is not None else str(item.get("answers", ""))
            task_type = "key_value_extraction" if structured is not None else "transcription"
            rows.append({
                "id": f"hf-cheques-{index}", "image_path": str(path), "ground_truth": truth,
                "category": "cheque_extraction" if structured else "handwritten_text",
                "description": str(item.get("query", "Document manuscrit")), "source": "hf_cheques",
                "source_revision": cfg["revision"], "split": cfg["split"], "task_type": task_type,
                "prompt": str(item.get("query", "")), "label_provenance": "dataset_annotation",
                "is_scorable": bool(truth.strip()), "license": "apache-2.0",
            })
        return rows

    def load_kaggle_iam(cfg):
        import kagglehub
        if KAGGLE_INTERACTIVE_LOGIN:
            kagglehub.login()
        if cfg.get("download_full_dataset"):
            raise ValueError("Téléchargement complet désactivé ici: retirez explicitement cette protection (archive ~4,6 Go).")
        rows = []
        for index, remote_path in enumerate(cfg["sample_files"][: int(cfg["max_samples"])]):
            downloaded = Path(kagglehub.dataset_download(cfg["handle"], path=remote_path))
            source = downloaded if downloaded.is_file() else next(downloaded.rglob(Path(remote_path).name))
            fingerprint = hashlib.sha1(remote_path.encode("utf-8")).hexdigest()[:12]
            path = save_any_image(str(source), DATASET_DIR / "kaggle_iam" / f"{fingerprint}.png")
            rows.append({
                "id": f"kaggle-iam-{fingerprint}", "image_path": str(path), "ground_truth": "",
                "category": "handwritten_form", "description": remote_path, "source": "kaggle_iam",
                "source_revision": cfg["source_revision"], "split": "sample_files",
                "task_type": "qualitative_ocr", "prompt": "", "label_provenance": "none",
                "is_scorable": False, "license": "unknown_verify_on_kaggle",
            })
        return rows

    def github_headers():
        headers = {"User-Agent": "ocr-model-selection-colab", "Accept": "application/vnd.github+json"}
        if GITHUB_TOKEN:
            headers["Authorization"] = f"Bearer {GITHUB_TOKEN}"
        return headers

    def github_api_json(repo, revision, path):
        encoded = quote(path, safe="/")
        url = f"https://api.github.com/repos/{repo}/contents/{encoded}?ref={quote(revision)}"
        with urlopen(Request(url, headers=github_headers()), timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))

    def fetch_optional_text(url):
        try:
            with urlopen(Request(url, headers=github_headers()), timeout=60) as response:
                return response.read().decode("utf-8", errors="replace")
        except HTTPError as exc:
            if exc.code == 404:
                return None
            raise

    def collect_github_images(repo, revision, maximum):
        roots = sorted(
            (item for item in github_api_json(repo, revision, "Dataset") if item.get("type") == "dir"),
            key=lambda item: item["path"],
        )
        quota = max(1, math.ceil(maximum / max(1, len(roots))))
        collected = []
        for root in roots:
            stack, per_root = [root["path"]], []
            while stack and len(per_root) < quota:
                items = sorted(github_api_json(repo, revision, stack.pop(0)), key=lambda item: item["path"])
                stack.extend(item["path"] for item in items if item.get("type") == "dir")
                per_root.extend(
                    item for item in items
                    if item.get("type") == "file" and Path(item["path"]).suffix.lower() in ALLOWED_EXTENSIONS
                )
            collected.extend(per_root[:quota])
        return collected[:maximum]

    def load_github_forms(cfg):
        repo, revision = cfg["repo"], cfg["revision"]
        images = collect_github_images(repo, revision, int(cfg["max_samples"]))
        rows = []
        for index, item in enumerate(images):
            raw = urlopen(Request(item["download_url"], headers=github_headers()), timeout=60).read()
            path = save_any_image(raw, DATASET_DIR / "github_forms" / f"{index:05d}.png")
            relative = item["path"].split("Dataset/", 1)[-1]
            stem = str(PurePosixPath(relative).with_suffix(""))
            label_folder = "OCR_text" if cfg["label_mode"] == "ocr_text" else "JSON_file"
            label_extension = ".txt" if cfg["label_mode"] == "ocr_text" else ".json"
            label_path = f"{label_folder}/{stem}{label_extension}"
            raw_base = f"https://raw.githubusercontent.com/{repo}/{revision}/"
            label = fetch_optional_text(raw_base + quote(label_path, safe="/")) or ""
            category = relative.split("/", 1)[0].strip().lower().replace(" ", "_")
            rows.append({
                "id": f"github-forms-{index}", "image_path": str(path), "ground_truth": label,
                "category": category, "description": relative, "source": "github_forms",
                "source_revision": revision, "split": "repository_sample",
                "task_type": "transcription" if cfg["label_mode"] == "ocr_text" else "key_value_extraction",
                "prompt": "", "label_provenance": "derived_ocr",
                "is_scorable": bool(label.strip()) and ALLOW_DERIVED_LABELS_IN_RANKING,
                "license": "research_only_verify_original",
            })
        return rows

    SOURCE_LOADERS = {
        "hf_multifin": load_hf_multifin,
        "hf_cheques": load_hf_cheques,
        "kaggle_iam": load_kaggle_iam,
        "github_forms": load_github_forms,
    }
    print("Chargeurs prêts: Hugging Face, Kaggle fichier-par-fichier, GitHub et ZIP personnel.")
"""))

cells.append(code(r"""
    # ÉTAPE 4C — Télécharger quelques exemples, auditer et dédupliquer
    records = []
    source_errors = []
    for source_name, config in DATASET_SOURCES.items():
        if not config.get("enabled"):
            continue
        try:
            print(f"Téléchargement {source_name} — maximum {config['max_samples']}...")
            records.extend(SOURCE_LOADERS[source_name](config))
        except Exception as exc:
            source_errors.append({"source": source_name, "error": str(exc)})
            print(f"Source indisponible: {source_name} — {exc}")

    if not records and USE_SYNTHETIC_FALLBACK:
        print("Aucune source distante disponible: activation du repli synthétique.")
        records.extend(build_sample_dataset())
    if not records:
        raise RuntimeError("Aucun document chargé. Vérifiez les secrets et la connexion réseau.")

    dataset_manifest_df = pd.DataFrame(records)
    required_columns = {
        "id", "image_path", "ground_truth", "category", "source", "source_revision",
        "task_type", "prompt", "label_provenance", "is_scorable",
    }
    missing = sorted(required_columns - set(dataset_manifest_df.columns))
    if missing:
        raise ValueError(f"Schéma dataset incomplet: {missing}")

    def inspect_image(path):
        path = Path(path)
        with Image.open(path) as image:
            width, height = image.size
        return pd.Series({
            "width": width,
            "height": height,
            "image_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        })

    dataset_manifest_df = pd.concat(
        [dataset_manifest_df.reset_index(drop=True), dataset_manifest_df["image_path"].apply(inspect_image)],
        axis=1,
    )
    dataset_manifest_df["ground_truth"] = dataset_manifest_df["ground_truth"].fillna("").astype(str)
    dataset_manifest_df["is_scorable"] = (
        dataset_manifest_df["is_scorable"].astype(bool)
        & dataset_manifest_df["ground_truth"].str.strip().ne("")
    )
    dataset_manifest_df["ground_truth_chars"] = dataset_manifest_df["ground_truth"].str.len()

    provenance_priority = {
        "dataset_annotation": 0, "user_provided": 0, "synthetic_exact": 1,
        "none": 2, "derived_ocr": 3,
    }
    dataset_manifest_df["dedupe_priority"] = dataset_manifest_df.apply(
        lambda row: (0 if row.is_scorable else 1, provenance_priority.get(row.label_provenance, 4)),
        axis=1,
    )
    dataset_manifest_df = dataset_manifest_df.sort_values(
        ["dedupe_priority", "source", "id"], kind="stable"
    ).reset_index(drop=True)
    dataset_manifest_df["duplicate_of"] = dataset_manifest_df.groupby("image_sha256")["id"].transform("first")
    dataset_manifest_df["is_duplicate"] = dataset_manifest_df["id"] != dataset_manifest_df["duplicate_of"]

    dataset_df = dataset_manifest_df.loc[
        ~dataset_manifest_df["is_duplicate"] if EXCLUDE_DUPLICATE_IMAGES else pd.Series(True, index=dataset_manifest_df.index)
    ].reset_index(drop=True)
    dataset_manifest_df.to_csv(WORK_DIR / "dataset_manifest.csv", index=False)

    display(dataset_df[["source", "category", "task_type", "is_scorable"]].value_counts().rename("documents").to_frame())
    display(dataset_df[["id", "source", "category", "is_scorable", "width", "height"]].head(20))
    print(
        f"Actifs: {len(dataset_df)} | scorables: {int(dataset_df.is_scorable.sum())} | "
        f"qualitatifs: {int((~dataset_df.is_scorable).sum())} | "
        f"doublons exclus: {int(dataset_manifest_df.is_duplicate.sum())}"
    )
    if int(dataset_df.is_scorable.sum()) == 0:
        print("AVERTISSEMENT: aucun label scorable; les sorties seront qualitatives, sans recommandation.")
    elif int(dataset_df.is_scorable.sum()) < 30:
        print("AVERTISSEMENT: moins de 30 documents scorables; la recommandation restera exploratoire.")
    if source_errors:
        display(pd.DataFrame(source_errors))
"""))

cells.append(code(r"""
    # ÉTAPE 4D — Aperçu visuel des données réellement chargées
    import matplotlib.pyplot as plt

    preview_df = dataset_df.head(min(8, len(dataset_df)))
    columns = min(4, max(1, len(preview_df)))
    rows_count = math.ceil(len(preview_df) / columns)
    preview_fig, axes = plt.subplots(rows_count, columns, figsize=(4 * columns, 3.2 * rows_count))
    axes = np.array(axes, dtype=object).reshape(-1)
    for axis in axes:
        axis.axis("off")
    for axis, (_, row) in zip(axes, preview_df.iterrows()):
        axis.imshow(Image.open(row.image_path).convert("RGB"))
        axis.set_title(f"{row.source} · {row.category}\n{'scorable' if row.is_scorable else 'qualitatif'}", fontsize=9)
        axis.axis("off")
    preview_fig.suptitle("Échantillons qui entreront dans le benchmark", fontsize=14, fontweight="bold")
    preview_fig.tight_layout()
    plt.show()
"""))

cells.append(markdown(r"""
    ## 5. Choisir et télécharger les modèles

    Cette cellule est la liste explicite des poids à télécharger. **Par défaut, tous les modèles du catalogue sont cochés** : le téléchargement peut représenter plusieurs dizaines de Go et certains modèles nécessitent un profil/runtine GPU séparé. Les poids sont téléchargés une fois, puis Gradio ne charge en mémoire que les modèles sélectionnés pour le run, un par un.

    - **EasyOCR** : baseline traditionnelle, CPU/GPU, sans notion de token génératif ;
    - **PP‑OCRv6 medium** : détection + reconnaissance OCR classique, rapide, CPU/GPU ;
    - **GLM‑OCR 1B** : modèle vision‑langage génératif dédié OCR ;
    - **Granite Docling 258M** : conversion structurée de document, exécutée en `float32` sur T4 pour éviter le problème BF16 documenté.

    Le catalogue contient aussi PaddleOCR‑VL 1.6, Qwen OCR 0.8B GGUF, MiniCPM‑V 4.6, Chandra OCR 2, LightOnOCR 2, dots.ocr, Unlimited‑OCR et LocateAnything. **LocateAnything est un modèle de localisation, pas de transcription** : il ne peut pas être classé avec CER/WER. Les modèles incompatibles avec le runtime courant restent sélectionnables pour voir la raison précise de leur exclusion, sans casser le notebook.

    Sources officielles utiles : [PP‑OCRv6](https://huggingface.co/blog/PaddlePaddle/pp-ocrv6), [GLM‑OCR](https://huggingface.co/zai-org/GLM-OCR), [Granite Docling 258M](https://huggingface.co/ibm-granite/granite-docling-258M), [PaddleOCR‑VL 1.6](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6).
"""))

cells.append(code(r"""
    # ÉTAPE 5A — Catalogue matériel et dépendances

    MODEL_CATALOG = {
        "easyocr": {
            "display_name": "EasyOCR (baseline)", "adapter_kind": "easyocr", "model_id": "easyocr-fr-en",
            "profile": "CORE", "supports_cpu": True, "t4_supported": True, "min_vram_gb": 0,
            "weights_gb": 0.2, "license": "Apache-2.0", "ranking_task": "transcription",
            "prompt": "", "max_new_tokens": None, "download_strategy": "runtime",
            "note": "OCR classique; tokens/s non applicable.",
        },
        "pp_ocrv6": {
            "display_name": "PP-OCRv6 medium", "adapter_kind": "pp_ocrv6", "model_id": "PP-OCRv6_medium_det+rec",
            "profile": "CORE", "supports_cpu": True, "t4_supported": True, "min_vram_gb": 0,
            "weights_gb": 0.17, "license": "Apache-2.0", "ranking_task": "transcription",
            "prompt": "", "max_new_tokens": None, "download_strategy": "pp_ocrv6",
            "note": "Paire officielle détection + reconnaissance, moteur Transformers.",
        },
        "paddleocr_vl_1_6": {
            "display_name": "PaddleOCR-VL 1.6", "adapter_kind": "paddleocr_vl", "model_id": "PaddlePaddle/PaddleOCR-VL-1.6",
            "profile": "CORE", "supports_cpu": False, "t4_supported": True, "min_vram_gb": 8,
            "weights_gb": 1.93, "license": "Apache-2.0", "ranking_task": "transcription",
            "prompt": "OCR:", "max_new_tokens": 512, "download_strategy": "snapshot",
            "note": "Chemin Transformers élémentaire; la pipeline complète Paddle ajoute layout/crops.",
        },
        "glm_ocr": {
            "display_name": "GLM-OCR 1B", "adapter_kind": "glm_ocr", "model_id": "zai-org/GLM-OCR",
            "profile": "CORE", "supports_cpu": False, "t4_supported": True, "min_vram_gb": 8,
            "weights_gb": 2.65, "license": "MIT", "ranking_task": "transcription",
            "prompt": "Text Recognition:", "max_new_tokens": 8192, "download_strategy": "snapshot",
            "note": "VLM OCR génératif; FP16 sur T4, chargé seul.",
        },
        "granite_docling_258m": {
            "display_name": "Granite Docling 258M", "adapter_kind": "granite_docling", "model_id": "ibm-granite/granite-docling-258M",
            "profile": "CORE", "supports_cpu": True, "t4_supported": True, "min_vram_gb": 0,
            "weights_gb": 0.52, "license": "Apache-2.0", "ranking_task": "transcription",
            "prompt": "Convert this page to docling.", "max_new_tokens": 8192, "download_strategy": "snapshot",
            "note": "Produit des DocTags puis du Markdown; float32+SDPA sur T4.",
        },
        "qwen_ocr_0_8b": {
            "display_name": "Qwen3.5 OCR 0.8B GGUF (communauté)", "adapter_kind": "qwen_gguf", "model_id": "loay/English-Document-OCR-Qwen3.5-0.8B",
            "profile": "LLAMA_CPP", "supports_cpu": True, "t4_supported": True, "min_vram_gb": 0,
            "weights_gb": 0.9, "license": "CC-BY-NC-SA (vérifier)", "ranking_task": "transcription",
            "prompt": "Extract all visible text from this document image and return only the transcription in reading order using a markdown-first format. Use HTML only for tables. Use LaTeX only for formulas.",
            "max_new_tokens": 2048, "download_strategy": "gguf",
            "note": "Checkpoint GGUF communautaire; nécessite llama-cli avec support multimodal.",
        },
        "minicpm_v_4_6": {
            "display_name": "MiniCPM-V 4.6", "adapter_kind": "minicpm", "model_id": "openbmb/MiniCPM-V-4.6",
            "profile": "CORE", "supports_cpu": False, "t4_supported": True, "min_vram_gb": 10,
            "weights_gb": 3.0, "license": "Apache-2.0", "ranking_task": "transcription",
            "prompt": "Transcribe every visible character. Preserve reading order and line breaks.", "max_new_tokens": 2048,
            "download_strategy": "snapshot", "note": "FP16 sur T4; slicing d'image 4x.",
        },
        "chandra_ocr_2": {
            "display_name": "Chandra OCR 2", "adapter_kind": "chandra", "model_id": "datalab-to/chandra-ocr-2",
            "profile": "CORE_CHANDRA", "supports_cpu": False, "t4_supported": False, "min_vram_gb": 24,
            "weights_gb": 10.0, "license": "OpenRAIL modifiée", "ranking_task": "transcription",
            "prompt": "ocr_layout", "max_new_tokens": 4096, "download_strategy": "snapshot",
            "note": "L4/A100 recommandé; package chandra-ocr[hf] requis.",
        },
        "lightonocr_2_1b": {
            "display_name": "LightOnOCR-2 1B", "adapter_kind": "lighton", "model_id": "lightonai/LightOnOCR-2-1B",
            "profile": "CORE", "supports_cpu": False, "t4_supported": True, "min_vram_gb": 8,
            "weights_gb": 2.2, "license": "Apache-2.0", "ranking_task": "transcription",
            "prompt": "", "max_new_tokens": 2048, "download_strategy": "snapshot",
            "note": "Modèle OCR génératif compact, FP16 sur T4.",
        },
        "dots_ocr": {
            "display_name": "dots.ocr", "adapter_kind": "dots_ocr", "model_id": "rednote-hilab/dots.ocr",
            "profile": "DOTS_456", "supports_cpu": False, "t4_supported": False, "min_vram_gb": 24,
            "weights_gb": 6.0, "license": "vérifier la carte", "ranking_task": "transcription",
            "prompt": "prompt_ocr", "max_new_tokens": 24000, "download_strategy": "dots_local",
            "note": "Transformers 4.56.1 + FlashAttention; runtime isolé nécessaire.",
        },
        "unlimited_ocr": {
            "display_name": "Unlimited-OCR 3B", "adapter_kind": "unlimited_ocr", "model_id": "baidu/Unlimited-OCR",
            "profile": "LEGACY_457", "supports_cpu": False, "t4_supported": False, "min_vram_gb": 24,
            "weights_gb": 6.8, "license": "MIT", "ranking_task": "transcription",
            "prompt": "<image>document parsing.", "max_new_tokens": 32768, "download_strategy": "snapshot",
            "note": "Transformers 4.57.1/BF16; L4 ou A100 et runtime isolé recommandés.",
        },
        "locateanything_3b": {
            "display_name": "LocateAnything", "adapter_kind": "locate_anything", "model_id": "nvidia/LocateAnything-3B",
            "profile": "LEGACY_457", "supports_cpu": False, "t4_supported": False, "min_vram_gb": 40,
            "weights_gb": 7.8, "license": "NVIDIA non-commercial", "ranking_task": "localization",
            "prompt": "Detect all the text in box format.", "max_new_tokens": 8192, "download_strategy": "snapshot",
            "note": "Détection qualitative de boîtes; A100 40 Go/H100 recommandé, IoU/F1 exige des labels de boîtes.",
        },
    }

    # Modifiez uniquement cette liste si vous ne voulez pas télécharger tout le catalogue.
    # Exemple rapide : MODELS_TO_DOWNLOAD = ["easyocr", "pp_ocrv6", "glm_ocr"]
    ALL_MODEL_NAMES = list(MODEL_CATALOG)
    DEFAULT_SELECTED_MODELS = [
        "easyocr", "pp_ocrv6", "paddleocr_vl_1_6", "glm_ocr", "granite_docling_258m",
        "qwen_ocr_0_8b", "minicpm_v_4_6", "chandra_ocr_2", "lightonocr_2_1b",
        "dots_ocr", "unlimited_ocr", "locateanything_3b",
    ]
    MODELS_TO_DOWNLOAD = ALL_MODEL_NAMES.copy()
    SELECTED_MODELS = MODELS_TO_DOWNLOAD.copy()
    DOWNLOAD_SELECTED_WEIGHTS = True
    print("Modèles à télécharger (modifiez MODELS_TO_DOWNLOAD ici) :")
    print("\n".join(f"  - {name}" for name in MODELS_TO_DOWNLOAD))
    print(f"Total sélectionné : {len(MODELS_TO_DOWNLOAD)} / {len(ALL_MODEL_NAMES)}")

    def model_readiness(model_name):
        cfg = MODEL_CATALOG[model_name]
        compatible_profile = cfg["profile"] == RUNTIME_PROFILE or (
            RUNTIME_PROFILE == "CORE_CHANDRA" and cfg["profile"] == "CORE"
        )
        if not compatible_profile:
            if cfg["adapter_kind"] == "qwen_gguf" and shutil.which("llama-cli"):
                return True, "llama-cli détecté"
            return False, f"profil requis: {cfg['profile']} (profil actif: {RUNTIME_PROFILE})"
        if DEVICE == "cpu" and not cfg["supports_cpu"]:
            return False, "GPU requis"
        if IS_T4 and not cfg["t4_supported"]:
            return False, "T4 non supporté; L4/A100 recommandé"
        if (
            DEVICE == "cuda"
            and cfg["profile"] in {"CORE_CHANDRA", "DOTS_456", "LEGACY_457"}
            and GPU_CAPABILITY[0] < 8
        ):
            return False, f"GPU Ampere ou plus récent requis (capability actuelle: {GPU_CAPABILITY})"
        if DEVICE == "cuda" and cfg["min_vram_gb"] and GPU_VRAM_GB < 0.90 * cfg["min_vram_gb"]:
            return False, f"VRAM insuffisante: {GPU_VRAM_GB:.1f} Go < {cfg['min_vram_gb']} Go"
        return True, "compatible avec ce runtime"

    catalog_rows = []
    for model_name, cfg in MODEL_CATALOG.items():
        ready, reason = model_readiness(model_name)
        catalog_rows.append({
            "clé": model_name, "modèle": cfg["display_name"], "prêt": ready,
            "raison": reason, "poids estimés (Go)": cfg["weights_gb"],
            "CPU": cfg["supports_cpu"], "T4": cfg["t4_supported"],
            "classement": cfg["ranking_task"], "licence": cfg["license"],
        })
    model_catalog_df = pd.DataFrame(catalog_rows)
    display(model_catalog_df)
"""))

cells.append(code(r"""
    # ÉTAPE 5B — Télécharger la liste MODELS_TO_DOWNLOAD
    from huggingface_hub import hf_hub_download, model_info, snapshot_download

    os.environ.setdefault("HF_HOME", str(MODEL_DIR))
    MODEL_RESOLVED_REVISIONS = {}
    MODEL_LOCAL_PATHS = {}

    def download_model_weights(model_name):
        cfg = MODEL_CATALOG[model_name]
        strategy = cfg["download_strategy"]
        if strategy == "runtime":
            MODEL_RESOLVED_REVISIONS.setdefault(model_name, "managed-by-runtime")
            return {"model": model_name, "status": "deferred", "detail": "téléchargé par l'adaptateur au premier lancement"}
        if strategy == "pp_ocrv6":
            resolved = {}
            for repo_id in (
                "PaddlePaddle/PP-OCRv6_medium_det_safetensors",
                "PaddlePaddle/PP-OCRv6_medium_rec_safetensors",
            ):
                snapshot_path = snapshot_download(repo_id, cache_dir=MODEL_DIR, token=HF_TOKEN or None)
                resolved[repo_id] = Path(snapshot_path).name
                MODEL_LOCAL_PATHS.setdefault(model_name, {})[repo_id] = str(snapshot_path)
            MODEL_RESOLVED_REVISIONS[model_name] = resolved
            return {"model": model_name, "status": "downloaded", "detail": "détection + reconnaissance"}
        if strategy == "gguf":
            resolved_revision = None
            for filename in (
                "english-document-ocr-qwen3.5-0.8b-q4_k_m.gguf",
                "mmproj-english-document-ocr-qwen3.5-0.8b-f16.gguf",
            ):
                downloaded_path = hf_hub_download(
                    cfg["model_id"], filename, cache_dir=MODEL_DIR, token=HF_TOKEN or None
                )
                resolved_revision = Path(downloaded_path).parent.name
            MODEL_RESOLVED_REVISIONS[model_name] = resolved_revision
            return {"model": model_name, "status": "downloaded", "detail": "GGUF + projection vision"}
        if strategy == "dots_local":
            # Le point dans `dots.ocr` casse le nom du module Python dynamique lorsqu'il est chargé
            # directement depuis le cache Hub. Le dossier local sans point suit le correctif officiel.
            resolved_revision = model_info(cfg["model_id"], token=HF_TOKEN or None).sha
            local_dir = MODEL_DIR / "DotsOCR"
            snapshot_download(
                cfg["model_id"], revision=resolved_revision, local_dir=local_dir,
                token=HF_TOKEN or None,
            )
            MODEL_RESOLVED_REVISIONS[model_name] = resolved_revision
            MODEL_LOCAL_PATHS[model_name] = str(local_dir)
            return {"model": model_name, "status": "downloaded", "detail": str(local_dir)}
        snapshot_path = snapshot_download(cfg["model_id"], cache_dir=MODEL_DIR, token=HF_TOKEN or None)
        MODEL_LOCAL_PATHS[model_name] = str(snapshot_path)
        MODEL_RESOLVED_REVISIONS[model_name] = Path(snapshot_path).name
        return {"model": model_name, "status": "downloaded", "detail": cfg["model_id"]}

    download_report = []
    if DOWNLOAD_SELECTED_WEIGHTS:
        for selected_name in SELECTED_MODELS:
            if selected_name not in MODEL_CATALOG:
                download_report.append({"model": selected_name, "status": "error", "detail": "clé inconnue"})
                continue
            try:
                print("Préparation:", selected_name)
                download_report.append(download_model_weights(selected_name))
            except Exception as exc:
                download_report.append({"model": selected_name, "status": "error", "detail": str(exc)})
    display(pd.DataFrame(download_report))
    print("Téléchargement terminé. Gradio vérifie la compatibilité du profil puis charge chaque modèle sélectionné un par un.")
"""))

cells.append(markdown(r"""
    ## 6. Comprendre les métriques avant de comparer

    - **CER — Character Error Rate** : nombre minimal de caractères à insérer, supprimer ou remplacer, divisé par le nombre de caractères attendus. `0 %` est parfait. La casse et les accents comptent ; il peut dépasser `100 %` si le modèle ajoute beaucoup de texte.
    - **WER — Word Error Rate** : même idée au niveau des mots, avec casse et accents conservés. Plus bas est meilleur. Il pénalise davantage une petite faute qui change un mot entier.
    - **Exact match normalisé** : pourcentage de documents dont le texte devient exactement identique après minuscules, espaces et accents normalisés. Plus haut est meilleur.
    - **Latence médiane** : temps typique pour une image. La moitié des images sont plus rapides, l'autre moitié plus lentes.
    - **P95** : 95 % des images finissent sous cette durée. C'est plus utile que la moyenne pour prévoir les cas lents en production.
    - **Documents/minute** : débit séquentiel estimé. Plus haut est meilleur.
    - **Tokens de sortie / tokens par seconde** : uniquement pour les modèles génératifs. Un token est un morceau de texte propre au tokenizer du modèle ; ce nombre n'est donc pas directement comparable entre familles. Pour EasyOCR et PP‑OCRv6, la valeur est `N/A`.
    - **Caractères/seconde** : débit du texte final, un peu plus comparable entre familles, mais dépend aussi de la longueur des pages.
    - **Boîtes/seconde** : uniquement pour LocateAnything ; nombre de zones de texte localisées par seconde. Sans boîtes attendues dans le dataset, cette mesure décrit le débit mais pas la précision IoU/F1.
    - **Pic VRAM PyTorch** : maximum alloué par PyTorch pendant l'image ; il n'inclut pas toute la mémoire du pilote. **RAM RSS** est une photographie de la mémoire du processus après l'inférence, pas un pic système complet.
    - **Taux de réussite** : part des tentatives ayant produit une sortie exploitable. Les échecs et incompatibilités sont exclus de la qualité et de la latence, mais restent visibles.

    Le score de décision privilégie la qualité (`70 %`), puis la fiabilité (`20 %`) et la vitesse P95 (`10 %`). Il est calculé uniquement sur les tâches de **transcription**. Les lignes d'extraction JSON utilisent séparément le **Field‑F1** afin de ne pas mélanger deux problèmes incompatibles. Il ne remplace pas vos contraintes métier : utilisez aussi les résultats par catégorie.
"""))

cells.append(code(r"""
    # ÉTAPE 6 — Fonctions de métriques, testées indépendamment des modèles
    def normalize_text(value):
        value = unicodedata.normalize("NFKC", str(value or "")).lower()
        value = "".join(char for char in unicodedata.normalize("NFD", value) if unicodedata.category(char) != "Mn")
        value = re.sub(r"\s+", " ", value).strip()
        return value

    def metric_text(value):
        value = unicodedata.normalize("NFKC", str(value or ""))
        return value.replace("\r\n", "\n").replace("\r", "\n").strip()

    def sequence_edit_distance(reference_items, hypothesis_items):
        previous = list(range(len(hypothesis_items) + 1))
        for ref_index, ref_item in enumerate(reference_items, start=1):
            current = [ref_index]
            for hyp_index, hyp_item in enumerate(hypothesis_items, start=1):
                current.append(min(
                    current[-1] + 1,
                    previous[hyp_index] + 1,
                    previous[hyp_index - 1] + (ref_item != hyp_item),
                ))
            previous = current
        return previous[-1]

    def metric_details(reference, hypothesis):
        ref_text, hyp_text = metric_text(reference), metric_text(hypothesis)
        ref_words, hyp_words = ref_text.split(), hyp_text.split()
        return {
            "char_edits": sequence_edit_distance(list(ref_text), list(hyp_text)),
            "reference_chars": len(ref_text),
            "word_edits": sequence_edit_distance(ref_words, hyp_words),
            "reference_words": len(ref_words),
            "normalized_exact_match": float(normalize_text(reference) == normalize_text(hypothesis)),
        }

    def cer(reference, hypothesis):
        details = metric_details(reference, hypothesis)
        return details["char_edits"] / details["reference_chars"] if details["reference_chars"] else np.nan

    def wer(reference, hypothesis):
        details = metric_details(reference, hypothesis)
        return details["word_edits"] / details["reference_words"] if details["reference_words"] else np.nan

    def extract_json_object(text):
        try:
            return json.loads(str(text))
        except Exception:
            match = re.search(r"\{.*\}", str(text), flags=re.DOTALL)
            if not match:
                return None
            try:
                return json.loads(match.group(0))
            except Exception:
                return None

    def structured_field_scores(reference, hypothesis):
        reference_obj, hypothesis_obj = extract_json_object(reference), extract_json_object(hypothesis)
        if not isinstance(reference_obj, dict):
            return {"field_precision": np.nan, "field_recall": np.nan, "field_f1": np.nan}
        if not isinstance(hypothesis_obj, dict):
            return {"field_precision": 0.0, "field_recall": 0.0, "field_f1": 0.0}
        def flatten_fields(value, prefix=""):
            fields = set()
            if isinstance(value, dict):
                for key, child in value.items():
                    child_prefix = f"{prefix}.{normalize_text(key)}" if prefix else normalize_text(key)
                    fields.update(flatten_fields(child, child_prefix))
            elif isinstance(value, list):
                for index, child in enumerate(value):
                    fields.update(flatten_fields(child, f"{prefix}[{index}]"))
            else:
                fields.add((prefix, normalize_text(value)))
            return fields

        expected = flatten_fields(reference_obj)
        predicted = flatten_fields(hypothesis_obj)
        true_positive = len(expected & predicted)
        precision = true_positive / len(predicted) if predicted else 0.0
        recall = true_positive / len(expected) if expected else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return {"field_precision": precision, "field_recall": recall, "field_f1": f1}

    def clean_model_text(value):
        text = str(value or "").strip()
        text = re.sub(r"^```(?:markdown|md|text|html|json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
        return text.strip()

    assert wer("alpha beta", "alpha gamma") == 0.5
    assert cer("abc", "abc") == 0.0
    print("Métriques prêtes. Les assertions CER/WER passent.")
"""))

cells.append(markdown(r"""
    ## 7. Adaptateurs : une interface commune, une implémentation propre à chaque moteur

    Tous les adaptateurs renvoient le même schéma (`text`, sortie brute, temps, statut, erreur, tokens). Cela permet au runner et à Gradio d'ignorer les détails internes. En revanche, les appels d'inférence ne sont pas généralisés artificiellement : PP‑OCRv6 utilise `PaddleOCR.predict`, GLM utilise son chat template, et Granite convertit ses DocTags en Markdown.

    Le paramètre de temps est volontairement un **budget souple**, sauf pour le processus `llama-cli` qui peut être interrompu réellement. GLM‑OCR, PaddleOCR‑VL, Granite Docling, MiniCPM, LightOnOCR et dots.ocr reçoivent `max_time` et s'arrêtent dès que leur boucle de génération le permet. EasyOCR, PP‑OCRv6, Chandra, Unlimited‑OCR et LocateAnything ne fournissent pas tous une interruption sûre au milieu d'une image : leur sortie tardive est donc conservée avec le statut `slow_success`. Le notebook ne prétend jamais qu'un thread GPU a été tué alors que ce n'est pas le cas.
"""))

cells.append(code(r"""
    # ÉTAPE 7 — Adaptateurs de modèles
    import tempfile
    from dataclasses import dataclass

    OCR_PROMPT = "Transcribe all visible text faithfully. Preserve reading order and line breaks. Return only the transcription."
    MODEL_RESOLVED_REVISIONS = globals().get("MODEL_RESOLVED_REVISIONS", {})
    MODEL_LOCAL_PATHS = globals().get("MODEL_LOCAL_PATHS", {})

    def hf_revision_for(model_name):
        value = MODEL_RESOLVED_REVISIONS.get(model_name)
        return value if isinstance(value, str) and value != "managed-by-runtime" else None

    def model_source_for(model_name):
        # Utilise exactement le snapshot téléchargé par l'étape 5 quand il existe.
        return MODEL_LOCAL_PATHS.get(model_name) or MODEL_CATALOG[model_name]["model_id"]

    def prediction_payload(
        text="", latency=np.nan, status="failed", raw_response=None, error=None,
        output_tokens=np.nan, output_tokens_kind="not_available", inference_device=None,
        scoring_text=None,
    ):
        return {
            "text": str(text or ""), "latency": latency, "status": status,
            "raw_response": raw_response, "error": error,
            "output_tokens": output_tokens, "output_tokens_kind": output_tokens_kind,
            "inference_device": inference_device,
            "scoring_text": scoring_text,
        }

    def markdown_to_plain_text(value):
        from bs4 import BeautifulSoup
        from markdown import markdown as render_markdown
        rendered = render_markdown(str(value or ""), extensions=["tables", "fenced_code"])
        return BeautifulSoup(rendered, "html.parser").get_text("\n")

    def completed_reasoning_blocks(value):
        return re.findall(r"<think>(.*?)</think>", str(value or ""), flags=re.DOTALL | re.IGNORECASE)

    def remove_completed_reasoning(value):
        # Les blocs complets sont retirés du score mais restent intacts dans raw_response.
        # Un bloc partiel sans </think> est conservé pour ne jamais jeter une sortie de timeout.
        return re.sub(r"<think>.*?</think>", "", str(value or ""), flags=re.DOTALL | re.IGNORECASE)

    def preferred_dtype(force_float32=False):
        if torch is None or DEVICE == "cpu" or force_float32:
            return torch.float32 if torch else None
        if GPU_CAPABILITY[0] >= 8 and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16

    def model_device(model):
        try:
            return next(model.parameters()).device
        except Exception:
            return torch.device(DEVICE) if torch else DEVICE

    def move_inputs(inputs, device, float_dtype=None):
        moved = {}
        for key, value in inputs.items():
            if hasattr(value, "to"):
                if float_dtype is not None and getattr(value, "is_floating_point", lambda: False)():
                    moved[key] = value.to(device=device, dtype=float_dtype)
                else:
                    moved[key] = value.to(device)
            else:
                moved[key] = value
        return moved

    class BaseAdapter:
        def __init__(self, name, config):
            self.name = name
            self.config = config

        def predict(self, image_path, prompt=None, max_seconds=120):
            raise NotImplementedError

        def close(self):
            for attribute in ("model", "processor", "reader", "ocr"):
                if hasattr(self, attribute):
                    try:
                        delattr(self, attribute)
                    except Exception:
                        pass

    class EasyOCRAdapter(BaseAdapter):
        def __init__(self, name, config):
            super().__init__(name, config)
            import easyocr
            self.reader = easyocr.Reader(["fr", "en"], gpu=DEVICE == "cuda", verbose=False)

        def predict(self, image_path, prompt=None, max_seconds=120):
            started = time.perf_counter()
            try:
                lines = self.reader.readtext(str(image_path), detail=0, paragraph=False)
                text = "\n".join(str(line) for line in lines)
                return prediction_payload(
                    text=text, latency=time.perf_counter() - started, status="success",
                    raw_response=json.dumps(lines, ensure_ascii=False),
                    output_tokens=np.nan, output_tokens_kind="not_applicable",
                )
            except Exception as exc:
                return prediction_payload(latency=time.perf_counter() - started, error=str(exc))

    class PPOCRv6Adapter(BaseAdapter):
        def __init__(self, name, config):
            super().__init__(name, config)
            from paddleocr import PaddleOCR
            self.engine_device = "gpu:0" if DEVICE == "cuda" else "cpu"
            local_paths = MODEL_LOCAL_PATHS.get(name, {})
            local_model_options = {}
            detection_path = local_paths.get("PaddlePaddle/PP-OCRv6_medium_det_safetensors")
            recognition_path = local_paths.get("PaddlePaddle/PP-OCRv6_medium_rec_safetensors")
            if detection_path and recognition_path:
                local_model_options = {
                    "text_detection_model_dir": detection_path,
                    "text_recognition_model_dir": recognition_path,
                }
            self.ocr = PaddleOCR(
                text_detection_model_name="PP-OCRv6_medium_det",
                text_recognition_model_name="PP-OCRv6_medium_rec",
                engine="transformers",
                device=self.engine_device,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                **local_model_options,
            )

        def predict(self, image_path, prompt=None, max_seconds=120):
            started = time.perf_counter()
            try:
                outputs = list(self.ocr.predict(str(image_path)))
                texts, raw_items = [], []
                for output in outputs:
                    payload = getattr(output, "json", output)
                    payload = payload() if callable(payload) else payload
                    if not isinstance(payload, dict):
                        payload = {"value": str(payload)}
                    raw_items.append(payload)
                    result = payload.get("res", payload)
                    texts.extend(str(item) for item in result.get("rec_texts", []) if str(item).strip())
                return prediction_payload(
                    text="\n".join(texts), latency=time.perf_counter() - started, status="success",
                    raw_response=json.dumps(raw_items, ensure_ascii=False, default=str),
                    output_tokens=np.nan, output_tokens_kind="not_applicable",
                    inference_device=self.engine_device,
                )
            except Exception as exc:
                return prediction_payload(latency=time.perf_counter() - started, error=str(exc))

    class GLMOCRAdapter(BaseAdapter):
        def __init__(self, name, config):
            super().__init__(name, config)
            from transformers import AutoModelForImageTextToText, AutoProcessor
            # GLM-OCR publie des poids BF16 et recommande torch_dtype="auto".
            # Cela évite un failed_load sur les runtimes où le type du checkpoint
            # ne peut pas être converti automatiquement au moment du chargement.
            dtype = "auto"
            model_source = MODEL_LOCAL_PATHS.get(name) or config["model_id"]
            local_only = Path(model_source).is_dir()
            self.processor = AutoProcessor.from_pretrained(
                model_source, revision=None if local_only else hf_revision_for(name),
                token=None if local_only else (HF_TOKEN or None), cache_dir=MODEL_DIR,
                local_files_only=local_only,
            )
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_source, torch_dtype=dtype, device_map="auto",
                revision=None if local_only else hf_revision_for(name),
                token=None if local_only else (HF_TOKEN or None), cache_dir=MODEL_DIR,
                local_files_only=local_only,
            ).eval()

        def predict(self, image_path, prompt=None, max_seconds=120):
            started = time.perf_counter()
            try:
                messages = [{
                    "role": "user",
                    "content": [
                        {"type": "image", "url": str(Path(image_path).resolve())},
                        {"type": "text", "text": prompt or self.config["prompt"]},
                    ],
                }]
                inputs = self.processor.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=True,
                    return_dict=True, return_tensors="pt",
                )
                inputs.pop("token_type_ids", None)
                inputs = move_inputs(inputs, model_device(self.model))
                input_length = inputs["input_ids"].shape[1]
                generated = self.model.generate(
                    **inputs, max_new_tokens=int(self.config["max_new_tokens"]),
                    max_time=float(max_seconds), do_sample=False,
                )
                generated_ids = generated[:, input_length:]
                text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                return prediction_payload(
                    text=text, latency=time.perf_counter() - started, status="success",
                    raw_response=text, output_tokens=int(generated_ids.shape[1]),
                    output_tokens_kind="exact_model_tokens",
                    scoring_text=markdown_to_plain_text(text),
                )
            except Exception as exc:
                return prediction_payload(latency=time.perf_counter() - started, error=str(exc))

    class PaddleOCRVLAdapter(BaseAdapter):
        def __init__(self, name, config):
            super().__init__(name, config)
            from transformers import AutoModelForImageTextToText, AutoProcessor
            dtype = preferred_dtype()
            self.processor = AutoProcessor.from_pretrained(
                config["model_id"], revision=hf_revision_for(name),
                token=HF_TOKEN or None, cache_dir=MODEL_DIR,
            )
            self.model = AutoModelForImageTextToText.from_pretrained(
                config["model_id"], torch_dtype=dtype, device_map="auto",
                revision=hf_revision_for(name), token=HF_TOKEN or None, cache_dir=MODEL_DIR,
            ).eval()

        def predict(self, image_path, prompt=None, max_seconds=120):
            started = time.perf_counter()
            try:
                image = Image.open(image_path).convert("RGB")
                messages = [{"role": "user", "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt or "OCR:"},
                ]}]
                max_pixels = 1280 * 28 * 28
                inputs = self.processor.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=True,
                    return_dict=True, return_tensors="pt",
                    images_kwargs={
                        "size": {
                            "shortest_edge": self.processor.image_processor.min_pixels,
                            "longest_edge": max_pixels,
                        }
                    },
                )
                inputs = move_inputs(inputs, model_device(self.model), preferred_dtype())
                input_length = inputs["input_ids"].shape[1]
                generated = self.model.generate(
                    **inputs, max_new_tokens=int(self.config["max_new_tokens"]),
                    max_time=float(max_seconds), do_sample=False,
                )
                generated_ids = generated[:, input_length:]
                text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                return prediction_payload(
                    text=text, latency=time.perf_counter() - started, status="success",
                    raw_response=text, output_tokens=int(generated_ids.shape[1]), output_tokens_kind="exact_model_tokens",
                    scoring_text=markdown_to_plain_text(text),
                )
            except Exception as exc:
                return prediction_payload(latency=time.perf_counter() - started, error=str(exc))

    class GraniteDoclingAdapter(BaseAdapter):
        def __init__(self, name, config):
            super().__init__(name, config)
            from transformers import AutoModelForVision2Seq, AutoProcessor
            force_float32 = IS_T4 or DEVICE == "cpu"
            dtype = preferred_dtype(force_float32=force_float32)
            attention = "sdpa" if IS_T4 or DEVICE == "cpu" else "sdpa"
            self.dtype = dtype
            self.processor = AutoProcessor.from_pretrained(
                config["model_id"], revision=hf_revision_for(name),
                token=HF_TOKEN or None, cache_dir=MODEL_DIR,
            )
            self.model = AutoModelForVision2Seq.from_pretrained(
                config["model_id"], torch_dtype=dtype, device_map="auto",
                revision=hf_revision_for(name), token=HF_TOKEN or None, cache_dir=MODEL_DIR,
                _attn_implementation=attention,
            ).eval()

        def _doctags_to_outputs(self, raw_doctags, image):
            try:
                from docling_core.types.doc import DoclingDocument
                from docling_core.types.doc.document import DocTagsDocument
                doctags_document = DocTagsDocument.from_doctags_and_image_pairs([raw_doctags], [image])
                document = DoclingDocument.load_from_doctags(
                    doctags_document, document_name="OCR benchmark page"
                )
                return document.export_to_markdown(), document.export_to_text(traverse_pictures=True)
            except Exception:
                return raw_doctags, markdown_to_plain_text(raw_doctags)

        def predict(self, image_path, prompt=None, max_seconds=120):
            started = time.perf_counter()
            try:
                image = Image.open(image_path).convert("RGB")
                messages = [{"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt or self.config["prompt"]},
                ]}]
                chat_prompt = self.processor.apply_chat_template(
                    messages, add_generation_prompt=True
                )
                inputs = self.processor(
                    images=[image], text=chat_prompt, return_tensors="pt"
                )
                inputs = move_inputs(inputs, model_device(self.model), self.dtype)
                input_length = inputs["input_ids"].shape[1]
                generated = self.model.generate(
                    **inputs, max_new_tokens=int(self.config["max_new_tokens"]),
                    max_time=float(max_seconds), do_sample=False,
                )
                generated_ids = generated[:, input_length:]
                raw_doctags = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
                raw_doctags = raw_doctags.replace("<|end_of_text|>", "").strip()
                markdown, plain_text = self._doctags_to_outputs(raw_doctags, image)
                return prediction_payload(
                    text=markdown, latency=time.perf_counter() - started, status="success",
                    raw_response=raw_doctags, output_tokens=int(generated_ids.shape[1]),
                    output_tokens_kind="exact_model_tokens", scoring_text=plain_text,
                )
            except Exception as exc:
                return prediction_payload(latency=time.perf_counter() - started, error=str(exc))

    class MiniCPMAdapter(BaseAdapter):
        def __init__(self, name, config):
            super().__init__(name, config)
            from transformers import AutoModelForImageTextToText, AutoProcessor
            dtype = torch.float16 if IS_T4 else preferred_dtype()
            self.dtype = dtype
            self.processor = AutoProcessor.from_pretrained(
                config["model_id"], revision=hf_revision_for(name),
                token=HF_TOKEN or None, cache_dir=MODEL_DIR,
            )
            self.model = AutoModelForImageTextToText.from_pretrained(
                config["model_id"], torch_dtype=dtype, device_map="auto",
                revision=hf_revision_for(name), token=HF_TOKEN or None, cache_dir=MODEL_DIR,
            ).eval()

        def predict(self, image_path, prompt=None, max_seconds=120):
            started = time.perf_counter()
            try:
                image = Image.open(image_path).convert("RGB")
                messages = [{"role": "user", "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt or self.config["prompt"]},
                ]}]
                inputs = self.processor.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=True,
                    return_dict=True, return_tensors="pt", downsample_mode="4x", max_slice_nums=36,
                )
                inputs = move_inputs(inputs, model_device(self.model), self.dtype)
                input_length = inputs["input_ids"].shape[1]
                generated = self.model.generate(
                    **inputs, downsample_mode="4x", max_new_tokens=int(self.config["max_new_tokens"]),
                    max_time=float(max_seconds), do_sample=False,
                )
                generated_ids = generated[:, input_length:]
                text = self.processor.batch_decode(
                    generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0]
                return prediction_payload(
                    text=text, latency=time.perf_counter() - started, status="success",
                    raw_response=text, output_tokens=int(generated_ids.shape[1]), output_tokens_kind="exact_model_tokens",
                )
            except Exception as exc:
                return prediction_payload(latency=time.perf_counter() - started, error=str(exc))

    class LightOnOCRAdapter(BaseAdapter):
        def __init__(self, name, config):
            super().__init__(name, config)
            from transformers import LightOnOcrForConditionalGeneration, LightOnOcrProcessor
            self.dtype = preferred_dtype()
            self.processor = LightOnOcrProcessor.from_pretrained(
                config["model_id"], revision=hf_revision_for(name),
                token=HF_TOKEN or None, cache_dir=MODEL_DIR,
            )
            self.model = LightOnOcrForConditionalGeneration.from_pretrained(
                config["model_id"], torch_dtype=self.dtype,
                revision=hf_revision_for(name), token=HF_TOKEN or None, cache_dir=MODEL_DIR,
            ).to(DEVICE).eval()

        def predict(self, image_path, prompt=None, max_seconds=120):
            started = time.perf_counter()
            try:
                image = Image.open(image_path).convert("RGB")
                conversation = [{"role": "user", "content": [{"type": "image", "image": image}]}]
                inputs = self.processor.apply_chat_template(
                    conversation, add_generation_prompt=True, tokenize=True,
                    return_dict=True, return_tensors="pt",
                )
                inputs = move_inputs(inputs, DEVICE, self.dtype)
                input_length = inputs["input_ids"].shape[1]
                generated = self.model.generate(
                    **inputs, max_new_tokens=int(self.config["max_new_tokens"]),
                    max_time=float(max_seconds), do_sample=False,
                )
                generated_ids = generated[:, input_length:]
                text = self.processor.decode(generated_ids[0], skip_special_tokens=True)
                return prediction_payload(
                    text=text, latency=time.perf_counter() - started, status="success",
                    raw_response=text, output_tokens=int(generated_ids.shape[1]), output_tokens_kind="exact_model_tokens",
                    scoring_text=markdown_to_plain_text(text),
                )
            except Exception as exc:
                return prediction_payload(latency=time.perf_counter() - started, error=str(exc))

    class QwenGGUFAdapter(BaseAdapter):
        def __init__(self, name, config):
            super().__init__(name, config)
            self.cli = shutil.which("llama-cli")
            if not self.cli:
                raise RuntimeError("llama-cli multimodal n'est pas installé dans ce runtime.")
            self.model_path = hf_hub_download(
                config["model_id"], "english-document-ocr-qwen3.5-0.8b-q4_k_m.gguf",
                revision=hf_revision_for(name), cache_dir=MODEL_DIR, token=HF_TOKEN or None,
            )
            self.mmproj_path = hf_hub_download(
                config["model_id"], "mmproj-english-document-ocr-qwen3.5-0.8b-f16.gguf",
                revision=hf_revision_for(name), cache_dir=MODEL_DIR, token=HF_TOKEN or None,
            )

        def predict(self, image_path, prompt=None, max_seconds=120):
            command = [
                self.cli, "--model", self.model_path, "--mmproj", self.mmproj_path,
                "--image", str(image_path), "-p", prompt or self.config["prompt"],
                "-n", str(self.config["max_new_tokens"]), "--temp", "0", "--no-display-prompt",
                "--single-turn",
            ]
            started = time.perf_counter()
            try:
                completed = subprocess.run(
                    command, text=True, capture_output=True, stdin=subprocess.DEVNULL,
                    timeout=max_seconds, check=False,
                )
                status = "success" if completed.returncode == 0 else "failed"
                return prediction_payload(
                    text=completed.stdout, latency=time.perf_counter() - started, status=status,
                    raw_response=completed.stdout + "\n" + completed.stderr,
                    error=None if completed.returncode == 0 else completed.stderr,
                    output_tokens=np.nan, output_tokens_kind="runtime_not_reported",
                    scoring_text=markdown_to_plain_text(completed.stdout),
                )
            except subprocess.TimeoutExpired as exc:
                partial = exc.stdout.decode(errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
                return prediction_payload(
                    text=partial, latency=time.perf_counter() - started,
                    status="timeout_with_output" if partial.strip() else "timeout",
                    raw_response=partial, error="Temps maximal atteint; sortie partielle conservée.",
                    output_tokens=np.nan, output_tokens_kind="runtime_not_reported",
                    scoring_text=markdown_to_plain_text(partial),
                )

    class DotsOCRAdapter(BaseAdapter):
        def __init__(self, name, config):
            super().__init__(name, config)
            from qwen_vl_utils import process_vision_info
            from transformers import AutoModelForCausalLM, AutoProcessor
            self.process_vision_info = process_vision_info
            try:
                from dots_ocr.utils import dict_promptmode_to_prompt
                self.default_prompt = dict_promptmode_to_prompt["prompt_ocr"]
            except Exception:
                self.default_prompt = "Extract all text from this image and preserve its reading order."
            model_source = MODEL_LOCAL_PATHS.get(name) or config["model_id"]
            revision = None if MODEL_LOCAL_PATHS.get(name) else hf_revision_for(name)
            self.processor = AutoProcessor.from_pretrained(
                model_source, trust_remote_code=True,
                revision=revision, token=HF_TOKEN or None, cache_dir=MODEL_DIR,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_source, attn_implementation="flash_attention_2",
                torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
                revision=revision, token=HF_TOKEN or None, cache_dir=MODEL_DIR,
            ).eval()

        def predict(self, image_path, prompt=None, max_seconds=120):
            started = time.perf_counter()
            try:
                effective_prompt = self.default_prompt if not prompt or prompt == "prompt_ocr" else prompt
                messages = [{"role": "user", "content": [
                    {"type": "image", "image": str(Path(image_path).resolve())},
                    {"type": "text", "text": effective_prompt},
                ]}]
                template = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                image_inputs, video_inputs = self.process_vision_info(messages)
                inputs = self.processor(
                    text=[template], images=image_inputs, videos=video_inputs,
                    padding=True, return_tensors="pt",
                )
                inputs = move_inputs(inputs, model_device(self.model), torch.bfloat16)
                input_length = inputs["input_ids"].shape[1]
                generated = self.model.generate(
                    **inputs, max_new_tokens=int(self.config["max_new_tokens"]),
                    max_time=float(max_seconds), do_sample=False,
                )
                generated_ids = generated[:, input_length:]
                text = self.processor.batch_decode(
                    generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0]
                payload = prediction_payload(
                    text=text, latency=time.perf_counter() - started, status="success",
                    raw_response=text, output_tokens=int(generated_ids.shape[1]),
                    output_tokens_kind="exact_model_tokens",
                    scoring_text=markdown_to_plain_text(text),
                )
                payload["prompt_used"] = effective_prompt
                return payload
            except Exception as exc:
                return prediction_payload(latency=time.perf_counter() - started, error=str(exc))

    class UnlimitedOCRAdapter(BaseAdapter):
        def __init__(self, name, config):
            super().__init__(name, config)
            from transformers import AutoModel, AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                config["model_id"], trust_remote_code=True,
                revision=hf_revision_for(name), token=HF_TOKEN or None, cache_dir=MODEL_DIR,
            )
            self.model = AutoModel.from_pretrained(
                config["model_id"], trust_remote_code=True, use_safetensors=True,
                torch_dtype=torch.bfloat16, revision=hf_revision_for(name),
                token=HF_TOKEN or None, cache_dir=MODEL_DIR,
            ).eval().cuda()
            self.output_dir = WORK_DIR / "unlimited_ocr_outputs"
            self.output_dir.mkdir(parents=True, exist_ok=True)

        def predict(self, image_path, prompt=None, max_seconds=120):
            started = time.perf_counter()
            try:
                effective_prompt = prompt or self.config["prompt"]
                if "<image>" not in effective_prompt:
                    effective_prompt = "<image>\n" + effective_prompt
                response = self.model.infer(
                    self.tokenizer,
                    prompt=effective_prompt,
                    image_file=str(image_path), output_path=str(self.output_dir),
                    base_size=1024, image_size=640, crop_mode=True,
                    max_length=int(self.config["max_new_tokens"]),
                    no_repeat_ngram_size=35, ngram_window=128,
                    save_results=False, eval_mode=True,
                )
                text = str(response or "")
                try:
                    token_count = len(self.tokenizer.encode(text, add_special_tokens=False))
                except Exception:
                    token_count = np.nan
                payload = prediction_payload(
                    text=text, latency=time.perf_counter() - started, status="success",
                    raw_response=text, output_tokens=token_count,
                    output_tokens_kind="estimated_tokenizer_tokens",
                    scoring_text=markdown_to_plain_text(text),
                )
                payload["prompt_used"] = effective_prompt
                return payload
            except Exception as exc:
                return prediction_payload(latency=time.perf_counter() - started, error=str(exc))

    class LocateAnythingAdapter(BaseAdapter):
        def __init__(self, name, config):
            super().__init__(name, config)
            from transformers import AutoModel, AutoProcessor, AutoTokenizer
            revision = hf_revision_for(name)
            self.tokenizer = AutoTokenizer.from_pretrained(
                config["model_id"], trust_remote_code=True, revision=revision,
                token=HF_TOKEN or None, cache_dir=MODEL_DIR,
            )
            self.processor = AutoProcessor.from_pretrained(
                config["model_id"], trust_remote_code=True, revision=revision,
                token=HF_TOKEN or None, cache_dir=MODEL_DIR,
            )
            self.model = AutoModel.from_pretrained(
                config["model_id"], trust_remote_code=True, torch_dtype=torch.bfloat16,
                revision=revision, token=HF_TOKEN or None, cache_dir=MODEL_DIR,
            ).to("cuda").eval()
            self.preview_dir = WORK_DIR / "locateanything_previews"
            self.preview_dir.mkdir(parents=True, exist_ok=True)

        @staticmethod
        def parse_boxes(answer, width, height):
            boxes = []
            for match in re.finditer(r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>", str(answer)):
                x1, y1, x2, y2 = [int(group) for group in match.groups()]
                boxes.append({
                    "x1": x1 / 1000 * width, "y1": y1 / 1000 * height,
                    "x2": x2 / 1000 * width, "y2": y2 / 1000 * height,
                })
            return boxes

        def annotate(self, image, boxes, image_path):
            preview = image.copy()
            draw = ImageDraw.Draw(preview)
            line_width = max(2, round(max(image.size) / 500))
            for index, box in enumerate(boxes, start=1):
                coordinates = (box["x1"], box["y1"], box["x2"], box["y2"])
                draw.rectangle(coordinates, outline="#EF4444", width=line_width)
                draw.text((box["x1"] + 3, box["y1"] + 3), str(index), fill="#EF4444")
            fingerprint = hashlib.sha1(str(image_path).encode("utf-8")).hexdigest()[:12]
            preview_path = self.preview_dir / f"{fingerprint}.png"
            preview.save(preview_path)
            return str(preview_path)

        def predict(self, image_path, prompt=None, max_seconds=120):
            started = time.perf_counter()
            try:
                image = Image.open(image_path).convert("RGB")
                question = prompt or self.config["prompt"]
                messages = [{"role": "user", "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question},
                ]}]
                template = self.processor.py_apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                images, videos = self.processor.process_vision_info(messages)
                inputs = self.processor(
                    text=[template], images=images, videos=videos, return_tensors="pt"
                ).to("cuda")
                seed = int(globals().get("SEED", 42))
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                response = self.model.generate(
                    pixel_values=inputs["pixel_values"].to(torch.bfloat16),
                    input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
                    image_grid_hws=inputs.get("image_grid_hws", None), tokenizer=self.tokenizer,
                    max_new_tokens=int(self.config["max_new_tokens"]), use_cache=True,
                    generation_mode="hybrid", temperature=0.7, do_sample=True,
                    top_p=0.9, repetition_penalty=1.1, verbose=False,
                )
                answer = response[0] if isinstance(response, tuple) else response
                answer = str(answer)
                boxes = self.parse_boxes(answer, *image.size)
                preview_path = self.annotate(image, boxes, image_path)
                try:
                    token_count = len(self.tokenizer.encode(answer, add_special_tokens=False))
                except Exception:
                    token_count = np.nan
                payload = prediction_payload(
                    text=answer, latency=time.perf_counter() - started, status="success",
                    raw_response=json.dumps({"answer": answer, "boxes": boxes}, ensure_ascii=False),
                    output_tokens=token_count, output_tokens_kind="estimated_tokenizer_tokens",
                    inference_device="cuda",
                )
                payload.update({"annotated_image_path": preview_path, "detected_boxes": len(boxes)})
                return payload
            except Exception as exc:
                return prediction_payload(latency=time.perf_counter() - started, error=str(exc))

    class ChandraAdapter(BaseAdapter):
        def __init__(self, name, config):
            super().__init__(name, config)
            from chandra.model.hf import generate_hf
            from chandra.model.schema import BatchInputItem
            from chandra.output import parse_markdown
            from transformers import AutoModelForImageTextToText, AutoProcessor
            self.generate_hf = generate_hf
            self.BatchInputItem = BatchInputItem
            self.parse_markdown = parse_markdown
            self.model = AutoModelForImageTextToText.from_pretrained(
                config["model_id"], torch_dtype=torch.bfloat16, device_map="auto",
                revision=hf_revision_for(name), token=HF_TOKEN or None, cache_dir=MODEL_DIR,
            ).eval()
            self.model.processor = AutoProcessor.from_pretrained(
                config["model_id"], revision=hf_revision_for(name),
                token=HF_TOKEN or None, cache_dir=MODEL_DIR
            )
            self.model.processor.tokenizer.padding_side = "left"

        def predict(self, image_path, prompt=None, max_seconds=120):
            started = time.perf_counter()
            try:
                image = Image.open(image_path).convert("RGB")
                result = self.generate_hf(
                    [self.BatchInputItem(image=image, prompt_type="ocr_layout")],
                    self.model, max_output_tokens=int(self.config["max_new_tokens"]),
                )[0]
                markdown_text = self.parse_markdown(result.raw)
                return prediction_payload(
                    text=markdown_text, latency=time.perf_counter() - started,
                    status="success", raw_response=result.raw,
                    output_tokens=result.token_count, output_tokens_kind="exact_model_tokens",
                    scoring_text=markdown_to_plain_text(markdown_text),
                )
            except Exception as exc:
                return prediction_payload(latency=time.perf_counter() - started, error=str(exc))

    def build_adapter(model_name):
        cfg = MODEL_CATALOG[model_name]
        ready, reason = model_readiness(model_name)
        if not ready:
            raise RuntimeError(reason)
        adapter_class = {
            "easyocr": EasyOCRAdapter,
            "pp_ocrv6": PPOCRv6Adapter,
            "paddleocr_vl": PaddleOCRVLAdapter,
            "glm_ocr": GLMOCRAdapter,
            "granite_docling": GraniteDoclingAdapter,
            "minicpm": MiniCPMAdapter,
            "lighton": LightOnOCRAdapter,
            "qwen_gguf": QwenGGUFAdapter,
            "chandra": ChandraAdapter,
            "dots_ocr": DotsOCRAdapter,
            "unlimited_ocr": UnlimitedOCRAdapter,
            "locate_anything": LocateAnythingAdapter,
        }.get(cfg["adapter_kind"])
        if adapter_class is None:
            raise RuntimeError(
                f"{cfg['display_name']} nécessite le profil isolé {cfg['profile']}; "
                "aucun pip downgrade n'est fait dans un kernel déjà importé."
            )
        return adapter_class(model_name, cfg)

    print("Adaptateurs prêts:", sorted({cfg["adapter_kind"] for cfg in MODEL_CATALOG.values()}))
"""))

cells.append(markdown(r"""
    ## 8. Sélectionner les documents et exécuter un benchmark checkpointé

    Trois modes sont disponibles : tout le dataset, une quantité globale, ou la même quantité dans chaque catégorie. Le runner écrit `results.json` et `details.csv` après **chaque image**. Si Colab se coupe, les résultats déjà terminés restent dans le dossier du run ; le notebook ne prétend pas reprendre automatiquement le modèle au milieu d'une génération.

    `AUTO_RUN_BENCHMARK` est désactivé par défaut pour éviter de lancer plusieurs gigaoctets de modèles dès l'exécution complète du notebook. L'interface Gradio de l'étape 11 appelle exactement la même fonction. Passez-le à `True` si vous préférez tout lancer directement depuis cette cellule.
"""))

cells.append(code(r"""
    # ÉTAPE 8A — Sélection réutilisable par le notebook et Gradio
    SELECTION_MODE = "Quantité globale"  # Tout le dataset | Quantité globale | Par catégorie
    GLOBAL_QUANTITY = min(12, len(dataset_df))
    PER_CATEGORY_QUANTITY = 3
    SELECTED_CATEGORIES = sorted(dataset_df["category"].unique().tolist())
    SHUFFLE = True
    SEED = 42

    def select_dataset(
        frame, mode="Quantité globale", quantity=12, categories=None,
        shuffle=True, seed=42,
    ):
        selected = frame.copy()
        categories = (
            sorted(selected["category"].unique().tolist())
            if categories is None else list(categories)
        )
        selected = selected[selected["category"].isin(categories)]
        if mode == "Tout le dataset":
            return selected.reset_index(drop=True)
        if mode == "Quantité globale":
            if shuffle and len(selected):
                selected = selected.sample(frac=1, random_state=seed)
            return selected.head(min(int(quantity), len(selected))).reset_index(drop=True)
        if mode == "Par catégorie":
            parts = []
            for category in categories:
                part = selected[selected["category"] == category]
                if shuffle and len(part):
                    part = part.sample(frac=1, random_state=seed)
                parts.append(part.head(int(quantity)))
            return pd.concat(parts, ignore_index=True) if parts else selected.iloc[0:0].copy()
        raise ValueError(f"Mode inconnu: {mode}")

    selected_df = select_dataset(
        dataset_df,
        mode=SELECTION_MODE,
        quantity=GLOBAL_QUANTITY if SELECTION_MODE != "Par catégorie" else PER_CATEGORY_QUANTITY,
        categories=SELECTED_CATEGORIES,
        shuffle=SHUFFLE,
        seed=SEED,
    )
    print(f"Sélection par défaut: {len(selected_df)} / {len(dataset_df)} documents")
    display(selected_df[["id", "source", "category", "task_type", "is_scorable"]])
"""))

cells.append(code(r"""
    # ÉTAPE 8B — Runner générateur avec progression image par image
    import importlib.metadata as package_metadata
    import psutil

    SUCCESS_STATUSES = {"success", "slow_success", "timeout_with_output"}
    QUALITY_OUTPUT_STATUSES = SUCCESS_STATUSES | {"empty_output"}
    FAILURE_STATUSES = {"failed", "failed_load", "timeout", "empty_output"}
    SKIP_STATUSES = {"skipped_incompatible", "skipped_task"}
    MAX_SECONDS_PER_IMAGE = 120

    def checkpoint_results(rows, run_directory):
        frame = pd.DataFrame(rows)
        frame.to_json(run_directory / "results.json", orient="records", force_ascii=False, indent=2)
        frame.to_csv(run_directory / "details.csv", index=False)

    def installed_versions(package_names):
        versions = {}
        for package_name in package_names:
            try:
                versions[package_name] = package_metadata.version(package_name)
            except package_metadata.PackageNotFoundError:
                versions[package_name] = None
        return versions

    def selection_signature(frame):
        columns = [
            "id", "image_sha256", "ground_truth", "category", "task_type",
            "is_scorable", "source", "source_revision",
        ]
        missing = [column for column in columns if column not in frame.columns]
        if missing:
            raise ValueError(f"Impossible de signer la sélection; colonnes absentes: {missing}")
        canonical = frame[columns].copy().fillna("")
        for column in columns:
            canonical[column] = canonical[column].astype(str)
        canonical = canonical.sort_values("id", kind="stable")
        payload = canonical.to_csv(index=False, lineterminator="\n")
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def write_run_metadata(
        run_directory, run_id, model_names, selection_frame,
        max_seconds, prompt_override, run_status,
    ):
        metadata = {
            "run_id": run_id,
            "run_status": run_status,
            "created_or_updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "runtime_profile": RUNTIME_PROFILE,
            "device": DEVICE,
            "gpu_name": GPU_NAME,
            "gpu_vram_gb": GPU_VRAM_GB,
            "gpu_capability": list(GPU_CAPABILITY),
            "torch_version": getattr(torch, "__version__", None) if torch else None,
            "cuda_version": getattr(getattr(torch, "version", None), "cuda", None) if torch else None,
            "python_version": platform.python_version(),
            "selected_models": list(model_names),
            "resolved_model_revisions": MODEL_RESOLVED_REVISIONS,
            "selected_document_ids": selection_frame["id"].astype(str).tolist(),
            "selection_signature": selection_signature(selection_frame),
            "selected_categories": sorted(selection_frame["category"].astype(str).unique().tolist()),
            "source_revisions": sorted(selection_frame["source_revision"].astype(str).unique().tolist()),
            "max_seconds_per_image": float(max_seconds),
            "prompt_override": str(prompt_override or ""),
            "shuffle": SHUFFLE,
            "seed": SEED,
            "package_versions": installed_versions([
                "torch", "transformers", "paddleocr", "easyocr", "docling-core",
                "datasets", "huggingface-hub", "kagglehub", "gradio", "pandas", "numpy",
            ]),
        }
        (Path(run_directory) / "run_metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )

    def prompt_for_document(model_name, row, override=""):
        cfg = MODEL_CATALOG[model_name]
        if cfg["adapter_kind"] in {"easyocr", "pp_ocrv6"}:
            return "Aucun prompt — moteur OCR non génératif"
        if cfg["adapter_kind"] == "lighton":
            return "Aucun prompt textuel — LightOnOCR reçoit uniquement l'image"
        if cfg["adapter_kind"] == "chandra":
            return "prompt_type=ocr_layout"
        if str(override or "").strip():
            return str(override).strip()
        if (
            row.task_type == "key_value_extraction"
            and cfg["adapter_kind"] in {"glm_ocr", "minicpm", "qwen_gguf", "dots_ocr", "unlimited_ocr"}
        ):
            reference_object = extract_json_object(row.ground_truth)
            if isinstance(reference_object, dict):
                def blank_schema(value):
                    if isinstance(value, dict):
                        return {key: blank_schema(child) for key, child in value.items()}
                    if isinstance(value, list):
                        return [blank_schema(value[0])] if value else []
                    return ""
                schema = json.dumps(blank_schema(reference_object), ensure_ascii=False, indent=2)
                return (
                    "Extract the requested information from the image. Return only valid JSON "
                    "with exactly this schema and no extra keys:\n" + schema
                )
        if cfg["adapter_kind"] in {"glm_ocr", "paddleocr_vl", "locate_anything"}:
            return cfg["prompt"]
        if row.task_type == "key_value_extraction" and str(row.prompt).strip():
            return str(row.prompt).strip()
        return cfg.get("prompt") or OCR_PROMPT

    def base_result_row(run_id, model_name, row, load_time, status, error=None):
        cfg = MODEL_CATALOG[model_name]
        return {
            "run_id": run_id, "model": model_name, "model_name": cfg["display_name"],
            "model_id": cfg["model_id"], "ranking_task": cfg["ranking_task"],
            "source": row.source, "source_revision": row.source_revision,
            "document_id": row.id, "image_path": row.image_path,
            "category": row.category, "task_type": row.task_type,
            "is_scorable": bool(row.is_scorable), "label_provenance": row.label_provenance,
            "ground_truth": row.ground_truth, "prompt_used": "", "raw_text": "", "text": "",
            "latency": np.nan, "load_time": load_time, "status": status,
            "raw_response": "", "error": error, "device": DEVICE,
            "runtime_profile": RUNTIME_PROFILE, "hardware_name": GPU_NAME,
            "preview_image_path": row.image_path, "detected_boxes": np.nan, "boxes_per_second": np.nan,
            "output_tokens": np.nan, "output_tokens_kind": "not_available",
            "tokens_per_second": np.nan, "chars_per_second": np.nan,
            "gpu_peak_mb": np.nan, "ram_rss_mb": psutil.Process(os.getpid()).memory_info().rss / 1024**2,
            "cer": np.nan, "wer": np.nan, "normalized_exact_match": np.nan,
            "char_edits": np.nan, "reference_chars": np.nan,
            "word_edits": np.nan, "reference_words": np.nan,
            "field_precision": np.nan, "field_recall": np.nan, "field_f1": np.nan,
        }

    def stream_loaded_adapter(
        adapter, model_name, config, selection_frame, rows, progress_state,
        run_id, run_directory, load_time, max_seconds, prompt_override, total,
    ):
        try:
            for _, document in selection_frame.iterrows():
                completed = progress_state["completed"]
                prompt_used = prompt_for_document(model_name, document, prompt_override)
                yield {
                    "phase": "analyzing", "model": model_name, "model_name": config["display_name"],
                    "document": document, "message": f"Analyse de {Path(document.image_path).name}",
                    "completed": completed, "total": total, "rows": rows,
                    "run_id": run_id, "run_dir": run_directory,
                }
                if torch and DEVICE == "cuda":
                    torch.cuda.reset_peak_memory_stats()
                try:
                    model_prompt = None if config["adapter_kind"] in {"easyocr", "pp_ocrv6"} else prompt_used
                    prediction = adapter.predict(
                        document.image_path, model_prompt, max_seconds=float(max_seconds)
                    )
                except Exception as exc:
                    prediction = prediction_payload(error=str(exc))

                model_output = str(prediction.get("text", ""))
                scoring_source = prediction.get("scoring_text")
                if scoring_source is None:
                    scoring_source = model_output
                reasoning = completed_reasoning_blocks(scoring_source)
                scoring_source = remove_completed_reasoning(scoring_source)
                cleaned = clean_model_text(scoring_source)
                latency = prediction.get("latency", np.nan)
                status = prediction.get("status", "failed")
                if status == "success" and pd.notna(latency) and latency > float(max_seconds):
                    status = "slow_success"
                if status in SUCCESS_STATUSES and bool(document.is_scorable) and not cleaned:
                    status = "empty_output"
                output_tokens = prediction.get("output_tokens", np.nan)
                result = base_result_row(
                    run_id, model_name, document, load_time, status, prediction.get("error")
                )
                result.update({
                    "prompt_used": prediction.get("prompt_used", prompt_used),
                    "raw_text": model_output, "text": cleaned,
                    "reasoning_text": "\n\n".join(reasoning),
                    "latency": latency, "raw_response": prediction.get("raw_response", ""),
                    "output_tokens": output_tokens,
                    "output_tokens_kind": prediction.get("output_tokens_kind", "not_available"),
                    "tokens_per_second": (
                        output_tokens / latency
                        if pd.notna(output_tokens) and pd.notna(latency) and latency > 0 else np.nan
                    ),
                    "chars_per_second": len(cleaned) / latency if pd.notna(latency) and latency > 0 else np.nan,
                    "gpu_peak_mb": (
                        torch.cuda.max_memory_allocated() / 1024**2
                        if torch and DEVICE == "cuda" else np.nan
                    ),
                    "ram_rss_mb": psutil.Process(os.getpid()).memory_info().rss / 1024**2,
                    "device": prediction.get("inference_device") or DEVICE,
                    "preview_image_path": prediction.get("annotated_image_path") or document.image_path,
                    "detected_boxes": prediction.get("detected_boxes", np.nan),
                    "boxes_per_second": (
                        prediction.get("detected_boxes") / latency
                        if prediction.get("detected_boxes") is not None
                        and pd.notna(latency) and latency > 0 else np.nan
                    ),
                })
                if (
                    result["is_scorable"]
                    and status in QUALITY_OUTPUT_STATUSES
                    and document.task_type == "transcription"
                    and config["ranking_task"] == "transcription"
                ):
                    result.update(metric_details(result["ground_truth"], cleaned))
                    result["cer"] = cer(result["ground_truth"], cleaned)
                    result["wer"] = wer(result["ground_truth"], cleaned)
                elif (
                    result["is_scorable"]
                    and status in QUALITY_OUTPUT_STATUSES
                    and document.task_type == "key_value_extraction"
                    and config["ranking_task"] == "transcription"
                ):
                    result.update(structured_field_scores(result["ground_truth"], cleaned))
                rows.append(result)
                progress_state["completed"] += 1
                checkpoint_results(rows, run_directory)
                yield {
                    "phase": "result", "model": model_name, "model_name": config["display_name"],
                    "document": document, "result": result,
                    "completed": progress_state["completed"],
                    "total": total, "rows": rows, "run_id": run_id, "run_dir": run_directory,
                }
        finally:
            try:
                adapter.close()
            finally:
                gc.collect()
                if torch and DEVICE == "cuda":
                    torch.cuda.empty_cache()

    def benchmark_stream(model_names, selection_frame, max_seconds=120, prompt_override=""):
        model_names = list(dict.fromkeys(model_names or []))
        if not model_names:
            raise ValueError("Sélectionnez au moins un modèle.")
        unknown = [name for name in model_names if name not in MODEL_CATALOG]
        if unknown:
            raise ValueError(f"Modèles inconnus: {unknown}")
        if selection_frame.empty:
            raise ValueError("Aucun document sélectionné.")

        run_id = time.strftime("%Y%m%d-%H%M%S") + f"-{int(time.time() * 1000) % 1000:03d}"
        run_directory = RUNS_DIR / run_id
        run_directory.mkdir(parents=True, exist_ok=True)
        total = len(model_names) * len(selection_frame)
        completed = 0
        rows = []
        write_run_metadata(
            run_directory, run_id, model_names, selection_frame,
            max_seconds, prompt_override, "running",
        )

        for model_name in model_names:
            cfg = MODEL_CATALOG[model_name]
            ready, reason = model_readiness(model_name)
            yield {
                "phase": "loading", "model": model_name, "model_name": cfg["display_name"],
                "message": f"Préparation de {cfg['display_name']}…", "completed": completed,
                "total": total, "rows": rows, "run_id": run_id, "run_dir": run_directory,
            }

            skip_status = "skipped_incompatible"

            if not ready:
                for _, document in selection_frame.iterrows():
                    result = base_result_row(run_id, model_name, document, 0.0, skip_status, reason)
                    result["prompt_used"] = prompt_for_document(model_name, document, prompt_override)
                    rows.append(result)
                    completed += 1
                    checkpoint_results(rows, run_directory)
                    yield {
                        "phase": "result", "model": model_name, "model_name": cfg["display_name"],
                        "document": document, "result": result, "completed": completed,
                        "total": total, "rows": rows, "run_id": run_id, "run_dir": run_directory,
                    }
                continue

            adapter = None
            load_started = time.perf_counter()
            try:
                download_model_weights(model_name)
                adapter = build_adapter(model_name)
                load_error = None
            except Exception as exc:
                load_error = str(exc)
            load_time = time.perf_counter() - load_started
            write_run_metadata(
                run_directory, run_id, model_names, selection_frame,
                max_seconds, prompt_override, "running",
            )

            if adapter is None:
                for _, document in selection_frame.iterrows():
                    result = base_result_row(run_id, model_name, document, load_time, "failed_load", load_error)
                    result["prompt_used"] = prompt_for_document(model_name, document, prompt_override)
                    rows.append(result)
                    completed += 1
                    checkpoint_results(rows, run_directory)
                    yield {
                        "phase": "result", "model": model_name, "model_name": cfg["display_name"],
                        "document": document, "result": result, "completed": completed,
                        "total": total, "rows": rows, "run_id": run_id, "run_dir": run_directory,
                    }
                continue

            progress_state = {"completed": completed}
            yield from stream_loaded_adapter(
                adapter, model_name, cfg, selection_frame, rows, progress_state,
                run_id, run_directory, load_time, max_seconds, prompt_override, total,
            )
            completed = progress_state["completed"]
            adapter = None

        results_frame = pd.DataFrame(rows)
        write_run_metadata(
            run_directory, run_id, model_names, selection_frame,
            max_seconds, prompt_override, "complete",
        )
        yield {
            "phase": "done", "completed": completed, "total": total,
            "rows": rows, "results_df": results_frame, "run_id": run_id, "run_dir": run_directory,
        }

    AUTO_RUN_BENCHMARK = False
    results_df = pd.DataFrame()
    run_dir = None
    RUN_ID = "not-run-yet"
    if AUTO_RUN_BENCHMARK:
        for event in benchmark_stream(SELECTED_MODELS, selected_df, MAX_SECONDS_PER_IMAGE):
            if event["phase"] == "result":
                result = event["result"]
                print(
                    f"{event['completed']}/{event['total']} · {event['model_name']} · "
                    f"{result['status']} · {result['latency'] if pd.notna(result['latency']) else 'n/a'}"
                )
            elif event["phase"] == "done":
                results_df, run_dir, RUN_ID = event["results_df"], event["run_dir"], event["run_id"]
        display(results_df.head())
    else:
        print("Runner prêt. Le benchmark démarrera avec le bouton Gradio à l'étape 11.")
"""))

cells.append(markdown(r"""
    ## 9. Construire le classement sans cacher les échecs

    Les fonctions ci-dessous excluent les échecs, les lignes ignorées et les documents sans vérité terrain des moyennes de qualité/latence. Elles les conservent toutefois dans les colonnes `failed` et `skipped`. Un modèle n'est éligible que s'il atteint au moins 95 % de réussite et couvre toutes les catégories scorables sélectionnées.
"""))

cells.append(code(r"""
    # ÉTAPE 9 — Résumé et recommandation réutilisables
    MIN_TECHNICAL_SUCCESS_RATE = 0.95
    MIN_SCORED_DOCUMENTS = 1
    SERIOUS_MIN_DOCS_PER_CATEGORY = 30

    def summarize_results(results_frame, selection_frame):
        if results_frame is None or results_frame.empty:
            return pd.DataFrame(), pd.DataFrame()
        scorable_categories = set(selection_frame.loc[
            selection_frame.is_scorable & selection_frame.task_type.eq("transcription"), "category"
        ].unique())
        summary_rows = []
        for model_name, group in results_frame.groupby("model", sort=False):
            all_successful = group[group.status.isin(SUCCESS_STATUSES) & group.latency.notna()]
            transcription_attempts = group[group.task_type.eq("transcription")]
            successful = transcription_attempts[
                transcription_attempts.status.isin(SUCCESS_STATUSES)
                & transcription_attempts.latency.notna()
            ]
            quality_transcription = transcription_attempts[
                transcription_attempts.status.isin(QUALITY_OUTPUT_STATUSES)
            ]
            scored = quality_transcription[
                quality_transcription.is_scorable
                & quality_transcription.reference_chars.fillna(0).gt(0)
            ]
            extraction_attempts = group[group.task_type.eq("key_value_extraction")]
            extraction_successful = extraction_attempts[
                extraction_attempts.status.isin(SUCCESS_STATUSES)
                & extraction_attempts.latency.notna()
            ]
            extraction_quality = extraction_attempts[
                extraction_attempts.status.isin(QUALITY_OUTPUT_STATUSES)
            ]
            extraction_scored = extraction_quality[
                extraction_quality.is_scorable & extraction_quality.field_f1.notna()
            ]
            covered = set(scored.category.unique()) & scorable_categories
            category_coverage = len(covered) / len(scorable_categories) if scorable_categories else 1.0
            ref_chars, ref_words = scored.reference_chars.sum(), scored.reference_words.sum()
            corpus_cer = scored.char_edits.sum() / ref_chars if ref_chars else np.nan
            corpus_wer = scored.word_edits.sum() / ref_words if ref_words else np.nan
            latencies = successful.latency.astype(float)
            cfg = MODEL_CATALOG[model_name]
            summary_rows.append({
                "model": model_name, "model_name": cfg["display_name"], "evaluations": len(group),
                "succeeded": len(all_successful), "failed": int(group.status.isin(FAILURE_STATUSES).sum()),
                "skipped": int(group.status.isin(SKIP_STATUSES).sum()),
                "overall_success_rate": len(all_successful) / len(group) if len(group) else 0.0,
                "transcription_evaluations": len(transcription_attempts),
                "success_rate": len(successful) / len(transcription_attempts) if len(transcription_attempts) else 0.0,
                "scored_documents": len(scored), "category_coverage": category_coverage,
                "corpus_cer": corpus_cer, "corpus_wer": corpus_wer,
                "macro_cer": scored.cer.mean(),
                "exact_match_rate": scored.normalized_exact_match.mean(),
                "median_latency": latencies.median(), "p95_latency": latencies.quantile(0.95),
                "avg_latency": latencies.mean(),
                "documents_per_minute": 60 / latencies.mean() if len(latencies) and latencies.mean() > 0 else np.nan,
                "tokens_per_second": successful.tokens_per_second.mean(),
                "characters_per_second": successful.chars_per_second.mean(),
                "boxes_per_second": all_successful.boxes_per_second.mean(),
                "max_gpu_mb": successful.gpu_peak_mb.max(), "max_ram_mb": successful.ram_rss_mb.max(),
                "extraction_evaluations": len(extraction_attempts),
                "extraction_success_rate": (
                    len(extraction_successful) / len(extraction_attempts) if len(extraction_attempts) else np.nan
                ),
                "extraction_median_latency": extraction_successful.latency.median(),
                "extraction_documents": len(extraction_scored),
                "field_f1": extraction_scored.field_f1.mean(), "load_time": group.load_time.max(),
                "device": group.device.iloc[0], "ranking_task": cfg["ranking_task"],
            })

        summary = pd.DataFrame(summary_rows)
        summary["eligible"] = (
            (summary.success_rate >= MIN_TECHNICAL_SUCCESS_RATE)
            & (summary.scored_documents >= MIN_SCORED_DOCUMENTS)
            & (summary.category_coverage >= 1.0)
            & summary.ranking_task.eq("transcription")
        )
        summary["quality_score"] = 100 * (
            1 - (0.70 * summary.corpus_cer.clip(0, 1) + 0.30 * summary.corpus_wer.clip(0, 1))
        )
        fastest_p95 = summary.loc[summary.eligible & summary.p95_latency.gt(0), "p95_latency"].min()
        summary["speed_score"] = np.where(
            summary.p95_latency.gt(0) & pd.notna(fastest_p95),
            100 * fastest_p95 / summary.p95_latency,
            np.nan,
        )
        summary["decision_score"] = (
            0.70 * summary.quality_score
            + 0.20 * (100 * summary.success_rate)
            + 0.10 * summary.speed_score.fillna(0)
        )
        summary.loc[~summary.eligible, "decision_score"] = np.nan

        def decision_note(row):
            if row.eligible:
                return "éligible"
            reasons = []
            if row.ranking_task != "transcription": reasons.append("tâche différente")
            if row.success_rate < MIN_TECHNICAL_SUCCESS_RATE: reasons.append("réussite insuffisante")
            if row.scored_documents < MIN_SCORED_DOCUMENTS: reasons.append("aucun label scorable")
            if row.category_coverage < 1.0: reasons.append("catégories incomplètes")
            return "exclu: " + ", ".join(reasons or ["données insuffisantes"])

        summary["decision_note"] = summary.apply(decision_note, axis=1)
        summary = summary.sort_values(
            ["eligible", "decision_score", "corpus_cer"], ascending=[False, False, True], na_position="last"
        ).reset_index(drop=True)

        category_rows = []
        scored_all = results_frame[
            results_frame.status.isin(QUALITY_OUTPUT_STATUSES)
            & results_frame.is_scorable
            & results_frame.task_type.eq("transcription")
            & results_frame.reference_chars.fillna(0).gt(0)
        ]
        for (model_name, category), group in scored_all.groupby(["model", "category"]):
            reference_words = group.reference_words.sum()
            category_rows.append({
                "model": model_name, "model_name": MODEL_CATALOG[model_name]["display_name"],
                "category": category, "documents": len(group),
                "corpus_cer": group.char_edits.sum() / group.reference_chars.sum(),
                "corpus_wer": group.word_edits.sum() / reference_words if reference_words else np.nan,
            })
        return summary, pd.DataFrame(category_rows)

    summary_df, category_summary_df = summarize_results(results_df, selected_df)
    if len(summary_df):
        display(summary_df)
    else:
        print("Le résumé apparaîtra après le premier benchmark.")
"""))

cells.append(code(r"""
    # ÉTAPE 10 — Dashboard Plotly compact
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    COLORS = {
        "blue": "#2563EB", "teal": "#0F9D8A", "orange": "#F59E0B",
        "red": "#DC2626", "ink": "#172033", "muted": "#667085", "bg": "#F5F7FB",
    }
    PALETTE = ["#2563EB", "#0F9D8A", "#7C3AED", "#F59E0B", "#DB2777", "#0891B2", "#65A30D"]

    def empty_dashboard(message="Lancez un benchmark pour afficher les graphiques."):
        figure = go.Figure()
        figure.add_annotation(text=message, x=0.5, y=0.5, showarrow=False, font={"size": 18, "color": COLORS["muted"]})
        figure.update_layout(height=500, paper_bgcolor=COLORS["bg"], plot_bgcolor="white", xaxis={"visible": False}, yaxis={"visible": False})
        return figure

    def build_dashboard(summary, category_summary, results_frame, selection_frame, run_id):
        if summary is None or summary.empty:
            return empty_dashboard()
        summary = summary.copy()
        color_by_model = {name: PALETTE[index % len(PALETTE)] for index, name in enumerate(summary.model)}
        summary["pareto"] = False
        for index, row in summary[summary.eligible].iterrows():
            others = summary[summary.eligible & (summary.index != index)]
            dominated = (
                (others.corpus_cer <= row.corpus_cer)
                & (others.p95_latency <= row.p95_latency)
                & ((others.corpus_cer < row.corpus_cer) | (others.p95_latency < row.p95_latency))
            ).any()
            summary.loc[index, "pareto"] = not dominated

        figure = make_subplots(
            rows=2, cols=2,
            specs=[[{"type": "scatter"}, {"type": "heatmap"}], [{"type": "bar"}, {"type": "box"}]],
            subplot_titles=(
                "Qualité vs P95 — idéal en haut à gauche",
                "Qualité caractères par catégorie",
                "Issue de toutes les tentatives",
                "Distribution des latences réussies",
            ),
            vertical_spacing=0.16, horizontal_spacing=0.12,
        )
        for _, row in summary.iterrows():
            if pd.isna(row.p95_latency) or pd.isna(row.quality_score):
                continue
            figure.add_trace(go.Scatter(
                x=[row.p95_latency], y=[row.quality_score], mode="markers+text",
                text=[row.model_name], textposition="top center",
                marker={
                    "size": 16 if row.eligible else 11,
                    "color": color_by_model[row.model], "opacity": 1 if row.eligible else 0.42,
                    "line": {"color": COLORS["ink"] if row.pareto else "white", "width": 3 if row.pareto else 1},
                },
                customdata=[[row.success_rate, row.corpus_cer, row.corpus_wer, row.scored_documents, row.decision_note]],
                hovertemplate=(
                    "<b>%{text}</b><br>P95: %{x:.2f}s<br>Qualité: %{y:.1f}/100"
                    "<br>Réussite: %{customdata[0]:.1%}<br>CER: %{customdata[1]:.1%}"
                    "<br>WER: %{customdata[2]:.1%}<br>Documents scorés: %{customdata[3]}"
                    "<br>%{customdata[4]}<extra></extra>"
                ), showlegend=False,
            ), row=1, col=1)

        if category_summary is not None and len(category_summary):
            heat = category_summary.assign(
                character_quality=100 * (1 - category_summary.corpus_cer.clip(0, 1))
            ).pivot(index="model_name", columns="category", values="character_quality")
            figure.add_trace(go.Heatmap(
                z=heat.values, x=heat.columns, y=heat.index, zmin=0, zmax=100,
                colorscale=[[0, "#FEE2E2"], [0.5, "#FEF3C7"], [1, "#D1FAE5"]],
                colorbar={"title": "Qualité %"}, text=np.round(heat.values, 1), texttemplate="%{text}%",
                hovertemplate="%{y}<br>%{x}<br>Qualité: %{z:.1f}%<extra></extra>",
            ), row=1, col=2)
        else:
            figure.add_annotation(text="Aucun label scorable", x=0.80, y=0.82, xref="paper", yref="paper", showarrow=False)

        figure.add_trace(go.Bar(name="Succès", x=summary.model_name, y=summary.succeeded, marker_color=COLORS["teal"]), row=2, col=1)
        figure.add_trace(go.Bar(name="Échecs", x=summary.model_name, y=summary.failed, marker_color=COLORS["red"]), row=2, col=1)
        figure.add_trace(go.Bar(name="Ignorés", x=summary.model_name, y=summary.skipped, marker_color=COLORS["orange"]), row=2, col=1)
        successful_latency = results_frame[
            results_frame.status.isin(SUCCESS_STATUSES) & results_frame.latency.notna()
        ]
        for model_name, group in successful_latency.groupby("model"):
            figure.add_trace(go.Box(
                y=group.latency, name=MODEL_CATALOG[model_name]["display_name"],
                marker_color=color_by_model.get(model_name, COLORS["blue"]), boxmean=True,
                showlegend=False, hovertemplate="%{y:.2f}s<extra></extra>",
            ), row=2, col=2)

        figure.update_xaxes(title_text="Secondes/image (P95) — plus bas = mieux", row=1, col=1)
        figure.update_yaxes(title_text="Qualité caractères /100", range=[0, 105], row=1, col=1)
        figure.update_yaxes(title_text="Tentatives", row=2, col=1)
        figure.update_yaxes(title_text="Secondes/image", row=2, col=2)
        figure.update_layout(
            height=760, barmode="stack", paper_bgcolor=COLORS["bg"], plot_bgcolor="white",
            font={"color": COLORS["ink"], "family": "Inter, Arial"},
            margin={"l": 55, "r": 30, "t": 82, "b": 48},
            title={"text": f"OCR Benchmark · {run_id} · {DEVICE.upper()} · {len(selection_frame)} documents", "x": 0.02},
            legend={"orientation": "h", "y": -0.10},
        )
        return figure

    dashboard_fig = build_dashboard(summary_df, category_summary_df, results_df, selected_df, RUN_ID)
    if AUTO_RUN_BENCHMARK:
        dashboard_fig.show()
"""))

cells.append(markdown(r"""
    ## 11. Lancer et explorer avec Gradio

    L'interface suit l'ordre de travail : **Lancer → Graphiques → Explorer → Comprendre les métriques → Dataset → Paramètres**. La page garde son défilement normal, mais les zones live sont limitées en hauteur pour tenir autant que possible dans un écran.

    Pendant l'exécution, l'image active et ses mesures sont côte à côte. Après chaque tentative, le compteur, la barre de progression, le statut et le résultat changent. L'explorateur fournit une liste déroulante, les boutons précédent/suivant, le texte attendu face au texte extrait, la sortie brute, le Markdown/HTML, le prompt exact et un diff.
"""))

cells.append(code(r"""
    # ÉTAPE 11 — Application Gradio autonome dans Colab
    import difflib
    import gradio as gr

    UI_STATE = {
        "results": results_df.copy(),
        "summary": summary_df.copy(),
        "category_summary": category_summary_df.copy(),
        "selection": selected_df.copy(),
        "run_id": RUN_ID,
        "run_dir": run_dir,
    }

    METRIC_GUIDE = '''
    ### Lire les chiffres sans être spécialiste

    | Mesure | Question à laquelle elle répond | Bon résultat | Limite importante |
    |---|---|---:|---|
    | **CER** | Combien de caractères faut-il corriger ? | proche de 0 % | peut dépasser 100 % si le modèle ajoute beaucoup de texte |
    | **WER** | Combien de mots faut-il corriger ? | proche de 0 % | plus sévère qu'un CER sur une petite faute |
    | **Exact match** | La transcription entière est-elle identique ? | proche de 100 % | très strict pour les longs documents |
    | **Médiane** | Quel est le temps d'une image typique ? | faible | ne décrit pas les cas très lents |
    | **P95** | Sous quel temps finissent 95 % des images ? | faible | demande assez d'images pour être stable |
    | **Docs/min** | Quel débit séquentiel peut-on espérer ? | élevé | dépend du matériel et de la taille des pages |
    | **Tokens/s** | À quelle vitesse un VLM génère-t-il ses tokens ? | élevé | tokenizers différents; N/A pour OCR classique |
        | **Caractères/s** | Combien de caractères finaux sont produits ? | élevé | dépend de la longueur des documents |
        | **Boîtes/s** | Combien de zones LocateAnything détecte-t-il ? | élevé | débit seulement sans labels de boîtes IoU/F1 |
        | **VRAM PyTorch / RAM RSS** | Quelle mémoire le processus utilise-t-il ? | sous la capacité cible | VRAM=max PyTorch; RAM=photo RSS après inférence |
    | **Réussite** | Combien de tentatives donnent une sortie ? | proche de 100 % | une sortie peut réussir techniquement mais être mauvaise |

        **Score de décision = 70 % qualité + 20 % fiabilité + 10 % vitesse P95**, uniquement pour la transcription.
        L'extraction JSON est présentée séparément avec le **Field‑F1** ; elle n'influence jamais le CER/WER.
    Une décision basée sur moins de 30 documents scorables par catégorie reste exploratoire.
    '''

    def ui_update(component_class, **kwargs):
        if hasattr(gr, "update"):
            return gr.update(**kwargs)
        return component_class(**kwargs)

    def result_labels(frame=None):
        frame = UI_STATE["results"] if frame is None else frame
        if frame is None or frame.empty:
            return []
        return [
            f"{index:04d} · {row.model_name} · {row.category} · {Path(row.image_path).name}"
            for index, row in frame.reset_index(drop=True).iterrows()
        ]

    def resolve_result_index(label):
        labels = result_labels()
        if not labels:
            return None
        if label in labels:
            return labels.index(label)
        try:
            return max(0, min(int(str(label).split("·", 1)[0].strip()), len(labels) - 1))
        except Exception:
            return 0

    def format_number(value, digits=2, suffix=""):
        return "N/A" if value is None or pd.isna(value) else f"{float(value):.{digits}f}{suffix}"

    def live_metrics_frame(result=None, completed=0, total=0, model_name=""):
        if result is None:
            return pd.DataFrame([
                {"Métrique": "Progression", "Valeur": f"{completed}/{total}"},
                {"Métrique": "Modèle", "Valeur": model_name or "En attente"},
                {"Métrique": "Statut", "Valeur": "Analyse en cours"},
            ])
        return pd.DataFrame([
            {"Métrique": "Progression", "Valeur": f"{completed}/{total}"},
            {"Métrique": "Modèle", "Valeur": result.get("model_name", model_name)},
            {"Métrique": "Statut", "Valeur": result.get("status", "N/A")},
            {"Métrique": "Temps image", "Valeur": format_number(result.get("latency"), 2, " s")},
            {"Métrique": "CER", "Valeur": format_number(result.get("cer") * 100 if pd.notna(result.get("cer", np.nan)) else np.nan, 1, " %")},
            {"Métrique": "WER", "Valeur": format_number(result.get("wer") * 100 if pd.notna(result.get("wer", np.nan)) else np.nan, 1, " %")},
            {"Métrique": "Field-F1", "Valeur": format_number(result.get("field_f1") * 100 if pd.notna(result.get("field_f1", np.nan)) else np.nan, 1, " %")},
            {"Métrique": "Tokens sortie", "Valeur": format_number(result.get("output_tokens"), 0)},
            {"Métrique": "Tokens/s", "Valeur": format_number(result.get("tokens_per_second"), 1)},
            {"Métrique": "Caractères/s", "Valeur": format_number(result.get("chars_per_second"), 1)},
            {"Métrique": "Boîtes détectées", "Valeur": format_number(result.get("detected_boxes"), 0)},
            {"Métrique": "Boîtes/s", "Valeur": format_number(result.get("boxes_per_second"), 1)},
            {"Métrique": "Pic GPU", "Valeur": format_number(result.get("gpu_peak_mb"), 0, " Mo")},
            {"Métrique": "Détail chargement/erreur", "Valeur": str(result.get("error") or "Aucune")},
        ])

    def summary_for_ui(summary):
        if summary is None or summary.empty:
            return pd.DataFrame(columns=["Modèle", "Réussite transcription", "CER", "WER", "Field-F1", "Réussite extraction", "P95", "Score", "Décision"])
        output = summary[[
            "model_name", "success_rate", "corpus_cer", "corpus_wer",
            "field_f1", "extraction_success_rate", "p95_latency", "documents_per_minute",
            "max_gpu_mb", "decision_score", "decision_note",
        ]].copy()
        output.columns = ["Modèle", "Réussite transcription", "CER", "WER", "Field-F1", "Réussite extraction", "P95 (s)", "Docs/min", "VRAM pic (Mo)", "Score /100", "Décision"]
        return output

    def make_run_archive(run_directory, summary, category_summary, selection):
        if run_directory is None:
            return None
        run_directory = Path(run_directory)
        summary.to_csv(run_directory / "summary.csv", index=False)
        category_summary.to_csv(run_directory / "summary_by_category.csv", index=False)
        selection.to_csv(run_directory / "selected_dataset.csv", index=False)
        (run_directory / "metric_definitions.md").write_text(METRIC_GUIDE, encoding="utf-8")
        dashboard = build_dashboard(summary, category_summary, UI_STATE["results"], selection, run_directory.name)
        dashboard.write_html(run_directory / "dashboard.html", include_plotlyjs="cdn")
        archive = shutil.make_archive(str(run_directory), "zip", root_dir=run_directory)
        return archive

    def ui_run_benchmark(
        model_names, categories, selection_mode, quantity,
        timeout_seconds, prompt_override, progress=gr.Progress(track_tqdm=False),
    ):
        global results_df, summary_df, category_summary_df, selected_df, run_dir, RUN_ID
        model_names = list(model_names or [])
        categories = list(categories or [])
        selection = select_dataset(
            dataset_df, mode=selection_mode, quantity=int(quantity), categories=categories,
            shuffle=True, seed=SEED,
        )
        if not model_names:
            raise gr.Error("Sélectionnez au moins un modèle.")
        if selection.empty:
            raise gr.Error("Aucun document ne correspond aux catégories choisies.")

        previous_summary = summary_for_ui(UI_STATE["summary"])
        previous_dashboard = build_dashboard(
            UI_STATE["summary"], UI_STATE["category_summary"], UI_STATE["results"],
            UI_STATE["selection"], UI_STATE["run_id"],
        )
        last_image = None
        last_metrics = live_metrics_frame(None, 0, len(model_names) * len(selection))
        last_text = "Le texte extrait de la dernière image apparaîtra ici."
        selector_update = ui_update(gr.Dropdown, choices=result_labels(), value=(result_labels() or [None])[0])

        for event in benchmark_stream(
            model_names, selection, max_seconds=float(timeout_seconds), prompt_override=prompt_override,
        ):
            phase = event["phase"]
            completed, total = event["completed"], event["total"]
            progress(completed / total if total else 0, desc=event.get("message", phase))
            progress_text = f"### {completed} / {total} tentatives · {100 * completed / total:.1f} %"

            if phase == "loading":
                status_text = f"**Chargement :** {event['model_name']} — téléchargement mis en cache si nécessaire."
            elif phase == "analyzing":
                document = event["document"]
                last_image = document.image_path
                last_metrics = live_metrics_frame(None, completed, total, event["model_name"])
                last_text = "Analyse en cours…"
                status_text = f"**En cours :** {event['model_name']} analyse `{Path(document.image_path).name}`"
            elif phase == "result":
                result = event["result"]
                last_image = result.get("preview_image_path") or result["image_path"]
                last_metrics = live_metrics_frame(result, completed, total, event["model_name"])
                last_text = str(result.get("text") or result.get("raw_text") or "(sortie vide)")
                status_text = (
                    f"**Dernier résultat :** {event['model_name']} · `{result['status']}` · "
                    f"{format_number(result['latency'], 2, ' s')}"
                    + (f"\n\n> Détail : `{result.get('error')}`" if result.get("error") else "")
                )
            else:
                current_results = event["results_df"]
                current_summary, current_category = summarize_results(current_results, selection)
                current_dashboard = build_dashboard(
                    current_summary, current_category, current_results, selection, event["run_id"]
                )
                results_df, summary_df, category_summary_df = current_results, current_summary, current_category
                selected_df, run_dir, RUN_ID = selection, event["run_dir"], event["run_id"]
                UI_STATE.update({
                    "results": current_results, "summary": current_summary,
                    "category_summary": current_category, "selection": selection,
                    "run_id": RUN_ID, "run_dir": run_dir,
                })
                labels = result_labels(current_results)
                selector_update = ui_update(gr.Dropdown, choices=labels, value=labels[0] if labels else None)
                eligible = current_summary[current_summary.eligible] if len(current_summary) else pd.DataFrame()
                if eligible.empty:
                    decision = "Aucun modèle recommandable : consultez les erreurs et les lignes ignorées."
                else:
                    best = eligible.iloc[0]
                    minimum_per_category = (
                        current_category[current_category.model == best.model].documents.min()
                        if len(current_category) else 0
                    )
                    confidence = "Recommandation exploratoire" if minimum_per_category < SERIOUS_MIN_DOCS_PER_CATEGORY else "Recommandation"
                    decision = f"{confidence} : **{best.model_name}**, score {best.decision_score:.1f}/100."
                status_text = f"## Benchmark terminé\n{decision}\n\nRésultats : `{run_dir}`"
                archive_path = make_run_archive(run_dir, current_summary, current_category, selection)
                yield (
                    status_text, progress_text, last_image, last_metrics, last_text,
                    summary_for_ui(current_summary), current_dashboard, selector_update, archive_path,
                )
                continue

            yield (
                status_text, progress_text, last_image, last_metrics, last_text,
                previous_summary, previous_dashboard, selector_update, None,
            )

    def show_result(label):
        index = resolve_result_index(label)
        if index is None:
            empty_metrics = pd.DataFrame([{"Métrique": "Statut", "Valeur": "Aucun résultat"}])
            return None, empty_metrics, "", "", "", "", "", "", ""
        row = UI_STATE["results"].reset_index(drop=True).iloc[index]
        metrics = pd.DataFrame([
            {"Métrique": "Modèle", "Valeur": row.model_name},
            {"Métrique": "Document", "Valeur": row.document_id},
            {"Métrique": "Catégorie", "Valeur": row.category},
            {"Métrique": "Statut", "Valeur": row.status},
            {"Métrique": "Temps", "Valeur": format_number(row.latency, 2, " s")},
            {"Métrique": "CER", "Valeur": format_number(row.cer * 100 if pd.notna(row.cer) else np.nan, 1, " %")},
            {"Métrique": "WER", "Valeur": format_number(row.wer * 100 if pd.notna(row.wer) else np.nan, 1, " %")},
            {"Métrique": "Field-F1", "Valeur": format_number(row.field_f1 * 100 if pd.notna(row.field_f1) else np.nan, 1, " %")},
            {"Métrique": "Tokens", "Valeur": format_number(row.output_tokens, 0)},
            {"Métrique": "Type tokens", "Valeur": row.output_tokens_kind},
            {"Métrique": "Tokens/s", "Valeur": format_number(row.tokens_per_second, 1)},
            {"Métrique": "Caractères/s", "Valeur": format_number(row.chars_per_second, 1)},
            {"Métrique": "Boîtes détectées", "Valeur": format_number(row.detected_boxes, 0)},
            {"Métrique": "Boîtes/s", "Valeur": format_number(row.boxes_per_second, 1)},
            {"Métrique": "VRAM pic", "Valeur": format_number(row.gpu_peak_mb, 0, " Mo")},
            {"Métrique": "RAM RSS après inférence", "Valeur": format_number(row.ram_rss_mb, 0, " Mo")},
            {"Métrique": "Erreur", "Valeur": str(row.error or "")},
        ])
        expected, extracted = str(row.ground_truth), str(row.text)
        raw = str(row.raw_response or row.raw_text or "")
        formatted_output = str(row.raw_text or extracted)
        html_source = formatted_output
        safe_preview = f"<pre class='safe-preview'>{html.escape(extracted)}</pre>"
        diff = difflib.HtmlDiff(wrapcolumn=80).make_table(
            expected.splitlines(), extracted.splitlines(), "Texte attendu", "Texte extrait", context=True, numlines=3
        )
        preview_path = row.preview_image_path if str(row.preview_image_path).strip() else row.image_path
        return preview_path, metrics, expected, extracted, raw, formatted_output, html_source, safe_preview + diff, str(row.prompt_used)

    def navigate_result(current_label, delta):
        labels = result_labels()
        if not labels:
            return None
        index = resolve_result_index(current_label) or 0
        return labels[(index + int(delta)) % len(labels)]

    def ui_import_dataset(zip_file):
        global dataset_df, dataset_manifest_df
        if not zip_file:
            raise gr.Error("Choisissez un fichier ZIP.")
        extract_dir = WORK_DIR / "imports" / time.strftime("%Y%m%d-%H%M%S")
        extract_dir.mkdir(parents=True, exist_ok=True)
        safe_extract_zip(str(zip_file), extract_dir)
        imported_rows = read_records_from_folder(extract_dir)
        imported = pd.DataFrame(imported_rows)
        imported = pd.concat([imported.reset_index(drop=True), imported["image_path"].apply(inspect_image)], axis=1)
        imported["ground_truth_chars"] = imported.ground_truth.astype(str).str.len()
        imported["dedupe_priority"] = [(0, 0)] * len(imported)
        imported["duplicate_of"] = imported.groupby("image_sha256")["id"].transform("first")
        imported["is_duplicate"] = imported["id"] != imported["duplicate_of"]

        # Une vérité terrain fournie par l'utilisateur remplace une copie distante identique dans le benchmark actif.
        imported_active = imported[~imported.is_duplicate].copy()
        imported_hashes = set(imported_active.image_sha256)
        replacement_by_hash = imported_active.set_index("image_sha256")["id"].to_dict()
        existing_duplicate_mask = dataset_manifest_df.image_sha256.isin(imported_hashes)
        dataset_manifest_df.loc[existing_duplicate_mask, "is_duplicate"] = True
        dataset_manifest_df.loc[existing_duplicate_mask, "duplicate_of"] = dataset_manifest_df.loc[
            existing_duplicate_mask, "image_sha256"
        ].map(replacement_by_hash)
        dataset_df = pd.concat(
            [dataset_df[~dataset_df.image_sha256.isin(imported_hashes)], imported_active], ignore_index=True
        )
        dataset_manifest_df = pd.concat([dataset_manifest_df, imported], ignore_index=True)
        dataset_manifest_df.to_csv(WORK_DIR / "dataset_manifest.csv", index=False)
        categories = sorted(dataset_df.category.unique().tolist())
        status = (
            f"{len(imported)} lignes validées, {len(imported_active)} images uniques activées. "
            f"Dataset actif: {len(dataset_df)} documents, {len(categories)} catégories."
        )
        preview = dataset_df[["id", "source", "category", "is_scorable", "ground_truth_chars"]].tail(30)
        return (
            status, preview,
            ui_update(gr.CheckboxGroup, choices=categories, value=categories),
            ui_update(gr.Slider, maximum=max(2, len(dataset_df)), value=max(1, min(12, len(dataset_df)))),
        )

    RUN_ARCHIVE_MAX_BYTES = 512 * 1024**2

    def read_run_archive(archive_path):
        archive_path = Path(getattr(archive_path, "name", archive_path))
        if archive_path.suffix.lower() != ".zip":
            raise ValueError(f"Archive attendue au format ZIP: {archive_path.name}")
        with zipfile.ZipFile(archive_path) as archive:
            members = archive.infolist()
            if len(members) > MAX_ZIP_FILES:
                raise ValueError(f"Archive {archive_path.name}: trop de fichiers.")
            if sum(member.file_size for member in members) > RUN_ARCHIVE_MAX_BYTES:
                raise ValueError(f"Archive {archive_path.name}: contenu décompressé supérieur à 512 Mo.")

            def member_bytes(basename):
                matches = [member for member in members if PurePosixPath(member.filename).name == basename]
                if len(matches) != 1:
                    raise ValueError(f"Archive {archive_path.name}: {basename} absent ou ambigu.")
                member = matches[0]
                if member.flag_bits & 0x1:
                    raise ValueError(f"Archive chiffrée non supportée: {archive_path.name}")
                if member.file_size > RUN_ARCHIVE_MAX_BYTES:
                    raise ValueError(f"Fichier trop grand dans {archive_path.name}: {basename}")
                return archive.read(member)

            details = pd.read_csv(BytesIO(member_bytes("details.csv")))
            selection = pd.read_csv(BytesIO(member_bytes("selected_dataset.csv")), keep_default_na=False)
            metadata = json.loads(member_bytes("run_metadata.json").decode("utf-8"))
        return details, selection, metadata, archive_path.name

    def ui_import_run_archives(archive_files):
        global results_df, summary_df, category_summary_df, selected_df, run_dir, RUN_ID
        paths = list(archive_files or [])
        if not paths:
            raise gr.Error("Choisissez au moins une archive ZIP exportée par ce notebook.")

        loaded = []
        try:
            loaded = [read_run_archive(path) for path in paths]
        except Exception as exc:
            raise gr.Error(str(exc)) from exc

        signatures = []
        for _, archive_selection, metadata, _ in loaded:
            computed_signature = selection_signature(archive_selection)
            recorded_signature = metadata.get("selection_signature")
            if recorded_signature and recorded_signature != computed_signature:
                raise gr.Error("Une archive a été modifiée: sa signature dataset ne correspond plus.")
            signatures.append(recorded_signature or computed_signature)
        if len(set(signatures)) != 1:
            raise gr.Error("Les archives n'utilisent pas exactement les mêmes documents et labels.")

        timeout_contracts = {
            round(float(metadata.get("max_seconds_per_image", -1)), 6)
            for _, _, metadata, _ in loaded
        }
        prompt_contracts = {str(metadata.get("prompt_override", "")) for _, _, metadata, _ in loaded}
        hardware_contracts = {
            (str(metadata.get("device", "")), str(metadata.get("gpu_name", "")))
            for _, _, metadata, _ in loaded
        }
        if len(timeout_contracts) != 1 or len(prompt_contracts) != 1:
            raise gr.Error("Même dataset, mais délai ou prompt différent: comparaison refusée.")
        if len(hardware_contracts) != 1:
            raise gr.Error(
                "GPU/CPU différents entre les archives: les vitesses et le score de décision seraient trompeurs. "
                "Relancez tous les profils sur le même GPU."
            )

        reference_selection = loaded[0][1]
        selected_ids = reference_selection["id"].astype(str).tolist()
        local_by_id = dataset_df.assign(id=dataset_df.id.astype(str)).set_index("id", drop=False)
        missing_ids = [document_id for document_id in selected_ids if document_id not in local_by_id.index]
        if missing_ids:
            raise gr.Error(
                "Le runtime courant ne possède pas les mêmes images. Rechargez les mêmes sources ou le même ZIP "
                f"personnel avant la fusion. Manquants: {missing_ids[:5]}"
            )
        local_selection = local_by_id.loc[selected_ids].reset_index(drop=True)
        if selection_signature(local_selection) != signatures[0]:
            raise gr.Error("Les identifiants correspondent, mais les images ou labels locaux sont différents.")

        merged = pd.concat([details for details, _, _, _ in loaded], ignore_index=True, sort=False)
        required_result_columns = {
            "model", "model_name", "document_id", "status", "task_type", "is_scorable",
            "latency", "ranking_task", "ground_truth", "text", "raw_text", "raw_response",
        }
        missing_columns = sorted(required_result_columns - set(merged.columns))
        if missing_columns:
            raise gr.Error(f"Archives incompatibles; colonnes résultat absentes: {missing_columns}")
        unknown_models = sorted(set(merged.model.astype(str)) - set(MODEL_CATALOG))
        if unknown_models:
            raise gr.Error(f"Modèles absents de ce notebook: {unknown_models}")

        merged["document_id"] = merged.document_id.astype(str)
        image_by_id = local_selection.set_index("id")["image_path"].astype(str).to_dict()
        merged["image_path"] = merged.document_id.map(image_by_id)
        merged["preview_image_path"] = merged["image_path"]
        for column in (
            "run_id", "model", "model_name", "status", "task_type", "ranking_task",
            "ground_truth", "raw_text", "text", "raw_response", "error", "prompt_used", "reasoning_text",
            "category", "source", "source_revision", "label_provenance", "output_tokens_kind",
            "device", "runtime_profile", "hardware_name",
        ):
            if column not in merged:
                merged[column] = ""
            merged[column] = merged[column].fillna("").astype(str)
        merged["is_scorable"] = merged.is_scorable.map(
            lambda value: value if isinstance(value, bool) else str(value).strip().lower() in {"1", "true", "yes"}
        )
        numeric_columns = (
            "latency", "load_time", "output_tokens", "tokens_per_second", "chars_per_second",
            "detected_boxes", "boxes_per_second", "gpu_peak_mb", "ram_rss_mb", "cer", "wer",
            "normalized_exact_match", "char_edits", "reference_chars", "word_edits", "reference_words",
            "field_precision", "field_recall", "field_f1",
        )
        for column in numeric_columns:
            if column not in merged:
                merged[column] = np.nan
            merged[column] = pd.to_numeric(merged[column], errors="coerce")
        merged = merged.drop_duplicates(["model", "document_id"], keep="last").reset_index(drop=True)

        current_summary, current_category = summarize_results(merged, local_selection)
        merged_id = "merged-" + time.strftime("%Y%m%d-%H%M%S")
        merged_directory = RUNS_DIR / merged_id
        merged_directory.mkdir(parents=True, exist_ok=True)
        checkpoint_results(merged.to_dict("records"), merged_directory)
        local_selection.to_csv(merged_directory / "selected_dataset.csv", index=False)
        (merged_directory / "run_metadata.json").write_text(json.dumps({
            "run_id": merged_id, "run_status": "merged", "selection_signature": signatures[0],
            "source_archives": [name for _, _, _, name in loaded],
            "runtime_profile": "MERGED",
            "source_profiles": sorted({
                str(metadata.get("runtime_profile", "inconnu")) for _, _, metadata, _ in loaded
            }),
            "device": next(iter(hardware_contracts))[0], "gpu_name": next(iter(hardware_contracts))[1],
            "max_seconds_per_image": next(iter(timeout_contracts)),
            "prompt_override": next(iter(prompt_contracts)),
        }, ensure_ascii=False, indent=2), encoding="utf-8")

        results_df, summary_df, category_summary_df = merged, current_summary, current_category
        selected_df, run_dir, RUN_ID = local_selection, merged_directory, merged_id
        UI_STATE.update({
            "results": merged, "summary": current_summary, "category_summary": current_category,
            "selection": local_selection, "run_id": merged_id, "run_dir": merged_directory,
        })
        dashboard = build_dashboard(current_summary, current_category, merged, local_selection, merged_id)
        labels = result_labels(merged)
        selector_update = ui_update(gr.Dropdown, choices=labels, value=labels[0] if labels else None)
        archive_path = make_run_archive(merged_directory, current_summary, current_category, local_selection)
        profiles = sorted({str(metadata.get("runtime_profile", "inconnu")) for _, _, metadata, _ in loaded})
        status = (
            f"Fusion validée: **{len(loaded)} archives**, {merged.model.nunique()} modèles, "
            f"{len(local_selection)} documents, même matériel `{next(iter(hardware_contracts))[1]}`. "
            f"Profils: {', '.join(profiles)}."
        )
        return status, summary_for_ui(current_summary), dashboard, selector_update, archive_path

    model_choices = []
    default_model_values = []
    for model_key, config in MODEL_CATALOG.items():
        ready, reason = model_readiness(model_key)
        marker = "prêt" if ready else f"indisponible: {reason}"
        model_choices.append((f"{config['display_name']} — {marker}", model_key))
        if model_key in DEFAULT_SELECTED_MODELS and ready:
            default_model_values.append(model_key)

    category_choices = sorted(dataset_df.category.unique().tolist())
    initial_labels = result_labels()

    APP_CSS = '''
    :root { --lab-bg: #F5F7FB; --lab-panel: #FFFFFF; --lab-ink: #172033; --lab-muted: #667085; --lab-blue: #2563EB; }
    body, .gradio-container { background: var(--lab-bg) !important; color: var(--lab-ink); }
    .gradio-container { max-width: 1480px !important; margin: 0 auto !important; padding: 14px 18px 28px !important; }
    .lab-header { background: #FFFFFF; border: 1px solid #E4E7EC; border-left: 5px solid var(--lab-blue); border-radius: 14px; padding: 14px 18px; margin-bottom: 10px; }
    .lab-header h1 { font-size: 1.45rem; line-height: 1.2; margin: 0 0 4px; color: var(--lab-ink); }
    .lab-header p { margin: 0; color: var(--lab-muted); }
    .live-image img { max-height: 390px !important; object-fit: contain !important; background: #FFFFFF; }
    .compact-table { max-height: 390px; overflow: auto; }
    .live-metrics { max-height: 205px; overflow: auto; }
    .safe-preview { white-space: pre-wrap; background: #FFFFFF; color: #172033; border: 1px solid #E4E7EC; border-radius: 10px; padding: 12px; }
    .diff { width: 100%; font-size: 12px; border-collapse: collapse; background: #FFFFFF; }
    .diff td, .diff th { padding: 3px 6px; border: 1px solid #EAECF0; }
    footer { display: none !important; }
    '''

    with gr.Blocks(title="OCR Model Selection Lab") as demo:
        gr.HTML('''
        <div class="lab-header">
          <h1>OCR Model Selection Lab</h1>
          <p>Comparer des modèles sur vos documents, comprendre les métriques et choisir avec des preuves.</p>
        </div>
        ''')

        with gr.Tab("1 · Lancer un benchmark"):
            with gr.Row():
                with gr.Column(scale=4):
                    model_selector = gr.CheckboxGroup(
                        choices=model_choices, value=default_model_values,
                        label="Modèles — un seul reste chargé à la fois",
                    )
                with gr.Column(scale=2):
                    category_selector = gr.CheckboxGroup(
                        choices=category_choices, value=category_choices, label="Catégories"
                    )
                    selection_mode = gr.Radio(
                        ["Quantité globale", "Par catégorie", "Tout le dataset"],
                        value="Quantité globale", label="Mode de quantité",
                    )
                    quantity_slider = gr.Slider(
                        1, max(2, len(dataset_df)), value=max(1, min(12, len(dataset_df))), step=1,
                        label="Quantité (globale ou par catégorie)",
                    )
                    run_button = gr.Button("Valider et lancer", variant="primary")
            launch_status = gr.Markdown("Prêt. Vérifiez les modèles, les catégories et les paramètres.")
            live_progress = gr.Markdown("### 0 / 0 tentative")
            with gr.Row(equal_height=True):
                live_image = gr.Image(label="Image en cours / dernier résultat", type="filepath", elem_classes="live-image", height=410)
                with gr.Column():
                    live_metrics = gr.Dataframe(
                        value=live_metrics_frame(), headers=["Métrique", "Valeur"],
                        interactive=False, label="Mesures live", elem_classes="live-metrics",
                    )
                    live_text = gr.Textbox(
                        value="Le texte extrait de la dernière image apparaîtra ici.",
                        label="Texte extrait en direct", lines=7, max_lines=7,
                        interactive=False, buttons=["copy"],
                    )

        with gr.Tab("2 · Graphiques"):
            summary_table = gr.Dataframe(
                value=summary_for_ui(summary_df), interactive=False,
                label="Classement et diagnostic technique", elem_classes="compact-table",
            )
            dashboard_plot = gr.Plot(value=dashboard_fig, label="Dashboard qualité · vitesse · fiabilité")
            export_file = gr.File(label="Archive ZIP reproductible du dernier run", interactive=False)
            gr.Markdown('''
            **Comparer plusieurs profils Colab :** téléchargez le ZIP de chaque run, redémarrez avec le profil suivant,
            puis importez toutes les archives ici. La fusion n'est acceptée que si documents, labels, prompt, délai et
            GPU sont identiques. Pour inclure LocateAnything, exécutez donc aussi les profils CORE/DOTS/CHANDRA sur
            le même A100 40 Go ; une vitesse A100 ne doit pas être comparée à une vitesse T4.
            ''')
            with gr.Row():
                run_archives = gr.File(
                    label="Archives ZIP de runs compatibles", file_types=[".zip"],
                    file_count="multiple", type="filepath",
                )
                merge_runs_button = gr.Button("Vérifier et fusionner les runs", variant="secondary")
            merge_runs_status = gr.Markdown()

        with gr.Tab("3 · Explorer les résultats"):
            with gr.Row():
                previous_button = gr.Button("← Précédent")
                result_selector = gr.Dropdown(
                    choices=initial_labels, value=initial_labels[0] if initial_labels else None,
                    label="Liste des documents testés — ouvrez puis faites défiler",
                    allow_custom_value=False,
                )
                next_button = gr.Button("Suivant →")
            with gr.Row(equal_height=True):
                detail_image = gr.Image(label="Document", type="filepath", height=430, elem_classes="live-image")
                detail_metrics = gr.Dataframe(
                    value=pd.DataFrame(), headers=["Métrique", "Valeur"], interactive=False,
                    label="Performance de ce résultat", elem_classes="compact-table",
                )
            with gr.Row(equal_height=True):
                expected_text = gr.Textbox(label="Texte attendu", lines=14, max_lines=22, buttons=["copy"])
                extracted_text = gr.Textbox(label="Texte extrait", lines=14, max_lines=22, buttons=["copy"])
            with gr.Tabs():
                with gr.Tab("Sortie brute"):
                    raw_output = gr.Textbox(lines=18, buttons=["copy"], label="Réponse complète conservée (y compris raisonnement éventuel)")
                with gr.Tab("Markdown"):
                    markdown_output = gr.Markdown(sanitize_html=True)
                with gr.Tab("HTML source"):
                    html_source_output = gr.Code(language="html", label="Source HTML/texte — non exécutée")
                with gr.Tab("Diff attendu/extrait"):
                    diff_output = gr.HTML()
                with gr.Tab("Prompt envoyé"):
                    prompt_output = gr.Textbox(lines=8, buttons=["copy"], label="Prompt exact de cette tentative")

        with gr.Tab("4 · Comprendre les métriques"):
            gr.Markdown(METRIC_GUIDE)

        with gr.Tab("5 · Dataset"):
            gr.Markdown('''
            ### Ajouter vos données avec leurs labels
            ZIP attendu : images + `labels.csv`. Colonnes obligatoires :
            `image_path,ground_truth,category`. Exemple :

            ```csv
            image_path,ground_truth,category,description,task_type,prompt
            images/cheque_01.png,"Montant: 1250 EUR",cheque,Chèque manuscrit,transcription,
            ```

            Les chemins absolus et les sorties de dossier (`../`) sont refusés. Une image importée avec un label utilisateur remplace un doublon distant dans l'échantillon actif.
            ''')
            with gr.Row():
                dataset_zip = gr.File(label="Dataset ZIP", file_types=[".zip"], type="filepath")
                import_button = gr.Button("Importer et valider le ZIP", variant="secondary")
            import_status = gr.Markdown()
            dataset_preview = gr.Dataframe(
                value=dataset_df[["id", "source", "category", "is_scorable", "ground_truth_chars"]].head(30),
                interactive=False, label="Dataset actif", elem_classes="compact-table",
            )

        with gr.Tab("6 · Paramètres"):
            with gr.Row():
                timeout_seconds = gr.Slider(
                    10, 900, value=MAX_SECONDS_PER_IMAGE, step=10,
                    label="Temps cible maximal par image (secondes)",
                )
                prompt_override = gr.Textbox(
                    value="", lines=5,
                    label="Prompt global facultatif — vide = prompt adapté à chaque modèle/tâche",
                    placeholder=OCR_PROMPT,
                )
            gr.Markdown('''
            **Temps cible, pas toujours un arrêt forcé :**

            - `llama-cli` (Qwen GGUF) : arrêt dur du sous-processus ; sa sortie partielle est conservée.
            - GLM‑OCR, PaddleOCR‑VL, Granite Docling, MiniCPM, LightOnOCR et dots.ocr : `max_time` demande à la génération de s'arrêter dès que possible ; les tokens déjà produits restent disponibles.
            - EasyOCR, PP‑OCRv6, Chandra, Unlimited‑OCR et LocateAnything : seuil de mesure souple, car leur API ne permet pas d'interrompre sûrement une image en cours. Une réponse tardive devient `slow_success` et reste enregistrée.

            Le prompt réellement envoyé est sauvegardé ligne par ligne et visible dans l'onglet Explorer. Les moteurs non génératifs affichent explicitement qu'aucun prompt n'est utilisé.
            ''')
            gr.Dataframe(value=model_catalog_df, interactive=False, label="Compatibilité du catalogue", elem_classes="compact-table")

        run_event = run_button.click(
            fn=ui_run_benchmark,
            inputs=[
                model_selector, category_selector, selection_mode, quantity_slider,
                timeout_seconds, prompt_override,
            ],
            outputs=[
                launch_status, live_progress, live_image, live_metrics, live_text,
                summary_table, dashboard_plot, result_selector, export_file,
            ],
            show_progress="full",
        )
        # Une mise à jour serveur du Dropdown ne déclenche pas de façon
        # uniforme `change` selon les versions de Gradio. Recharge donc
        # explicitement le premier résultat quand le benchmark est terminé,
        # afin que le texte attendu soit immédiatement visible.
        run_event.then(
            fn=show_result, inputs=result_selector,
            outputs=[
                detail_image, detail_metrics, expected_text, extracted_text,
                raw_output, markdown_output, html_source_output, diff_output, prompt_output,
            ],
        )
        result_selector.change(
            fn=show_result, inputs=result_selector,
            outputs=[
                detail_image, detail_metrics, expected_text, extracted_text,
                raw_output, markdown_output, html_source_output, diff_output, prompt_output,
            ],
        )
        previous_button.click(
            fn=lambda label: navigate_result(label, -1), inputs=result_selector, outputs=result_selector
        )
        next_button.click(
            fn=lambda label: navigate_result(label, 1), inputs=result_selector, outputs=result_selector
        )
        import_button.click(
            fn=ui_import_dataset, inputs=dataset_zip,
            outputs=[import_status, dataset_preview, category_selector, quantity_slider],
        )
        merge_runs_button.click(
            fn=ui_import_run_archives, inputs=run_archives,
            outputs=[merge_runs_status, summary_table, dashboard_plot, result_selector, export_file],
        )

    demo.queue(default_concurrency_limit=1, max_size=8)
    demo.launch(
        share=IS_COLAB, inline=IS_COLAB, debug=False, show_error=True, prevent_thread_lock=True,
        theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"), css=APP_CSS,
        auth=GRADIO_AUTH, max_file_size="1GB", allowed_paths=[str(WORK_DIR)],
    )
"""))

cells.append(markdown(r"""
    ## 12. Export optionnel vers Google Drive

    Le bouton de l'onglet Graphiques télécharge déjà un ZIP complet. Cette dernière cellule permet en plus de copier le dernier run dans votre Drive. Elle ne s'exécute que si un benchmark a terminé et si vous activez `COPY_LAST_RUN_TO_DRIVE`.
"""))

cells.append(code(r"""
    # ÉTAPE 12 — Sauvegarde durable facultative
    COPY_LAST_RUN_TO_DRIVE = False

    if COPY_LAST_RUN_TO_DRIVE:
        if not IS_COLAB:
            raise RuntimeError("Google Drive mount est disponible seulement dans Colab.")
        if UI_STATE.get("run_dir") is None:
            raise RuntimeError("Lancez d'abord un benchmark dans Gradio.")
        from google.colab import drive
        drive.mount("/content/drive")
        destination = Path("/content/drive/MyDrive/OCR-Model-Selection-Lab") / UI_STATE["run_id"]
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(UI_STATE["run_dir"], destination)
        print("Run copié vers:", destination)
    else:
        print("Export Drive désactivé. Le ZIP du dernier run reste disponible dans Gradio.")
"""))


notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"name": "OCR_Model_Selection_Lab.ipynb", "provenance": []},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.x"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(f"Wrote {OUTPUT} with {len(cells)} cells")
