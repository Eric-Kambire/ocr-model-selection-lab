"""Generate six self-contained Colab notebooks (two OCR models per notebook).

The generated notebooks deliberately keep the execution path simple: one model is
loaded, smoke-tested, benchmarked, and unloaded before the next model starts.
They are independent of the local Gradio application and never clone this repo.
"""

from __future__ import annotations

import json
import textwrap
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks"

PAIRS = [
    ("01_classic_ocr", "EasyOCR", "PP-OCRv6"),
    ("02_transformers_documents", "GLM-OCR", "Granite Docling 258M"),
    ("03_paddle_qwen", "PaddleOCR-VL 1.6", "Qwen3.5 OCR 0.8B"),
    ("04_compact_vlm", "MiniCPM-V 4.6", "LightOnOCR-2 1B"),
    ("05_specialized_gpu", "Chandra OCR 2", "dots.ocr"),
    ("06_legacy_localization", "Unlimited-OCR", "LocateAnything-3B"),
]


def src(value: str) -> list[str]:
    return (textwrap.dedent(value).strip("\n") + "\n").splitlines(keepends=True)


def md(value: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": src(value)}


def code(value: str) -> dict:
    return {
        "cell_type": "code", "execution_count": None,
        "metadata": {"collapsed": False}, "outputs": [], "source": src(value),
    }


COMMON_INSTALL = r'''
# Installation reproductible. Exécutez cette cellule dans un runtime Colab frais.
import os, subprocess, sys, importlib

PINNED = [
    "numpy==1.26.4", "pillow==11.1.0",
    "huggingface_hub>=0.30,<1", "datasets>=3.5,<4", "plotly>=5.24,<7",
    "transformers>=4.51,<5", "accelerate>=1.6,<2",
]
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--upgrade", "--force-reinstall", "--no-cache-dir", *PINNED], check=True)
print("Dépendances réinstallées ensemble. IMPORTANT : redémarrez maintenant le runtime Colab (Exécution → Redémarrer la session), puis reprenez à la cellule de vérification.")
'''

NUMPY_GUARD = r'''
# Garde explicite contre NumPy 2.x (incompatibilités ABI avec certaines roues
# OCR/vision). Cette cellule est volontairement séparée pour être identifiable.
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--upgrade", "--force-reinstall", "--no-cache-dir", "numpy<2.0.0"], check=True)
print("numpy<2.0.0 installé. Redémarrez le runtime avant la cellule suivante si Colab indique que NumPy était déjà chargé.")
'''

RUNTIME = r'''
import gc, json, os, platform, subprocess, sys, time, traceback, threading
from pathlib import Path

def _check_binary_stack():
    """Fail early with an actionable message instead of a cryptic ABI traceback."""
    try:
        import numpy as np
        from PIL import Image, ImageOps
        print({"numpy": np.__version__, "pillow": Image.__version__})
        return np, Image
    except (ValueError, ImportError) as exc:
        raise RuntimeError(
            "Incompatibilité binaire NumPy/Pillow. Exécutez la cellule d'installation, "
            "redémarrez le runtime Colab, puis reprenez ici. Détail: " + repr(exc)
        ) from exc

np, Image = _check_binary_stack()

ROOT = Path("/content/ocr_pair_benchmark")
ROOT.mkdir(parents=True, exist_ok=True)
ARTIFACTS = ROOT / "artifacts"; ARTIFACTS.mkdir(exist_ok=True)
TIMEOUT_SECONDS = 120
DOWNLOAD_TIMEOUT_SECONDS = 180
MAX_DOCUMENTS = 30
SELECTED_MODELS = list(MODELS)  # Vous pouvez réduire à un seul modèle.
assert 1 <= len(SELECTED_MODELS) <= 2
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", str(DOWNLOAD_TIMEOUT_SECONDS))
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "30")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

try:
    import torch
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print({"device": DEVICE, "gpu": torch.cuda.get_device_name(0) if DEVICE == "cuda" else None,
           "vram_gb": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2) if DEVICE == "cuda" else None})
except Exception as exc:
    DEVICE = "cpu"; print("Torch indisponible:", repr(exc))
'''

DATASET = r'''
# Jeu de données public, borné et reproductible. Aucun fichier local du projet n'est requis.
from datasets import load_dataset

DATASET_SOURCES = [
    ("hf_multifin", "TheFinAI/MultiFinBen-EnglishOCR", "train", "08cbac5db10834b6cbce428364e0bd8c52eea6fb", 15),
    ("hf_cheques", "arunchincheti/handwritten_and_cheques_dataset", "test", "4d81a7c9b1af2fcbb9abc7c1f85f1c7b789c01a2", 15),
]

def _find_columns(ds):
    image_cols = [c for c in ds.column_names if c.lower() in {"image", "img", "images", "filepath", "file"}]
    text_cols = [c for c in ds.column_names if any(k in c.lower() for k in ("text", "label", "transcription", "ground_truth", "gt"))]
    return image_cols, text_cols

def _as_image(value):
    if isinstance(value, Image.Image): return value.convert("RGB")
    if isinstance(value, dict) and value.get("bytes") is not None:
        import io; return Image.open(io.BytesIO(value["bytes"])).convert("RGB")
    if isinstance(value, (str, Path)): return Image.open(value).convert("RGB")
    return None

def load_cases(limit=MAX_DOCUMENTS):
    cases = []
    for source_name, dataset_id, split, revision, quota in DATASET_SOURCES:
        try:
            ds, status = _run_with_timeout(lambda: load_dataset(dataset_id, split=split, revision=revision, streaming=False, trust_remote_code=False), DOWNLOAD_TIMEOUT_SECONDS)
            if status != "success" or ds is None: raise TimeoutError(status)
            image_cols, text_cols = _find_columns(ds)
            if not image_cols: raise ValueError(f"Aucune colonne image détectée: {ds.column_names}")
            image_col = image_cols[0]; text_col = text_cols[0] if text_cols else None
            for row in ds.select(range(min(quota, len(ds)))):
                image = _as_image(row[image_col])
                if image is None: continue
                expected = str(row[text_col]) if text_col else ""
                path = ARTIFACTS / f"{source_name}_{len(cases):03d}.png"; image.save(path)
                cases.append({"id": path.stem, "image_path": str(path), "expected": expected, "source": source_name})
                if len(cases) >= limit: return cases
        except Exception as exc:
            print(f"Source {source_name} indisponible: {exc!r}")
    return cases

CASES = load_cases()
print(f"Cas chargés: {len(CASES)}. Les scores CER/WER sont calculés uniquement si une vérité terrain existe.")

def _run_with_timeout(fn, seconds):
    box = {}; done = threading.Event()
    def worker():
        try: box["value"] = fn()
        except Exception as exc: box["error"] = repr(exc)
        finally: done.set()
    threading.Thread(target=worker, daemon=True).start(); done.wait(seconds)
    if not done.is_set(): return None, "timeout"
    if "error" in box: return None, box["error"]
    return box.get("value", ""), "success"
'''

ADAPTER = r'''
from dataclasses import dataclass
import re, threading

MODEL_META = {
    "EasyOCR": {"id": "easyocr", "kind": "easyocr", "min_gpu_gb": 0},
    "PP-OCRv6": {"id": "PaddlePaddle/PP-OCRv6_medium_det_safetensors", "kind": "paddle", "min_gpu_gb": 4},
    "GLM-OCR": {"id": "zai-org/GLM-OCR", "kind": "transformers", "min_gpu_gb": 8},
    "Granite Docling 258M": {"id": "ibm-granite/granite-docling-258M", "kind": "transformers", "min_gpu_gb": 8},
    "PaddleOCR-VL 1.6": {"id": "PaddlePaddle/PaddleOCR-VL-1.6", "kind": "transformers", "min_gpu_gb": 8},
    "Qwen3.5 OCR 0.8B": {"id": "loay/English-Document-OCR-Qwen3.5-0.8B", "kind": "gguf", "min_gpu_gb": 0},
    "MiniCPM-V 4.6": {"id": "openbmb/MiniCPM-V-4.6", "kind": "transformers", "min_gpu_gb": 12},
    "LightOnOCR-2 1B": {"id": "lightonai/LightOnOCR-2-1B", "kind": "transformers", "min_gpu_gb": 8},
    "Chandra OCR 2": {"id": "datalab-to/chandra-ocr-2", "kind": "chandra", "min_gpu_gb": 24},
    "dots.ocr": {"id": "rednote-hilab/dots.ocr", "kind": "dots", "min_gpu_gb": 16},
    "Unlimited-OCR": {"id": "baidu/Unlimited-OCR", "kind": "legacy", "min_gpu_gb": 24},
    "LocateAnything-3B": {"id": "nvidia/LocateAnything-3B", "kind": "legacy", "min_gpu_gb": 24},
}

def _norm(s): return re.sub(r"\s+", " ", str(s or "")).strip()

class Adapter:
    def __init__(self, name):
        self.name = name; self.meta = MODEL_META[name]; self.obj = None
    def download(self):
        from huggingface_hub import snapshot_download
        if self.meta["kind"] == "easyocr": return "pip/easyocr"
        return snapshot_download(self.meta["id"], token=os.environ.get("HF_TOKEN"), local_files_only=False)
    def load(self):
        kind = self.meta["kind"]
        if kind == "easyocr":
            import easyocr; self.obj = easyocr.Reader(["fr", "en"], gpu=(DEVICE == "cuda")); return
        if kind == "paddle":
            from paddleocr import PaddleOCR
            try:
                self.obj = PaddleOCR(lang="fr", use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False)
            except TypeError:
                self.obj = PaddleOCR(lang="fr")
            return
        if kind in {"chandra", "dots", "legacy", "gguf"}:
            raise RuntimeError(f"{self.name} nécessite son runtime officiel dédié ({kind}); profil non activé dans ce notebook CORE.")
        from transformers import AutoProcessor
        # Les classes diffèrent selon la fiche officielle; on essaie la classe
        # recommandée puis un fallback compatible sans masquer l'erreur finale.
        from transformers import AutoModelForImageTextToText, AutoModelForVision2Seq
        try:
            from transformers import AutoModelForMultimodalLM
        except ImportError:
            AutoModelForMultimodalLM = None
        dtype = "auto" if DEVICE == "cuda" else None
        kwargs = {"torch_dtype": dtype, "device_map": "auto"} if dtype else {}
        self.processor = AutoProcessor.from_pretrained(self.meta["id"], token=os.environ.get("HF_TOKEN"), trust_remote_code=True)
        classes = ([AutoModelForMultimodalLM] if self.name == "GLM-OCR" and AutoModelForMultimodalLM else []) + [AutoModelForImageTextToText, AutoModelForVision2Seq]
        last = None
        for cls in classes:
            try:
                self.obj = cls.from_pretrained(self.meta["id"], token=os.environ.get("HF_TOKEN"), trust_remote_code=True, **kwargs)
                break
            except Exception as exc:
                last = exc
        if self.obj is None: raise last
    def predict(self, path):
        image = Image.open(path).convert("RGB")
        if self.meta["kind"] == "easyocr": return "\n".join(self.obj.readtext(np.array(image), detail=0))
        if self.meta["kind"] == "paddle":
            result = self.obj.predict(np.array(image)) if hasattr(self.obj, "predict") else self.obj.ocr(np.array(image))
            return _norm(result)
        prompt = "Text Recognition:"
        messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[image], return_tensors="pt", padding=True)
        if DEVICE == "cuda": inputs = {k: v.to("cuda") if hasattr(v, "to") else v for k, v in inputs.items()}
        out = self.obj.generate(**inputs, max_new_tokens=512)
        return _norm(self.processor.batch_decode(out, skip_special_tokens=True)[0])
    def close(self):
        self.obj = None; gc.collect()
        if DEVICE == "cuda":
            import torch; torch.cuda.empty_cache()

# Les téléchargements et chargements sont bornés eux aussi. Ainsi une cellule
# ne reste pas bloquée avant l'inférence.
_raw_download, _raw_load = Adapter.download, Adapter.load
def _bounded_download(self):
    value, status = _run_with_timeout(lambda: _raw_download(self), DOWNLOAD_TIMEOUT_SECONDS)
    if status != "success": raise TimeoutError(f"download_status={status}")
    return value
def _bounded_load(self):
    value, status = _run_with_timeout(lambda: _raw_load(self), DOWNLOAD_TIMEOUT_SECONDS)
    if status != "success": raise TimeoutError(f"load_status={status}")
    return value
Adapter.download, Adapter.load = _bounded_download, _bounded_load
'''

def pair_adapter_source(names: tuple[str, str]) -> str:
    """Keep only the two selected model definitions in each notebook."""
    marker = "MODEL_META = "
    start = ADAPTER.index(marker)
    end = ADAPTER.index("\n\ndef _norm", start)
    dict_start = start + len(marker)
    metadata = ast.literal_eval(ADAPTER[dict_start:end].strip())
    selected = {name: metadata[name] for name in names}
    return ADAPTER[:start] + marker + repr(selected) + ADAPTER[end:]

BENCH = r'''
def _run_with_timeout(fn, seconds):
    box = {}; done = threading.Event()
    def worker():
        try: box["value"] = fn()
        except Exception as exc: box["error"] = repr(exc)
        finally: done.set()
    threading.Thread(target=worker, daemon=True).start()
    done.wait(seconds)
    if not done.is_set(): return None, "timeout"
    if "error" in box: return None, box["error"]
    return box.get("value", ""), "success"

def _cer(ref, hyp):
    ref, hyp = _norm(ref), _norm(hyp)
    if not ref: return None
    prev = list(range(len(hyp)+1))
    for i, a in enumerate(ref, 1):
        cur = [i]
        for j, b in enumerate(hyp, 1): cur.append(min(cur[-1]+1, prev[j]+1, prev[j-1]+(a != b)))
        prev = cur
    return prev[-1] / max(1, len(ref))

def run_benchmark():
    rows = []; raw_path = ARTIFACTS / "raw_outputs.jsonl"
    for name in SELECTED_MODELS:  # strictement séquentiel: un seul modèle en mémoire
        adapter = Adapter(name); t0 = time.perf_counter()
        row_base = {"model": name, "model_id": adapter.meta["id"]}
        try:
            _, download_status = _run_with_timeout(adapter.download, DOWNLOAD_TIMEOUT_SECONDS)
            if download_status != "success": raise TimeoutError(f"download_status={download_status}")
            _, load_status = _run_with_timeout(adapter.load, DOWNLOAD_TIMEOUT_SECONDS)
            if load_status != "success": raise TimeoutError(f"load_status={load_status}")
            for case in CASES:
                started = time.perf_counter(); output, status = _run_with_timeout(lambda p=case["image_path"]: adapter.predict(p), TIMEOUT_SECONDS)
                record = {**row_base, **case, "status": status, "output": output or "", "latency_s": time.perf_counter()-started,
                          "output_chars": len(output or ""), "cer": _cer(case["expected"], output or "")}
                with raw_path.open("a", encoding="utf-8") as f: f.write(json.dumps(record, ensure_ascii=False)+"\n")
                rows.append(record); print(record)
        except Exception as exc:
            record = {**row_base, "status": "failed_load", "error": repr(exc), "output": "", "latency_s": time.perf_counter()-t0}
            rows.append(record); print(record)
        finally: adapter.close()
    import csv
    result_path = ARTIFACTS / "results.csv"
    fields = sorted({k for row in rows for k in row})
    with result_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    return rows

RESULTS = run_benchmark()
display(RESULTS)
'''

PLOTS = r'''
import plotly.graph_objects as go
if len(RESULTS):
    models = sorted({r.get("model", "") for r in RESULTS})
    fig = go.Figure()
    for model in models:
        vals = [float(r.get("latency_s", 0)) for r in RESULTS if r.get("model") == model]
        fig.add_trace(go.Box(y=vals, name=model))
    fig.update_layout(title="Latence par modèle", yaxis_title="secondes")
    display(fig)
    ok = [r for r in RESULTS if r.get("status") == "success"]
    if ok:
        means = [sum(float(r.get("output_chars", 0)) for r in ok if r.get("model") == m) / max(1, sum(r.get("model") == m for r in ok)) for m in models]
        fig2 = go.Figure(go.Bar(x=models, y=means)); fig2.update_layout(title="Volume moyen de texte extrait", yaxis_title="caractères"); display(fig2)
print(f"Résultats persistés dans {ARTIFACTS}. Les sorties brutes restent disponibles même pour timeout/erreur.")
'''


def make_notebook(number: str, left: str, right: str) -> dict:
    extras = {
        "01_classic_ocr": ["easyocr>=1.7,<2", "paddleocr>=2.9,<3", "paddlepaddle>=3.0,<4"],
        "02_transformers_documents": [], "03_paddle_qwen": ["paddleocr>=2.9,<3", "paddlepaddle>=3.0,<4"],
        "04_compact_vlm": [], "05_specialized_gpu": ["chandra-ocr[hf]"], "06_legacy_localization": [],
    }[number]
    cells = [
        md(f"# OCR benchmark — {left} + {right}\n\nNotebook autonome Colab. **Deux modèles maximum**, chargés l'un après l'autre pour protéger la mémoire. Les poids viennent des dépôts officiels Hugging Face ou du paquet officiel du modèle.\n\nOrdre : installer → vérifier le runtime → télécharger → smoke test → benchmark → graphiques."),
        code(f"MODELS = {left!r}, {right!r}\nprint('Modèles de ce notebook:', MODELS)"),
        code(COMMON_INSTALL),
        code(NUMPY_GUARD),
        code("EXTRA_PACKAGES = " + repr(extras) + "\nif EXTRA_PACKAGES:\n    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--upgrade', *EXTRA_PACKAGES], check=True)\nprint('Dépendances spécifiques:', EXTRA_PACKAGES or 'aucune')"),
        code(RUNTIME),
        md("## Secrets (facultatif)\nAjoutez `HF_TOKEN` dans Colab → Secrets si un modèle gated le demande. Le token n'est jamais affiché. Kaggle reste optionnel : ce notebook utilise par défaut un dataset public Hugging Face."),
        code("try:\n    from google.colab import userdata\n    os.environ['HF_TOKEN'] = userdata.get('HF_TOKEN') or ''\nexcept Exception:\n    os.environ.setdefault('HF_TOKEN', '')\nprint('HF_TOKEN présent:', bool(os.environ.get('HF_TOKEN')))"),
        code(DATASET),
        code(pair_adapter_source((left, right))),
        md("## Téléchargement et smoke test\nCette cellule vérifie réellement le téléchargement, l'instanciation et une inférence sur le premier document. Un `failed_load` est conservé avec l'erreur complète; il n'est pas transformé en faux succès."),
        code("SMOKE = []\nfor name in SELECTED_MODELS:\n    adapter = Adapter(name); started = time.perf_counter()\n    try:\n        location = adapter.download(); adapter.load()\n        if CASES: output, status = _run_with_timeout(lambda: adapter.predict(CASES[0]['image_path']), TIMEOUT_SECONDS)\n        else: output, status = '', 'no_dataset'\n        SMOKE.append({'model': name, 'status': status, 'load_seconds': time.perf_counter()-started, 'output_chars': len(output or ''), 'error': ''})\n    except Exception as exc:\n        SMOKE.append({'model': name, 'status': 'failed_load', 'load_seconds': time.perf_counter()-started, 'output_chars': 0, 'error': repr(exc)})\n    finally: adapter.close()\nprint(SMOKE)"),
        md("## Benchmark sérialisé\nLe benchmark garde chaque sortie brute dans `raw_outputs.jsonl`, y compris un timeout. `latency_s` est le temps par image (chargement exclu), `output_chars` mesure le volume produit, et CER est le taux d'erreur caractère quand le dataset fournit une vérité terrain."),
        code(BENCH),
        code(PLOTS),
        md("## Interprétation\nUn modèle est exploitable seulement si son smoke test est `success`, ses sorties ne sont pas vides et sa latence reste compatible avec votre usage. Un score CER plus bas est meilleur. Pour les modèles marqués `failed_load`, l'erreur indique explicitement le runtime ou la mémoire manquante; ne relancez pas en boucle sans corriger cette cause."),
    ]
    return {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.x"}}, "nbformat": 4, "nbformat_minor": 5}


def main() -> None:
    OUT.mkdir(exist_ok=True)
    for number, left, right in PAIRS:
        path = OUT / f"{number}.ipynb"
        path.write_text(json.dumps(make_notebook(number, left, right), ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
        print(path)


if __name__ == "__main__":
    main()
