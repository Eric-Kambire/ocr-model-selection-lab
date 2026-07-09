"""Import 30 labeled FUNSD forms from a Kaggle dataset.

The Kaggle copy has some image/annotation pairs in different split folders.
Files are therefore paired by stem across all splits.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi

DATASET = "senju14/ocr-dataset-of-multi-type-documents"


def list_dataset_files(api: KaggleApi) -> list[str]:
    names: list[str] = []
    token = None
    while True:
        response = api.dataset_list_files(
            DATASET,
            page_token=token,
            page_size=200,
        )
        names.extend(file.name for file in response.dataset_files)
        token = response.next_page_token
        if not token:
            return names


def build_pairs(names: list[str]) -> list[tuple[str, str, str]]:
    images: dict[str, str] = {}
    annotations: dict[str, str] = {}
    for name in names:
        path = Path(name)
        if not name.startswith("form/"):
            continue
        if "/images/" in name and path.suffix.lower() == ".png":
            images[path.stem] = name
        elif "/annotations/" in name and path.suffix.lower() == ".json":
            annotations[path.stem] = name
    return [
        (stem, images[stem], annotations[stem])
        for stem in sorted(images.keys() & annotations.keys())
    ]


def annotation_to_text(annotation: dict) -> str:
    entries = [
        item
        for item in annotation.get("form", [])
        if str(item.get("text", "")).strip() and len(item.get("box", [])) == 4
    ]
    entries.sort(key=lambda item: (item["box"][1], item["box"][0]))
    lines: list[list[dict]] = []
    for entry in entries:
        center_y = (entry["box"][1] + entry["box"][3]) / 2
        if not lines:
            lines.append([entry])
            continue
        last_center = sum(
            (item["box"][1] + item["box"][3]) / 2 for item in lines[-1]
        ) / len(lines[-1])
        if abs(center_y - last_center) <= 10:
            lines[-1].append(entry)
        else:
            lines.append([entry])
    rendered = []
    for line in lines:
        line.sort(key=lambda item: item["box"][0])
        rendered.append(" | ".join(str(item["text"]).strip() for item in line))
    return "\n".join(rendered)


def download_file(api: KaggleApi, remote_name: str, destination: Path) -> Path:
    with tempfile.TemporaryDirectory() as temporary:
        api.dataset_download_file(
            DATASET,
            remote_name,
            path=temporary,
            force=True,
            quiet=True,
        )
        candidates = list(Path(temporary).glob("*"))
        if len(candidates) != 1:
            raise RuntimeError(f"Unexpected download for {remote_name}: {candidates}")
        downloaded = candidates[0]
        if downloaded.suffix == ".zip":
            shutil.unpack_archive(downloaded, temporary)
            extracted = [
                path
                for path in Path(temporary).rglob("*")
                if path.is_file() and path != downloaded
            ]
            if len(extracted) != 1:
                raise RuntimeError(f"Unexpected archive for {remote_name}: {extracted}")
            downloaded = extracted[0]
        shutil.copy2(downloaded, destination)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    args = parser.parse_args()

    root = args.project_root.resolve()
    output = root / "dataset" / "kaggle_forms"
    catalog_path = root / "dataset" / "dataset.json"
    output.mkdir(parents=True, exist_ok=True)

    api = KaggleApi()
    names = list_dataset_files(api)
    pairs = build_pairs(names)[: args.count]
    if len(pairs) < args.count:
        raise RuntimeError(f"Only {len(pairs)} labeled pairs are available.")

    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    existing_sources = {item.get("source_id") for item in catalog}
    imported = 0
    for index, (stem, image_remote, annotation_remote) in enumerate(pairs, start=1):
        source_id = f"kaggle:{DATASET}:{stem}"
        if source_id in existing_sources:
            continue
        image_path = output / f"{stem}.png"
        annotation_path = output / f"{stem}.json"
        download_file(api, image_remote, image_path)
        download_file(api, annotation_remote, annotation_path)
        annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
        ground_truth = annotation_to_text(annotation)
        if not ground_truth:
            raise RuntimeError(f"Empty annotation for {stem}")
        catalog.append(
            {
                "image_path": image_path.relative_to(root).as_posix(),
                "ground_truth": ground_truth,
                "category": "handwritten_form",
                "description": (
                    "Formulaire FUNSD rempli manuellement, avec transcription humaine. "
                    f"Source Kaggle MIT: {DATASET}."
                ),
                "source": "kaggle",
                "source_id": source_id,
                "license": "MIT",
                "annotation_path": annotation_path.relative_to(root).as_posix(),
            }
        )
        imported += 1
        print(f"[{index}/{len(pairs)}] {stem}")

    temporary = catalog_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(catalog, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary.replace(catalog_path)
    print(f"Imported {imported} new labeled forms.")


if __name__ == "__main__":
    main()
