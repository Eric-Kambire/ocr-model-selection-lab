import json

import pytest
from PIL import Image

from ocr_benchmark.dataset_repository import DatasetRepository


def test_add_labeled_image_copies_image_and_updates_catalog(tmp_path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "dataset.json").write_text("[]", encoding="utf-8")
    source = tmp_path / "source.png"
    Image.new("RGB", (20, 10), "white").save(source)

    record = DatasetRepository(tmp_path).add_labeled_image(
        source,
        "Name: Jane Doe",
        "Handwritten Form",
        "A test form",
    )

    catalog = json.loads((dataset / "dataset.json").read_text(encoding="utf-8"))
    assert catalog == [record]
    assert record["category"] == "handwritten_form"
    assert (tmp_path / record["image_path"]).is_file()


def test_add_labeled_image_rejects_empty_label(tmp_path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "dataset.json").write_text("[]", encoding="utf-8")
    source = tmp_path / "source.png"
    Image.new("RGB", (20, 10), "white").save(source)

    with pytest.raises(ValueError, match="label"):
        DatasetRepository(tmp_path).add_labeled_image(
            source,
            " ",
            "form",
        )
