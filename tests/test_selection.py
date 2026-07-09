from main import select_dataset_items


DATASET = [
    {"image_path": "a1.png", "ground_truth": "a1", "category": "a"},
    {"image_path": "a2.png", "ground_truth": "a2", "category": "a"},
    {"image_path": "b1.png", "ground_truth": "b1", "category": "b"},
    {"image_path": "b2.png", "ground_truth": "b2", "category": "b"},
]


def test_global_selection_is_reproducible():
    first = select_dataset_items(
        DATASET,
        "Quantité globale",
        2,
        [],
        shuffle=True,
        seed=42,
    )
    second = select_dataset_items(
        DATASET,
        "Quantité globale",
        2,
        [],
        shuffle=True,
        seed=42,
    )
    assert first == second
    assert len(first) == 2


def test_per_category_selection_respects_requested_counts():
    selected = select_dataset_items(
        DATASET,
        "Par catégorie",
        None,
        [["a", 2, 1], ["b", 2, 2]],
        shuffle=False,
        seed=42,
    )
    assert [item["category"] for item in selected] == ["a", "b", "b"]
