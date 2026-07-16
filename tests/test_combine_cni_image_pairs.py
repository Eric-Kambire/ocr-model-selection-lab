from pathlib import Path

from PIL import Image

from scripts.combine_cni_image_pairs import combine_vertical, find_image_pairs


def test_find_image_pairs_detects_shared_prefix(tmp_path: Path):
    Image.new("RGB", (100, 50), "red").save(tmp_path / "007_CIN_Recto.png")
    Image.new("RGB", (80, 60), "blue").save(tmp_path / "007_CIN_Verso.png")
    Image.new("RGB", (40, 40), "white").save(tmp_path / "008_CIN_Recto.png")

    pairs, unmatched, issues = find_image_pairs(tmp_path)

    assert len(pairs) == 1
    assert pairs[0]["pair_key"] == "007_CIN"
    assert [path.name for path in unmatched] == ["008_CIN_Recto.png"]
    assert any("verso absent" in issue for issue in issues)


def test_combine_vertical_keeps_pixels_and_adds_separator(tmp_path: Path):
    recto = tmp_path / "recto.png"
    verso = tmp_path / "verso.png"
    output = tmp_path / "combined.png"
    Image.new("RGB", (100, 50), "red").save(recto)
    Image.new("RGB", (80, 60), "blue").save(verso)

    combine_vertical(recto, verso, output, separator_px=20)

    with Image.open(output) as combined:
        assert combined.size == (100, 130)
        assert combined.getpixel((50, 10)) == (255, 0, 0)
        assert combined.getpixel((50, 60)) == (255, 255, 255)
        assert combined.getpixel((50, 100)) == (0, 0, 255)
