from pathlib import Path

from scripts.group_cni_pdf_pairs import create_client_directories, find_pdf_pairs


def test_find_pdf_pairs_uses_prefix_before_recto_verso(tmp_path: Path):
    (tmp_path / "007_CIN_Recto.pdf").write_bytes(b"recto")
    (tmp_path / "007_CIN_Verso.pdf").write_bytes(b"verso")
    (tmp_path / "008_CIN_Recto.pdf").write_bytes(b"orphan")
    (tmp_path / "not_a_cni.pdf").write_bytes(b"ignored")

    pairs, unmatched, issues = find_pdf_pairs(tmp_path)

    assert len(pairs) == 1
    assert pairs[0]["pair_key"] == "007_CIN"
    assert {path.name for path in unmatched} == {"008_CIN_Recto.pdf", "not_a_cni.pdf"}
    assert any("verso absent" in issue for issue in issues)


def test_create_client_directories_copies_pair_and_writes_manifest(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    recto = source / "123_CIN_Recto.pdf"
    verso = source / "123_CIN_Verso.pdf"
    recto.write_bytes(b"recto")
    verso.write_bytes(b"verso")
    output = tmp_path / "clients"

    mapping = create_client_directories(
        [{"pair_key": "123_CIN", "recto": recto, "verso": verso}],
        output,
        id_prefix="client-",
        id_length=8,
    )

    client_id = mapping[0]["folder_client_id"]
    assert client_id.startswith("client-")
    assert (output / client_id / recto.name).read_bytes() == b"recto"
    assert (output / client_id / verso.name).read_bytes() == b"verso"
    assert recto.exists()  # Copie par défaut : les originaux restent en place.
    assert (output / "client_mapping.json").is_file()
