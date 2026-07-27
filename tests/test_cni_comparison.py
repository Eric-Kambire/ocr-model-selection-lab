"""Tests du score et des états de comparaison CNI."""

from ocr_benchmark.cni_comparison import compare_cni_extraction, field_state_map


def test_comparison_scores_expected_values_and_normalises_dates_and_accents():
    label = {
        "cin": "XA 12-34",
        "prenom": "ÉRIC",
        "nom": "KABIRE",
        "date_naissance": "01/02/1990",
        "field_confidence": {"cin": 100, "prenom": 98},
    }
    extraction = {
        "cin_fusionne": "xa1234",
        "prenom": "ERIC",
        "nom": "AUTRE",
        "date_naissance": "1990-02-01",
    }

    comparison = compare_cni_extraction(label, extraction)
    states = field_state_map(comparison)

    assert comparison["accuracy"] == 3 / 4
    assert comparison["score_status"] == "scored"
    assert states["cin"] == "correct"
    assert states["prenom"] == "correct"
    assert states["nom"] == "different"
    assert states["date_naissance"] == "correct"
    assert states["adresse"] == "reference_missing"


def test_comparison_distinguishes_missing_extraction_from_unavailable_extraction():
    label = {"cin": "XA1234"}

    present_but_blank = compare_cni_extraction(label, {"cin_fusionne": None})
    unavailable = compare_cni_extraction(label, None)

    assert field_state_map(present_but_blank)["cin"] == "extracted_missing"
    assert present_but_blank["accuracy"] == 0.0
    assert field_state_map(unavailable)["cin"] == "extraction_unavailable"
    assert unavailable["accuracy"] is None


def test_comparison_reads_qlickeer_raw_customer_data_without_normalising_file():
    """La réponse API est conservée telle quelle ; le comparateur lit ses clés."""
    qlickeer_response = {
        "response": {
            "status_code": 200,
            "body": {
                "response_data": {
                    "customer": {
                        "customer_data": {
                            "cin_id": "A0000000",
                            "first_name": "PRENOM",
                            "last_name": "NOM",
                            "birth_date": "01/01/1990",
                            "birth_place": "VILLE DE NAISSANCE",
                            "validity_date": "01/01/2030",
                            "address": "ADRESSE TEST",
                            "cin_id_confidence": 100,
                            "first_name_confidence": 99,
                            "last_name_confidence": 98,
                            "birth_date_confidence": 97,
                            "birth_place_confidence": 96,
                            "validity_date_confidence": 95,
                            "address_confidence": 94,
                        }
                    }
                }
            },
        }
    }
    extraction = {
        "cin_fusionne": "A0000000",
        "prenom": "PRENOM",
        "nom": "NOM",
        "date_naissance": "1990-01-01",
        "ville_naissance": "VILLE DE NAISSANCE",
        "date_validite_fusionnee": "2030-01-01",
        "adresse": "ADRESSE TEST",
    }

    comparison = compare_cni_extraction(qlickeer_response, extraction)
    rows = {row["field"]: row for row in comparison["rows"]}

    assert comparison["accuracy"] == 1.0
    assert rows["cin"]["expected"] == "A0000000"
    assert rows["ville_naissance"]["expected"] == "VILLE DE NAISSANCE"
    assert rows["adresse"]["reference_confidence"] == 94
    # La réponse brute n'est pas modifiée et peut être conservée telle quelle.
    assert qlickeer_response["response"]["body"]["response_data"]["customer"]["customer_data"]["cin_id"] == "A0000000"
