"""Tests sans réseau du parseur URL inspiré de Postman."""

from scripts.qlicker_url_parser_lab import parse_url_to_rows, rows_to_query_pairs


def test_parse_url_keeps_blank_and_duplicate_query_parameters():
    endpoint, rows, message = parse_url_to_rows(
        "http://qlicker.local/api/GetCustomers?page=1&filter=&tag=a&tag=b"
    )

    assert endpoint == "http://qlicker.local/api/GetCustomers"
    assert rows == [["page", "1", True], ["filter", "", True], ["tag", "a", True], ["tag", "b", True]]
    assert "4 paramètre" in message


def test_rows_to_query_pairs_omits_disabled_row():
    assert rows_to_query_pairs([["page", "1", True], ["filter", "", True], ["ignored", "x", False]]) == [
        ("page", "1"),
        ("filter", ""),
    ]
