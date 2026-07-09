import pytest

from evaluator import (
    assess_structure_preservation,
    calculate_cer,
    evaluate_bankmark,
)


def test_cer_can_exceed_one_for_large_hallucination():
    assert calculate_cer("a", "abcdefgh", normalize=False) > 1


def test_absent_structure_is_not_scored_as_preserved():
    metrics = assess_structure_preservation("plain text", "plain text")
    assert metrics["table_preservation_score"] is None
    assert metrics["math_preservation_score"] is None


def test_bank_metrics_are_not_applicable_when_entities_are_absent():
    metrics = evaluate_bankmark("hello", "hello")
    assert metrics["iban_emr"] is None
    assert metrics["amount_emr"] is None
    assert metrics["iban_valid_rate"] is None
    assert metrics["bankmark_score"] == pytest.approx(1.0)


def test_missing_expected_iban_is_a_quality_failure_not_a_validity_success():
    metrics = evaluate_bankmark("FR7630006000011234567890189", "")
    assert metrics["iban_emr"] == 0
    assert metrics["iban_valid_rate"] is None
