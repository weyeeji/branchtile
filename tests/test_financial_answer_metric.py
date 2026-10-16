"""Synthetic fixtures fixed independently of any held-out model predictions."""
import pytest

from financial_answer_metric import (
    ParseError, extract_final, paired_label_sensitivity, parse_value, score_answer, score_pairs,
)


@pytest.mark.parametrize("pred,gold", [
    ("FINAL: 1,250", "1250"), ("FINAL: .125", "0.125"),
    ("FINAL: 2.5e3", "2500"), ("FINAL: −4.5", "-4.5"),
    ("FINAL: (12.5)", "-12.5"), ("FINAL: -$12.5", "$ -12.5"),
    ("FINAL: ($12.5 million)", "$ -12500000"),
    ("FINAL: 1.25 million USD", "$1250000"),
    ("FINAL: EUR 2 thousand", "€2000"),
    ("FINAL: 12.5 percent", "12.5%"),
    ("FINAL: 1250 basis points", "12.5%"),
    ("FINAL: 0.125 ratio", "12.5%"),
    ("FINAL: 2:4", "50%"), ("FINAL: 2 times", "2:1"),
    ("FINAL: 3 million shares", "3000000 shares"),
    ("FINAL: YES", "yes"), ("FINAL: no", "no"),
    ("Reasoning with 991 discarded.\nFINAL: 17", "17"),
    ("FINAL: $ 250", "$ 250\\n"),
])
def test_exact_positive_synthetic(pred, gold):
    row = score_answer(pred, gold)
    assert row["exact_match"] and row["rounded_match"], row


@pytest.mark.parametrize("pred,gold", [
    ("FINAL: 12.5", "12.5%"), ("FINAL: 0.125", "12.5%"),
    ("FINAL: 0.125%", "12.5%"), ("FINAL: 12.5 pp", "12.5%"),
    ("FINAL: EUR 12.5", "$12.5"), ("FINAL: 12.5", "$12.5"),
    ("FINAL: $1.25 billion", "$1.25 million"),
    ("FINAL: -2.5", "2.5"), ("FINAL: yes", "1"),
    ("FINAL: 3 days", "3 years"),
])
def test_unit_sign_and_kind_are_not_guessed(pred, gold):
    assert not score_answer(pred, gold)["exact_match"]


@pytest.mark.parametrize("text", [
    "The answer is 20", "FINAL: 20\nmore text", "FINAL: 20\nFINAL: 21",
    "final: 20", "FINAL:", "<think>FINAL: 20", "<think>\nFINAL: 20",
    "```\nFINAL: 20", "FINAL: **20**", "FINAL: 2 + 18", "FINAL: 20 or 21",
    "FINAL: 20 (or 21)", "FINAL: 1,20", "FINAL: 1.2.3", "FINAL: NaN",
    "FINAL: inf", "FINAL: __import__('os').system('id')", "FINAL: 2**1000000",
    "FINAL: 1e999", "FINAL: 1:0", "FINAL: (-20)", "FINAL: -$-20",
    "FINAL: 20\\n", "FINAL: 20 approximately", "FINAL: 20.",
])
def test_no_salvaging_ambiguous_or_executable_output(text):
    result = score_answer(text, "20")
    assert result["prediction_parse_error"] is not None, result
    assert not result["exact_match"]


def test_completed_reasoning_and_blank_lines():
    assert extract_final("<think>wrong 30</think>\n\nFINAL: 20\n\n") == "20"


def test_bare_gold_explicitly_cannot_verify_units():
    row = score_answer("FINAL: $2.5 million", "2.5")
    assert row["exact_match"] and row["unit_unverified"] and not row["unit_verified"]
    # No guessing that the missing gold units were millions.
    assert not score_answer("FINAL: $2500000", "2.5")["exact_match"]


@pytest.mark.parametrize("pred,gold,expected", [
    ("12.344%", "12.34%", True), ("12.345%", "12.34%", True),
    ("12.346%", "12.34%", False), ("-12.345%", "-12.34%", True),
    ("7.5", "8", True), ("8.5", "8", True),
    ("8.501", "8", False), ("0.49", "0", True),
    ("$1234567", "$1.23 million", True),
])
def test_rounding_sensitivity_is_not_primary_exact(pred, gold, expected):
    row = score_answer("FINAL: " + pred, gold)
    assert row["rounded_match"] is expected
    assert not row["exact_match"]


def test_unsupported_labels_stay_in_denominator():
    rows = score_pairs(["FINAL: 5", "FINAL: 5", "bad"], ["5", "5 or 6", "5"])
    assert rows["denominator"] == 3 and rows["exact_accuracy"] == 1 / 3
    assert rows["gold_parse_failures"] == 1 and rows["prediction_parse_failures"] == 1
    assert rows["per_example"][1]["gold"] == "5 or 6"


def test_shared_label_adversarial_delta_not_independent_cherry_picking():
    row = paired_label_sensitivity("FINAL: 3%", "FINAL: 9%", "3%", ["9%", "3%"])
    assert row["per_label"][0]["gold"] == "3%"
    assert row["per_label"][0]["exact_delta_a_minus_b"] == 1
    assert row["exact_delta_a_minus_b_bounds"] == [-1, 1]
    with pytest.raises(ValueError):
        paired_label_sensitivity("FINAL: 3%", "FINAL: 9%", "3%", ["9%"])


def test_nonfinite_unicode_digits_and_whole_payload_guard():
    for text in ["２０", "١٢", "1_000", "1/2", "1; print(2)"]:
        with pytest.raises(ParseError):
            parse_value(text)
