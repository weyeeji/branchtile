"""Preregistered FINAL-answer metric; NOT FinQA/DocFinQA execution accuracy.

Pure parsing and Decimal arithmetic only: no model/gold program is executed.
Primary: exact_match. Mandatory sensitivity: rounded_match (reference precision).
See results/metric_calibration_docfinqa.md for the frozen protocol and limits.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from collections import Counter
from decimal import Decimal, InvalidOperation, ROUND_HALF_EVEN, localcontext
import re

METRIC_ID = "strict-final-financial-answer-v1-20261016"
NUMBER = r"[+-]?(?:(?:[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)(?:\.[0-9]+)?|\.[0-9]+)(?:[eE][+-]?[0-9]{1,3})?"
CURRENCIES = {
    "$": "USD", "us$": "USD", "usd": "USD", "dollar": "USD", "dollars": "USD",
    "€": "EUR", "eur": "EUR", "euro": "EUR", "euros": "EUR",
    "£": "GBP", "gbp": "GBP", "pound": "GBP", "pounds": "GBP",
    "c$": "CAD", "cad": "CAD", "a$": "AUD", "aud": "AUD", "jpy": "JPY",
}
MAGNITUDES = {"thousand": 3, "million": 6, "billion": 9, "trillion": 12}
UNITS = {
    "%": ("ratio", -2), "percent": ("ratio", -2), "per cent": ("ratio", -2),
    "bp": ("ratio", -4), "bps": ("ratio", -4),
    "basis point": ("ratio", -4), "basis points": ("ratio", -4),
    "ratio": ("ratio", 0), "times": ("ratio", 0), "x": ("ratio", 0),
    "pp": ("percentage_point", 0), "percentage point": ("percentage_point", 0),
    "percentage points": ("percentage_point", 0),
}
for _unit in ["share", "year", "day", "barrel", "ton", "tonne", "employee", "tower"]:
    UNITS[_unit] = UNITS[_unit + "s"] = (_unit, 0)
UNITS.update({"people": ("person", 0), "person": ("person", 0),
              "tbd": ("tbd", 0), "tpd": ("tpd", 0)})


class ParseError(ValueError):
    """A stable machine-readable failure reason."""


@dataclass(frozen=True)
class ParsedAnswer:
    raw: str
    kind: str
    coefficient: str
    value: str
    dimension: str
    explicit_unit: bool
    quantum: str | None
    normalizations: tuple[str, ...] = ()


def extract_final(text: str) -> str:
    """Exactly one FINAL marker, at the last nonempty line, outside reasoning."""
    if not isinstance(text, str):
        raise ParseError("not_string")
    if len(text) > 2_000_000:
        raise ParseError("response_too_long")
    markers = list(re.finditer(r"(?im)^[ \t]*FINAL:", text))
    if len(markers) != 1:
        raise ParseError("missing_final" if not markers else "multiple_final")
    lines = text.rstrip().splitlines()
    if not lines or not re.fullmatch(r"[ \t]*FINAL:[ \t]*\S.*", lines[-1]):
        raise ParseError("final_not_last_or_empty_or_wrong_case")
    before = text[:markers[0].start()]
    depth = 0
    for tag in re.findall(r"</?think>", text):
        depth += -1 if tag == "</think>" else 1
        if depth < 0:
            raise ParseError("malformed_reasoning_tags")
    if depth:
        raise ParseError("unfinished_reasoning")
    if before.count("```") % 2:
        raise ParseError("final_inside_code_fence")
    return lines[-1].split(":", 1)[1].strip()


def _decimal(text: str) -> Decimal:
    if len(text) > 80 or not re.fullmatch(NUMBER, text):
        raise ParseError("invalid_number")
    try:
        value = Decimal(text.replace(",", ""))
    except InvalidOperation as exc:
        raise ParseError("invalid_number") from exc
    if not value.is_finite() or abs(value.adjusted()) > 100 or abs(value.as_tuple().exponent) > 100:
        raise ParseError("number_out_of_bounds")
    return value


def parse_value(text: str, *, gold: bool = False) -> ParsedAnswer:
    """Parse one complete quantity; do not search text for a convenient number."""
    if not isinstance(text, str) or len(text) > 256:
        raise ParseError("invalid_payload_type_or_length")
    raw = text
    normalized = []
    text = text.strip()
    # Documented serialization cleanup, gold only; original gold is always retained.
    if gold and text.endswith(r"\n"):
        text = text[:-2].rstrip()
        normalized.append("gold_literal_trailing_backslash_n")
    if "−" in text:
        text = text.replace("−", "-")
        normalized.append("unicode_minus")
    text = " ".join(text.split())
    if text.lower() in {"yes", "no"}:
        answer = text.lower()
        return ParsedAnswer(raw, "boolean", answer, answer, "boolean", True, None,
                            tuple(normalized))
    negative = text.startswith("(") and text.endswith(")")
    if negative:
        text = text[1:-1].strip()
    # A colon ratio is a small fixed grammar, not an arithmetic expression.
    ratio = re.fullmatch(rf"({NUMBER})\s*:\s*({NUMBER})", text)
    if ratio:
        a, b = map(_decimal, ratio.groups())
        if b == 0 or negative:
            raise ParseError("invalid_ratio")
        with localcontext() as ctx:
            ctx.prec = 180
            value = str(a / b)
        return ParsedAnswer(raw, "numeric", value, value, "ratio", True, None,
                            tuple(normalized))
    currency = None
    # Permit a sign before a currency symbol, but never two signs.
    front_sign = ""
    if text[:1] in {"+", "-"} and re.match(r"[+-](?:US\$|C\$|A\$|\$|€|£)", text, re.I):
        front_sign, text = text[0], text[1:]
    prefixes = sorted(CURRENCIES, key=len, reverse=True)
    for prefix in prefixes:
        if text.lower().startswith(prefix):
            remaining = text[len(prefix):]
            if remaining and (remaining[0].isspace() or remaining[0] in "+-.0123456789"):
                currency, text = CURRENCIES[prefix], remaining.strip()
                break
    match = re.match(NUMBER, text)
    if not match:
        raise ParseError("not_single_quantity")
    number_text = match.group()
    if front_sign and number_text.startswith(("+", "-")):
        raise ParseError("double_sign")
    value = _decimal(front_sign + number_text)
    if negative:
        if number_text.startswith(("+", "-")) or front_sign:
            raise ParseError("double_sign")
        value = value.copy_negate()
    tail = text[match.end():].strip().lower()
    magnitude = 0
    for word, exponent in MAGNITUDES.items():
        if tail == word or tail.startswith(word + " "):
            magnitude, tail = exponent, tail[len(word):].strip()
            break
    if tail in CURRENCIES:
        if currency is not None:
            raise ParseError("duplicate_currency")
        currency, tail = CURRENCIES[tail], ""
    if currency:
        if tail:
            raise ParseError("unsupported_currency_suffix")
        dimension, scale_exponent = currency, magnitude
        explicit = True
    elif tail:
        if tail not in UNITS:
            raise ParseError("unsupported_unit_or_extra_text")
        dimension, extra_exponent = UNITS[tail]
        if magnitude and dimension in {"ratio", "percentage_point"}:
            raise ParseError("scaled_percentage_not_supported")
        scale_exponent, explicit = magnitude + extra_exponent, True
    else:
        dimension, scale_exponent, explicit = "unspecified", magnitude, bool(magnitude)
    with localcontext() as ctx:
        ctx.prec = 180
        scale = Decimal(10) ** scale_exponent
        canonical = value * scale
        quantum = Decimal(10) ** value.as_tuple().exponent * scale
    return ParsedAnswer(raw, "numeric", str(value), str(canonical), dimension, explicit,
                        str(quantum), tuple(normalized))


def score_answer(prediction: str, gold: str) -> dict:
    """Score without changing gold or excluding failures from the denominator."""
    result = {"metric_id": METRIC_ID, "prediction": prediction, "gold": gold,
              "exact_match": False, "rounded_match": False,
              "prediction_parse_error": None, "gold_parse_error": None,
              "unit_verified": False, "unit_unverified": False}
    parsed_pred = parsed_gold = None
    try:
        parsed_pred = parse_value(extract_final(prediction))
        result["parsed_prediction"] = asdict(parsed_pred)
    except ParseError as exc:
        result["prediction_parse_error"] = str(exc)
    try:
        parsed_gold = parse_value(gold, gold=True)
        result["parsed_gold"] = asdict(parsed_gold)
    except ParseError as exc:
        result["gold_parse_error"] = str(exc)
    if parsed_gold is None or parsed_pred is None:
        result["status"] = "gold_parse_failure" if parsed_gold is None else "prediction_parse_failure"
        return result
    if parsed_gold.kind != parsed_pred.kind:
        result["status"] = "kind_mismatch"
        return result
    if parsed_gold.kind == "boolean":
        result["exact_match"] = parsed_gold.value == parsed_pred.value
        result["rounded_match"] = result["exact_match"]
        result["status"] = "correct" if result["exact_match"] else "incorrect"
        return result
    if not parsed_gold.explicit_unit:
        # Bare gold does not tell us whether its number means dollars/millions/%.
        # Compare the written coefficient only. Never guess a factor from a model.
        pred_value, gold_value = Decimal(parsed_pred.coefficient), Decimal(parsed_gold.coefficient)
        result["unit_unverified"] = True
    elif not parsed_pred.explicit_unit or parsed_pred.dimension != parsed_gold.dimension:
        result["status"] = "unit_mismatch"
        return result
    else:
        pred_value, gold_value = Decimal(parsed_pred.value), Decimal(parsed_gold.value)
        result["unit_verified"] = True
    result["exact_match"] = pred_value == gold_value
    result["rounded_match"] = result["exact_match"]
    if parsed_gold.quantum is not None:
        with localcontext() as ctx:
            ctx.prec = 300
            q = Decimal(parsed_gold.quantum)
            rounded = (pred_value / q).quantize(Decimal(1), rounding=ROUND_HALF_EVEN) * q
            result["rounded_match"] = rounded == gold_value
    result["status"] = "correct" if result["exact_match"] else "incorrect"
    return result


def score_pairs(predictions, references) -> dict:
    if len(predictions) != len(references) or not predictions:
        raise ValueError("Expected nonempty aligned prediction/reference sequences")
    rows = [score_answer(p, r) for p, r in zip(predictions, references)]
    return {"metric_id": METRIC_ID, "denominator": len(rows), "per_example": rows,
            "exact_accuracy": sum(r["exact_match"] for r in rows) / len(rows),
            "rounded_accuracy": sum(r["rounded_match"] for r in rows) / len(rows),
            "prediction_parse_failures": sum(r["prediction_parse_error"] is not None for r in rows),
            "gold_parse_failures": sum(r["gold_parse_error"] is not None for r in rows),
            "unit_unverified": sum(r["unit_unverified"] for r in rows),
            "unit_verified": sum(r["unit_verified"] for r in rows),
            "status_counts": dict(Counter(r["status"] for r in rows)),
            "gold_kind_counts": dict(Counter(
                r.get("parsed_gold", {}).get("kind", "unparsed") for r in rows
            ))}


def paired_label_sensitivity(prediction_a, prediction_b, primary_gold, source_variants):
    """Use one common alternate label for both arms; never pick closest labels."""
    if primary_gold not in source_variants:
        raise ValueError("Frozen primary label must be among source variants")
    labels = list(dict.fromkeys([primary_gold, *source_variants]))
    rows = []
    for label in labels:
        a, b = score_answer(prediction_a, label), score_answer(prediction_b, label)
        rows.append({"gold": label, "primary": label == primary_gold,
                     "exact_delta_a_minus_b": int(a["exact_match"]) - int(b["exact_match"]),
                     "rounded_delta_a_minus_b": int(a["rounded_match"]) - int(b["rounded_match"])})
    return {"per_label": rows, **{
        name + "_bounds": [min(r[name] for r in rows), max(r[name] for r in rows)]
        for name in ["exact_delta_a_minus_b", "rounded_delta_a_minus_b"]
    }}
