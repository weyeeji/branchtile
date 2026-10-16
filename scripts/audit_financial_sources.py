#!/usr/bin/env python3
"""Source-only audit of frozen DocFinQA development questions; never scores models.

Run: .venv/bin/python scripts/audit_financial_sources.py
Reads only the named manifest, its eight documents, and official FinQA JSON.
Arithmetic uses a bounded allowlist grammar and exact rational arithmetic.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import os
import re
import sys
from collections import Counter, defaultdict
from decimal import Decimal, localcontext, ROUND_HALF_EVEN, ROUND_HALF_UP
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = Path("data/manifests/docfinqa_public_development8_v1.json")
EXPECTED_MANIFEST_SHA = "17a52d72fde5a2b7e430ff6113905d3eece0589240d30a27794848be226e2d33"
SOURCE_COMMIT = "0f16e2867befa6840783e58be38c9efb9229d742"
COMPANY_ALIASES = {"GS": "goldman sachs", "MAS": "masco corporation",
                   "ANSS": "ansys", "MRO": "marathon oil corporation",
                   "JPM": "jpmorgan chase", "UNP": "union pacific corporation"}
NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)"
CALL = re.compile(r"([a-z_]+)\(([^(),]+),\s*([^(),]+)\)")
OPS = {"add", "subtract", "multiply", "divide", "table_min", "table_max", "table_average"}

# These source-only observations were fixed after reading questions, annotations,
# and report headers, without reading any model outputs. They are not metric rules.
NOTES = {
    "35beedc6": {
        "semantic_status": "question_program_quantity_conflict",
        "unit_status": "question_program_conflict",
        "question_requested_unit": "percentage",
        "program_result_unit": "USD billion",
        "evidence": "Question asks percentage. table_5 is 20,234 million USD; text_13 is 16.77 billion USD. Program divides 20234 by 1000 then subtracts 16.77, yielding a monetary residual 3.464 billion, not a percentage.",
    },
    "be3d1ef2": {
        "semantic_status": "incomplete_question",
        "evidence": "Question ends 'is comprised of' without naming the component. Program selects cabinet net goodwill 181 from table_1 divided by total 1891 in table_6; that numerator cannot be uniquely inferred from the question alone.",
    },
    "cbee1270": {
        "semantic_status": "unit_abbreviation_ambiguous",
        "unit_status": "table_context_resolves_question_typo",
        "question_requested_unit": "literal tpd (ambiguous abbreviation)",
        "program_result_unit": "thousand barrels per day, summed annual rates",
        "evidence": "Question says tpd. Table header explicitly says thousands of barrels per day; footnote lists matching buy/sell volumes 24, 77, 71 mbpd. Program sums annual daily-volume rates; it is not an integrated three-year barrel count.",
    },
    "ede08a0a": {
        "semantic_status": "gold_omits_unit_multiplier",
        "unit_status": "derived_from_explicit_table_dimensions",
        "question_requested_unit": "sales revenue; question does not state display multiplier",
        "program_result_unit": "USD thousand",
        "evidence": "table_7: 1425 thousand barrels/day; table_8: 77.76 USD/barrel; question specifies 365 days. Their product 40444920 is thousand USD. Original gold '$ 40444920\\n' has a dollar sign but omits 'thousand'. No label correction is applied.",
    },
    "16dc3dc7": {
        "semantic_status": "direction_interpretation_ambiguous",
        "evidence": "Program subtracts 2009 impact -1594 from 2008 impact +336, giving +1930 million USD: magnitude of deterioration. Signed 2009-minus-2008 change is -1930. Question 'how much more would the impact be' does not explicitly fix direction; retain this distinction without declaring a replacement label.",
    },
    "7e76d941": {
        "semantic_status": "direction_interpretation_ambiguous",
        "evidence": "Program subtracts 2009 impact -554 from 2008 impact +672, giving +1226 million USD: magnitude of deterioration. Signed 2009-minus-2008 change is -1226. Question 'how much more would the impact be' does not explicitly fix direction; retain this distinction without declaring a replacement label.",
    },
    "237a9c0b": {
        "semantic_status": "malformed_question_operation_unknown",
        "evidence": "Question is the fragment '5 development costs incurred during the period 1654 1251 1030', with no requested operation. Source program averages the three row values; that operation is not specified by the question. Table header provides millions of dollars, but requested statistic remains unknown.",
    },
    "389f92bc": {
        "semantic_status": "question_program_year_conflict",
        "evidence": "Question asks the 2014 cash-investment/operating ratio. In the source table columns [2015, 2014, 2013], operands 3405 and 6823 are in the 2013 column. Program uses positive cash-used magnitude although table cash-used entry is negative. This audit preserves original 0.499 and does not substitute another answer.",
    },
}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", html.unescape(re.sub(r"<[^>]+>", "", text)).lower())


def rational_payload(value: Fraction) -> dict:
    with localcontext() as ctx:
        ctx.prec = 50
        decimal = Decimal(value.numerator) / Decimal(value.denominator)
    return {"numerator": str(value.numerator), "denominator": str(value.denominator),
            "decimal_50_digits": str(decimal)}


def cell_number(raw: str) -> Fraction:
    """Strict number only; recognize documented FinQA duplicate negative rendering."""
    value = html.unescape(re.sub(r"<[^>]+>", "", str(raw))).strip().replace(",", "")
    value = re.sub(r"^\$\s*", "", value).strip()
    duplicate = re.fullmatch(rf"(-\d+(?:\.\d+)?)\s*\(\s*(\d+(?:\.\d+)?)\s*\)", value)
    if duplicate:
        a, b = map(Fraction, duplicate.groups())
        if a != -b:
            raise ValueError("inconsistent duplicated negative cell")
        return a
    negative = re.fullmatch(rf"\(\s*({NUMBER})\s*\)", value)
    if negative:
        if Fraction(negative[1]) < 0:
            raise ValueError("ambiguous double negative")
        return -Fraction(negative[1])
    if not re.fullmatch(NUMBER, value):
        raise ValueError(f"not an unambiguous numeric cell: {raw!r}")
    return Fraction(value)


def interpret(program: str, table: list[list[str]]) -> tuple[Fraction, list[dict]]:
    if not isinstance(program, str) or not program or len(program) > 4096:
        raise ValueError("invalid program length")
    values, trace = [], []
    position = 0
    while position < len(program):
        match = CALL.match(program, position)
        if not match or match[1] not in OPS:
            raise ValueError(f"unsupported syntax/op at position {position}")
        op, a, b = match[1], match[2].strip(), match[3].strip()
        step = {"operation": op, "arguments": [a, b]}
        if op.startswith("table_"):
            if b != "none":
                raise ValueError("table second argument must be none")
            rows = [(i, row) for i, row in enumerate(table[1:], 1) if norm(row[0]) == norm(a)]
            if len(rows) != 1:
                raise ValueError("table row missing or ambiguous")
            row_index, row = rows[0]
            cells = [cell_number(v) for v in row[1:]]
            if not cells:
                raise ValueError("empty table row")
            result = min(cells) if op == "table_min" else max(cells) if op == "table_max" else sum(cells) / len(cells)
            step.update(table_row_index=row_index, raw_cells=row[1:], numeric_cells=[rational_payload(v) for v in cells])
        else:
            def operand(text: str) -> Fraction:
                if re.fullmatch(r"#\d+", text):
                    index = int(text[1:])
                    if index >= len(values):
                        raise ValueError("forward or out-of-bounds reference")
                    return values[index]
                if text.startswith("const_"):
                    text = text[len("const_"):]
                if not re.fullmatch(NUMBER, text):
                    raise ValueError("unsupported scalar operand")
                return Fraction(text)
            left, right = operand(a), operand(b)
            if op == "add":
                result = left + right
            elif op == "subtract":
                result = left - right
            elif op == "multiply":
                result = left * right
            else:
                if right == 0:
                    raise ValueError("division by zero")
                result = left / right
            step["resolved_operands"] = [rational_payload(left), rational_payload(right)]
        if result.numerator.bit_length() > 4096 or result.denominator.bit_length() > 4096:
            raise ValueError("arithmetic size limit")
        values.append(result)
        step["result"] = rational_payload(result)
        trace.append(step)
        if len(trace) > 32:
            raise ValueError("too many steps")
        position = match.end()
        if position < len(program):
            separator = re.match(r",\s*", program[position:])
            if not separator or position + separator.end() == len(program):
                raise ValueError("invalid or trailing separator")
            position += separator.end()
    return values[-1], trace


def rounding_check(value: Fraction, raw: str) -> dict:
    # This is a diagnostic of source display precision, NOT an answer normalizer.
    cleaned = str(raw).strip()
    transforms = []
    for escape in ("\\\\n", "\\n"):
        if escape in cleaned:
            cleaned = cleaned.replace(escape, " ").strip()
            transforms.append(f"remove literal {escape!r}")
    match = re.fullmatch(rf"(\$\s*)?({NUMBER})(\s*%)?", cleaned)
    if not match:
        return {"status": "unknown_display_grammar", "raw": raw, "diagnostic_transforms": transforms}
    target = Fraction(match[2])
    scale = 100 if match[3] else 1
    scaled = value * scale
    precision = len(match[2].split(".", 1)[1]) if "." in match[2] else 0
    with localcontext() as ctx:
        ctx.prec = 60
        number = Decimal(scaled.numerator) / Decimal(scaled.denominator)
        quantum = Decimal(1).scaleb(-precision)
        even = number.quantize(quantum, rounding=ROUND_HALF_EVEN)
        up = number.quantize(quantum, rounding=ROUND_HALF_UP)
    exact = scaled == target
    matches = target in (Fraction(even), Fraction(up))
    return {"raw": raw, "status": "exact_numeric" if exact else "rounded_display_matches" if matches else "display_conflict",
            "explicit_percent_scale": scale, "display_decimals": precision,
            "exact_before_rounding": rational_payload(scaled), "half_even": str(even), "half_up": str(up),
            "rounding_convention_ambiguous": even != up, "diagnostic_transforms": transforms}


def units(question: dict, record: dict) -> dict:
    text = question["question"].lower()
    table = record["table"]
    if "percent" in text or "%" in question["answer"]:
        unit, status, evidence = "dimensionless percentage", "explicit_question", "Question requests a percent; a literal % display multiplies a program ratio by 100."
    elif "ratio" in text:
        unit, status, evidence = "dimensionless ratio", "explicit_question", "Question explicitly requests a ratio."
    elif "expected life" in text:
        unit, status, evidence = "years", "explicit_table_row", "Source row is 'expected life in years'."
    elif "barrels per day" in " ".join(table[0]):
        unit, status, evidence = "thousand barrels per day", "explicit_table_header", str(table[0])
    elif "million" in " ".join(table[0]).lower() or "million" in text or "tax benefits" in text:
        unit, status, evidence = "USD million", "table_or_relevant_context", "Table header, question, or cited dollar-denominated source context states millions."
    else:
        unit, status, evidence = "unknown", "unknown", "No unique dimensional interpretation assigned."
    result = {"question_requested_unit": unit, "program_result_unit": unit, "unit_status": status,
              "evidence": evidence, "bare_numeric_gold": bool(re.fullmatch(NUMBER, question["answer"].strip()))}
    note = NOTES.get(question["question_sha256"][:8], {})
    result.update({key: note[key] for key in ("question_requested_unit", "program_result_unit", "unit_status") if key in note})
    if any(key in note for key in ("question_requested_unit", "program_result_unit", "unit_status")):
        result["evidence"] = note["evidence"]
    return result


def self_test() -> dict:
    accepted = [
        ("divide(5099, 39623)", [], Fraction(5099, 39623)),
        ("divide(20234, const_1000), subtract(#0, 16.77)", [], Fraction("3.464")),
        ("table_min(x, none)", [["x", "year1", "year2"], ["x", "$ -8683 ( 8683 )", "1936"]], Fraction(-8683)),
        ("table_average(x, none)", [["", "1", "2", "3"], ["x", "23", "22", "22"]], Fraction(67, 3)),
    ]
    for program, table, result in accepted:
        assert interpret(program, table)[0] == result
    rejected = ["__import__('os').system('id')", "add(1, 2); print(3)", "divide(1, 0)", "add(#0, 1)",
                "add(1, 2),", "power(2, 3)", "table_average(missing, none)", "add(1, nan)"]
    for program in rejected:
        try:
            interpret(program, [])
        except ValueError:
            pass
        else:
            raise AssertionError(f"unsafe/unsupported program accepted: {program}")
    assert cell_number("$(8,683)") == -8683
    assert rounding_check(Fraction(5099, 39623), "13%")["status"] == "rounded_display_matches"
    assert rounding_check(Fraction(40444920), "$ 40444920\\n")["status"] == "exact_numeric"
    return {"valid_program_cases": len(accepted), "rejected_program_cases": len(rejected), "cell_and_display_checks": 3, "passed": True}


def run(outdir: Path) -> dict:
    assert os.getuid() != 0, "ordinary user required"
    assert sys.prefix != sys.base_prefix, "virtual environment required"
    inputs = []
    def read(path: Path, kind: str):
        data = (ROOT / path).read_bytes()
        inputs.append({"path": str(path), "sha256": digest(data), "bytes": len(data), "kind": kind})
        return data
    manifest_bytes = read(MANIFEST, "frozen_development_manifest")
    assert digest(manifest_bytes) == EXPECTED_MANIFEST_SHA, "frozen manifest changed"
    manifest = json.loads(manifest_bytes)
    commit = json.loads(read(Path("data/raw/finqa/source_commit.json"), "source_revision_metadata"))
    assert commit["sha"] == SOURCE_COMMIT
    by_question = defaultdict(list)
    for split in ("train", "dev", "test"):
        path = Path(f"data/raw/finqa/{split}.json")
        records = json.loads(read(path, "official_source_annotations"))
        for index, record in enumerate(records):
            by_question[record["qa"]["question"]].append((split, index, record))
    packets, reports = [], []
    for group in manifest["groups"]:
        document_bytes = read(Path(group["document_path"]), "frozen_source_report")
        assert digest(document_bytes) == group["document_sha256"]
        document = document_bytes.decode("utf-8")
        header = re.sub(r"\s+", " ", document[:2000]).strip()
        year_match = re.search(r"for the fiscal year ended [a-z]+\s+\d+,?\s+(\d{4})", header, re.I)
        year = year_match[1] if year_match else None
        header_companies = [code for code, alias in COMPANY_ALIASES.items() if alias in header.lower()]
        year_evidence = {"method": "fiscal_year_header" if year else "unknown", "excerpt": year_match[0] if year_match else None}
        # JPM's extracted cover omits the year. Use an explicit company/report
        # running title in the same full document, never a source-ID inferred year.
        if year is None and len(header_companies) == 1:
            alias = COMPANY_ALIASES[header_companies[0]]
            titles = list(re.finditer(re.escape(alias) + r"[^\n]{0,40}?/\s*(\d{4})\s+annual report", re.sub(r"\s+", " ", document), re.I))
            title_years = {match[1] for match in titles}
            if len(title_years) == 1:
                year = next(iter(title_years))
                year_evidence = {"method": "explicit_company_annual_report_running_title", "excerpt": titles[0][0]}
        identity = {"document_id": group["document_id"], "document_sha256": group["document_sha256"],
                    "header_excerpt": header, "header_year": year_match[1] if year_match else None,
                    "verified_report_year": year, "report_year_evidence": year_evidence, "header_company_candidates": header_companies,
                    "manifest_company_year_candidates_unchanged": group["company_year_candidates"],
                    "verified_unique_report_identity": len(header_companies) == 1 and year is not None}
        reports.append(identity)
        normalized_document = norm(document)
        for question in group["questions"]:
            assert digest(question["question"].encode()) == question["question_sha256"]
            sources = []
            for split, index, record in by_question[question["question"]]:
                qa = record["qa"]
                code, source_year = record["id"].split("/")[:2]
                same_company = header_companies == [code]
                association = "same_company_and_report_year" if same_company and year == source_year else "same_company_cross_year_alias_retained" if same_company and year is not None else "identity_unknown_or_conflict_retained"
                value, trace = interpret(qa["program"], record["table"])
                gold_rows = []
                for key, annotation in qa["gold_inds"].items():
                    row_match = re.fullmatch(r"table_(\d+)", key)
                    row = int(row_match[1]) if row_match else None
                    gold_rows.append({"key": key, "annotation": annotation,
                                      "normalized_table_row": record["table"][row] if row is not None and row < len(record["table"]) else None})
                # Independently check label text and numeric anchors in the full report.
                # These anchors support identity; they do not prove contiguous row layout.
                anchors = []
                for entry in gold_rows:
                    row = entry["normalized_table_row"]
                    if row:
                        anchors.append({"key": entry["key"], "row_label": row[0], "row_label_found_normalized": norm(row[0]) in normalized_document})
                sources.append({"split": split, "row_index_zero_based": index, "id": record["id"],
                                "filename": record["filename"], "document_association": association,
                                "source_revision_url": f"https://github.com/czyssrs/FinQA/blob/{SOURCE_COMMIT}/dataset/{split}.json",
                                "question_exact": qa["question"], "answer_original": qa["answer"],
                                "program_original": qa["program"], "exe_ans_original": qa["exe_ans"],
                                "annotation_steps": qa.get("steps"), "annotation_explanation": qa.get("explanation"),
                                "gold_inds_original": qa["gold_inds"], "relevant_table_rows": gold_rows,
                                "table_normalized_original": record["table"], "table_ori_original": record["table_ori"],
                                "pre_text_original": record["pre_text"], "post_text_original": record["post_text"],
                                "report_anchor_checks": anchors, "program_result": rational_payload(value), "program_trace": trace,
                                "exe_ans_check": {"exact_match": value == Fraction(str(qa["exe_ans"])),
                                                  "at_five_decimal_display": rounding_check(value, f"{Decimal(str(qa['exe_ans'])):.5f}")},
                                "source_answer_display_check": rounding_check(value, qa["answer"]),
                                "docfinqa_answer_display_check": rounding_check(value, question["answer"]),
                                "unit_audit": units(question, record)})
            note = NOTES.get(question["question_sha256"][:8], {})
            source_answers = sorted({str(source["answer_original"]) for source in sources})
            packets.append({"question_sha256": question["question_sha256"], "document_id": group["document_id"],
                            "question_original": question["question"], "docfinqa_gold_original": question["answer"],
                            "docfinqa_source_occurrences_original": question["source_occurrences"],
                            "docfinqa_answer_variants_original": question["source_answer_variants"],
                            "exact_question_source_count": len(sources), "source_match_ambiguity": len(sources) != 1,
                            "finqa_answer_variants": source_answers,
                            "source_label_conflict": len(source_answers) > 1 or any(answer != question["answer"] for answer in source_answers),
                            "source_semantic_status": note.get("semantic_status", "no_specific_conflict_identified"),
                            "source_semantic_note": note.get("evidence", "No specific inconsistency identified in this source-only review; this is not a correctness guarantee."),
                            "sources": sources})
    assert len(packets) == 24 and len(reports) == 8
    assert all(packet["sources"] for packet in packets)
    outdir.mkdir(parents=True, exist_ok=True)
    source_counts = Counter(s["document_association"] for p in packets for s in p["sources"])
    display_counts = Counter(p["sources"][0]["docfinqa_answer_display_check"]["status"] for p in packets)
    summary = {
        "schema": "financial_source_audit_v1", "scope": "24 frozen public development questions; source-only, no model-output access",
        "command": ".venv/bin/python scripts/audit_financial_sources.py", "uid": os.getuid(),
        "python_executable": sys.executable, "virtual_environment": sys.prefix,
        "manifest_sha256": EXPECTED_MANIFEST_SHA, "finqa_revision": SOURCE_COMMIT,
        "script_sha256": digest(Path(__file__).read_bytes()), "inputs_read": inputs,
        "question_count": len(packets), "report_count": len(reports),
        "source_record_count": sum(len(p["sources"]) for p in packets),
        "questions_without_source": sum(not p["sources"] for p in packets),
        "questions_with_multiple_exact_sources": sum(p["source_match_ambiguity"] for p in packets),
        "source_association_counts": dict(source_counts), "report_identities_verified": sum(r["verified_unique_report_identity"] for r in reports),
        "report_year_evidence_counts": dict(Counter(r["report_year_evidence"]["method"] for r in reports)),
        "questions_with_conflicting_source_labels": sum(p["source_label_conflict"] for p in packets),
        "source_programs_match_exe_ans_at_five_decimals": sum(s["exe_ans_check"]["at_five_decimal_display"]["status"] in {"exact_numeric", "rounded_display_matches"} for p in packets for s in p["sources"]),
        "question_display_status_counts": dict(display_counts),
        "question_semantic_status_counts": dict(Counter(p["source_semantic_status"] for p in packets)),
        "bare_numeric_gold_questions": sum(p["sources"][0]["unit_audit"]["bare_numeric_gold"] for p in packets),
        "unit_status_counts": dict(Counter(p["sources"][0]["unit_audit"]["unit_status"] for p in packets)),
        "interpreter_checks": self_test(),
        "method": {
            "association": "Exact original question text across all official splits, followed by independent company/year header or explicit annual-report running-title checks. All candidate records retained, including year mismatch. No answer-proximity matching.",
            "rounding": "Exact rational arithmetic. Literal percent display uses factor 100; other displays factor 1. Diagnostic rounding at displayed decimals reports half-even and half-up. Five-decimal exe_ans diagnostic is not an evaluation tolerance.",
            "units": "Source-only human observations fixed in script; uncertainty and question/program conflicts preserved. No automatic unit conversion or relabeling.",
            "source_scope": "Official released FinQA annotations and DocFinQA full report text; no fresh SEC PDF verification. Row-label anchor checks are not contiguous-layout proofs.",
            "unknowns": "Two incomplete/malformed questions do not uniquely determine the requested component/statistic; two JPM questions leave signed direction implicit; GS percentage question conflicts with program quantity. Header resolves UNP report year but cross-year exact-question alias remains retained.",
            "unchanged": "No gold, manifest, scoring metric, or model output is modified. This audit does not produce corrected accuracy or replacement labels.",
        },
    }
    for name, value in (("summary.json", summary), ("report_identities.json", reports), ("evidence_packets.json", packets)):
        (outdir / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    fields = ["question_sha256", "document_id", "gold_original", "source_ids", "source_count", "source_label_conflict", "semantic_status", "display_status", "program_decimal", "unit", "unit_status"]
    with (outdir / "question_audit.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for packet in packets:
            first = packet["sources"][0]
            writer.writerow(dict(question_sha256=packet["question_sha256"], document_id=packet["document_id"], gold_original=packet["docfinqa_gold_original"],
                                 source_ids=";".join(source["id"] for source in packet["sources"]), source_count=len(packet["sources"]), source_label_conflict=packet["source_label_conflict"],
                                 semantic_status=packet["source_semantic_status"], display_status=first["docfinqa_answer_display_check"]["status"], program_decimal=first["program_result"]["decimal_50_digits"],
                                 unit=first["unit_audit"]["program_result_unit"], unit_status=first["unit_audit"]["unit_status"]))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=ROOT / "results/financial_source_audit")
    args = parser.parse_args()
    result = run(args.outdir)
    print(json.dumps({key: value for key, value in result.items() if key.endswith("count") or key.endswith("counts") or key == "interpreter_checks"}, indent=2))
