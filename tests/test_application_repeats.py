"""Ensure unserviceable repeats never look like exact zero-work successes."""
import copy
import importlib.util
from pathlib import Path

import pytest

path = Path(__file__).resolve().parents[1] / "scripts/summarize_application_repeats.py"
spec = importlib.util.spec_from_file_location("repeats", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture():
    runs = []
    for group in [0, 1]:
        for repeat in range(4):
            run = dict(group_index=group, arm="dense", document_id=str(group), repeat=repeat,
                       cache="cold" if repeat < 3 else "warm", question_ids=["q0"],
                       status="unserviceable_context" if group == 1 else "complete")
            if group == 0:
                run.update(group_jct_seconds=[10., 2., 4., 1.][repeat], requests=[{
                    "output_token_ids": [1, 2] if repeat < 3 else [1, 3],
                    "num_cached_tokens": 0, "finish_reason": "stop"}])
            runs.append(run)
    return dict(status="complete", model="fixture", manifest="fixture", manifest_sha256="fixture",
                initialization_seconds=1., config={"groups": "0,1"}, runs=runs)


def test_ineligible_documents_are_preserved_without_empty_equality():
    result = module.summarize(fixture())
    arm = result["aggregates"]["dense"]
    assert arm["documents"] == 2 and arm["ineligible_documents_retained"] == 1
    assert arm["requests"] == arm["cold_token_equal_requests"] == 1
    assert arm["warm_token_equal_requests"] == 0
    assert arm["total_steady_cold_seconds"] == 3.
    assert "cold_repeats_token_equal_by_request" not in result["records"][1]


def test_truncated_or_reordered_request_sets_cannot_claim_identity():
    data = fixture()
    data["runs"][1]["requests"] = []
    with pytest.raises(ValueError, match="request sets"):
        module.summarize(data)
    data = fixture()
    data["runs"][1]["question_ids"] = ["another"]
    with pytest.raises(ValueError, match="identities"):
        module.summarize(data)
    data = fixture()
    data["runs"] = data["runs"][:4]
    with pytest.raises(ValueError, match="denominator"):
        module.summarize(data)
