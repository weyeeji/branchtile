"""Scoring boundaries must reject incomplete reasoning and missing answers."""
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("screen_application", Path(__file__).resolve().parents[1] / "scripts/screen_application.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_unfinished_reasoning_is_not_an_answer():
    assert module.extract_final("Maybe the answer is 42", True)[0] == ""
    assert module.extract_final("thought</think>Final summary", True)[0] == "Final summary"


def test_joint_requires_every_id_once():
    assert module.parse_joint('{"q0":"first", "q1":"second"}', 2) == (["first", "second"], "ok")
    assert module.parse_joint('{"q0":"first"}', 2)[1] != "ok"
    assert module.parse_joint('{"q0":"first", "q0":"replacement"}', 1)[1] == "duplicate_ids"
    assert module.parse_joint('{"q0":42}', 1)[1] != "ok"
    assert module.parse_joint('["first"]', 1)[1] == "not_object"
    assert module.parse_joint('prose {"q0":"first"}', 1)[1] == "invalid_json"


def test_exact_fence_is_allowed_without_salvaging_partial_json():
    assert module.parse_joint('```json\n{"q0":"first"}\n```', 1) == (["first"], "ok")
    assert module.parse_joint('```json\n{"q0":"first"', 1)[1] != "ok"
