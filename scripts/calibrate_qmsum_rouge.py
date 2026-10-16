#!/usr/bin/env python3
"""CPU-only QMSum published-output calibration against the original Perl ROUGE.

See results/metric_calibration_qmsum.md for provenance and dependency preparation.
No metric parameters are tuned to the published numbers. --predictions and
--references also accept another aligned corpus; --require-fixture verifies the
published calibration corpus, without imposing its scores as a pass condition.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import statistics
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
HMNET_REV = "416966c63e3cb7a57dc59b4ce8fa76f11fd048be"
QMSUM_REV = "83d7768c1f2b4dfeb091385d3dc7e239b8e5bb7e"
MATCHSUM_REV = "c7754245a454d0ba3535db0e4cc1a13b3d35680d"
FIXTURE_SHA = {
    "preds": "c2c4ed2cc5be7eb7ea1ee843f05570c18a2562f6908de64c661b4a8d354a3fd9",
    "refs": "32ff266e626e8492ca10e9ddbef046f32621d78f385f59b9c768f368384cd467",
}
# Exactly Evaluation/ROUGEEval.py lines 74 and 94 in pinned HMNet.
SENTENCE_PATTERN = r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s"
# QMSum author links MatchSum's eval_rouge in issue #5 (2021-07-30).
AUTHOR_FLAGS = ["-c", "95", "-r", "1000", "-n", "2", "-m"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def split_summary(text: str) -> str:
    return "\n".join(re.split(SENTENCE_PATTERN, text))


def run_perl(predictions, references, hmnet_root, pyrouge_root, output_prefix):
    """Run pinned author's scorer; preserve lexical, unpadded MatchSum IDs.

    Returns published-style bootstrap Average_F and diagnostic per-case F1.
    The latter can be used for paired/cluster bootstrap in application tests.
    """
    sys.path.insert(0, str(pyrouge_root))
    from pyrouge import Rouge155

    perl_root = hmnet_root / "ThirdParty/ROUGE/ROUGE-1.5.5"
    records = []
    with tempfile.TemporaryDirectory(prefix="qmsum-rouge-") as work:
        base = Path(work)
        for name, summaries in [("dec", predictions), ("ref", references)]:
            directory = base / name
            directory.mkdir()
            for i, summary in enumerate(summaries):
                # HMNet escapes these characters before writing SEE HTML.
                safe = summary.replace("<", "&lt;").replace(">", "&gt;")
                html = Rouge155.convert_text_to_rouge_format(split_summary(safe))
                (directory / f"{i}.{name}").write_text(html, encoding="utf-8")
        config = base / "settings.xml"
        Rouge155.write_config_static(
            str(base / "dec"), r"(\d+).dec", str(base / "ref"), "#ID#.ref",
            str(config), system_id=1,
        )
        # Explicit perl and -I replace chmod/system installation only.
        # -d is an output-only addition for auditable per-example scores.
        command = ["perl", "-I", str(perl_root),
                   str(perl_root / "ROUGE-1.5.5.pl"),
                   "-e", str(perl_root / "data"), *AUTHOR_FLAGS,
                   "-d", "-a", str(config)]
        proc = subprocess.run(command, check=True, capture_output=True, text=True)
        output_prefix.with_suffix(".perl.txt").write_text(proc.stdout)
        output_prefix.with_suffix(".stderr.txt").write_text(proc.stderr)
        averages = {}
        for metric, stat, value in re.findall(
            r"^1 ROUGE-(\S+) Average_([RPF]): ([\d.]+)", proc.stdout, re.M
        ):
            averages.setdefault(metric, {})[stat] = 100 * float(value)
        # Config tasks are 1-based lexical filenames; Perl bootstraps string IDs.
        lexical_indices = sorted(range(len(predictions)), key=lambda i: f"{i}.dec")
        records = [{"index": i} for i in range(len(predictions))]
        for metric, task, recall, precision, f1 in re.findall(
            r"^1 ROUGE-(\S+) Eval (\d+)\.1 R:([\d.]+) P:([\d.]+) F:([\d.]+)",
            proc.stdout, re.M,
        ):
            index = lexical_indices[int(task) - 1]
            records[index][metric] = {
                "recall": float(recall), "precision": float(precision),
                "f1": float(f1),
            }
        for row in records:
            if not all(metric in row for metric in ["1", "2", "L"]):
                raise RuntimeError(f"Missing Perl per-case output: {row['index']}")
        output_prefix.with_suffix(".per_example.json").write_text(
            json.dumps(records, indent=2) + "\n"
        )
        return {
            "reported_average_percent": averages,
            "per_example": records,
            "per_case_macro_f1_percent_from_printed_5dp": {
                metric: 100 * statistics.mean(row[metric]["f1"] for row in records)
                for metric in ["1", "2", "L"]
            },
            "command": command[:-1] + ["<temporary-settings.xml>"],
            "eval_order_original_indices": lexical_indices,
        }


def score_pairs(preds, refs, *, output_prefix=None, hmnet_root=None, pyrouge_root=None):
    """Application API for aligned single-reference strings (CPU, no GPU imports).

    Returns `per_example` in the original order, P/R/F on the [0,1] scale,
    plus `reported_average_percent` (Perl bootstrap aggregates on [0,100]).
    Set output_prefix to retain raw output and per-example artifacts.
    """
    if not preds or len(preds) != len(refs):
        raise ValueError("Predictions/references must be nonempty and aligned")
    if not all(isinstance(x, str) for x in [*preds, *refs]):
        raise TypeError("score_pairs expects one string reference per prediction")
    hmnet_root = Path(hmnet_root or ROOT / "cache/metric_calibration/HMNet")
    pyrouge_root = Path(pyrouge_root or ROOT / "cache/metric_calibration/pyrouge-0.1.3")
    if output_prefix is None:
        with tempfile.TemporaryDirectory(prefix="qmsum-score-") as directory:
            return run_perl(preds, refs, hmnet_root, pyrouge_root, Path(directory) / "scores")
    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    return run_perl(preds, refs, hmnet_root, pyrouge_root, prefix)


def google_diagnostics(predictions, references):
    from rouge_score import rouge_scorer

    result = {}
    for stem in [False, True]:
        scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL", "rougeLsum"],
            use_stemmer=stem, split_summaries=False,
        )
        rows = [scorer.score(split_summary(r), split_summary(p))
                for p, r in zip(predictions, references)]
        result[f"stem_{stem}"] = {
            name: 100 * statistics.mean(row[name].fmeasure for row in rows)
            for name in rows[0]
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path,
                        default=ROOT / "data/raw/qmsum/rouge_calibration/preds.txt")
    parser.add_argument("--references", type=Path,
                        default=ROOT / "data/raw/qmsum/rouge_calibration/refs.txt")
    parser.add_argument("--hmnet-root", type=Path,
                        default=ROOT / "cache/metric_calibration/HMNet")
    parser.add_argument("--pyrouge-root", type=Path,
                        default=ROOT / "cache/metric_calibration/pyrouge-0.1.3")
    parser.add_argument("--output-prefix", type=Path,
                        default=ROOT / "results/metric_calibration_qmsum")
    parser.add_argument("--require-fixture", action="store_true")
    parser.add_argument("--skip-google", action="store_true")
    args = parser.parse_args()
    if os.geteuid() == 0:
        raise RuntimeError("Run as an unprivileged user, not root")
    if sys.prefix == sys.base_prefix:
        raise RuntimeError("Run using the project's virtual environment")
    predictions = args.predictions.read_text(encoding="utf-8").splitlines()
    references = args.references.read_text(encoding="utf-8").splitlines()
    if not predictions or len(predictions) != len(references):
        raise ValueError("Predictions/references must be nonempty and aligned")
    hashes = {"preds": sha256(args.predictions), "refs": sha256(args.references)}
    if args.require_fixture and (hashes != FIXTURE_SHA or len(predictions) != 279):
        raise ValueError("Published QMSum fixture hash/count mismatch")
    hmnet_rev = subprocess.check_output(
        ["git", "-C", str(args.hmnet_root), "rev-parse", "HEAD"], text=True
    ).strip()
    if hmnet_rev != HMNET_REV:
        raise ValueError(f"Expected pinned HMNet {HMNET_REV}, got {hmnet_rev}")
    if sha256(args.pyrouge_root / "pyrouge/Rouge155.py") != (
        "8f0a447c9528ca53fe205371f4788657d36b8a1035eab400551791b60ae5fa8d"
    ):
        raise ValueError("Expected original PyPI pyrouge 0.1.3 Rouge155.py")
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "date": "2026-10-16", "uid": os.geteuid(), "python": platform.python_version(),
        "python_executable": sys.executable,
        "perl_version": subprocess.check_output(
            ["perl", "-e", 'print "$^V"'], text=True
        ),
        "system_perl_modules": subprocess.check_output(
            ["perl", "-MXML::Parser", "-MDB_File", "-e",
             'print "XML::Parser=$XML::Parser::VERSION;DB_File=$DB_File::VERSION"'],
            text=True,
        ),
        "samples": len(predictions), "input_sha256": hashes,
        "input_paths": {"preds": str(args.predictions), "refs": str(args.references)},
        "qmsum_revision": QMSUM_REV, "hmnet_revision": hmnet_rev,
        "matchsum_revision": MATCHSUM_REV, "pyrouge_version": "0.1.3",
        "primary_sources": {
            "author_metric_version": "https://github.com/Yale-LILY/QMSum/issues/5#issuecomment-887155639",
            "author_metric_code": "https://github.com/Yale-LILY/QMSum/issues/5#issuecomment-890003212",
            "flags_and_filenames": f"https://github.com/maszhongming/MatchSum/blob/{MATCHSUM_REV}/metrics.py#L137",
            "sentence_split": f"https://github.com/microsoft/HMNet/blob/{HMNET_REV}/Evaluation/ROUGEEval.py#L70",
            "published_fixture": f"https://github.com/Yale-LILY/QMSum/tree/{QMSUM_REV}/model_output",
            "readme_scores": f"https://github.com/Yale-LILY/QMSum/blob/{QMSUM_REV}/README.md#model-outputs",
            "paper": "https://aclanthology.org/2021.naacl-main.472.pdf",
            "pyrouge_package": "https://pypi.org/project/pyrouge/0.1.3/",
        },
        "sentence_split_regex": SENTENCE_PATTERN,
        "sentence_split_evidence_scope": "HMNet implementation; QMSum issue only pins scorer, not its own sentence splitter",
        "author_flags": AUTHOR_FLAGS,
        "readme_published_f1_percent": {"1": 36.51, "2": 11.41, "L": 31.60},
        "paper_three_run_mean_f1_percent_not_fixture_target": {"1": 36.06, "2": 11.36, "L": 31.27},
    }
    result["perl"] = score_pairs(
        predictions, references, hmnet_root=args.hmnet_root,
        pyrouge_root=args.pyrouge_root, output_prefix=args.output_prefix,
    )
    result["perl"].pop("per_example")  # Already persisted in per_example.json.
    result["delta_vs_readme_percent_points"] = {
        m: result["perl"]["reported_average_percent"][m]["F"] - value
        for m, value in result["readme_published_f1_percent"].items()
    }
    # Record all scorer code/data, including WordNet exception DB and XML modules.
    files = [args.hmnet_root / "Evaluation/ROUGEEval.py"]
    files += [p for p in (args.hmnet_root / "ThirdParty/ROUGE/ROUGE-1.5.5").rglob("*")
              if p.is_file() and p.suffix in [".pl", ".pm", ".db", ".txt", ".exc"]]
    files += list((args.pyrouge_root / "pyrouge").rglob("*.py"))
    result["scorer_file_sha256"] = {str(p): sha256(p) for p in sorted(files)}
    if not args.skip_google:
        result["google_rouge_score_version"] = importlib.metadata.version("rouge-score")
        result["nltk_version"] = importlib.metadata.version("nltk")
        result["google_same_sentence_split_macro_f1_percent"] = google_diagnostics(
            predictions, references
        )
    result["calibration_status"] = (
        "Near published fixture scores; not bit-exact. No parameter search or fitted offsets."
    )
    args.output_prefix.with_suffix(".json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps({k: result[k] for k in [
        "samples", "delta_vs_readme_percent_points", "calibration_status"
    ]}, indent=2))


if __name__ == "__main__":
    main()
