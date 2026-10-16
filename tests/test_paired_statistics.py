import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from paired_statistics import paired_cluster_interval


def fixture(seeds=(1,)):
    return [dict(document_id=f"d{i}", question_id=f"q{j}", cluster_id=f"c{i}",
                 seed=seed, reference=.5, candidate=score)
            for i, scores in enumerate([[1., 1., 1.], [0.]])
            for j, score in enumerate(scores) for seed in seeds]


class PairedStatisticsTest(unittest.TestCase):
    def test_cluster_weighting_and_replicate_collapse(self):
        one = paired_cluster_interval(fixture(), expected_seeds=[1])
        three = paired_cluster_interval(fixture([1, 2, 3]), expected_seeds=[1, 2, 3], formal=True)
        self.assertEqual(one["questions"], 4)
        self.assertEqual(three["questions"], 4)
        self.assertEqual(one["question_micro"]["paired_difference"], .25)
        self.assertEqual(one["equal_cluster"]["paired_difference"], 0)
        for name in ["question_micro", "equal_cluster"]:
            self.assertEqual(one[name]["percentile_interval"], three[name]["percentile_interval"])
            self.assertEqual(one[name]["percentile_interval"], [-.5, .5])

    def test_zero_width_cannot_certify_noninferiority(self):
        rows = [dict(document_id=f"d{i}", question_id="q", cluster_id=f"c{i}",
                     seed=seed, reference=1., candidate=1.) for i in range(59) for seed in [1, 2, 3]]
        result = paired_cluster_interval(rows, expected_seeds=[1, 2, 3], formal=True)
        self.assertEqual(result["question_micro"]["percentile_interval"], [0., 0.])
        self.assertFalse(result["question_micro"]["formal_noninferiority_supported_by_interval_only"])
        self.assertGreater(result["no_observed_cluster_loss_sensitivity"]["iid_cluster_event_loss_probability_upper_if_zero"], .02)

    def test_missing_duplicate_and_nonfinite_rows_rejected(self):
        for rows, seeds in [(fixture(), [1, 2]), (fixture() + fixture()[:1], [1])]:
            with self.assertRaises(ValueError):
                paired_cluster_interval(rows, expected_seeds=seeds)
        bad = fixture(); bad[0]["candidate"] = float("nan")
        with self.assertRaises(ValueError):
            paired_cluster_interval(bad, expected_seeds=[1])
        with self.assertRaises(ValueError):
            paired_cluster_interval(fixture(), expected_seeds=[1], formal=True)


if __name__ == "__main__":
    unittest.main()
