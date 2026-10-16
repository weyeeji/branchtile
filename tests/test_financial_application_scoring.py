import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from score_application_docfinqa import completed_answer


class CompletionEligibilityTest(unittest.TestCase):
    def test_request_specific_caps(self):
        run = {"status": "complete", "arm": "dense",
               "requests": [{"finish_reason": "stop"}, {"finish_reason": "length"}]}
        self.assertEqual(completed_answer(run, 0, "FINAL: 1")[0], "FINAL: 1")
        self.assertEqual(completed_answer(run, 1, "FINAL: 1")[0], "")

    def test_joint_and_service_failures(self):
        run = {"status": "complete", "arm": "joint", "requests": [{"finish_reason": "length"}]}
        self.assertTrue(all(completed_answer(run, i, "FINAL: 1")[0] == "" for i in range(4)))
        self.assertEqual(completed_answer({"status": "unserviceable_context"}, 0, "FINAL: 1")[0], "")
        self.assertEqual(completed_answer({"status": "complete", "arm": "dense", "requests": []}, 0, "FINAL: 1")[0], "")


if __name__ == "__main__":
    unittest.main()
