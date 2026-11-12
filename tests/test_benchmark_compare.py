import unittest

from scholar_daily.benchmark_compare import paired_delta


class PairedComparisonTests(unittest.TestCase):
    def test_identical_predictions_have_zero_interval(self):
        rows = [{"question_id": "a", "official_answer_f1": 0.3},
                {"question_id": "b", "official_answer_f1": 0.0}]
        result = paired_delta(rows, rows)
        self.assertEqual(result["mean_delta"], 0)
        self.assertEqual(result["bootstrap_95_percentile_interval"], [0, 0])

    def test_known_gain_and_reproducibility(self):
        baseline = [{"question_id": "a", "official_answer_f1": 0}]
        candidate = [{"question_id": "a", "official_answer_f1": 1}]
        result = paired_delta(baseline, candidate)
        self.assertEqual(result["mean_delta"], 1)
        self.assertEqual(result["bootstrap_95_percentile_interval"], [1, 1])
        self.assertEqual(result, paired_delta(baseline, candidate))

    def test_question_mismatch_and_empty_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            paired_delta([], [])
        with self.assertRaises(ValueError):
            paired_delta([{"question_id": "a"}], [{"question_id": "b"}])


if __name__ == "__main__":
    unittest.main()
