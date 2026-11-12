import unittest

from scholar_daily.benchmark_rerank import select_ids


class RerankerTests(unittest.TestCase):
    def test_local_ids_map_back_to_original_batch_order(self):
        tasks = [{"paragraphs": [{"id": "a"}, {"id": "b"}]}, {"paragraphs": [{"id": "c"}]}]
        outputs = [{"question_id": "q1", "paragraph_ids": ["c"]}, {"question_id": "q0", "paragraph_ids": ["b"]}]
        self.assertEqual(select_ids(outputs, tasks, 1), [["b"], ["c"]])

    def test_unknown_duplicate_and_wrong_count_ids_are_rejected(self):
        tasks = [{"paragraphs": [{"id": "a"}, {"id": "b"}]}]
        for ids in (["a", "a"], ["a", "x"], ["a"]):
            with self.assertRaises(ValueError):
                select_ids([{"question_id": "q0", "paragraph_ids": ids}], tasks, 2)
        with self.assertRaises(ValueError):
            select_ids([{"question_id": "q1", "paragraph_ids": ["a"]}], tasks, 1)

    def test_small_candidate_pool_and_empty_context(self):
        tasks = [{"paragraphs": [{"id": "a"}]}, {"paragraphs": []}]
        self.assertEqual(select_ids([{"question_id": "q0", "paragraph_ids": ["a"]},
                                     {"question_id": "q1", "paragraph_ids": []}], tasks, 10), [["a"], []])


if __name__ == "__main__":
    unittest.main()
