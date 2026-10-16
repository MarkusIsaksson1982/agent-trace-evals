import unittest

from agent_trace_evals import rag_eval as r

DOCS = {
    "refund": "Refunds are paid back to the original card within 14 days after the return is approved.",
    "password": "To reset your password open settings, choose security and follow the reset link sent by email.",
    "gdpr": "Personal data is deleted on request. Contact the data protection officer to request deletion of personal data.",
    "ship": "Standard shipping takes three to five working days inside Sweden.",
}


class RagEvalTests(unittest.TestCase):
    def test_chunk_windows_and_overlap(self):
        c = r.chunk(" ".join(f"w{i}" for i in range(10)), size=4, overlap=1)
        self.assertEqual(c[:2], ["w0 w1 w2 w3", "w3 w4 w5 w6"])
        self.assertTrue(c[-1].endswith("w9"))
        self.assertEqual(r.chunk("", size=4, overlap=1), [])
        with self.assertRaises(ValueError):
            r.chunk("a b c", size=3, overlap=3)

    def test_retrieve(self):
        hits = r.retrieve("how do I reset my password", DOCS, k=3)
        self.assertEqual(hits[0][0], "password")
        self.assertEqual(r.retrieve("zebra quantum", DOCS), [])

    def test_metrics(self):
        self.assertEqual(r.recall_at_k(["a", "b", "c"], ["c", "d"], 3), 0.5)
        self.assertEqual(r.mrr(["a", "b", "c"], ["b"]), 0.5)

    def test_pack_context_budget(self):
        self.assertEqual(r.pack_context([("a", "x" * 50), ("b", "y" * 80), ("c", "z" * 30)], 90), ["a", "c"])

    def test_grounding(self):
        self.assertTrue(r.grounding("Refunds take 14 days [refund].", ["refund"])["grounded"])
        g = r.grounding("See [refund] and [made-up].", ["refund"])
        self.assertFalse(g["grounded"])
        self.assertEqual(g["unknown"], ["made-up"])

    def test_evaluate(self):
        cases = [{"query": "reset password link", "relevant": ["password"]},
                 {"query": "delete my personal data", "relevant": ["gdpr"]}]
        self.assertEqual(r.evaluate(cases, DOCS, k=2), {"cases": 2, "recall_at_k": 1.0, "mrr": 1.0})


if __name__ == "__main__":
    unittest.main()
