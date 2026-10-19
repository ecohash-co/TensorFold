"""The replay comparison counts every record in either run, so a missing result cannot pass the gate by omission."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import replay_run


def record(id_, bucket, text=None, error=None, ids=None):
    r = {"id": id_, "bucket": bucket, "text": text}
    if error:
        r["error"] = error
    if ids is not None:
        r["token_ids"] = ids
    return r


class ReplayCompareTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        self.reference = self.write("ref", [record("s1", "8-16K", "hello"), record("l1", "128-256K", "world")])

    def write(self, name, records):
        path = self.root / f"{name}.jsonl"
        path.write_text("".join(json.dumps(r) + "\n" for r in records))
        return str(path)

    def gate(self, candidate):
        rows = replay_run.pair(self.reference, candidate)
        return rows, replay_run.gate(replay_run.summarize(rows), None, 0.05, 0.01)

    def test_matching_records_pass(self):
        rows, fails = self.gate(self.write("b", [record("s1", "8-16K", "hello"), record("l1", "128-256K", "world")]))
        self.assertEqual(fails, [])
        self.assertTrue(all(r["identical"] for r in rows))

    def test_an_errored_record_fails_its_bucket(self):
        _, fails = self.gate(self.write("b", [record("s1", "8-16K", "hello"), record("l1", "128-256K", error="timeout")]))
        self.assertEqual(len(fails), 1)
        self.assertIn("128-256K", fails[0])

    def test_an_absent_record_fails_its_bucket(self):
        rows, fails = self.gate(self.write("b", [record("s1", "8-16K", "hello")]))
        self.assertEqual([r["missing"] for r in rows if r.get("missing")], ["B"])
        self.assertEqual(len(fails), 1)
        self.assertIn("128-256K", fails[0])

    def test_a_record_only_in_the_candidate_fails_its_bucket(self):
        rows, fails = self.gate(self.write("b", [record("s1", "8-16K", "hello"), record("l1", "128-256K", "world"),
                                                 record("x9", "32-64K", "extra")]))
        self.assertEqual([r["missing"] for r in rows if r.get("missing")], ["A"])
        self.assertEqual(len(fails), 1)
        self.assertIn("32-64K", fails[0])

    def test_token_ids_decide_identity_when_both_runs_have_them(self):
        ref = self.write("ra", [record("s1", "8-16K", "hello", ids=[1, 2, 3])])
        rows = replay_run.pair(ref, self.write("rb", [record("s1", "8-16K", "hello<eos>", ids=[1, 2, 3])]))
        self.assertTrue(rows[0]["identical"])

    def test_compare_gate_exits_nonzero_on_a_missing_bucket(self):
        candidate = self.write("b", [record("s1", "8-16K", "hello")])
        argv = ["compare", self.reference, candidate, "--gate"]
        with contextlib.redirect_stdout(io.StringIO()) as out, self.assertRaises(SystemExit) as stop:
            replay_run.main(argv)
        self.assertEqual(stop.exception.code, 1)
        self.assertIn("GATE: FAIL", out.getvalue())


if __name__ == "__main__":
    unittest.main()
