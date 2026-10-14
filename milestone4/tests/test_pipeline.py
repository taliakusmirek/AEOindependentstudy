"""Checks agent.py / brief.py / tagging.py against known answers. Run: python -m unittest discover -s tests -v"""
import json
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import synthetic_demo as S  # noqa: E402

M4 = S.M4


class AgentAndBrief(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.rep_path, cls.brief_path, cls.moved = S.build(cls.tmp.name)
        cls.rep = json.loads(Path(cls.rep_path).read_text())

    def test_flags_exactly_the_planted_changes(self):
        flagged = {f["category"]: f["direction"] for f in self.rep["flagged"]}
        self.assertEqual(flagged, {"competitor institution": "down", "forum/community": "up"})

    def test_before_after_values_are_consistent(self):
        c = self.rep["categories"]
        self.assertAlmostEqual(sum(v["before_share_pct"] for v in c.values()), 100, delta=0.2)
        self.assertAlmostEqual(sum(v["after_share_pct"] for v in c.values()), 100, delta=0.2)
        for v in c.values():
            self.assertAlmostEqual(v["delta_pp"], v["after_share_pct"] - v["before_share_pct"], delta=0.02)

    def test_lens_excludes_rebrand_sim_and_new_platform_rows(self):
        ex = self.rep["comparable_lens"]["excluded_current"]
        self.assertEqual(ex["rebrand_sim_rows"], S.N_REBRAND)
        self.assertGreaterEqual(ex["other_rows"], S.N_NEWPLAT)
        self.assertNotIn("perplexity", self.rep["comparable_lens"]["platforms_in_both"])

    def test_brief_has_five_fixed_sections_in_order(self):
        heads = re.findall(r"^## (\d)\. (.+)$", Path(self.brief_path).read_text(), flags=re.M)
        self.assertEqual([h[1] for h in heads],
                         ["Headline", "Comparable lens", "Metrics", "Diagnosis", "Recommended action"])

    def test_brief_carries_synthetic_banner(self):
        self.assertIn("SYNTHETIC", Path(self.brief_path).read_text().splitlines()[0])

    def test_agent_refuses_out_of_order_snapshots(self):
        a = Path(self.tmp.name) / "snapshot_a_SYNTHETIC.csv"; b = Path(self.tmp.name) / "snapshot_b_SYNTHETIC.csv"
        r = subprocess.run([sys.executable, str(M4 / "agent.py"), "--current", str(a), "--previous", str(b),
                            "--out", str(Path(self.tmp.name) / "x.json")], capture_output=True, text=True)
        self.assertNotEqual(r.returncode, 0)


class Tagging(unittest.TestCase):
    def test_change_log_prompt_becomes_rebrand_sim(self):
        export = M4.parent / "milestone2" / "profound_citations_raw.csv"
        target = pd.read_csv(M4.parent / "milestone2" / "prompts.csv")["Prompt"].iloc[0]
        with tempfile.TemporaryDirectory() as d:
            log = Path(d) / "change_log.csv"
            log.write_text(f'prompt_text,date,reason\n"{target}",2026-11-18,test\n')
            subprocess.run([sys.executable, str(M4 / "tagging.py"), "--export", str(export),
                            "--change-log", str(log), "--out-dir", d], check=True, capture_output=True)
            snap = pd.read_csv(next(Path(d).glob("20*.csv")))
        self.assertEqual(set(snap.loc[snap["cohort"] == "rebrand-sim", "prompt"]), {target})
        self.assertTrue((snap.loc[snap["prompt"] != target, "cohort"] == "legacy").all())
        self.assertEqual(set(snap["platform"]), {"chatgpt", "google-ai-overviews", "google-gemini"})
        self.assertEqual(set(snap["intent"]), {"informational", "comparison", "recommendation"})


if __name__ == "__main__":
    unittest.main()
