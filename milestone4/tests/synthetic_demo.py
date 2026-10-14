#!/usr/bin/env python3
"""SYNTHETIC fixtures for testing agent.py and brief.py. NOT real data.

Real weekly snapshots need a fresh Profound export each week. Until a second real snapshot
exists, this builds a fake 'next week' from the first real snapshot with deliberate, known
changes, so the agent's logic can be checked against an answer key:
  * ~a third of 'competitor institution' citations become 'forum/community'  (competitor DOWN, forum UP)
  * 40 rows are tagged cohort=rebrand-sim                                   (must be EXCLUDED from the lens)
  * 30 rows are on a platform absent from the prior snapshot                (must be EXCLUDED from the lens)
Outputs go to tests/example_output/ and are labeled SYNTHETIC.

  python tests/synthetic_demo.py        # regenerate tests/example_output/
"""
import subprocess
import sys
from pathlib import Path

import pandas as pd

M4 = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "example_output"
NOTE = "SYNTHETIC FIXTURE - fabricated changes to test the agent; not real citation data"
N_REBRAND, N_NEWPLAT = 40, 30


def build(out_dir=OUT):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    real = sorted((M4 / "snapshots").glob("*.csv"))
    if not real:
        sys.exit("no real snapshot found; run tagging.py first")
    prior = pd.read_csv(real[0])
    cur = prior.copy()
    cur["date"] = "2099-01-08"; prior = prior.assign(date="2099-01-01")   # dates only need to be ordered
    comp = cur.index[cur["predicted_category"] == "competitor institution"]
    moved = comp[: len(comp) // 3]
    cur.loc[moved, "predicted_category"] = "forum/community"
    legacy_idx = cur.index[cur["cohort"] == "legacy"]
    cur.loc[legacy_idx[-N_REBRAND:], "cohort"] = "rebrand-sim"
    cur.loc[legacy_idx[:N_NEWPLAT], "platform"] = "perplexity"
    a, b = out_dir / "snapshot_a_SYNTHETIC.csv", out_dir / "snapshot_b_SYNTHETIC.csv"
    prior.to_csv(a, index=False); cur.to_csv(b, index=False)
    rep, brief = out_dir / "diff_report.json", out_dir / "brief.md"
    subprocess.run([sys.executable, str(M4 / "agent.py"), "--current", str(b), "--previous", str(a),
                    "--out", str(rep), "--note", NOTE], check=True, capture_output=True)
    subprocess.run([sys.executable, str(M4 / "brief.py"), "--report", str(rep), "--out", str(brief)],
                   check=True, capture_output=True)
    return rep, brief, len(moved)


if __name__ == "__main__":
    rep, brief, n = build()
    print(f"wrote {rep} and {brief} (SYNTHETIC; {n} competitor rows relabeled forum/community)")
