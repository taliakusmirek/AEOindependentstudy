#!/usr/bin/env python3
"""agent.py -- Milestone 4, pipeline stage 5a (diff agent).

Takes this week's snapshot and the immediately prior snapshot (both written by tagging.py),
computes each category's citation share in each, and flags every category whose share moved
by MORE than a fixed materiality threshold (default 5 percentage points). Writes diff_report.json.

Comparable lens (what keeps the comparison statistically valid):
  * only cohort == legacy rows (rebrand-sim prompts are a different prompt set and are excluded)
  * only prompts present in BOTH snapshots
  * only platforms present in BOTH snapshots (a change in engine mix would otherwise move shares)
Everything excluded is counted in the report, so nothing is dropped silently.

Citation share of a category = its citations / all citations in the lens. The report also gives an
approximate one-sigma sampling noise for each change; citations are not independent (one page is
often cited repeatedly for a prompt), so that figure is a lower bound.

RUN (from this directory):
  python agent.py --current snapshots/2026-10-21.csv --previous snapshots/2026-10-14.csv
  python agent.py --current ... --previous ... --threshold-pp 5 --out diff_report.json
"""
import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

CATEGORIES = ["BC-owned", "earned media", "education directory/ranking site",
              "forum/community", "competitor institution", "other"]
REQUIRED = {"date", "domain", "prompt", "cohort", "platform", "predicted_category"}


def load(path):
    df = pd.read_csv(path)
    missing = REQUIRED - set(df.columns)
    if missing:
        sys.exit(f"{path} is not a tagged snapshot (missing {sorted(missing)})")
    return df


def shares(df):
    n = len(df)
    counts = df["predicted_category"].value_counts()
    return {c: int(counts.get(c, 0)) for c in CATEGORIES}, n


def build_report(cur, prev, cur_path, prev_path, threshold_pp, note=None):
    cur_l, prev_l = cur[cur["cohort"] == "legacy"], prev[prev["cohort"] == "legacy"]
    prompts = set(cur_l["prompt"]) & set(prev_l["prompt"])
    platforms = set(cur_l["platform"]) & set(prev_l["platform"])
    cur_c = cur_l[cur_l["prompt"].isin(prompts) & cur_l["platform"].isin(platforms)]
    prev_c = prev_l[prev_l["prompt"].isin(prompts) & prev_l["platform"].isin(platforms)]
    if cur_c.empty or prev_c.empty:
        sys.exit("comparable lens is empty: no legacy prompts/platforms are shared by both snapshots")

    (cn, ctot), (pn, ptot) = shares(cur_c), shares(prev_c)
    cats = {}
    for c in CATEGORIES:
        after, before = 100 * cn[c] / ctot, 100 * pn[c] / ptot
        p1, p0 = cn[c] / ctot, pn[c] / ptot
        se = 100 * math.sqrt(p1 * (1 - p1) / ctot + p0 * (1 - p0) / ptot)
        cats[c] = {"before_share_pct": round(before, 2), "after_share_pct": round(after, 2),
                   "delta_pp": round(after - before, 2), "delta_se_pp": round(se, 2),
                   "before_n": pn[c], "after_n": cn[c], "flagged": abs(after - before) > threshold_pp}
    flagged = [{"category": c, "before_share_pct": v["before_share_pct"], "after_share_pct": v["after_share_pct"],
                "delta_pp": v["delta_pp"], "direction": "up" if v["delta_pp"] > 0 else "down"}
               for c, v in cats.items() if v["flagged"]]
    flagged.sort(key=lambda f: -abs(f["delta_pp"]))
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "note": note,
        "current_snapshot": {"path": str(cur_path), "date": str(cur["date"].iloc[0]), "rows": int(len(cur))},
        "previous_snapshot": {"path": str(prev_path), "date": str(prev["date"].iloc[0]), "rows": int(len(prev))},
        "threshold_pp": threshold_pp,
        "comparable_lens": {
            "cohort": "legacy", "prompts_in_both": len(prompts), "platforms_in_both": sorted(platforms),
            "rows_current": int(len(cur_c)), "rows_previous": int(len(prev_c)),
            "excluded_current": {"rebrand_sim_rows": int((cur["cohort"] == "rebrand-sim").sum()),
                                 "other_rows": int(len(cur_l) - len(cur_c))},
            "excluded_previous": {"rebrand_sim_rows": int((prev["cohort"] == "rebrand-sim").sum()),
                                  "other_rows": int(len(prev_l) - len(prev_c))}},
        "categories": cats,
        "flagged": flagged,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--current", required=True)
    ap.add_argument("--previous", required=True)
    ap.add_argument("--threshold-pp", type=float, default=5.0)
    ap.add_argument("--out", default="diff_report.json")
    ap.add_argument("--note", default=None, help="free text carried into the report/brief (e.g. 'SYNTHETIC FIXTURE')")
    a = ap.parse_args()
    cur, prev = load(a.current), load(a.previous)
    if str(cur["date"].iloc[0]) <= str(prev["date"].iloc[0]):
        sys.exit("--current must be a LATER snapshot than --previous")
    report = build_report(cur, prev, a.current, a.previous, a.threshold_pp, a.note)
    Path(a.out).write_text(json.dumps(report, indent=2))
    print(f"wrote {a.out}: {len(report['flagged'])} of {len(CATEGORIES)} categories moved more than {a.threshold_pp} pp")
    for f in report["flagged"]:
        print(f"  {f['category']}: {f['before_share_pct']}% -> {f['after_share_pct']}% ({f['delta_pp']:+} pp)")


if __name__ == "__main__":
    main()
