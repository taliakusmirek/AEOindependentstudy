#!/usr/bin/env python3
"""brief.py -- Milestone 4, pipeline stage 5b (brief generator).

Reads diff_report.json (written by agent.py) and writes brief.md with five fixed sections:
  1. Headline   2. Comparable lens   3. Metrics table   4. Diagnosis   5. Recommended action

The text is template-driven and deterministic: every sentence is built from numbers in
diff_report.json plus the fixed wording below, so the brief cannot assert anything the data
does not contain. Diagnoses are framed as hypotheses to check, not established causes.

RUN (from this directory):
  python brief.py --report diff_report.json --out brief.md
"""
import argparse
import json
from pathlib import Path

MEANING = {
    "BC-owned": "BC's own pages (bc.edu and BC-run properties)",
    "earned media": "third-party news and editorial coverage",
    "education directory/ranking site": "college directory, ranking and advising sites",
    "forum/community": "forums and social platforms",
    "competitor institution": "other colleges' and universities' own sites",
    "other": "all other sources",
}
ACTION = {
    ("BC-owned", "up"): "List the bc.edu pages that gained citations and apply what they share (structure, freshness, clear facts) to pages that are not being cited.",
    ("BC-owned", "down"): "Check whether key bc.edu pages (admissions, aid, programs) stopped being cited; audit their crawlability and freshness first.",
    ("earned media", "up"): "Identify which outlets are now cited and whether the coverage is accurate and current.",
    ("earned media", "down"): "Check whether previously cited articles were updated or removed; consider refreshing the underlying facts media draws on.",
    ("education directory/ranking site", "up"): "Audit BC's profile on the directories gaining share (accuracy of cost, aid and program data).",
    ("education directory/ranking site", "down"): "Check whether BC's directory profiles were changed or delisted; confirm data on the major ranking sites is current.",
    ("forum/community", "up"): "Read the cited threads for outdated or wrong claims about BC and consider correcting them through official accounts where the platform allows.",
    ("forum/community", "down"): "No forum action needed unless BC-owned or competitor share absorbed the loss; check which category gained.",
    ("competitor institution", "up"): "Look at which competitor pages are cited on prompts where BC is absent and compare their content with BC's equivalent pages.",
    ("competitor institution", "down"): "Check which prompts stopped citing competitors and whether BC pages took their place.",
    ("other", "up"): "Inspect the new 'other' sources; if one source type recurs, it may deserve its own category.",
    ("other", "down"): "No action needed on its own; check which category gained.",
}


def build(rep):
    thr, lens, cats = rep["threshold_pp"], rep["comparable_lens"], rep["categories"]
    cur, prev, fl = rep["current_snapshot"], rep["previous_snapshot"], rep["flagged"]
    out = []
    if rep.get("note"):
        out += [f"> **{rep['note']}**", ""]
    out += [f"# AEO weekly brief: Boston College citations, {prev['date']} to {cur['date']}", ""]

    # 1. headline
    out += ["## 1. Headline", ""]
    if fl:
        names = "; ".join(f"{f['category']} {f['delta_pp']:+.1f} pp" for f in fl)
        out += [f"{len(fl)} of 6 source categories moved more than {thr:g} percentage points in citation share: {names}.", ""]
    else:
        out += [f"No source category moved more than {thr:g} percentage points in citation share between "
                f"{prev['date']} and {cur['date']}.", ""]

    # 2. comparable lens
    ec, ep = lens["excluded_current"], lens["excluded_previous"]
    out += ["## 2. Comparable lens", "",
            f"Shares compare only like with like: cohort `{lens['cohort']}`, the {lens['prompts_in_both']} prompts present in both "
            f"snapshots, and platforms present in both ({', '.join(lens['platforms_in_both'])}). "
            f"That is {lens['rows_previous']} citations on {prev['date']} and {lens['rows_current']} on {cur['date']}. "
            f"Excluded: {ec['rebrand_sim_rows']} rebrand-sim and {ec['other_rows']} non-comparable citations this week; "
            f"{ep['rebrand_sim_rows']} and {ep['other_rows']} last week.", ""]

    # 3. metrics
    out += ["## 3. Metrics", "",
            "| Category | Before | After | Change (pp) | ~1-sigma noise (pp) | Flagged |",
            "|---|---:|---:|---:|---:|:---:|"]
    for c, v in cats.items():
        out.append(f"| {c} | {v['before_share_pct']:.1f}% ({v['before_n']}) | {v['after_share_pct']:.1f}% ({v['after_n']}) | "
                   f"{v['delta_pp']:+.1f} | {v['delta_se_pp']:.1f} | {'YES' if v['flagged'] else ''} |")
    out += ["", "Counts in parentheses. Noise is a lower bound because repeated citations of the same page are not independent.", ""]

    # 4. diagnosis
    out += ["## 4. Diagnosis", ""]
    if fl:
        for f in fl:
            v = cats[f["category"]]
            sure = "larger than" if abs(v["delta_pp"]) > 2 * v["delta_se_pp"] else "not clearly larger than"
            out.append(f"- **{f['category']}** ({MEANING[f['category']]}) went {f['direction']} from {f['before_share_pct']:.1f}% to "
                       f"{f['after_share_pct']:.1f}%. The change is {sure} twice the approximate sampling noise "
                       f"({v['delta_se_pp']:.1f} pp), so treat it as a hypothesis to verify in the snapshot rows, not a confirmed cause.")
    else:
        out.append("All six categories stayed inside the threshold, so there is nothing to diagnose this week.")
    out.append("")

    # 5. recommended action
    out += ["## 5. Recommended action", ""]
    if fl:
        for f in fl:
            out.append(f"- **{f['category']} ({f['direction']}):** {ACTION[(f['category'], f['direction'])]}")
    else:
        out.append("No action. Keep the weekly cadence and re-check next week.")
    out.append("")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", default="diff_report.json")
    ap.add_argument("--out", default="brief.md")
    a = ap.parse_args()
    Path(a.out).write_text(build(json.loads(Path(a.report).read_text())))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
