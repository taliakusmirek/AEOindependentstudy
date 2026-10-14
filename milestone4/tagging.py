#!/usr/bin/env python3
"""tagging.py -- Milestone 4, pipeline stage 4 (tagging layer).

Reads a raw Profound citation export and writes ONE tagged snapshot CSV. Appends:
  intent    informational | comparison | recommendation   (from milestone2/prompts.csv Tags)
  cohort    legacy | rebrand-sim                          (rebrand-sim = prompt listed in change_log.csv)
  platform  chatgpt | google-ai-overviews | google-gemini (slug of the engine name)
plus the Milestone 3 classifier's predicted_category and confidence, so the snapshot is the
single input the agent needs.

Why cohorts: at week 9 five new/edited prompts are added (logged in change_log.csv). Their
citations are tagged cohort=rebrand-sim; everything collected before stays cohort=legacy, so
agent.py can compare like with like and the new prompts cannot contaminate the time series.

RUN (weekly, from this directory, after exporting a fresh Profound citations CSV):
  python tagging.py --export ../milestone2/profound_citations_raw.csv
  python tagging.py --export ~/Downloads/profound_citations_data.csv --model hybrid_seed_public

Writes snapshots/<collection-date>.csv (date = the export's own date column, not today's date).
Scope: citation rows whose prompt is in milestone2/prompts.csv or change_log.csv; exact
duplicates (host, url, prompt, platform) are dropped, as in Milestone 2.
"""
import argparse
import re
import sys
from pathlib import Path

import joblib
import pandas as pd

HERE = Path(__file__).resolve().parent
M3 = HERE.parent / "milestone3"
sys.path.insert(0, str(M3))
import classifier as C  # noqa: E402  (Milestone 3 classifier: host/path features + models)

INTENTS = ("informational", "comparison", "recommendation")
SLUGS = {"ChatGPT": "chatgpt", "Google AI Overviews": "google-ai-overviews", "Google Gemini": "google-gemini"}


def slug(name):
    return SLUGS.get(name, re.sub(r"[^a-z0-9]+", "-", str(name).lower()).strip("-"))


def first_intent(tags):
    for t in str(tags).split(","):
        if t.strip().lower() in INTENTS:
            return t.strip().lower()
    return None


def load_change_log(path):
    cols = ["prompt_text", "date", "reason"]
    if not Path(path).exists():
        return pd.DataFrame(columns=cols)
    df = pd.read_csv(path, dtype=str).fillna("")
    missing = set(cols) - set(df.columns)
    if missing:
        sys.exit(f"{path} is missing columns: {sorted(missing)}")
    return df[df["prompt_text"].str.strip() != ""]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--export", required=True, help="raw Profound citations CSV")
    ap.add_argument("--prompts", default=str(HERE.parent / "milestone2" / "prompts.csv"))
    ap.add_argument("--change-log", default=str(HERE / "change_log.csv"))
    ap.add_argument("--out-dir", default=str(HERE / "snapshots"))
    ap.add_argument("--date", default=None, help="YYYY-MM-DD; required only if the export spans several dates")
    ap.add_argument("--model", choices=C.MODELS, default=None, help="classifier variant (default: selected best)")
    args = ap.parse_args()

    raw = pd.read_csv(args.export, encoding="utf-8-sig")
    cit = raw[raw["type"] == "citation"].copy()

    prompts = pd.read_csv(args.prompts)
    intent_map = {r.Prompt: first_intent(r.Tags) for r in prompts.itertuples()}
    log = load_change_log(args.change_log)
    log_prompts = set(log["prompt_text"])
    absent = sorted(p for p in log_prompts if p not in set(cit["prompt"]))
    for p in absent:
        print(f"WARNING: change_log prompt has no citations in this export: {p!r}", file=sys.stderr)

    cit = cit[cit["prompt"].isin(set(intent_map) | log_prompts)]
    dates = sorted(cit["date"].dropna().unique())
    if args.date:
        cit = cit[cit["date"] == args.date]
        snap_date = args.date
    elif len(dates) == 1:
        snap_date = dates[0]
    else:
        sys.exit(f"export spans several dates {dates}; pass --date YYYY-MM-DD")
    if cit.empty:
        sys.exit("no citation rows in scope for that date")
    cit = cit.drop_duplicates(["hostname", "url", "prompt", "platform"]).reset_index(drop=True)

    # --- the three appended tag columns
    cit["intent"] = [intent_map.get(p) or first_intent(t) or "unknown" for p, t in zip(cit["prompt"], cit["tags"])]
    cit["cohort"] = ["rebrand-sim" if p in log_prompts else "legacy" for p in cit["prompt"]]
    cit["platform_slug"] = cit["platform"].map(slug)

    # --- Milestone 3 classifier
    bundle = joblib.load(M3 / "model.joblib")
    name = args.model or bundle["best"]
    feats = pd.DataFrame({"host": cit["hostname"].map(C.normalize_host), "path_text": cit["url"].map(C.path_text)})
    pred, conf = C.predict_with(name, feats, fitted=bundle["finals"].get(name))

    out = pd.DataFrame({
        "date": snap_date, "domain": cit["hostname"], "source_url": cit["url"], "title": cit["title"].fillna(""),
        "prompt": cit["prompt"], "intent": cit["intent"], "cohort": cit["cohort"], "platform": cit["platform_slug"],
        "predicted_category": pred, "confidence": conf.round(4), "model": name,
    })
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{snap_date}.csv"
    out.to_csv(path, index=False)

    print(f"wrote {path}  ({len(out)} citations, model={name})")
    for col in ("cohort", "intent", "platform", "predicted_category"):
        print(f"  {col}: " + ", ".join(f"{k}={v}" for k, v in out[col].value_counts().items()))


if __name__ == "__main__":
    main()
