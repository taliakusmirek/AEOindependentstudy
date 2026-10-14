#!/usr/bin/env python3
"""classifier.py -- Milestone 3: citation-source classifier (baseline + learned).

Sorts each AI-answer citation (domain + URL path) into one of the six Milestone 2
categories. Two models are built and compared:

  1. baseline       rule-based keyword / domain-pattern matcher (RULES below)
  2. learned        multinomial logistic regression on character n-grams of the host
                    plus word tokens of the URL path, trained on seed_dataset.csv and,
                    in the "learned_seed_public" variant, on extra public labeled
                    domains (UT1 category lists + Hipo US-university list)

RUN COMMANDS (from this directory; reads ../milestone2/seed_dataset.csv):

  python classifier.py run                       # full evaluation + outputs
  python classifier.py predict reddit.com bu.edu [--model hybrid_seed_public]
  python classifier.py agreement                 # score the filled-in manual-check sheet

`run` writes: results.json, classified_output.csv, manual_check_sheet.csv, errors.csv,
model.joblib. Requires: pandas, numpy, scikit-learn, joblib.

Evaluation protocol (see eval_report.pdf):
  * every split is GROUPED BY HOST, so no domain is in both train and test. seed_dataset.csv
    has 524 rows but only 156 hosts (www.bc.edu alone is 165 rows), so a random row split
    would leak and overstate accuracy; a random-split number is reported only to show that.
  * primary result: one stratified 80/20 grouped split (seed chosen by test-set size and
    category coverage, never by model performance).
  * robustness: 5 repeats x 5-fold grouped cross-validation.
  * model choice and classified_output.csv use out-of-fold predictions, so no citation is
    scored by a model that saw its own domain's label.
  * five variants are compared: baseline, LR (seed), LR (seed+public), and "hybrid" versions
    of both LRs in which a hard rule (host under bc.edu -> BC-owned) handles ownership and
    the LR is trained on the other five categories only.
  * manual_check_sheet.csv samples 30 citations from domains that are in the Profound export
    but NOT in seed_dataset.csv. Fill in your_category, then run `agreement`; it scores the
    selected model and every other model against your labels (an out-of-sample comparison).
"""
import argparse
import json
import re
import sys
import warnings
from pathlib import Path
from urllib.parse import urlsplit

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.pipeline import Pipeline

HERE = Path(__file__).resolve().parent
CATEGORIES = [
    "BC-owned",
    "earned media",
    "education directory/ranking site",
    "forum/community",
    "competitor institution",
    "other",
]
SEED = 42
C_VALUE = 10.0        # fixed a priori; not tuned on any test data
PUBLIC_WEIGHT = 0.3   # public rows count less than hand-labeled seed rows
PUBLIC_CAP = 300      # max public rows sampled per category
MODELS = ["baseline", "learned_seed", "learned_seed_public", "hybrid_seed", "hybrid_seed_public"]
# learned_*  : logistic regression over all six categories
# hybrid_*   : BC-ownership rule (host under bc.edu -> BC-owned) + logistic regression over the other five

warnings.filterwarnings("ignore", category=UserWarning)


# ----------------------------------------------------------------------------- helpers
def normalize_host(h):
    h = str(h).strip().lower()
    h = re.sub(r"^https?://", "", h).split("/")[0].split(":")[0]
    return re.sub(r"^www\.", "", h)


def registered_domain(host):
    parts = host.split(".")
    return ".".join(parts[-2:]) if len(parts) >= 2 else host


def is_bc(host):
    return host == "bc.edu" or host.endswith(".bc.edu")


def path_text(url):
    """Word tokens of the URL path (fragment/query dropped)."""
    try:
        path = urlsplit(str(url)).path.lower()
    except ValueError:
        return ""
    return " ".join(re.findall(r"[a-z]{3,}", path))


def load_seed(path):
    df = pd.read_csv(path)
    bad = set(df.category) - set(CATEGORIES)
    if bad:
        sys.exit(f"unknown categories in seed file: {bad}")
    df["host"] = df["domain"].map(normalize_host)
    df["path_text"] = df["source_url"].map(path_text)
    return df


# ------------------------------------------------------------------- 1. rule baseline
FORUM_PATTERNS = ("reddit.", "quora.", "facebook.", "instagram.", "tiktok.", "youtube.",
                  "youtu.be", "twitter.", "linkedin.", "discord.", "forum", "community",
                  "discuss", "collegeconfidential", "wallstreetoasis", "urbanmom", "talk.")
DIRECTORY_PATTERNS = ("niche.", "usnews", "petersons", "collegeboard", "princetonreview",
                      "collegevine", "collegeadvisor", "collegeraptor", "collegexpress",
                      "collegesimply", "cappex", "bestcolleges", "rank", "topcolleges",
                      "scholarship", "admission", "college", "universit", "guide", "compar")
DIRECTORY_PATH_PATTERNS = ("ranking", "best", "compare", "top")
NEWS_PATTERNS = ("news", "times", "post", "journal", "globe", "herald", "tribune",
                 "chronicle", "gazette", "forbes", "wsj", "reuters", "magazine",
                 "newsweek", "nytimes", "insider")


def rule_predict(host, ptext=""):
    """Ordered rules; returns (category, confidence). Confidence is 1.0 when a rule
    fired and 0.5 for the 'other' fallback (rules have no calibrated probability)."""
    if host == "bc.edu" or host.endswith(".bc.edu"):
        return "BC-owned", 1.0
    if any(p in host for p in FORUM_PATTERNS):
        return "forum/community", 1.0
    if host.endswith(".edu"):
        return "competitor institution", 1.0
    if any(p in host for p in DIRECTORY_PATTERNS) or any(p in ptext for p in DIRECTORY_PATH_PATTERNS):
        return "education directory/ranking site", 1.0
    if any(p in host for p in NEWS_PATTERNS):
        return "earned media", 1.0
    return "other", 0.5


# ----------------------------------------------------------------- 2. learned model
def load_public(public_dir, exclude_hosts, cap=PUBLIC_CAP, seed=SEED):
    """Extra labeled domains mapped onto the 6 categories (see eval_report.pdf):
         Hipo US university domains          -> competitor institution
         UT1 forums + social_networks lists  -> forum/community
         UT1 press list (.com/.org/.net)     -> earned media
         UT1 jobsearch + financial lists     -> other
    No public source exists for BC-owned or education directory/ranking site.
    Any public domain sharing a registered domain with a seed host is dropped so public
    data can never leak a seed label."""
    public_dir = Path(public_dir)
    rng = np.random.default_rng(seed)
    excl_reg = {registered_domain(h) for h in exclude_hosts} | {"bc.edu"}
    ip_re = re.compile(r"^\d+\.\d+\.\d+\.\d+$")

    def read_list(names, tlds=None):
        hosts = set()
        for n in names:
            for line in (public_dir / n).read_text(errors="ignore").splitlines():
                h = normalize_host(line)
                if not h or ip_re.match(h) or "." not in h:
                    continue
                if tlds and h.rsplit(".", 1)[-1] not in tlds:
                    continue
                hosts.add(h)
        return hosts

    hipo = set()
    for u in json.loads((public_dir / "hipo_universities.json").read_text()):
        if u.get("alpha_two_code") == "US":
            hipo.update(normalize_host(d) for d in u.get("domains", []))
    pools = {
        "competitor institution": hipo,
        "forum/community": read_list(["ut1_forums.txt", "ut1_social_networks.txt"]),
        "earned media": read_list(["ut1_press.txt"], tlds={"com", "org", "net"}),
        "other": read_list(["ut1_jobsearch.txt", "ut1_financial.txt"], tlds={"com", "org", "net"}),
    }
    rows = []
    for cat, hosts in pools.items():
        hosts = sorted(h for h in hosts if registered_domain(h) not in excl_reg)
        take = rng.permutation(len(hosts))[:cap]
        rows += [(hosts[i], "", cat) for i in take]
    pub = pd.DataFrame(rows, columns=["host", "path_text", "category"])
    pub["source"] = "public"
    return pub


def make_X(df):
    return pd.DataFrame({"host_text": df["host"].values, "path_text": df["path_text"].values})


def build_pipeline():
    pre = ColumnTransformer([
        ("host", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True), "host_text"),
        ("path", TfidfVectorizer(analyzer="word", token_pattern=r"[a-z]{3,}", sublinear_tf=True), "path_text"),
    ])
    clf = LogisticRegression(C=C_VALUE, max_iter=3000, class_weight="balanced")
    return Pipeline([("pre", pre), ("clf", clf)])


def train_model(train_df, public_df=None, hybrid=False):
    if hybrid:  # ownership is handled by rule, so the learner never sees BC-owned rows
        train_df = train_df[train_df["category"] != "BC-owned"]
    parts = [train_df[["host", "path_text", "category"]].assign(w=1.0)]
    if public_df is not None:
        parts.append(public_df[["host", "path_text", "category"]].assign(w=PUBLIC_WEIGHT))
    d = pd.concat(parts, ignore_index=True)
    pipe = build_pipeline()
    pipe.fit(make_X(d), d["category"], clf__sample_weight=d["w"].values)
    return pipe


def model_predict(pipe, df):
    proba = pipe.predict_proba(make_X(df))
    idx = proba.argmax(axis=1)
    return pipe.classes_[idx], proba.max(axis=1)


def fit_variant(model_name, train_df, public_df):
    return train_model(train_df, public_df if model_name.endswith("public") else None,
                       hybrid=model_name.startswith("hybrid"))


def predict_with(model_name, df, train_df=None, public_df=None, fitted=None):
    if model_name == "baseline":
        out = [rule_predict(h, p) for h, p in zip(df["host"], df["path_text"])]
        return np.array([o[0] for o in out], dtype=object), np.array([o[1] for o in out])
    pipe = fitted or fit_variant(model_name, train_df, public_df)
    pred, conf = model_predict(pipe, df)
    pred = np.array(pred, dtype=object)
    if model_name.startswith("hybrid"):
        bc = df["host"].map(is_bc).values
        pred[bc], conf[bc] = "BC-owned", 1.0
    return pred, conf


# ---------------------------------------------------------------------- evaluation
def metrics(y_true, y_pred):
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=CATEGORIES, zero_division=0)
    per = {c: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
           for i, c in enumerate(CATEGORIES)}
    present = s > 0
    y_true_a, y_pred_a = np.asarray(y_true), np.asarray(y_pred)
    non_bc = y_true_a != "BC-owned"
    return {"per_category": per, "accuracy": float(accuracy_score(y_true, y_pred)),
            "accuracy_excl_bc_owned": float((y_true_a[non_bc] == y_pred_a[non_bc]).mean()),
            "macro_f1": float(f[present].mean()),
            "confusion": confusion_matrix(y_true, y_pred, labels=CATEGORIES).tolist()}


def host_level_accuracy(df, y_pred):
    tmp = pd.DataFrame({"host": df["host"].values, "true": df["category"].values, "pred": y_pred})
    g = tmp.groupby("host").agg(true=("true", "first"), pred=("pred", lambda s: s.value_counts().index[0]))
    return float((g.true == g.pred).mean()), int(len(g))


def choose_primary_split(df):
    """First seed whose grouped hold-out has 17-23% of rows (target 20%) with all six
    categories present. Chosen on split shape only -- model performance is never consulted.
    4-way grouped folds are used because one host (bc.edu) holds 165 of 524 rows and must
    sit in train; with 5 folds every other fold is only ~14% of the data."""
    y, groups = df["category"].values, df["host"].values
    for s in range(500):
        sgkf = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=s)
        for tr, te in sgkf.split(df, y, groups):   # scan all folds of this seed
            if 0.17 <= len(te) / len(df) <= 0.23 and df.iloc[te]["category"].nunique() == len(CATEGORIES):
                return s, tr, te
    sys.exit("could not find a grouped split with all categories in test")


def grouped_cv(df, public_df, rep_seed):
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=rep_seed)
    pred = {m: np.empty(len(df), dtype=object) for m in MODELS}
    conf = {m: np.zeros(len(df)) for m in MODELS}
    for tr, te in sgkf.split(df, df["category"].values, df["host"].values):
        for m in MODELS:
            p, c = predict_with(m, df.iloc[te], df.iloc[tr], public_df)
            pred[m][te], conf[m][te] = p, c
    return pred, conf


def run(args):
    out = Path(args.out_dir)
    seed_df = load_seed(args.seed)
    public_df = load_public(args.public_dir, set(seed_df["host"]))
    print(f"seed: {len(seed_df)} rows, {seed_df.host.nunique()} hosts | public: {len(public_df)} rows")

    res = {"config": {"seed": SEED, "C": C_VALUE, "public_weight": PUBLIC_WEIGHT, "public_cap": PUBLIC_CAP},
           "data": {"rows": len(seed_df), "hosts": int(seed_df.host.nunique()),
                    "rows_by_category": seed_df.category.value_counts().to_dict(),
                    "hosts_by_category": seed_df.groupby("category").host.nunique().to_dict(),
                    "public_rows_by_category": public_df.category.value_counts().to_dict()}}

    # --- primary 80/20 grouped split
    split_seed, tr, te = choose_primary_split(seed_df)
    train, test = seed_df.iloc[tr], seed_df.iloc[te]
    res["primary_split"] = {
        "split_seed": split_seed, "train_rows": len(train), "test_rows": len(test),
        "test_fraction": len(test) / len(seed_df), "test_hosts": int(test.host.nunique()),
        "test_hosts_by_category": test.groupby("category").host.nunique().to_dict(),
        "test_rows_by_category": test.category.value_counts().to_dict(),
        "models": {}}
    fitted = {}
    for m in MODELS:
        if m != "baseline":
            fitted[m] = fit_variant(m, train, public_df)
        p, c = predict_with(m, test, train, public_df, fitted.get(m))
        met = metrics(test["category"].values, p)
        met["host_level_accuracy"], met["n_test_hosts"] = host_level_accuracy(test, p)
        res["primary_split"]["models"][m] = met
        print(f"  primary split  {m:20s} acc={met['accuracy']:.3f} macroF1={met['macro_f1']:.3f}")

    # --- leakage demonstration: naive random ROW split (domains shared across train/test)
    rtr, rte = train_test_split(np.arange(len(seed_df)), test_size=0.2, random_state=SEED,
                                stratify=seed_df["category"].values)
    p, _ = predict_with("learned_seed", seed_df.iloc[rte], seed_df.iloc[rtr])
    res["leakage_demo"] = {"random_row_split_accuracy_learned_seed":
                           float(accuracy_score(seed_df.iloc[rte]["category"], p))}
    print(f"  naive random-row split (leaky) learned_seed acc={res['leakage_demo']['random_row_split_accuracy_learned_seed']:.3f}")

    # --- repeated grouped CV (also the source of out-of-fold predictions)
    reps = 5
    cv = {m: {"acc": [], "acc_nobc": [], "macro_f1": [], "f1_by_cat": {c: [] for c in CATEGORIES}}
          for m in MODELS}
    oof0 = None
    for r in range(reps):
        pred, conf = grouped_cv(seed_df, public_df, 100 + r)
        if r == 0:
            oof0 = (pred, conf)
        for m in MODELS:
            mm = metrics(seed_df["category"].values, pred[m])
            cv[m]["acc"].append(mm["accuracy"]); cv[m]["macro_f1"].append(mm["macro_f1"])
            cv[m]["acc_nobc"].append(mm["accuracy_excl_bc_owned"])
            for c in CATEGORIES:
                cv[m]["f1_by_cat"][c].append(mm["per_category"][c]["f1"])
    res["cv"] = {"repeats": reps, "folds": 5, "models": {
        m: {"accuracy_mean": float(np.mean(v["acc"])), "accuracy_std": float(np.std(v["acc"])),
            "accuracy_excl_bc_owned_mean": float(np.mean(v["acc_nobc"])),
            "macro_f1_mean": float(np.mean(v["macro_f1"])), "macro_f1_std": float(np.std(v["macro_f1"])),
            "f1_by_category_mean": {c: float(np.mean(x)) for c, x in v["f1_by_cat"].items()}}
        for m, v in cv.items()}}
    for m in MODELS:
        s = res["cv"]["models"][m]
        print(f"  grouped CV     {m:20s} acc={s['accuracy_mean']:.3f}+/-{s['accuracy_std']:.3f} "
              f"accExclBC={s['accuracy_excl_bc_owned_mean']:.3f} macroF1={s['macro_f1_mean']:.3f}")

    # --- pick the better model by mean grouped-CV accuracy (stable; not the single split)
    best = max(MODELS, key=lambda m: res["cv"]["models"][m]["accuracy_mean"])
    res["best_model"] = best
    print(f"better-performing model (mean grouped-CV accuracy): {best}")

    # --- classified_output.csv: out-of-fold predictions of the best model for every citation
    pred0, conf0 = oof0
    keep = seed_df[["domain", "source_url", "engine"]].copy()
    keep["predicted_category"] = pred0[best]
    keep["confidence"] = np.round(conf0[best], 4)
    # also carry the best *learned* model (the only one with calibrated probabilities),
    # since the rule baseline's confidence is just 1.0 (rule fired) / 0.5 (fallback)
    learned = [m for m in MODELS if m != "baseline"]
    best_learned = max(learned, key=lambda m: res["cv"]["models"][m]["accuracy_mean"])
    res["best_learned_model"] = best_learned
    keep["learned_predicted_category"] = pred0[best_learned]
    keep["learned_confidence"] = np.round(conf0[best_learned], 4)
    keep[["domain", "predicted_category", "confidence", "learned_predicted_category",
          "learned_confidence", "source_url", "engine"]].to_csv(out / "classified_output.csv", index=False)
    oof_metrics = metrics(seed_df["category"].values, pred0[best])
    res["oof_best"] = {"accuracy": oof_metrics["accuracy"], "macro_f1": oof_metrics["macro_f1"],
                       "per_category": oof_metrics["per_category"]}

    # --- errors file for analysis (host-level, all three models, out-of-fold)
    err = seed_df[["domain", "category"]].copy()
    for m in MODELS:
        err[m] = pred0[m]
    err["best_confidence"] = conf0[best]
    err = err[(err[MODELS] != err[["category"]].values).any(axis=1)]
    (err.groupby(["domain", "category"] + MODELS).agg(rows=("domain", "size"),
                                                     best_confidence=("best_confidence", "mean"))
        .reset_index().sort_values("rows", ascending=False).to_csv(out / "errors.csv", index=False))

    # --- final models, trained on ALL seed rows (+ public where the variant uses it)
    finals = {m: fit_variant(m, seed_df, public_df) for m in MODELS if m != "baseline"}
    joblib.dump({"model": finals.get(best), "best": best, "finals": finals}, out / "model.joblib")

    # --- manual-check sheet: 30 citations from domains NEVER in the seed set.
    # The Profound export also holds citations for ~43 other BC-profile prompts that are
    # not part of seed_dataset.csv. Their domains are unseen by every model, and the
    # baseline's rules were written while looking at seed domains, so scoring seed domains
    # would flatter the baseline. Distinct hosts, up to 5 per predicted category.
    raw = Path(args.raw_export)
    res["agreement"] = None
    if raw.exists():
        r = pd.read_csv(raw, encoding="utf-8-sig", usecols=["type", "url", "hostname", "title", "prompt"])
        r = r[(r["type"] == "citation") & ~r["prompt"].isin(set(seed_df["prompt"]))].copy()
        r["host"] = r["hostname"].map(normalize_host)
        r = r[~r["host"].isin(set(seed_df["host"]))].drop_duplicates("host").reset_index(drop=True)
        r["path_text"] = r["url"].map(path_text)
        res["unseen_pool_hosts"] = int(len(r))
        mp = {}
        for m in MODELS:
            mp[m] = predict_with(m, r, fitted=finals.get(m))
            r[f"pred_{m}"], r[f"conf_{m}"] = mp[m][0], np.round(mp[m][1], 4)
        rng = np.random.default_rng(7)
        picks = []
        for c in CATEGORIES:
            pool = r[r[f"pred_{best}"] == c]
            picks += list(pool.index[rng.permutation(len(pool))[:5]])
        rest = [i for i in rng.permutation(r.index) if i not in picks]
        picks += rest[: max(0, 30 - len(picks))]
        sheet = r.loc[picks[:30]].sample(frac=1, random_state=7).reset_index(drop=True)
        sheet.insert(0, "sample_id", range(1, len(sheet) + 1))
        sheet = sheet.assign(domain=sheet["hostname"], source_url=sheet["url"], page_title=sheet["title"].fillna(""),
                             predicted_category=sheet[f"pred_{best}"], confidence=sheet[f"conf_{best}"],
                             your_category="")
        sheet_path = out / "manual_check_sheet.csv"
        filled = sheet_path.exists() and pd.read_csv(sheet_path)["your_category"].notna().any()
        if filled:   # never destroy the student's own labels
            print("manual_check_sheet.csv already has your_category entries -> left untouched "
                  "(manual_check_models.csv also left untouched)")
        else:
            sheet[["sample_id", "domain", "source_url", "page_title", "predicted_category", "confidence",
                   "your_category"]].to_csv(sheet_path, index=False)
            sheet[["sample_id", "domain"] + [f"pred_{m}" for m in MODELS]].to_csv(
                out / "manual_check_models.csv", index=False)
            print(f"manual_check_sheet.csv: {len(sheet)} unseen domains (pool of {len(r)})")
    else:
        print(f"WARNING: raw export {raw} not found; manual_check_sheet.csv not written")

    (out / "results.json").write_text(json.dumps(res, indent=2))
    print("wrote results.json classified_output.csv manual_check_sheet.csv errors.csv model.joblib")


# ----------------------------------------------------------------- predict / agreement
def predict_cmd(args):
    bundle = joblib.load(Path(args.out_dir) / "model.joblib")
    df = pd.DataFrame({"host": [normalize_host(h) for h in args.hosts], "path_text": ""})
    name = args.model or bundle["best"]
    pred, conf = predict_with(name, df, fitted=bundle["finals"].get(name))
    for h, p, c in zip(args.hosts, pred, conf):
        print(f"{h:40s} {p:35s} {c:.3f}")


def agreement_cmd(args):
    out = Path(args.out_dir)
    sheet = pd.read_csv(out / "manual_check_sheet.csv")
    mine = sheet["your_category"].fillna("").str.strip()
    if (mine == "").any():
        sys.exit(f"{(mine == '').sum()} rows of manual_check_sheet.csv still have no your_category")
    bad = sorted(set(mine) - set(CATEGORIES))
    if bad:
        sys.exit(f"your_category values not in the taxonomy: {bad}")
    agree = (mine == sheet["predicted_category"]).sum()
    rate = float(agree) / len(sheet)
    print(f"agreement: {agree}/{len(sheet)} = {rate:.1%}")
    for _, r in sheet[mine != sheet["predicted_category"]].iterrows():
        print(f"  disagree: {r.domain:35s} predicted={r.predicted_category} | you={r.your_category}")
    # score every model against the student's labels (same 30 unseen domains)
    key = pd.read_csv(out / "manual_check_models.csv").set_index("sample_id")
    per_model = {m: float((key.loc[sheet["sample_id"], f"pred_{m}"].values == mine.values).mean())
                 for m in MODELS}
    for m, v in per_model.items():
        print(f"  {m:20s} agreement with you: {v:.1%}")
    res = json.loads((out / "results.json").read_text())
    res["agreement"] = {"n": int(len(sheet)), "agree": int(agree), "rate": rate,
                        "per_model": per_model,
                        "disagreements": [{"domain": r.domain, "predicted": r.predicted_category,
                                           "student": r.your_category}
                                          for _, r in sheet[mine != sheet["predicted_category"]].iterrows()]}
    (out / "results.json").write_text(json.dumps(res, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=str(HERE))
    sub = ap.add_subparsers(dest="cmd")
    r = sub.add_parser("run", help="train, evaluate, and write all outputs")
    r.add_argument("--seed", default=str(HERE.parent / "milestone2" / "seed_dataset.csv"))
    r.add_argument("--public-dir", default=str(HERE / "data" / "public"))
    r.add_argument("--raw-export", default=str(HERE.parent / "milestone2" / "profound_citations_raw.csv"))
    p = sub.add_parser("predict", help="classify hostnames with a saved model")
    p.add_argument("hosts", nargs="+")
    p.add_argument("--model", choices=MODELS, default=None, help="default: the selected best model")
    sub.add_parser("agreement", help="score the filled-in manual_check_sheet.csv")
    args = ap.parse_args()
    {"run": run, "predict": predict_cmd, "agreement": agreement_cmd}.get(args.cmd or "run", run)(
        args if args.cmd else ap.parse_args(["run"]))


if __name__ == "__main__":
    main()
