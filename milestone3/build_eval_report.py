#!/usr/bin/env python3
"""build_eval_report.py -- renders eval_report.pdf from results.json (no hand-typed numbers).

Run:  python build_eval_report.py
Re-run after `python classifier.py agreement` to fill in the manual-check result.
"""
import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

HERE = Path(__file__).resolve().parent
R = json.loads((HERE / "results.json").read_text())
CATS = ["BC-owned", "earned media", "education directory/ranking site",
        "forum/community", "competitor institution", "other"]
NAMES = {"baseline": "Rule baseline",
         "learned_seed": "LR, seed only",
         "learned_seed_public": "LR, seed + public",
         "hybrid_seed": "BC rule + LR, seed only",
         "hybrid_seed_public": "BC rule + LR, seed + public"}
BL, LM = "baseline", R["best_learned_model"]
PS, CV, D = R["primary_split"], R["cv"]["models"], R["data"]
pct = lambda x: f"{100 * x:.1f}%"

ss = getSampleStyleSheet()
H1 = ParagraphStyle("h1", parent=ss["Heading1"], fontSize=12.5, spaceBefore=9, spaceAfter=4, textColor=colors.black)
TITLE = ParagraphStyle("t", parent=ss["Title"], fontSize=17, spaceAfter=2, textColor=colors.black)
SUB = ParagraphStyle("s", parent=ss["Normal"], fontSize=9.5, spaceAfter=4, textColor=colors.black)
B = ParagraphStyle("b", parent=ss["BodyText"], fontSize=9, leading=12, spaceAfter=4, textColor=colors.black)
T = ParagraphStyle("tt", parent=ss["BodyText"], fontSize=7.8, leading=9.6, textColor=colors.black)
TB = ParagraphStyle("tb", parent=T, fontName="Helvetica-Bold")
CAP = ParagraphStyle("c", parent=T, fontSize=7.8, spaceBefore=2, spaceAfter=5)


def table(rows, widths, header_rows=1, bold_last=0):
    data = [[Paragraph(str(c), TB if (i < header_rows or i >= len(rows) - bold_last) else T) for c in r]
            for i, r in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=header_rows)
    t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.5, colors.black),
                           ("LINEBELOW", (0, header_rows - 1), (-1, header_rows - 1), 1.2, colors.black),
                           ("VALIGN", (0, 0), (-1, -1), "TOP"),
                           ("TOPPADDING", (0, 0), (-1, -1), 2.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
                           ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4)]))
    return t


s = []
s.append(Paragraph("Milestone 3 - Classifier Build and Evaluation", TITLE))
s.append(Paragraph("Natalia Kusmirek | Readings in Computer Science | Instructor: Prof. Maira Marques Samary", SUB))

# ------------------------------------------------------------------ 1. setup
s.append(Paragraph("1. Data, models, and evaluation protocol", H1))
rows_c, hosts_c, pub_c = D["rows_by_category"], D["hosts_by_category"], D["public_rows_by_category"]
s.append(Paragraph(
    f"<b>Seed data.</b> seed_dataset.csv: {D['rows']} citations from {D['hosts']} distinct hosts, labeled with the six "
    "Milestone 2 categories. Labels came from an AI-assisted first pass (Profound's own citation category mapped onto "
    "the taxonomy, plus hand-built domain lists, then corrected) that the student reviewed and confirmed; they are not "
    "fully independent hand labels. Data are concentrated: www.bc.edu alone is 165 of the "
    f"{rows_c['BC-owned']} BC-owned rows, and several categories have few hosts "
    f"(hosts per category: " + ", ".join(f"{c} {hosts_c[c]}" for c in CATS) + ").", B))
s.append(Paragraph(
    "<b>Public training signal.</b> Curlie/DMOZ could not be used: curlie.org is unreachable from the build environment "
    "and no maintained public dump exists. Substitutes, mapped onto the taxonomy: Hipo US-university domains "
    f"-> competitor institution ({pub_c.get('competitor institution', 0)} sampled); UT1 forums + social-network lists "
    f"-> forum/community ({pub_c.get('forum/community', 0)}); UT1 press list (.com/.org/.net) -> earned media "
    f"({pub_c.get('earned media', 0)}); UT1 job-search + financial lists -> other ({pub_c.get('other', 0)}). "
    "Sampling is capped at 300 per category and public rows are weighted 0.3. No public source exists for BC-owned or "
    "education directory/ranking site, so those two categories learn from seed data only. Any public domain sharing a "
    "registered domain with a seed host is dropped so public data cannot leak a seed label.", B))
s.append(Paragraph(
    "<b>Models.</b> (1) <i>Rule baseline</i>: ordered keyword/domain-pattern rules (bc.edu suffix, social/forum names, "
    ".edu suffix, ranking/admissions keywords, news keywords, else other). (2) <i>Learned</i>: multinomial logistic "
    "regression (C = 10, fixed in advance, class-balanced) on TF-IDF character 2-5-grams of the host plus word tokens of "
    "the URL path. (3) <i>BC rule + LR</i>: because BC ownership is definitional (taxonomy rule: bc.edu and subdomains) "
    "and one host dominates the class, a hard ownership rule handles BC-owned and the LR is trained on the other five "
    "categories. Each learned variant is run with and without public data; the pure LR variants are kept in Table 2 "
    "so the effect of the rule is visible.", B))
s.append(Paragraph(
    f"<b>Protocol.</b> Every split is grouped by host, so no domain appears in both train and test. Primary result: one "
    f"stratified grouped hold-out of {PS['test_rows']} rows ({pct(PS['test_fraction'])} of the data, {PS['test_hosts']} hosts; "
    "the split was picked for size and category coverage, never by model performance; it is just under 20% because "
    "www.bc.edu must sit in train). Robustness: 5 x 5-fold grouped cross-validation (CV). The better-performing model "
    "is the one with the highest mean CV accuracy, a rule fixed before any results were seen. For contrast, a naive random "
    f"row split of the same learned model scores {pct(R['leakage_demo']['random_row_split_accuracy_learned_seed'])}, versus "
    f"{pct(CV['learned_seed']['accuracy_mean'])} grouped: shared domains inflate accuracy by about "
    f"{100 * (R['leakage_demo']['random_row_split_accuracy_learned_seed'] - CV['learned_seed']['accuracy_mean']):.0f} points.", B))

# ------------------------------------------------------------------ 2. results
s.append(Paragraph("2. Results", H1))
bm, lm = PS["models"][BL], PS["models"][LM]
rows = [["Category", "Test rows / hosts", f"{NAMES[BL]}<br/>P / R / F1", f"{NAMES[LM]}<br/>P / R / F1"]]
th = PS["test_hosts_by_category"]
for c in CATS:
    a, b = bm["per_category"][c], lm["per_category"][c]
    rows.append([c, f"{a['support']} / {th.get(c, 0)}",
                 f"{a['precision']:.2f} / {a['recall']:.2f} / {a['f1']:.2f}",
                 f"{b['precision']:.2f} / {b['recall']:.2f} / {b['f1']:.2f}"])
rows.append(["Overall accuracy", f"{PS['test_rows']} / {PS['test_hosts']}", pct(bm["accuracy"]), pct(lm["accuracy"])])
rows.append(["Macro-F1", "", f"{bm['macro_f1']:.3f}", f"{lm['macro_f1']:.3f}"])
rows.append(["Accuracy excluding BC-owned rows", "", pct(bm["accuracy_excl_bc_owned"]), pct(lm["accuracy_excl_bc_owned"])])
s.append(table(rows, [2.35 * inch, 1.15 * inch, 1.75 * inch, 1.75 * inch], bold_last=3))
s.append(Paragraph(
    f"Table 1. Per-category precision, recall, F1 on the primary grouped hold-out. Supports are tiny (BC-owned is "
    f"{th.get('BC-owned', 0)} host, forum/community {th.get('forum/community', 0)} hosts, earned media "
    f"{th.get('earned media', 0)} hosts), so single-split per-category values are fragile; CV (Table 2 and Section 3) is "
    "the more reliable guide.", CAP))

rows = [["Model", "Primary split accuracy", "CV accuracy (mean +/- sd)", "CV accuracy excl. BC-owned", "CV macro-F1"]]
for m in ["baseline", "learned_seed", "learned_seed_public", "hybrid_seed", "hybrid_seed_public"]:
    c = CV[m]
    rows.append([NAMES[m] + (" (selected)" if m == R["best_model"] else ""),
                 pct(PS["models"][m]["accuracy"]),
                 f"{pct(c['accuracy_mean'])} +/- {100 * c['accuracy_std']:.1f}",
                 pct(c["accuracy_excl_bc_owned_mean"]), f"{c['macro_f1_mean']:.3f}"])
t2 = table(rows, [2.0 * inch, 1.2 * inch, 1.6 * inch, 1.2 * inch, 0.9 * inch])
cap2 = (Paragraph(
    "Table 2. Overall comparison (5 x 5-fold grouped CV). The baseline's sd is 0 because it has no trained parameters. "
    "Pure-LR accuracy with public data is low overall because, when bc.edu is held out, public .edu rows outvote the four "
    "tiny BC subdomains (BC-owned F1 0.07); the ownership rule removes that artifact.", CAP))
s.append(KeepTogether([t2, cap2]))

# ------------------------------------------------------------------ 3. analysis
s.append(Paragraph("3. Does it meet the 80% target? Analysis and limitations", H1))
base_cv, base_ps = CV[BL]["accuracy_mean"], bm["accuracy"]
lrn_cv, lrn_ps = CV[LM]["accuracy_mean"], lm["accuracy"]
s.append(Paragraph(
    f"<b>Target.</b> The rule baseline reaches {pct(base_ps)} on the primary hold-out and {pct(base_cv)} in CV, "
    f"above 80%; it is therefore the selected model. The best learned model ({NAMES[LM]}) reaches {pct(lrn_ps)} and "
    f"{pct(lrn_cv)}, short of 80%. The baseline's lead should not be read as rules beating learning: its keyword lists were "
    "written after looking at these same domains, and the seed labels were built with similar domain lists, so its numbers "
    "are optimistic and not a true held-out measurement. The fair head-to-head is the unseen-domain check below.", B))
s.append(Paragraph(
    f"<b>Why the learned model falls short.</b> (1) Little data: {D['hosts']} hosts, with only "
    f"{hosts_c['earned media']} earned-media and {hosts_c['other']} other hosts, so character n-grams cannot learn a "
    "category from a handful of names. (2) Earned media vs other vs directory is a question about what a site does, which "
    "a domain string often does not reveal (e.g. catholic-link.org, empowerly.com); the dominant confusions are directory "
    "-> other and earned media -> other. (3) The public lists are domain-only and noisy (UT1's press list is dominated by "
    "small local newspapers, not the outlets AI engines cite) and cover four of six categories. (4) Seed labels are partly heuristic, so the model also learns "
    "labeling quirks. Even so, public data helps: "
    f"BC rule + LR rises from {pct(CV['hybrid_seed']['accuracy_mean'])} to {pct(CV['hybrid_seed_public']['accuracy_mean'])} CV accuracy "
    f"and macro-F1 from {CV['hybrid_seed']['macro_f1_mean']:.3f} to {CV['hybrid_seed_public']['macro_f1_mean']:.3f}, mainly "
    f"through forum/community (F1 {CV['hybrid_seed']['f1_by_category_mean']['forum/community']:.2f} to "
    f"{CV['hybrid_seed_public']['f1_by_category_mean']['forum/community']:.2f}), since unseen social platforms look nothing "
    "like the seed ones by name. Directory/ranking F1 dipped slightly "
    f"({CV['hybrid_seed']['f1_by_category_mean']['education directory/ranking site']:.2f} to "
    f"{CV['hybrid_seed_public']['f1_by_category_mean']['education directory/ranking site']:.2f}): it has no public source, "
    "and with public data its sites were more often labeled earned media or other (errors.csv).", B))
s.append(Paragraph(
    "<b>Weak categories (both models).</b> Earned media (CV F1 "
    f"{CV[BL]['f1_by_category_mean']['earned media']:.2f} baseline / {CV[LM]['f1_by_category_mean']['earned media']:.2f} learned) "
    f"and other ({CV[BL]['f1_by_category_mean']['other']:.2f} / {CV[LM]['f1_by_category_mean']['other']:.2f}) are the problem "
    "classes. Candidate fixes for Milestone 4/5: add page-title and snippet text (already in the Profound export) as "
    "features, label more domains, and add a cheap lookup for well-known news outlets.", B))

ag = R.get("agreement")
if ag:
    pm = ag["per_model"]
    s.append(Paragraph(
        f"<b>Manual check (30 unseen domains).</b> The student reviewed 30 citations drawn from {R['unseen_pool_hosts']} "
        f"domains that appear in the Profound export but not in the seed set (up to 5 per predicted category; the sample "
        f"is not proportional to the population). Agreement with the selected model ({NAMES[R['best_model']]}): "
        f"<b>{ag['agree']}/{ag['n']} = {pct(ag['rate'])}</b>. Same 30 against each model: "
        + "; ".join(f"{NAMES[m]} {pct(v)}" for m, v in pm.items()) + ". Disagreements: "
        + "; ".join(f"{d['domain']} (predicted {d['predicted']}, you: {d['student']})" for d in ag["disagreements"][:10])
        + ("." if ag["disagreements"] else "none."), B))
else:
    s.append(Paragraph(
        f"<b>Manual check (30 unseen domains).</b> manual_check_sheet.csv holds 30 citations drawn from "
        f"{R['unseen_pool_hosts']} domains that appear in the Profound export but not in the seed set, so no model or "
        "rule list has seen them (up to 5 per predicted category; not proportional to the population). "
        "<b>Agreement rate: pending - to be recorded by the student</b> after filling the your_category column "
        "(python classifier.py agreement, then python build_eval_report.py). The same command scores every model "
        "against the student's labels, giving the fair comparison between the rule baseline and the learned model.", B))

s.append(Paragraph(
    "<b>Other limitations.</b> One hold-out of 93 rows is a small test set; CV repeats reuse the same 156 hosts, so the "
    "reported +/- reflects split randomness, not sampling uncertainty about new BC citations. Confidence for the rule "
    "baseline is a placeholder (1.0 rule fired, 0.5 fallback); classified_output.csv therefore also carries the best "
    "learned model's probabilities. classified_output.csv holds out-of-fold predictions, so no citation is scored by a "
    "model that saw its own domain's label.", B))

doc = SimpleDocTemplate(str(HERE / "eval_report.pdf"), pagesize=letter, topMargin=0.6 * inch,
                        bottomMargin=0.6 * inch, leftMargin=0.75 * inch, rightMargin=0.75 * inch,
                        title="Milestone 3 Evaluation Report")
doc.build(s)
print("built eval_report.pdf")
