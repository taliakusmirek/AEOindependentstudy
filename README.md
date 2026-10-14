# AEO independent study: classifying Boston College's AI-answer citation sources

Readings in Computer Science (Prof. Maira Marques Samary). Builds a small pipeline that takes citations
AI answer engines give for Boston College prompts, sorts the cited sources into six types, tracks the mix
week over week, and writes a short brief. All data is Boston College's public citation footprint, collected
with a personal Profound seat, plus public domain lists. **Private repo:** it contains a raw Profound export
(third-party page snippets). No employer data is used.

Pipeline: Profound export -> citation extractor -> classifier -> tagging layer -> report generator.

## Status

| Milestone | State |
|---|---|
| 1. Literature review | Not in this repo yet. Add `milestone1/review.pdf` (or .md). |
| 2. Design doc + seed dataset | Done. `design_doc.pdf`, `seed_dataset.csv` (524 rows, 156 hosts). |
| 3. Classifier + evaluation | Done except step 5: the 30-row manual agreement check is **yours to fill in** (`milestone3/manual_check_sheet.csv`). |
| 4. Tagging layer + weekly brief | Code complete and tested. Real weekly data is still needed: only the first snapshot (2026-09-30) exists. See "Weekly workflow". |
| 5. Final evaluation + report | Not started (needs your own manual analysis and the real briefs). |

## Reproduce everything

    pip install -r requirements.txt
    bash run_all.sh          # regenerates every output; logs in logs/

Outputs regenerated this way were verified byte-identical to the delivered files (seed_dataset.csv,
classified_output.csv, results.json).

## Scripts and run commands

Run each from its own folder.

| Script | What it does | Command |
|---|---|---|
| milestone2/build_profound_csv.py | 26 prompts in Profound's upload template | `python build_profound_csv.py` |
| milestone2/process_citations.py | raw export -> `seed_dataset_raw.csv` (4 cols) and labeled `seed_dataset.csv` (5 cols) | `python process_citations.py` |
| milestone2/diagram.py, build_pdf.py | architecture diagram, `design_doc.pdf` | `python diagram.py && python build_pdf.py` |
| milestone3/classifier.py | baseline + learned classifiers, evaluation, outputs | `python classifier.py run` |
| milestone3/classifier.py predict | classify hostnames | `python classifier.py predict reddit.com bu.edu [--model hybrid_seed_public]` |
| milestone3/classifier.py agreement | score your filled manual-check sheet, all models | `python classifier.py agreement` then `python build_eval_report.py` |
| milestone3/build_eval_report.py | `eval_report.pdf` from `results.json` | `python build_eval_report.py` |
| milestone4/tagging.py | raw export -> tagged snapshot (intent, cohort, platform, predicted category) | `python tagging.py --export <profound_export.csv>` |
| milestone4/agent.py | two snapshots -> `diff_report.json` (flags share moves > 5 pp) | `python agent.py --current snapshots/B.csv --previous snapshots/A.csv` |
| milestone4/brief.py | `diff_report.json` -> `brief.md` (5 fixed sections) | `python brief.py` |
| milestone4/tests | unit tests on known answers (synthetic fixtures) | `python -m unittest discover -s tests -v` |

## Weekly workflow (Milestone 4)

1. Export a fresh citations CSV from Profound.
2. `cd milestone4 && python tagging.py --export ~/Downloads/<export>.csv` writes `snapshots/<collection date>.csv`.
3. From week 2: `python agent.py --current snapshots/<this week>.csv --previous snapshots/<last week>.csv`, then `python brief.py`. Save the brief as `brief_1.md`, `brief_2.md`, ... (two runs at least 7 days apart are required).
4. At week 9, add the 5 new/edited prompts to Profound, log each in `change_log.csv` (`prompt_text,date,reason`, exact prompt text), then run the tagging step as usual: those prompts' citations are tagged `rebrand-sim`, everything else stays `legacy`, and the agent compares legacy rows only.

`milestone4/tests/example_output/` holds a brief generated from **synthetic** fixtures (dates in 2099, banner on top) so the agent's logic can be seen working. It is not real data and is not one of the required weekly briefs.

## Things to know

- **Seed labels are AI-assisted.** First pass: Profound's own citation category mapped onto the six categories plus hand-built domain lists; then reviewed and confirmed by the student. Describe them that way in the final report.
- **Classifier choice is provisional.** The rule baseline has the best cross-validated accuracy (88.4%) but its rules were written after seeing the seed domains, so it is optimistic. The best learned model reaches 78.7% (below the 80% target; reasons in `eval_report.pdf`). The unseen-domain manual check decides which model Milestone 4 should use: `tagging.py --model <name>`.
- **Earned media is under-counted.** The baseline predicts only 3 earned-media citations in the first snapshot versus 23 in the labeled seed data, so earned-media share moves in briefs are unreliable until that category is improved (page-title features, more labels).
- **Re-running is safe for your labels:** `classifier.py run` will not overwrite `manual_check_sheet.csv` once `your_category` has entries.
- Public training data: see `milestone3/data/public/SOURCES.md` (Curlie/DMOZ was unavailable).
