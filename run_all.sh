#!/usr/bin/env bash
# Reproduces every generated output in this repo and writes a log per milestone to logs/.
# From the repo root:  bash run_all.sh        (use PYTHON=python3.11 bash run_all.sh to pick an interpreter)
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"
mkdir -p logs

echo "== Milestone 2: design doc + seed dataset" | tee logs/milestone2_run.log
( cd milestone2 \
  && $PY diagram.py && $PY build_profound_csv.py && $PY process_citations.py && $PY build_pdf.py ) 2>&1 | tee -a logs/milestone2_run.log

echo "== Milestone 3: classifier + evaluation report" | tee logs/milestone3_run.log
( cd milestone3 && $PY classifier.py run && $PY build_eval_report.py ) 2>&1 | tee -a logs/milestone3_run.log

echo "== Milestone 4: tagging layer, tests, synthetic agent demo" | tee logs/milestone4_run.log
( cd milestone4 \
  && $PY tagging.py --export ../milestone2/profound_citations_raw.csv \
  && $PY -m unittest discover -s tests -v \
  && $PY tests/synthetic_demo.py ) 2>&1 | tee -a logs/milestone4_run.log
echo "done"
