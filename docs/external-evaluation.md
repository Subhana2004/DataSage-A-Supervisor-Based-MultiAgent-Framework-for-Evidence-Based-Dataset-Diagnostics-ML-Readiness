# Phase 11 — External dataset comparison (pilot)

Two **documented** UCI datasets are selected:
- Adult, UCI ID 2, https://archive.ics.uci.edu/dataset/2/adult (CC BY 4.0).
- Wine Quality, UCI ID 186, https://archive.ics.uci.edu/dataset/186/wine+quality (CC BY 4.0).

Install:
```bash
pip install -e ".[dev,evaluation]"
python -m datasage.external_evaluation --fetch --output results/external-evaluation.json
```

The explicit `--fetch` flag accesses UCI over the network. The repository does **not** ship downloaded dataset copies. Run this before any claims about external performance. The result contains dataset-level agreement and disagreement on three **descriptive** categories: missing values, constant columns, and exact duplicated rows. A simple pandas baseline is deliberately transparent; it is **not an independent ground-truth reference**.

**Blinding protocol for follow-up:** pre-register feature availability and definition of a harmful data issue, sample flagged and unflagged records, have human assessors annotate independently without seeing which method flagged each case, adjudicate disagreements, then score precision/recall by review labels. Use a held-out dataset for final assessment. No accuracy, superiority, or confidence claims should be made from baseline agreement alone.

Adult missing values may involve special marker strings; verify their parsing before interpreting results. Wine Quality class frequency differences should not automatically be classed as a harmful defect.
