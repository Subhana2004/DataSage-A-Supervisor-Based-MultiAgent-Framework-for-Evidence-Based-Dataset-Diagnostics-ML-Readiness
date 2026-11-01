# Blinded external diagnostic assessment

This is a **review protocol and scoring utility**, not a completed study or a source of genuine issue labels.

Use `prepare_blinded_review(report, baseline_keys, dataset_id)` to produce two separate objects:
- `review_cases`: descriptions of observations with no information about which system found them.
- `sealed_predictions`: system predictions keyed by case ID. **Keep this hidden from reviewers.**

Two independently working reviewers assess each case using the actual dataset's task, collection details, and prediction context. Each produces a label (`actionable`, `not_actionable`, or `uncertain`) and a rationale. Neither reviewer should see the sealed predictions or the other reviewer's answers. The `adjudicate_reviews` function marks disagreements and uncertain cases as unresolved. `score_review` reports confusion-matrix metrics only on resolved cases and tracks exclusions.

Important limits:
- A context-free observation cannot be assumed actionable; reviewers need domain and collection context.
- Concordance is not automatically objective truth; an independent adjudicator and a documented rubric improve rigor.
- The supplied pandas baseline covers only missingness, constant features and duplicates; broader categories require category-matched baselines.
- If a warning is category-mismatched or unreviewed, do not turn it into a favorable performance claim.
- Before a publication-style evaluation: preregister metrics, use external datasets, justify sample selection, assess inter-rater agreement, report uncertainty intervals and conduct error analysis.

No external annotations have been collected as part of this implementation.
