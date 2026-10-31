# Research evaluation v0.1

Run the deterministic synthetic benchmark:

```bash
python -m datasage.benchmark --output results/benchmark-v0.1.json
```

The suite creates six data scenarios with explicit issue labels: missing values, exact duplicate rows, IQR outliers, imbalanced target, near-exact target copy, and a limited negative-control dataset.

## Measurement contract

The unit of comparison is `(category, column)`. The benchmark counts only *verified* findings in the five labeled categories. TP = predicted and labeled; FP = predicted but not labeled; FN = labeled but not predicted. Precision and recall are micro-averaged over the six scenarios. Unsupported, rejected and unverified findings are tracked separately, not counted as successful detections.

This is an engineering smoke benchmark, **not a validated paper evaluation**. The scenarios intentionally align with implemented rules, so even perfect scores cannot establish generalization. Some features are suspicious only in the absence of prediction context; exact target matches do not prove leakage. The negative-control case tests only its declared category labels. Future research should include blinded datasets, multiple random seeds, difficult near-miss controls, calibrated severity, matched EDA baselines and human review.
