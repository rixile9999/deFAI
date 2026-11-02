# FedAdam Kalman follow-up: stability, cost, usability and public-data validation

User-confirmed scope (2026-11-02): improve the FedAdam extension, not the other
algorithms or Bayesian candidate search. Base: `35e2027` (48 prior extension
commits, 625 passing regression tests). New commits retain that staged history.

## Acceptance matrix

| Area | Improvement and verification | Limits |
|---|---|---|
| Numerical stability | Stable reduction arithmetic, finite metrics, strict prior/checkpoint validation; reproduce failures before fixes | No optimizer convergence theorem |
| Performance | Avoid redundant copies and disabled adaptation work; record before/after timing and allocations | Local CPU measurements, no network speedup claim |
| Usability | Explicit optimizer reset, safe numeric checkpoint files, command-line controls and actionable errors | No full distributed/DP training checkpoint |
| Evaluation integrity | Complete paired comparisons, immutable split identity, target attainment and non-attainment, uncertainty summaries | No test-set tuning or selective winner reporting |
| Public data | Multiclass digits and binary Wisconsin diagnostic classification; train-only preprocessing; disjoint train/validation/test sets | Algorithm benchmarks, not clinical validation |
| Integration | Entire regression suite, resource inventory, sdist/wheel, separate installed execution and four-platform CI | No main-branch merge |

The public-data comparison uses the existing scikit-learn dependency. Its
`load_digits` loader supplies 1,797 examples (64 features) from the UCI digits
test subset; it is not the complete 5,620-example UCI release.
`load_breast_cancer` supplies 569 examples with 30 features.
Both UCI dataset pages list CC BY 4.0. The package ships attribution and loader
code, not a second copy of the records. Data hashes and split identifiers belong
in the experiment output; no patient identifiers are collected.

Sources (checked 2026-11-02):

- [UCI Optical Recognition of Handwritten Digits](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits), E. Alpaydin and C. Kaynak
- [scikit-learn digits loader](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html)
- [UCI Breast Cancer Wisconsin (Diagnostic)](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic), W. Wolberg, O. Mangasarian and W. Street
- [scikit-learn diagnostic dataset loader](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html)

Predeclared full public-data grid: both datasets × alpha 0.1/0.5/10 × local
steps 1/5/20 × participation 0.1/0.5/1 × seeds 0–4 × EMA/fixed/adaptive
Kalman = 810 runs. Stationary public labels are retained; synthetic label
reversal remains in the separate existing benchmark. A linear softmax model
tests the server optimizer without an external pretrained model. Fixed settings
and held-out performance must be disclosed even if a method is slower or worse.
