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

## Additional calibrated phase, declared before its execution

The original 810 public-data runs are retained. Their default fixed/adaptive
Kalman settings often reached validation targets later than the EMA control.
Inspecting the recurrence shows that a zero initial variance gives a much
smaller first observation weight, and adaptive Q can cause increasing smoothing.

A second, separately reported phase uses seeds 100–104, the same full grid and
no parameter search. `KalmanConfig.from_ema(beta_1)` derives `K=1-beta_1`,
`P=K*R`, `Q=K**2*R/(1-K)` for rho=1. Fixed-noise initialization recovers the
EMA recurrence; the adaptive variant additionally keeps `K >= 1-beta_1`.
This is an interpretable calibration and lower-bound safeguard, not evidence of
acceleration. Defaults are not silently changed. Public benchmark flags are
`--calibrated --seed-offset 100`. Actual configs are recorded per run.

The new splits reuse the same public datasets, so this is not an independent
external validation cohort. Target tracking uses validation scores only and each
model sees its held-out test labels only for the final evaluation. The first
phase's unfavorable results remain visible alongside the calibrated phase.

## Run and inspect

```sh
defai kalman-benchmark --dataset public --full --rounds 60 --output original-public.json
defai kalman-benchmark --dataset public --full --calibrated --seed-offset 100 --rounds 60 --output calibrated-public.json
defai kalman-profile --sizes 16 10000 1000000 --repeats 20 --output estimator-profile.json
python scripts/compare_kalman_profile.py --output paired-profile.json
defai run --strategy fedadam-kalman --dataset synthetic --clients 2 --rounds 3 --optimizer-checkpoint final.npz --output training-run
```

The paired profile script requires this Git checkout and the recorded reference
commit; the standalone `kalman-profile` command works in installed packages.
Use fresh output names. Profiling reports seconds and traced allocation bytes,
not whole-process memory or physical network costs.

`strategy.save_checkpoint(path, model=latest_arrays)` explicitly takes the
latest returned model, since the strategy's cached broadcast can precede the
last update. `restored_model = strategy.load_checkpoint(path)` restores numeric
model/optimizer state and returns model arrays. Continue with the round driver
at `strategy.state_dict()['last_round']+1`. Client state, RNG scheduling,
communication counters and privacy accounts are not restored. A new
`Strategy.start()` call is not a complete training-resume protocol.

Snapshots use numeric NPZ arrays and JSON metadata with dtype/shape/content
checks. Loading disables pickle. Content hashes detect damage, not authenticity.
Writes publish a complete temporary file atomically and refuse overwrite unless
explicitly requested. Failed loads validate on an isolated strategy shell.
`reset_optimizer()` clears moment/filter history and diagnostics while keeping
the configured model and existing privacy mechanism/accountant.

## Results and limits

Both 810-run phases, dataset fingerprints, condition-level paired seed
intervals and all six profile conditions are retained in
[the machine-readable receipt](fedadam_followup_results.json).

In the original public-data phase, adaptive Kalman reached the validation target
earlier in 18 paired cases and later in 136; two digits cases failed to reach
the target although EMA reached it. Its average held-out accuracy was 93.77%
on digits versus EMA's 95.59%, and 97.41% on the diagnostic dataset versus
EMA's 97.66%. These are descriptive means across a reused condition grid.

In the separately seeded calibrated phase, fixed Kalman reproduced EMA's
validation/test scores. Bounded adaptive Kalman had the same target round in
258 jointly attained cases, an earlier round in 3 and a later round in 1;
both methods missed the diagnostic target in 8 cases. Average held-out scores
were 95.63% on digits for both methods, and 97.41% adaptive versus 97.43% EMA
on the diagnostic dataset. Calibration mitigated degradation but does not
establish general training acceleration on these data/settings.

The two phases use different splits/seeds; comparing their absolute accuracy
means is not a paired estimate of the calibration's effect. The useful checks
are each phase's paired EMA comparison and the fixed-filter equivalence test.
The normalizer is fit centrally on training rows for this simulation; secure
distributed estimation of normalization statistics is not implemented.
Small repaired client partitions make lower-decile client accuracy noisy.
Intervals use five paired seeds within a fixed condition and have no correction
for the many comparisons; they are descriptive, not an overall superiority test.

| Completed | Not evaluated / unsupported |
|---|---|
| Numerical overflow, scalar state and diagnostic rollback fixes | A convergence theorem or calibrated uncertainty guarantee |
| Reduced copies/disabled work, reproducible latency and allocation profiles | General end-to-end training or network acceleration |
| Numeric snapshots, exact continuation and final-weight equality | Client/sampler/DP complete training resume |
| 1,620 public-data runs with train-only fitting and final test isolation | Other datasets, nonstationary real streams, parameter search, independent external cohorts |

The package is improved for controlled experiments; real institution deployment
and clinical use remain outside the validation scope.
