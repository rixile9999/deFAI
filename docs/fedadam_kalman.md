# FedAdam with a Kalman first moment

Research implementation; acceleration is a hypothesis, not a guaranteed outcome.

The user-supplied `FL_Kalman_Bayesian_Acceleration_Design_20261102.html`
(2026-11-02), sections 3, 4/FedAdam, 6 and 7, defines this extension:
replace the first-moment exponential moving average with a layer-wise scalar-gain
Kalman estimator. Retain the second moment of the **raw aggregated update**,
the positive local-minus-global update sign, tau-squared initialization and no
bias correction of this repository's FedAdam.

Baseline reference: [Reddi et al., Adaptive Federated Optimization, ICLR 2021](https://arxiv.org/abs/2003.00295).
The Kalman substitution is the supplied design's proposal, not a result claimed
by that paper. No external source code or private design document is redistributed.

## Staged acceptance plan

1. Define independent estimator/configuration skeletons and their validation
2. Implement prediction, correction, covariance floors and deterministic tests
3. Add a separate `fedadam-kalman` strategy and preserve raw second moments
4. Verify zero blend equals existing FedAdam exactly, including multiple rounds
5. Add optional bounded residual adaptation, gain bounds and trust-radius limits
6. Validate finite arithmetic, schema stability and transactional failed rounds
7. Add optimizer-state export/restore, metrics, factory and command-line config
8. Compare fixed EMA, fixed Kalman and adaptive Kalman with paired schedules/seeds
9. Run full regressions, inventory/build and installed-wheel execution checks

Scope excludes other algorithm extensions, state-space Gaussian processes,
Bayesian optimization and real institution deployment. Per the design, Bayesian
candidate search is a subsequent stage conditional on demonstrated benefit.

Small commits record each implementation or verification milestone; intermediate
skeleton commits are not final validated releases.

Source SHA-256: `713ba382ac153251e84d15be40ad7c37686f8ec358720532258e33bee5af2cb7`

## Install and run

Python 3.11/3.12, using the existing optional learning runtime; no new dependencies:

```sh
python -m pip install '.[flower]'
defai run --strategy fedadam-kalman --dataset synthetic --clients 4 --rounds 5 --output kf-run
defai run --strategy fedadam-kalman --set kf-blend=0.0 --output ema-control
defai run --strategy fedadam-kalman --set kf-process-adaptation=0.2 --set kf-max-step-norm=0.1 --output adaptive-run
defai kalman-benchmark --full --rounds 40 --output paired-benchmark.json
```

Use a fresh output directory/file for every run. The same run-config keys work
with the existing deployment entry point. Default `strategy=fedavg` is unchanged.

```python
from defai.kalman import KalmanConfig
from defai.strategies import FedAdamKalman

strategy = FedAdamKalman(
    server_learning_rate=0.01,
    kalman=KalmanConfig(process_noise=1e-4, observation_noise=1e-2),
    kalman_blend=1.0,
)
```

## Recurrence and controls

For each parameter-array key (a "layer" here), retain a vector mean and one
scalar variance. Predict `z-=rho*z`, `P-=rho**2*P+Q`; correct using
`K=clip(P-/(P-+R), gain_min, gain_max)` and `z=(1-K)*z-+K*d`.
Joseph variance `(1-K)**2*P-+K**2*R` preserves the interpretation when gains
are clipped. `variance_floor` prevents a collapsed variance.

The first moment is the Kalman mean, or an explicitly weighted blend of an
**independent** EMA mean and the Kalman mean. For `rho=1`, fixed `K=1-beta_1`
recovers the original EMA recurrence from a zero mean. `kf-blend=0` skips the
filter entirely and is the exact original FedAdam control when the trust radius
is disabled. There is no extra bias correction.

`v=beta_2*v+(1-beta_2)*d**2`, initialized at `tau**2`, always uses the raw
aggregate `d`. Model movement is `eta*m/(sqrt(v)+tau)`, with the repository's
positive local-minus-global convention. Optional `kf-max-step-norm` bounds the
Euclidean norm of the combined parameter displacement **before dtype rounding**.
It is a numerical trust radius, not a loss-decrease or privacy guarantee.

Every config field in `KalmanConfig` maps to a hyphenated `kf-` run-config key.
Additional keys are `kf-blend` (default 1) and `kf-max-step-norm` (default 0/off).
The defaults use fixed Q/R; `kf-process-adaptation` and
`kf-observation-adaptation` are both 0/off. Positive rates exponentially blend
bounded innovation-matching proxies for the next round, capped by
`kf-noise-ceiling`. A residual cannot separately identify true process noise,
sampling noise, legitimate client differences or DP noise. Simultaneous Q/R
adaptation is a heuristic; it can slow down tracking and must be compared.

## State, failures and privacy

Failed non-finite arithmetic, model schema changes and invalid inputs leave
optimizer state untouched. Retrying the rejected round is allowed; successful
round numbers must increase. Skipped rounds are allowed and correspond to one
filter observation per successful round, not physical elapsed-time prediction.

`state_dict()`/`load_state_dict()` copy and validate optimizer-only state.
Restore the matching global model separately before loading. This is not a
complete training checkpoint: client states, sampling schedules, communication
counters and privacy accounts are not included. DP-enabled optimizer checkpoint
export/restore is explicitly rejected to prevent an apparent budget reset.

During normal DP training, clipping/noise/accounting happen **before** the filter;
the filter and its diagnostics use only the released aggregate. It never reads
individual updates to estimate observation noise. This preserves the existing
post-processing boundary; it does not add a new privacy guarantee or repair
privacy problems in another component.

Metrics include gain bounds, mean variance, mean innovation squared error,
next-round Q/R, effective server learning rate and pre-rounding step norm.
`optimizer_state_bytes()` counts unique numeric array payloads and scalar values,
excluding Python containers, global models, clients and privacy-accountant state.

## Synthetic comparison and interpretation

The complete preset contains 270 condition/seed combinations per method (810
training runs): alpha 0.1/0.5/10; local steps 1/5/20; participation 0.1/0.5/1.0;
stationary or a label-reversal shift; seeds 0–4. Each run uses 40 rounds, 10
clients and 7 logistic parameters. Training and test samples are independent.
The synthetic data has no external dataset license or clinical interpretation.

At the predeclared 75% test-accuracy target, adaptive Kalman reached the target
in fewer rounds than the EMA control in 60 paired cases, more rounds in 28,
and the same number in 182. Fixed Kalman was earlier in 20, later in 54 and
tied in 196. All methods reached the target before the shift/end at least once;
first attainment is not sustained accuracy. Of the 135 shifted cases per method,
52 adaptive cases recovered to the target within the following 20 rounds,
versus 0 for the fixed Kalman and EMA controls under these settings.

These are descriptive results for this data and these fixed hyperparameters.
There is no general or statistically established acceleration claim. Different
methods have different tuning needs; the comparison does not establish optimal
tuned performance. The condition grid also includes identical pre-shift prefixes;
810 runs must not be treated as 810 independent scientific replications.
No settings were selected using held-out scores.

Measured wall times include client learning, server operations and evaluation,
but exclude setup/imports and real communication. They are local diagnostics;
concurrent host workloads and fixed method order prevent a controlled wall-time
speedup conclusion. Training/evaluation payload counters represent numeric
payloads, not network protocol bytes or latency. Numeric optimizer payloads are
112 bytes for EMA and 152 bytes for Kalman on this 7-parameter example.

| Completed | Not completed |
|---|---|
| Exact EMA equivalence, equations, finite checks, trust radius and rollback | A convergence theorem for the modified optimizer |
| Factory/CLI, real neural-client smoke training, optimizer restart tests | Full distributed-training/DP checkpoint resume |
| Paired synthetic grid and explicit unsuccessful outcomes | Real-data benchmark, tuning study, clinical or institutional validation |
| Package and test checks recorded in the validation receipt | Bayesian optimization or state-space Gaussian-process extension |

The implementation is available for experiments; the evidence supports only
the numerical and synthetic behaviors described above.
