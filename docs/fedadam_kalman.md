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
