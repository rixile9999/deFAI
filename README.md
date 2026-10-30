# deFAI 0.2: packaged algorithms and disease research

Common algorithms and all four disease-research source collections are available
through an installable Python package, a unified CLI, and Python APIs.
Start with the [Korean user guide](docs/user_guide_ko.md).

```powershell
python -m pip install ".[all]"
defai doctor
defai list
defai demo --output results/demo01
defai run --strategy fedprox --rounds 3 --output results/prox01
defai init studies/session01
defai research list
```

The default runtime is CPU. Select `--device cuda` explicitly after installing a
compatible CUDA PyTorch build. Discovery/workspace creation do not import training
libraries. Optional install groups are `analysis`, `learning`, `flower`, `research`,
`mobile`, `simulation`, `plot`, `all`, and `test`. Python 3.11/3.12 is recommended;
Windows Ray simulations need 3.11/3.12, while the in-process backend also runs on 3.13.

The maintained Android app can be built with `defai android --sdk SDK_PATH --output app.apk`.
It requires an installed Android SDK/JDK; Python dependencies do not install them.
The mobile parameter coordinator uses explicit rounds, validated client updates,
atomic persistence and NumPy, without a TensorFlow dependency.

For a locked source-checkout environment use `uv sync --locked --extra research --extra mobile --extra test`.
`uv.lock` records the resolved dependency graph; `defai doctor` records the active versions.

Research-stage software: synthetic smoke tests do not establish clinical validity,
causal policy effects, or distributed network performance. The publication manifests
from 2026-10-29 are historical snapshots; current resource hashes are in
`defai/research/catalog.json`. Unreleased research remains outside this package.

## Existing algorithm reference

# deFAI — Federated Learning Assignment

[![tests](https://github.com/haneol815/deFAI/actions/workflows/tests.yml/badge.svg)](https://github.com/haneol815/deFAI/actions/workflows/tests.yml)

A federated learning (FL) assignment template built on [Flower](https://flower.ai).
It simulates multiple clients (SuperNodes) on a single machine with the Flower Simulation Engine,
and the same code can be deployed to a real distributed setup with the Deployment Engine.

- Datasets: MNIST, CIFAR-10 (downloaded and partitioned with [Flower Datasets](https://flower.ai/docs/datasets/)),
  MNIST and Fashion-MNIST from plain-file mirrors (`mnist-mirror`, `fashion-mnist-mirror`, no Hugging Face needed),
  plus an offline `synthetic` dataset for smoke tests
- Partitioning: IID, Dirichlet non-IID
- Algorithms (all implemented in this repo against Flower's `Strategy` interface — see [Algorithms](#algorithms)):
  - non-IID / heterogeneity: FedProx, SCAFFOLD, MOON, FedNova, q-FedAvg (fairness)
  - server optimisation: FedAvgM, FedAdagrad, FedAdam, FedYogi
  - Byzantine-robust aggregation: FedMedian, FedTrimmedAvg, Krum, Multi-Krum, Bulyan
  - central differential privacy: fixed / adaptive clipping, server- or client-side, with an RDP accountant
- Malicious-client simulation: label flipping, sign flipping, Gaussian-noise updates
- Communication: traffic accounting, lossy update compression (top-k, quantisation, error feedback)
- Evaluation: per-client spread of accuracy, personalised accuracy after local fine-tuning, client dropout
- Evaluation: federated (on each client's local data) + centralized (on the full test set at the server)

## Structure
```
.
├── pyproject.toml            # Dependencies + Flower app settings (run config)
├── defai/
│   ├── server_app.py         # ServerApp: strategy selection, rounds, centralized eval, result saving
│   ├── client_app.py         # ClientApp: local training / evaluation, per-client state, attacks, client-side DP
│   ├── task.py               # Model, data partitioning/loading, local train/test (FedProx/SCAFFOLD/MOON hooks)
│   ├── config.py             # Run-config validation (fails fast on typos / out-of-range values)
│   ├── aggregation.py        # Pure-numpy rules: weighted mean, median, trimmed mean, Krum, Bulyan
│   ├── attacks.py            # Malicious-client behaviours
│   ├── compression.py        # Top-k / quantisation / error feedback for client updates
│   ├── mirror.py             # MNIST / Fashion-MNIST from plain-file mirrors (checksummed, cached)
│   ├── records.py            # ArrayRecord <-> numpy helpers, wire-format keys
│   ├── strategies/
│   │   ├── base.py           # FedAvg + the hooks every strategy plugs into
│   │   ├── heterogeneity.py  # FedProx, MOON, SCAFFOLD, FedNova, q-FedAvg
│   │   ├── fedopt.py         # FedAvgM, FedAdagrad, FedAdam, FedYogi
│   │   ├── robust.py         # FedMedian, FedTrimmedAvg, Krum, Multi-Krum, Bulyan
│   │   └── __init__.py       # registry + run-config -> strategy factory
│   └── privacy/
│       ├── dp.py             # Clipping + Gaussian noise (fixed / adaptive)
│       └── accountant.py     # Renyi-DP accountant
├── tests/                    # pytest suite (no network or GPU needed)
├── scripts/plot.py           # Metric-per-round plotting
├── scripts/compare.py        # Results table (mean ± std over seeds, rounds-to-target, epsilon)
├── scripts/partition_stats.py # Per-client class counts of a partitioning (+ heat map)
├── scripts/run_grid.py       # Resumable variants x seeds experiment grids
├── docs/report.md            # Report template
├── docs/experiments.md       # Results of the experiment grids, with caveats
├── .github/workflows/tests.yml # CI: ruff + the offline test suite
└── results/                  # Experiment outputs (git-ignored)
```

## Installation
Requires Python 3.11 or later.
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .            # add ".[plot]" for plotting, ".[test]" for the tests
```

## Usage
```bash
# Default run (MNIST, IID, FedAvg, 10 rounds, 10 SuperNodes)
flwr run . --stream --federation-config "num-supernodes=10"

# Non-IID (Dirichlet α=0.5) + FedProx
flwr run . --stream --federation-config "num-supernodes=10" \
  --run-config "partitioner='dirichlet' strategy='fedprox' proximal-mu=0.01"

# Change number of rounds, learning rate, etc.
flwr run . --stream --federation-config "num-supernodes=10" \
  --run-config "num-server-rounds=30 learning-rate=0.05"
```

- All configurable values are listed under `[tool.flwr.app.config]` in `pyproject.toml`.
- Since Flower 1.32 the default number of simulated SuperNodes is 2, so set it with `--federation-config`.
  To change the default permanently instead:
  `flwr federation simulation-config --num-supernodes 10`
- Results are saved to `results/<dataset>_<strategy>_<partitioner>[_dp-...][_<attack>x<n>]_<timestamp>/history.json`.
- The run config is validated before the first round (`defai/config.py`): a typo such as `attack='signflip'`, an
  out-of-range value, or DP combined with a strategy that cannot support it is rejected with the offending keys listed.
- Runs are reproducible for a given `seed`: model initialisation, client sampling (nodes are ordered by registration time,
  because node ids are random), every client's data order (seeded from `seed`, partition and round), the aggregation
  order and the DP noise are all fixed. Checked with `flwr run`: same seed gives bitwise identical `history.json`
  metrics on CPU; GPU kernels may add run-to-run noise.
- `history.json` is strict JSON: non-finite numbers (e.g. an infinite epsilon) are written as `null`.
- A `ServerApp` exception does not change the exit code of `flwr run`; look for a `Traceback` in the log.
- `dataset='synthetic'` generates MNIST-like images locally, so a run needs no download. This is handy for smoke tests
  and in environments without access to Hugging Face. It is too easy to compare algorithms on, though: on a
  Dirichlet α=0.1 split of 10 clients, FedAvg, FedProx, SCAFFOLD, MOON, FedNova and q-FedAvg all end at 100% test
  accuracy after 20 rounds and differ only in how soon they get there (9 to 19 rounds to reach 99%).
- `dataset='mnist-mirror'` and `'fashion-mnist-mirror'` are the original MNIST / Fashion-MNIST IDX archives, downloaded
  by `defai/mirror.py` from the mirrors torchvision (S3) and Keras (GCS) use, verified against pinned SHA-256 sums and
  cached in `~/.cache/defai` (`DEFAI_DATA_DIR` changes it). They are the standard 60k/10k splits (Fashion-MNIST is a harder task than
  MNIST) and work where the Hugging Face hub cannot be reached. The files are fetched by the first process that
  needs them, so run once locally before starting a long grid. CIFAR-10 has no such mirror here.
- Recent Flower versions build a fresh virtual environment per `flwr run` and reinstall the app's dependencies.
  To reuse the current environment instead (much faster, and works offline), run with
  `FLWR_DISABLE_RUNTIME_DEPENDENCY_INSTALLATION=1`.

```bash
pip install -e ".[plot]"
python scripts/plot.py results/*/history.json -o results/acc.png
python scripts/plot.py results/*/history.json -m dp-epsilon -o results/eps.png   # any history key
python scripts/plot.py results/*/history.json --group -o results/acc.png        # one curve per configuration: mean ± std over seeds
python scripts/compare.py results/*/history.json --target 0.95                   # one row per configuration
python scripts/compare.py results/*/history.json --extra eval_acc_min comm-mb-total --markdown  # more columns, as a Markdown table
python scripts/compare.py results/*/history.json --tail 3                          # accuracy averaged over the last 3 rounds (steadier than the final one)
```
`compare.py` groups runs that differ only in `seed` and reports mean ± std (a single seed is too noisy to rank
algorithms, especially on non-IID data). `scripts/run_grid.py` runs such a grid, variants × seeds, and calls
`compare.py` at the end:
```bash
python scripts/run_grid.py --output-dir results/noniid --seeds 1 2 3 --num-supernodes 10 \
  --base "partitioner='dirichlet' dirichlet-alpha=0.1 num-server-rounds=30" \
  --vary "strategy='fedavg'" "strategy='fedprox'" "strategy='scaffold'" "strategy='moon'"
```
Every `--vary` fragment is added to `--base` (same `key=value` syntax as `--run-config`); `seed` and `output-dir` are
set per run, and logs go to `<output-dir>/logs/`. The grid keeps a manifest, `<output-dir>/grid.json`, mapping each
finished run to its results. A run is identified by its complete run config (the `pyproject.toml` defaults plus the
fragments and the seed) and `--num-supernodes`, so an interrupted grid continues where it stopped, while changing any
of these, or `--no-resume`, runs the job again. The final comparison covers the runs of the invocation only.
`--dry-run` only prints the commands.

`flwr run` exits with 0 even when the ServerApp fails, so a run counts as successful only if it wrote a new
`history.json`; the script exits with 1 if any run did not. `flwr run --stream` hands the run to a background
SuperLink, so a run that exceeds `--timeout` (and a Ctrl-C) is stopped with `flwr stop`. Environment variables such as
`FLWR_DISABLE_RUNTIME_DEPENDENCY_INSTALLATION=1` (set for the runs unless `--install-deps` is given) and
`DEFAI_DATA_DIR` are read by that SuperLink, which the first run starts and later ones reuse; stop it
(`pkill -f flower-superlink`) to change them.

## Reading `history.json`
One row per round (round 0 is the initial model), merged from three sources:

| key | meaning |
|---|---|
| `test_acc`, `test_loss` | centralized evaluation of the global model on the full test set |
| `eval_acc`, `eval_loss` | federated evaluation: example-weighted mean over the sampled clients' held-out data |
| `eval_acc_min`, `eval_acc_max`, `eval_acc_std` | the same accuracy over clients, **unweighted**: the worst and best client and their spread. A good mean with a low minimum means the model fails some clients |
| `train_loss` | mean task loss of the local training (without the FedProx / MOON regularisers) |
| `comm-mb-down`, `comm-mb-up`, `comm-mb-total` | training traffic in MiB: model sent to the sampled clients, what came back, and the running total of the run (evaluation messages not counted). SCAFFOLD pays twice per direction for its control variates |
| `eval_acc_personalized`, `eval_loss_personalized` (+ `_min`/`_max`/`_std`) | with `personalize-epochs` > 0: the same evaluation after the client fine-tuned the received model locally. The gap to `eval_acc` is what a global model leaves on the table for each client |
| `compression-error` | with `compression`: share of the compressed update lost to compression, `‖sent − compressed‖ / ‖compressed‖`, averaged over clients (with `error-feedback` the compressed update is the client's update plus its carried-over residual) |
| `dp-*` | differential-privacy bookkeeping, see [Differential privacy](#differential-privacy) |

To choose a `dirichlet-alpha`, look at the split it produces:
```bash
python scripts/partition_stats.py --partitioner dirichlet --alpha 0.1 --num-partitions 10 --plot results/partition.png
```
It prints the per-client class counts and the mean share of each client's most frequent class (0.1 = uniform split of 10 classes).

## Algorithms
Select with `strategy='<name>'`. Every strategy is implemented in `defai/strategies/` on top of Flower's abstract
`Strategy` interface; Flower's built-in strategies are not imported. The structure follows Flower's own
`FedAvg`/`FedOpt`/`Krum`/DP wrappers (Flower 1.40; tested with 1.42).

| `strategy` | Idea | Main settings | Reference |
|---|---|---|---|
| `fedavg` | Example-weighted average of client models | `server-learning-rate` | McMahan et al., 2017 |
| `fedprox` | Proximal term `μ/2·‖w−w_global‖²` in the local loss | `proximal-mu` | Li et al., 2020 |
| `moon` | Model-contrastive loss vs. global and previous local representation | `moon-mu`, `moon-temperature` | Li et al., 2021 |
| `scaffold` | Control variates correct client drift (option II) | `learning-rate`, `local-epochs` | Karimireddy et al., 2020 |
| `fednova` | Normalised averaging: removes the bias from clients doing unequal numbers of local steps | – | Wang et al., 2020 |
| `qfedavg` | Weights clients by their loss to improve the worst client | `qfedavg-q`, `qfedavg-lipschitz` | Li et al., 2020 |
| `fedavgm` | Server momentum | `server-momentum`, `server-learning-rate` | Hsu et al., 2019 |
| `fedadagrad` `fedadam` `fedyogi` | Adaptive server optimisers | `fedopt-learning-rate`, `beta-1`, `beta-2`, `tau` | Reddi et al., 2021 |
| `fedmedian` | Coordinate-wise median | – | Yin et al., 2018 |
| `fedtrimmedavg` | Coordinate-wise trimmed mean | `trim-ratio` | Yin et al., 2018 |
| `krum` `multikrum` | Pick the update(s) closest to their neighbours | `num-byzantine`, `multikrum-select` | Blanchard et al., 2017 |
| `bulyan` | Recursive Krum selection, then the values closest to the median | `num-byzantine` (needs `4f+3` replies) | El Mhamdi et al., 2018 |

`fednova` and `qfedavg` need two extra numbers from the clients, which they report on their own: `local-steps`
(for FedNova) and `initial-loss`, the loss of the received model before local training (for q-FedAvg). Neither is
combined with differential privacy: the per-client rescaling and the loss weights are not bounded by clipping.

q-FedAvg's update is divided by `Σ_k [q·F_k^(q-1)·‖L·δ_k‖² + L·F_k^q]` with `L = 1/lr` (`qfedavg-lipschitz` replaces `L`; 0
keeps `1/lr`). With a small client learning rate `L` is large, the first term can dominate, and the step becomes much
smaller than FedAvg's. The history therefore has `qfedavg-step-scale`, the factor `Σ L·F_k^q / Σ h_k` (1 = the step is
not shrunk; it is 1 for `q = 0`), so a q-FedAvg run that learns slowly can be diagnosed.

SCAFFOLD and MOON keep per-client state (the client's control variate / previous local model) in
`context.state`, which the simulation engine preserves between rounds.

### Communication: compression
`compression='topk'` keeps the `compression-ratio` largest-magnitude entries of each client's update;
`compression='quantize'` rounds every tensor to `compression-bits` bits with stochastic rounding (unbiased);
`error-feedback=true` remembers what was lost and adds it to the client's next update (it is kept in the client's
state). The lossy update is sent as a dense array, so the server is unchanged, and the client reports what the
encoded message would have cost (`upload-bytes`, shown as `comm-mb-up`). A top-k message costs 8 bytes per kept
entry, so a ratio of 0.1 is about a fifth of the dense float32 upload; 8-bit quantisation is a quarter.
```bash
flwr run . --stream --federation-config "num-supernodes=10" --run-config \
  "compression='topk' compression-ratio=0.1 error-feedback=true"
```
Compression is applied last. With DP the server still clips every update, so the privacy guarantee holds, but the
compression error is not part of the accounting; SCAFFOLD's control variate is sent uncompressed and counted as such.

### Client failures and learning-rate schedule
`dropout-rate=0.3` makes every sampled client fail to answer a train message with probability 0.3 (decided from
`seed`, partition and round, so runs stay reproducible). The server skips the missing replies; with DP the
denominator stays the number of *sampled* clients, so the privacy guarantee is unaffected. `comm-mb-up` below
`comm-mb-down` in `history.json` shows how much was lost.

`lr-decay=0.95` multiplies the clients' learning rate by 0.95 each round (`learning-rate * lr-decay^(round-1)`).

### Robustness experiments
Clients with `partition-id < num-malicious` misbehave. `num-byzantine` is what the *defence* assumes (Krum's `f`);
it is independent of the real number of attackers.
```bash
# 2 of 10 clients flip the sign of their update (x3); every client trains each round
flwr run . --stream --federation-config "num-supernodes=10" --run-config \
  "strategy='fedavg' fraction-train=1.0 attack='sign-flip' num-malicious=2 attack-scale=3.0"
# ... and defend with Krum / trimmed mean / median
flwr run . --stream --federation-config "num-supernodes=10" --run-config \
  "strategy='krum' num-byzantine=2 fraction-train=1.0 attack='sign-flip' num-malicious=2 attack-scale=3.0"
```
`attack` is `label-flip` (labels y → 9−y while training), `sign-flip` (sends `global − scale·update`) or `gaussian`
(sends `global + N(0, scale²)`). Krum/Multi-Krum need at least `2f+3` replies per round and Bulyan `4f+3`; with fewer, they
log a warning (Bulyan lowers its `f`) because the guarantee no longer holds. `fedtrimmedavg` warns when
`trim-ratio × replies < 1`, i.e. when it would trim nothing.

Whatever the strategy, a reply is dropped (with a warning) if its parameters contain NaN/inf or its `num-examples`
is not a positive finite number — a single non-finite value would otherwise survive any averaging, clipping or noise.

### Differential privacy
`dp='fixed'` or `dp='adaptive'` adds central DP to any mean-based strategy (`fedavg`, `fedprox`, `moon`, `fedavgm`,
`fedadagrad`, `fedadam`, `fedyogi`). Each round the server releases
`(1/m)·Σ clip(update_i, C) + N(0, (z·C/m)²)` with `m` clients sampled per round, noise multiplier `z`
(`dp-noise-multiplier`) and clipping norm `C` (`dp-clip-norm`).
```bash
flwr run . --stream --federation-config "num-supernodes=10" --run-config \
  "dp='fixed' dp-noise-multiplier=1.0 dp-clip-norm=1.0 dp-delta=1e-5 dp-max-epsilon=8.0"
flwr run . --stream --federation-config "num-supernodes=10" --run-config \
  "dp='adaptive' dp-side='client' dp-target-quantile=0.5 dp-clip-lr=0.2"
```
- `dp-side='server'`: the server clips the updates. `dp-side='client'`: clients clip before sending (adaptive mode:
  they also report whether they were clipped); the server still enforces the bound, so it holds for faulty clients too.
- `dp='adaptive'` tracks the `dp-target-quantile` of the update norms (Andrew et al., 2019) and spends part of the
  noise budget on the clipped-count query so that the overall noise multiplier stays `dp-noise-multiplier`. The count's
  noise std defaults to `max(clients per round / 20, dp-noise-multiplier)`; if you set `dp-clipped-count-stddev`
  yourself it must exceed half the noise multiplier, which is checked before the first client trains.
- `epsilon` is tracked per round (`dp-epsilon` in `history.json`, plus a `privacy` summary; `null` plus
  `epsilon_unbounded: true` when the noise multiplier is 0). With `dp-max-epsilon > 0`, training stops before a round
  that would exceed the budget (evaluation continues, and `save-model` then saves the initial model if no round ran).
- Only quantities covered by the accounting are published: fixed clipping releases no clipping statistic, adaptive
  clipping releases `dp-noisy-unclipped-fraction` (the noisy fraction that drives the clip-norm update). The exact number
  of clipped clients is never written out. To tune `dp-clip-norm`, use the adaptive mode and read its `dp-clip-norm` curve.
- In client-side adaptive mode a reply whose `dp-unclipped-bit` is not 0 or 1 is dropped, so one faulty client cannot
  steer the clip norm.

What the guarantee does and does not cover:
- It is (ε, δ)-DP for the released model updates under add/remove-one-client adjacency, accounted with Rényi DP.
  **No amplification by client sampling is credited** (Flower samples a fixed number of clients, not Poisson), so ε is a
  conservative upper bound.
- The server-side optimiser runs *after* the noise, so it is post-processing and the guarantee holds for FedAvgM/FedOpt
  as well. Aggregated training/evaluation metrics and the client-sampling pattern are **not** privatised.
- `dp-noise-multiplier` is relative to the clipping norm and the noise is divided by `m`. With the handful of clients
  of a classroom simulation the noise swamps the signal at any meaningful ε — useful privacy needs hundreds of clients
  per round. The pipeline is the point here.
- The noise RNG is seeded with `seed` so experiments are reproducible. That is wrong for real deployments.
- `scaffold` and the robust rules are rejected together with DP: their sensitivity does not follow from clipping.

## Tests
```bash
pip install -e ".[all,test]"
pytest
```
The suite covers the aggregation rules, the accountant (cross-checked against Google's `dp-accounting` and against the
exact Gaussian-mechanism δ(ε)), DP clipping/noise/adaptive-quantile convergence, every server strategy through
`Strategy.start()` on a toy quadratic federation, the local-training hooks against hand-computed SGD, and
`client_app.train` with a real `Context` (per-client state, attacks, client-side DP, reproducibility), config validation
and the helper scripts. It needs no network. GitHub Actions runs `ruff check .` and the suite on every push.

## Notes on where this differs from Flower
- **FedOpt**: following Reddi et al., the second moment starts at `τ²` and there is no Adam bias correction
  (Flower initialises it with zeros and applies a bias correction).
- **Adaptive clipping**: the clip norm is updated with the fraction of *unclipped* updates, `C ← C·exp(−η(b̃ − γ))`
  (Andrew et al.), and the round's noise uses the norm that was used for clipping. Flower 1.42's
  `DifferentialPrivacyServerSideAdaptiveClipping` feeds the *clipped* fraction into that formula, which turns the
  feedback positive: in our check with fixed update norms `1..10` the clip norm shrank towards 0 from below the median
  and grew without bound from above it, instead of settling at the target quantile.
- **DP position**: noise is added to the mean update before the server optimiser, not to the final model.
- **Training loss** reported by clients excludes the FedProx / MOON regularisers so it is comparable across algorithms.
- **Multi-Krum** averages the selected updates uniformly (as in the paper); Flower weights them by example count.
- **Bulyan** selects with Krum applied recursively (as in the paper); Flower picks the `n−2f` selection with one Multi-Krum call.

## Extending
- **New aggregation algorithm**: subclass `defai.strategies.base.FedAvg` and override one of `select_updates`,
  `aggregate_updates`, `update_global` (and `_extend_train_config` / `_extra_train_records` to send extra data to
  clients), then register it in `STRATEGIES` and `build_strategy()` in `defai/strategies/__init__.py`.
- **New local training method**: `train()` in `task.py` and `train()` in `client_app.py`
- **New dataset/model**: `DATASETS` and `build_model()` in `task.py`
- **Real distributed deployment**: [Deployment Engine guide](https://flower.ai/docs/framework/how-to-run-flower-with-deployment-engine.html)

## License
[MIT](LICENSE)
