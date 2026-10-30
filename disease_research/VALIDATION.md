# Publication-time validation (2026-10-29)

Research-stage source publication. These checks were run against the submission
copies, not inferred from stored reports. No new clinical performance is claimed.

| Checked | Result | Environment |
|---|---|---|
| SHA-256 comparison with authorized local sources | 209/209 files match | Python 3.13 |
| Python source parsing | 114/114 source files parse | Python 3.13 |
| Unpublished-method path and text scan | No matches in publication files | Full text/path scan |
| GAN research: test_federated_algorithms.py, test_skin_bias.py, test_algorithm_report.py | 10 tests passed | Existing system Python/PyTorch |
| Policy research: test_policy_stats.py, test_region_expansion.py | 25 tests passed; two dependency deprecation warnings | Existing .venv-policy-research interpreter |
| HbA1c fl_comparison/test_learning.py | Seven learning invariants passed | Existing system Python/PyTorch |
| Paillier signed fixed-point sum self-test | Passed | Existing system Python |
| Federated Burr self-test | Analytic gradient and federated/pooled fit comparisons passed | Existing system Python/SciPy |
| NeoJaundice provenance | Local archive size and MD5 match official Figshare file metadata | Official API and local archive |

Run each command from the corresponding project directory:

```powershell
# 01/GAN_research
python -m pytest tests/test_federated_algorithms.py tests/test_skin_bias.py tests/test_algorithm_report.py -q
# 02/fcausal_HIERARCHICAL/fcausal_hierarchical (use its research interpreter)
python -m pytest tests/test_policy_stats.py tests/test_region_expansion.py -q
# 03/abm_fl_simulation
python fl_pipeline/paillier.py
python fl_pipeline/federated_burr.py
# 04/hba1c_fl_research/experiments/fl_comparison
python test_learning.py
```

| Done | Not done |
|---|---|
| Publish source snapshots and data references | Full model retraining and reproduction of all historical result tables |
| Validate selected algorithm/statistics invariants | Full data-dependent tests requiring omitted images, splits or results |
| Verify NeoJaundice archive provenance | Fresh verification of every historical dataset registry entry |
| Copy Android source and build configuration | Android compilation, device deployment, or end-to-end mobile validation |
| Test synthetic federated estimation | Full population simulation or verification of population/place input origins |
| Preserve research-stage limitations | Clinical validation or real policy-effect validation |

The system Python initially could not collect policy tests because Flower was
absent. Those tests were rerun successfully using the existing policy research
environment. No global dependencies were installed or changed.

Terms: SHA-256 and MD5 are file digests; HbA1c means glycated hemoglobin; GAN means
generative adversarial network; Paillier is additive homomorphic encryption.
