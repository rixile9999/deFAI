# Completed result interchange, version 1

`federated-result-v1` transfers a completed experiment's numeric results and
inference artifacts. The consumer does not need deFAI or Flower. This is a file
exchange contract, not a distributed training protocol or a training-resume file.
The first profile is PyTorch numeric weights plus a TorchScript inference model.
Other producers can implement the same contract without naming themselves deFAI.

## Export

Install the existing `flower` optional group, then:

```sh
defai run --strategy fedadam-kalman --dataset synthetic --clients 2 --rounds 2 --set save-model=true --output run-001
defai export-result run-001 --output exchange-001
```

Export requires a trusted local `history.json` and `final_model.pt` produced by
the common image-model workflow. Supported model datasets are those accepted by
`defai.task.build_model`, including synthetic, MNIST/Fashion-MNIST mirrors and
CIFAR-10. The exporter never downloads data. Disease-specific and vertical model
exports are outside this profile. It preserves the complete source result and
numeric state, loads only numeric model state, writes into a fresh directory,
and publishes after checking that the source files did not change.

## Bundle contract

| Field/file | Meaning |
|---|---|
| `result.json` / `format` | Exactly `federated-result-v1`; unsupported versions are rejected |
| `origin.engine` | Nonempty producer name; does not select an algorithm in the consumer |
| `result.config` | Original configuration object |
| `result.history` | Nonempty list of objects with strictly increasing, nonnegative integer `round`; round 0 may be initial evaluation |
| `metric_specs` | Metric name to nonempty `unit`, boolean `higher_is_better`, and optional inclusive numeric `range` |
| `model.framework` / `inference_format` | `pytorch` / `torchscript` in this profile |
| `model.input_schema` / `output_schema` | Input/output descriptions. deFAI emits pre-normalized tensors, not a raw-image preprocessing service |
| `model.tensors` | Array name to exact `shape` and NumPy `dtype.str` |
| `artifacts.weights` | Relative file name and SHA-256 of numeric-only `weights.npz` |
| `artifacts.inference` | Relative file name and SHA-256 of inference `model.ts` |
| `source_history_sha256` / `source_model_sha256` | Producer provenance for the original files; optional for other producers |

Example metric definitions:

```json
{
  "test_acc": {"unit": "ratio", "range": [0, 1], "higher_is_better": true},
  "test_loss": {"unit": "cross_entropy", "higher_is_better": false}
}
```

History cells are finite numeric scalars or null; unknown numeric diagnostic
columns remain in the original result. Every declared metric has an observed
value. Keep units and evaluation conditions: a ratio must not silently become a
percentage, and regression error must not be relabelled accuracy. Configuration
and producer-specific result metadata are retained as JSON data.

The manifest is UTF-8 without BOM and at most 4 MiB. Artifact paths are single
file names, distinct from each other and `result.json`; no absolute paths,
directory separators or symlinks outside the bundle. Each artifact is at most
256 MiB. Numeric weights contain at most 4,096 named, nonempty arrays, 512 MiB
total numeric data, and NPY version 1/2 members only. Names use ASCII letters,
digits, underscore, dot or hyphen (1–256 characters). Shapes have at most 32
positive integer dimensions; scalar shape `[]` is allowed. Only canonical NumPy
boolean, integer and floating dtypes are allowed; no object arrays/pickle.

## Platform consumption

The companion platform can stage the directory in its configured external-result
inbox, inspect it, and register the immutable copied inference artifact as an
ordinary model version. Native ridge/FedAvg jobs continue through their existing
paths. An externally produced FedAdam result must not be re-averaged as FedAvg.
See `pcnrnd/Federated_Learning`'s external-result API documentation.

Hashes detect changed bytes, not producer authenticity. Only operator-approved,
trusted inference artifacts belong in the inbox. Import never executes
TorchScript and does not prove its safety, accuracy, or deployment readiness;
packaging health and prediction checks remain separate. Optimizer/client state,
privacy-accountant resume, real network cost and institutional deployment are
not established by successful file exchange.
