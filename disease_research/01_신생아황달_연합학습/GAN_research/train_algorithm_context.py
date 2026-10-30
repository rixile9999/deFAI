"""Independent experiment worker; each FL run still trains clients sequentially."""
import argparse
import json

import numpy as np
import torch
from runtime import DEVICE, device_name, require_device

from run_algorithm_study import (ROOT, OUTPUT, CONTEXTS, CONFIGS, FAMILIES, cache,
                                 MultiModalNet, seed_all, cpu_state, train_one)
from skin_bias import normalize


def main(context_index):
    torch.set_num_threads(2); torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("high")
    condition, bias = CONTEXTS[context_index]
    data = np.load(ROOT / "data/real_data.npz"); tensors = cache(data)
    synthetic = np.load(ROOT / "data/gan_paired_synthetic.npz")
    source = synthetic["source_index"]
    assert (data["split"][source] == "train").all()
    np.testing.assert_array_equal(data["client"][source], synthetic["client"])
    extra = {"images": normalize(torch.tensor(synthetic["images"], dtype=torch.float32).permute(0, 3, 1, 2)/255),
             "source": torch.tensor(source)}
    training = [np.flatnonzero((data["split"] == "train") & (data["client"] == c)) for c in range(3)]
    synthetic_indices = [torch.tensor(np.flatnonzero(synthetic["client"] == c)) for c in range(3)]
    selected = []
    for seed in [17, 29, 43]:
        seed_all(seed); model = MultiModalNet().to(DEVICE); initial = cpu_state(model)
        configs = CONFIGS if seed == 17 else selected
        results = [train_one(model, initial, data, tensors, extra, training, synthetic_indices,
                             condition, bias, config, seed, 20) for config in configs]
        if seed == 17:
            selected = [min([r for r in results if r["config"]["family"] == family],
                            key=lambda r: r["validation_mae"])["config"] for family in FAMILIES]
            scaffold_lr = next(c["lr"] for c in selected if c["family"] == "scaffold")
            sgd_lr = next(c["lr"] for c in selected if c["family"] == "fedavg_sgd")
            if scaffold_lr != sgd_lr:
                selected.append(next(c for c in CONFIGS if c["family"] == "fedavg_sgd" and c["lr"] == scaffold_lr))
        del model, initial; torch.cuda.empty_cache()
    (OUTPUT / f"context_{context_index}_complete.json").write_text(json.dumps({"condition": condition, "bias": bias,
            "selected_configurations": selected, "test_evaluated": False}, indent=2))
    print(f"CONTEXT {context_index} TRAINING COMPLETE; TEST STILL LOCKED", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context-index", required=True, type=int, choices=range(4))
    main(parser.parse_args().context_index)
