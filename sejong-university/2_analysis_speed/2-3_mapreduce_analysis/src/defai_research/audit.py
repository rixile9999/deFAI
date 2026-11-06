from __future__ import annotations
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from sklearn.datasets import load_breast_cancer, load_digits
from sklearn.metrics import f1_score, log_loss
from sklearn.model_selection import train_test_split


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _predict(model, x):
    hidden = np.maximum(x @ model[0] + model[1], 0)
    logits = hidden @ model[2] + model[3]
    logits -= logits.max(axis=1, keepdims=True)
    p = np.exp(logits)
    return p / p.sum(axis=1, keepdims=True)


def verify_benchmark(run_dir):
    root = Path(run_dir)
    protocol, final = _read(root / "protocol.json"), _read(root / "final.json")
    source, wire = _read(root / "source_audit.json"), _read(root / "model_wire_audit.json")
    control, deploy = _read(root / "wire_audit.json"), _read(root / "codec_deployment_audit.json")
    with (root / "rounds.csv").open(encoding="utf-8-sig", newline="") as stream:
        rounds = list(csv.DictReader(stream))
    failures, count = [], 0

    def check(condition, label):
        nonlocal count
        count += 1
        if not bool(condition):
            failures.append(label)

    def close(a, b, label):
        check(np.isclose(float(a), float(b), rtol=1e-10, atol=1e-12), label)

    nrounds = int(protocol["rounds"])
    check(final["status"] == "completed", "completion status")
    check(len(rounds) == len(wire) == len(control) == nrounds, "complete round ledgers")
    check(protocol["clients"] == 12 and protocol["groups"] == 4, "declared topology")
    check(final["rounds"] == nrounds and final["codec"] == protocol["codec"], "final protocol identity")
    check(final["test_evaluated"] == protocol["test_evaluated"], "test evaluation identity")
    hashes = protocol["source_sha256"]
    check(isinstance(hashes, dict) and bool(hashes), "source manifest present")
    from .source_files import resolve_source
    for name, expected in hashes.items():
        path = resolve_source(name)
        check(hashlib.sha256(path.read_bytes()).hexdigest() == expected, "source SHA256: " + name)
    dataset = protocol["dataset"]
    if dataset not in {"digits", "breast_cancer"}:
        raise ValueError("Unknown benchmark dataset")
    fixture = load_digits() if dataset == "digits" else load_breast_cancer()
    x, y = fixture.data.astype(np.float32), fixture.target.astype(np.int64)
    train, other = train_test_split(np.arange(len(y)), train_size=.6, stratify=y,
                                   random_state=int(protocol["split_seed"]))
    val, test = train_test_split(other, test_size=.5, stratify=y[other],
                               random_state=int(protocol["split_seed"]) + 1)
    if dataset == "digits":
        x /= 16
    else:
        x = np.clip((x - x[train].mean(0)) / np.maximum(x[train].std(0), 1e-6), -5, 5).astype(np.float32)
    for label, ids in [("train_ids", train), ("validation_ids", val), ("test_ids", test)]:
        check(np.array_equal(source[label], ids), "fixed source split: " + label)
    split = dict(dataset=dataset, split_seed=int(protocol["split_seed"]), train_ids=train.tolist(),
                 validation_ids=val.tolist(), test_ids=test.tolist())
    split_sha = hashlib.sha256(json.dumps(split, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    check(final["fixed_split_sha256"] == split_sha, "fixed split SHA256")
    with np.load(root / "selected_model.npz", allow_pickle=False) as z:
        selected = [z[f"array_{i}"].copy() for i in range(4)]
    with np.load(root / "final_model.npz", allow_pickle=False) as z:
        last = [z[f"array_{i}"].copy() for i in range(4)]
    shapes = [list(a.shape) for a in selected]
    check(all(np.isfinite(a).all() for a in selected + last), "finite stored model parameters")
    sums, recalls, frame_index, processed, steps_total = {}, [], {}, 0, 0
    with np.load(root / "retained_masks.npz", allow_pickle=False) as masks:
        check(len(masks.files) == nrounds * 12, "complete retained masks")
        for number, (row, ledger, messages) in enumerate(zip(rounds, wire, control), 1):
            tag = f"round {number}: "
            check(int(row["round"]) == ledger["round"] == messages["round"] == number, tag + "identity")
            phase = "drift" if number >= protocol["drift_round"] else "initial"
            parts = [np.asarray(a, dtype=np.int64) for a in source["phase_source_ids"][phase]]
            check(row["phase"] == phase and len(parts) == 12, tag + "phase")
            check(set(np.concatenate(parts).tolist()) == set(train.tolist()), tag + "training-only sources")
            keep = [masks[f"r{number:02d}_c{cid:02d}"] for cid in range(12)]
            check(all(m.dtype == np.bool_ and m.shape == a.shape for m, a in zip(keep, parts)), tag + "mask dimensions")
            before, after = np.concatenate(parts), np.concatenate([a[m] for a, m in zip(parts, keep)])
            lost = len(set(before.tolist()) - set(after.tolist()))
            duplicates = len(before) - len(set(before.tolist()))
            remaining = len(after) - len(set(after.tolist()))
            for key, value in dict(records_before=len(before), records_after=len(after),
                                   true_duplicate_copies=duplicates, remaining_duplicate_copies=remaining,
                                   duplicate_removed=len(before)-len(after), lost_unique_records=lost).items():
                check(int(row[key]) == value, tag + key)
            check(lost == 0, tag + "unique source preservation")
            recall = (duplicates - remaining) / duplicates if duplicates else None
            if recall is None:
                check(row["dedup_recall"] in ("", "None", "null"), tag + "undefined duplicate recall")
            else:
                close(row["dedup_recall"], recall, tag + "duplicate recall")
                if phase == "drift":
                    recalls.append(recall)
            fitted = [cid for cid, m in enumerate(keep) if m.any()]
            examples = len(after) * int(protocol["local_epochs"])
            steps = sum(int(protocol["local_epochs"]) * int(np.ceil(keep[cid].sum() / protocol["batch_size"])) for cid in fitted)
            check(int(row["examples_processed"]) == examples and int(row["optimizer_steps"]) == steps, tag + "actual retained learning workload")
            processed += examples; steps_total += steps
            groups = {cid: (cid // 3 + int(phase == "drift" and cid % 3 == 0)) % 4 for cid in fitted}
            group_counts = {g: sum(int(keep[cid].sum()) for cid in fitted if groups[cid] == g) for g in set(groups.values())}
            expected_channels = {("cloud", "down", "central", g) for g in group_counts}
            expected_channels |= {("cloud", "up", g, "central") for g in group_counts}
            expected_channels |= {("edge", "down", groups[cid], cid) for cid in fitted}
            expected_channels |= {("edge", "up", cid, groups[cid]) for cid in fitted}
            frames = ledger["frames"]
            check(len(frames) == len(expected_channels) and {tuple(f["channel"]) for f in frames} == expected_channels, tag + "complete model links")
            payload, context, up, down, tf_bytes, tf_count, int8 = 0, 0, 0, 0, 0, 0, 0
            for frame in frames:
                channel, direction, kind = frame["channel"], frame["direction"], frame["kind"]
                frame_index[(number, tuple(channel))] = frame
                token = channel[3] if direction == "down" else channel[2]
                weight = 0 if direction == "down" else int(keep[token].sum()) if kind == "client_up" else group_counts[token]
                header = json.dumps(dict(round=number, kind=kind, token=int(token), n=weight, shapes=shapes,
                    codec_id=2 if protocol["codec"] == "transformer" and direction == "up" else 1),
                    sort_keys=True, separators=(",", ":")).encode()
                check(frame["json_header_bytes"] == len(header), tag + "context length")
                check(frame["json_header_sha256"] == hashlib.sha256(header).hexdigest(), tag + "context hash")
                check(frame["frame_bytes"] == frame["header_bytes"] + frame["payload_bytes"], tag + "frame accounting")
                check(frame["aggregation_weight"] == weight, tag + "received sample weight")
                payload += frame["frame_bytes"]; context += len(header)
                amount = frame["frame_bytes"] + len(header)
                up += amount if direction == "up" else 0
                down += amount if direction == "down" else 0
                if direction == "up":
                    tf_count += int(protocol["codec"] == "transformer")
                    tf_bytes += frame["frame_bytes"] if protocol["codec"] == "transformer" else 0
                    int8 += int(protocol["codec"] != "transformer")
            control_bytes = sum(len(bytes.fromhex(p["hex"])) for p in messages["packets"])
            check(int(row["metadata_bytes"]) == control_bytes, tag + "actual control buffers")
            deployment_bytes = sum(p["payload_bytes"] for p in deploy) if number == 1 else 0
            values = dict(model_payload_bytes=payload, model_protocol_header_bytes=context,
                model_wire_bytes=payload+context, up_bytes=up, down_bytes=down,
                graph_control_both_hops_bytes=2*control_bytes, codec_deployment_bytes=deployment_bytes,
                total_bytes=payload+context+2*control_bytes+deployment_bytes,
                transformer_frames=tf_count, transformer_frame_bytes=tf_bytes, int8_frames=int8)
            for key, value in values.items():
                check(int(row[key]) == ledger[key] == value, tag + key)
                sums[key] = sums.get(key, 0) + value
    for key, value in sums.items():
        check(final["sum_" + key] == value, "cumulative " + key)
    check(final["sum_examples_processed"] == processed and final["sum_optimizer_steps"] == steps_total, "cumulative learning workload")
    check(final["sum_lost_unique_records"] == 0, "cumulative unique-source preservation")
    samples = _read(root / "codec_samples.json")
    with np.load(root / "codec_samples.npz", allow_pickle=False) as z:
        for sample in samples:
            frame = frame_index[(sample["round"], tuple(sample["channel"]))]
            raw = bytes.fromhex(sample["frame_hex"])
            check(len(raw) == frame["frame_bytes"] and hashlib.sha256(raw).hexdigest() == frame["frame_sha256"], "saved frame sample bytes and hash")
            prefix = sample["array_prefix"]
            for suffix, key in [("corrected", "input_sha256"), ("received", "decoded_sha256")]:
                check(hashlib.sha256(z[prefix + "_" + suffix].tobytes()).hexdigest() == frame[key], "saved frame sample: " + suffix)
    endpoints = {tuple(p["destination"]) for p in deploy}
    expected = {("group", i) for i in range(4)} | {("client", i) for i in range(12)}
    check((len(deploy) == 16 and endpoints == expected) if protocol["codec"] == "transformer" else not deploy, "codec deployment endpoints")
    for packet in deploy:
        check(packet["hop_count"] == 1 and packet["payload_bytes"] == packet["deployment_header_bytes"] + packet["packed_artifact_bytes"], "deployment frame accounting")
    check(final["post_drift_dedup_recall_defined_rounds"] == len(recalls), "defined post-drift recall rounds")
    if recalls:
        close(final["post_drift_dedup_recall"], np.mean(recalls), "mean post-drift recall")
    else:
        check(final["post_drift_dedup_recall"] is None, "undefined mean post-drift recall")
    if protocol["validation_evaluated"]:
        with np.load(root / "validation_predictions.npz", allow_pickle=False) as z:
            check(np.array_equal(z["source_ids"], val) and np.array_equal(z["y"], y[val]), "validation identities")
            p = z["probabilities"]
        check(p.shape == (nrounds, len(val), len(np.unique(y))), "validation probability dimensions")
        losses = [float(log_loss(y[val], a, labels=np.arange(p.shape[2]))) for a in p]
        best = int(np.argmin(losses)) + 1
        check(final["selected_round"] == best, "minimum-validation-loss checkpoint")
        close(final["selected_val_log_loss"], losses[best-1], "selected validation loss")
        for row, loss in zip(rounds, losses):
            close(row["val_log_loss"], loss, "stored round validation loss")
        for row, probability in zip(rounds, p):
            close(row["val_macro_f1"], f1_score(y[val], probability.argmax(1), average="macro", zero_division=0), "stored round validation F1")
        close(final["selected_val_macro_f1"], rounds[best-1]["val_macro_f1"], "selected validation F1")
        check(np.allclose(_predict(selected, x[val]), p[best-1], rtol=1e-6, atol=1e-7), "selected model validation probabilities")
        check(np.allclose(_predict(last, x[val]), p[-1], rtol=1e-6, atol=1e-7), "final model validation probabilities")
    else:
        check(not (root / "validation_predictions.npz").exists() and final["selected_round"] == nrounds, "calibration excludes validation predictions")
    if protocol["test_evaluated"]:
        with np.load(root / "test_predictions.npz", allow_pickle=False) as z:
            check(np.array_equal(z["source_ids"], test) and np.array_equal(z["y"], y[test]), "test identities")
            p, q = z["probabilities"], z["final_probabilities"]
        check(np.allclose(_predict(selected, x[test]), p, rtol=1e-6, atol=1e-7), "selected model test probabilities")
        check(np.allclose(_predict(last, x[test]), q, rtol=1e-6, atol=1e-7), "final model test probabilities")
        close(final["test_macro_f1"], f1_score(y[test], p.argmax(1), average="macro", zero_division=0), "selected test F1")
        close(final["final_test_macro_f1"], f1_score(y[test], q.argmax(1), average="macro", zero_division=0), "final test F1")
        close(final["test_accuracy"], np.mean(p.argmax(1) == y[test]), "selected test accuracy")
    else:
        check(not (root / "test_predictions.npz").exists() and not any(k.startswith("test_") or k.startswith("final_test_") for k in final if k != "test_evaluated"), "test artifacts excluded")
    return {"passed": not failures, "check_count": count, "failures": failures,
            "scope": "Read-only source, application-ledger, source-preservation and saved-prediction checks; no training replay or WAN verification"}
