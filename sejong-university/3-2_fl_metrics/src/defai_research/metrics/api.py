from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import Any

import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


_DEFINITIONS = (
    (1, "Precision", "proportion", "Correctness of classwise positive predictions", "macro_precision"),
    (2, "Recall", "proportion", "Detection of true members of each class", "macro_recall"),
    (3, "F1", "proportion", "Per-class precision/recall balance", "macro_f1"),
    (4, "TC", "second; round", "Measured time to a fixed sustained validation target", "convergence"),
    (5, "Communication", "byte", "Actual application frames and bytes to equal quality", "communication"),
    (6, "MUI", "bit; coverage", "Alignment of available real modalities alongside sensor ablation utility", "multimodal"),
    (7, "Adaptation", "F1; F1-round; round", "Response to input loss/return against a fixed reference trajectory", "adaptation"),
    (8, "KID", "F1 difference", "Useful global knowledge transfer against isolated and ablated models", "knowledge_integration"),
    (9, "RU", "CPU-second; MiB", "Separate process CPU time and sampled resident memory", "resources"),
    (10, "Privacy", "epsilon; delta; attack score", "Declared mechanism/transcript accounting and empirical exposure", "privacy"),
    (11, "H(D)", "bit", "Empirical categorical or fixed-bin local data diversity", "data_entropy_bits"),
    (12, "Joint information", "bit", "Saved joint-count information with modality/node ablations", "joint_information"),
    (13, "Transmission loss", "relative L2; MSE; cosine", "Actual reference-versus-received update distortion", "distortion"),
    (14, "IRI", "recovered fraction", "Restored update error and utility with explicit available headroom", "recovery"),
    (15, "Transfer rate", "application bit/second", "Measured framed transport throughput with timing scope", "transfer_rate"),
    (16, "Cross-node information", "bit; F1 difference", "Conditional predictive information and node-ablation utility", "cross_node_information"),
)


def metric_definitions() -> list[dict[str, Any]]:
    return [dict(id=i, name=name, unit=unit, purpose=purpose, field=field)
            for i, name, unit, purpose, field in _DEFINITIONS]


def _array(value: Any, name: str, *, ndim: int | None = None) -> np.ndarray:
    a = np.asarray(value, dtype=np.float64)
    if ndim is not None and a.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} must contain finite values")
    return a


def _probabilities(value: Any) -> np.ndarray:
    p = _array(value, "probabilities", ndim=2)
    if not len(p) or p.shape[1] < 2 or np.any(p < 0) or not np.allclose(p.sum(1), 1, atol=1e-6, rtol=0):
        raise ValueError("Expected nonempty normalized N-by-C probabilities, C>=2")
    return p


def classification_metrics(labels: Any, probabilities: Any) -> dict[str, Any]:
    p = _probabilities(probabilities)
    y = np.asarray(labels)
    if y.ndim != 1 or len(y) != len(p) or not np.issubdtype(y.dtype, np.integer):
        raise ValueError("Labels must be a matching integer vector")
    if np.any(y < 0) or np.any(y >= p.shape[1]):
        raise ValueError("Label outside probability class range")
    classes = np.arange(p.shape[1])
    prediction = p.argmax(1)
    cm = confusion_matrix(y, prediction, labels=classes)
    precision, recall, f1, support = precision_recall_fscore_support(
        y, prediction, labels=classes, zero_division=0)
    return {
        "macro_precision": float(precision.mean()), "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()), "accuracy": float(np.mean(y == prediction)),
        "classes": classes.tolist(), "confusion_matrix": cm.tolist(),
        "per_class": [dict(class_id=int(c), precision=float(precision[c]), recall=float(recall[c]),
                           f1=float(f1[c]), support=int(support[c]), predicted=int(cm[:, c].sum())) for c in classes],
        "zero_division": 0,
    }


def entropy_counts(counts: Any) -> float:
    a = _array(counts, "counts")
    if np.any(a < 0):
        raise ValueError("Counts cannot be negative")
    if a.sum() == 0:
        return 0.
    p = a[a > 0] / a.sum()
    return float(-np.sum(p * np.log2(p)))


def categorical_joint(*variables: Any, sizes: Sequence[int]) -> np.ndarray:
    sizes = tuple(sizes)
    if not sizes or len(sizes) != len(variables) or any(not isinstance(s, (int, np.integer)) or s < 1 for s in sizes):
        raise ValueError("Positive integer size required for every variable")
    arrays = [np.asarray(v) for v in variables]
    if any(a.ndim != 1 or not np.issubdtype(a.dtype, np.integer) for a in arrays):
        raise ValueError("Variables must be one-dimensional integer arrays")
    if len({len(a) for a in arrays}) != 1:
        raise ValueError("Variables must have equal length")
    if any(np.any((a < 0) | (a >= size)) for a, size in zip(arrays, sizes)):
        raise ValueError("Category outside declared support")
    out = np.zeros(sizes, dtype=np.int64)
    np.add.at(out, tuple(arrays), 1)
    return out


def conditional_mi(joint: Any, left_axes: Sequence[int], right_axes: Sequence[int],
                   condition_axes: Sequence[int] = ()) -> float:
    joint = _array(joint, "joint")
    groups = [tuple(g) for g in (left_axes, right_axes, condition_axes)]
    axes = sum(groups, ())
    if not groups[0] or not groups[1] or len(set(axes)) != len(axes) or set(axes) != set(range(joint.ndim)):
        raise ValueError("Left, right and conditioning axes must partition joint axes")
    if np.any(joint < 0) or joint.sum() <= 0:
        raise ValueError("A nonempty nonnegative contingency tensor is required")
    def h(keep):
        drop = tuple(i for i in range(joint.ndim) if i not in keep)
        return entropy_counts(joint.sum(axis=drop))
    a, b, c = groups
    return float(max(0., h(a + c) + h(b + c) - h(c) - h(a + b + c)))


def information_metrics(labels: Any, probabilities: Any) -> dict[str, Any]:
    p = _probabilities(probabilities)
    classification_metrics(labels, p)
    y = np.asarray(labels)
    joint = categorical_joint(y, p.argmax(1), sizes=(p.shape[1], p.shape[1]))
    hy, hm, hj = entropy_counts(joint.sum(1)), entropy_counts(joint.sum(0)), entropy_counts(joint)
    return dict(data_entropy_bits=hy, prediction_entropy_bits=hm, joint_entropy_bits=hj,
                mutual_information_bits=max(0., hy + hm - hj), joint_counts=joint.tolist(),
                scope="Empirical categorical labels and hard predictions, not continuous feature entropy")


def convergence_time(quality: Any, elapsed_seconds: Any, *, threshold: float, streak: int = 3) -> dict[str, Any]:
    q, t = _array(quality, "quality", ndim=1), _array(elapsed_seconds, "elapsed_seconds", ndim=1)
    if len(q) != len(t) or not len(q) or not isinstance(streak, int) or streak < 1:
        raise ValueError("Matching nonempty observations and a positive integer streak are required")
    if not math.isfinite(threshold) or np.any(t < 0) or np.any(np.diff(t) < 0):
        raise ValueError("Finite threshold and monotonic nonnegative elapsed times are required")
    index = next((i for i in range(streak - 1, len(q)) if np.all(q[i-streak+1:i+1] >= threshold)), None)
    return dict(threshold=float(threshold), sustained_rounds=streak, reached=index is not None,
                completion_round=index + 1 if index is not None else None,
                elapsed_seconds=float(t[index]) if index is not None else None,
                observations=len(q))


def communication_metrics(frame_bytes: Any, *, reference_bytes: int | None = None) -> dict[str, Any]:
    a = _array(frame_bytes, "frame_bytes", ndim=1)
    if not len(a) or np.any(a < 0) or np.any(a != np.floor(a)):
        raise ValueError("Frame byte counts must be nonnegative integers")
    total = int(a.sum())
    if reference_bytes is not None and reference_bytes <= 0:
        raise ValueError("Reference byte count must be positive")
    return dict(total_framed_bytes=total, frame_count=len(a),
                savings_fraction=None if reference_bytes is None else 1-total/reference_bytes,
                scope="Application frames only; no unobserved TCP/IP/TLS overhead")


def multimodal_alignment(probabilities: Any, *, availability: Any | None = None) -> dict[str, Any]:
    p = _array(probabilities, "modality probabilities", ndim=3)
    if p.shape[0] < 2:
        raise ValueError("At least two real modalities are required")
    for view in p:
        _probabilities(view)
    available = np.ones(p.shape[:2], dtype=bool) if availability is None else np.asarray(availability)
    if available.shape != p.shape[:2] or np.any((available != 0) & (available != 1)):
        raise ValueError("Availability must be a matching binary modality-by-record mask")
    available = available.astype(bool)
    mask = available.all(0)
    divergence = None
    if mask.any():
        q = p[:, mask]
        midpoint = q.mean(0)
        with np.errstate(divide="ignore", invalid="ignore"):
            terms = np.where(q > 0, q * np.log2(np.maximum(q, 1e-300) / np.maximum(midpoint, 1e-300)), 0.)
        divergence = float(terms.sum(2).mean())
    return dict(predictive_jsd_bits=divergence, coverage_by_modality=available.mean(1).tolist(),
                jointly_available_records=int(mask.sum()), total_records=p.shape[1],
                interpretation="Alignment only; useful integration requires sensor-ablation utility")


def adaptation_metrics(clean_quality: Any, changed_quality: Any, *, shock_start: int,
                       return_start: int | None = None, tolerance: float = .02, streak: int = 3) -> dict[str, Any]:
    clean, changed = _array(clean_quality, "clean quality", ndim=1), _array(changed_quality, "changed quality", ndim=1)
    if len(clean) != len(changed) or not 2 <= shock_start <= len(clean) or tolerance < 0 or streak < 1:
        raise ValueError("Matching trajectories, a pre-shock observation and valid recovery settings are required")
    end = len(clean) if return_start is None else return_start - 1
    if end < shock_start or end > len(clean):
        raise ValueError("Return round must follow shock and lie within trajectory or immediately after it")
    baseline = float(clean[shock_start-2])
    def recovery(start, stop):
        q = changed[start-1:stop]
        found = next((i for i in range(streak-1, len(q)) if np.all(q[i-streak+1:i+1] >= baseline-tolerance)), None)
        return None if found is None else int(found + 1)
    return dict(pre_shock_quality=baseline, immediate_quality_drop=float(clean[shock_start-1]-changed[shock_start-1]),
                shock_mean_quality=float(changed[shock_start-1:end].mean()),
                positive_deficit_quality_rounds=float(np.maximum(clean-changed, 0)[shock_start-1:end].sum()),
                recovery_observations_after_shock=recovery(shock_start, end),
                recovery_observations_after_return=None if return_start is None or return_start > len(clean) else recovery(return_start, len(clean)),
                tolerance=float(tolerance), sustained_observations=int(streak),
                censoring="null means not observed within the declared observation period")


def knowledge_integration(global_score: float, isolated_scores: Any, *, ablated_score: float | None = None) -> dict[str, Any]:
    scores = _array(isolated_scores, "isolated scores", ndim=1)
    if not len(scores) or not math.isfinite(global_score) or (ablated_score is not None and not math.isfinite(ablated_score)):
        raise ValueError("Finite global/local/ablation scores are required")
    return dict(global_score=float(global_score), isolated_model_mean=float(scores.mean()),
                global_minus_isolated=float(global_score-scores.mean()),
                global_minus_ablated=None if ablated_score is None else float(global_score-ablated_score),
                isolated_model_count=len(scores))


def resource_usage(cpu_seconds: Any, sampled_rss_bytes: Any) -> dict[str, Any]:
    cpu, rss = _array(cpu_seconds, "CPU seconds", ndim=1), _array(sampled_rss_bytes, "sampled RSS", ndim=1)
    if not len(cpu) or not len(rss) or np.any(cpu < 0) or np.any(rss < 0):
        raise ValueError("Nonempty nonnegative resource observations are required")
    return dict(total_process_cpu_seconds=float(cpu.sum()), sampled_peak_rss_MiB=float(rss.max()/2**20),
                cpu_observation_count=len(cpu), rss_sample_count=len(rss),
                scope="Sum CPU across declared processes; RSS is a sampled peak, not an instantaneous bound")


def gaussian_privacy_accountant(rounds: int, sigma: float, delta: float = 1e-5) -> dict[str, Any]:
    if not isinstance(rounds, int) or rounds < 1 or not math.isfinite(sigma) or sigma < 0 or not 0 < delta < 1:
        raise ValueError("Positive integer rounds, nonnegative sigma and 0<delta<1 required")
    if sigma == 0:
        return dict(rho=None, epsilon=None, delta=delta, rounds=rounds, status="no_noise_no_finite_bound")
    rho = rounds/(2*sigma*sigma)
    return dict(rho=rho, epsilon=rho+2*math.sqrt(rho*math.log(1/delta)), delta=delta,
                rounds=rounds, status="ideal_Gaussian_full_transcript_bound")


def update_distortion(reference: Any, received: Any) -> dict[str, Any]:
    a, b = _array(reference, "reference"), _array(received, "received")
    if a.shape != b.shape or not a.size:
        raise ValueError("Matching nonempty update arrays are required")
    a, b = a.ravel(), b.ravel()
    error, norm = float(np.linalg.norm(a-b)), float(np.linalg.norm(a))
    product = norm*float(np.linalg.norm(b))
    return dict(relative_l2=error/norm if norm > 0 else None, mse=float(np.mean((a-b)**2)),
                cosine=float(np.dot(a,b)/product) if product > 0 else None,
                absolute_l2=error, reference_l2=norm, element_count=len(a))


def recovery_metrics(reference: Any, damaged: Any, restored: Any, *,
                     reference_utility: float | None = None, damaged_utility: float | None = None,
                     restored_utility: float | None = None, tolerance: float = 1e-8) -> dict[str, Any]:
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("Tolerance must be finite and nonnegative")
    damage, repair = update_distortion(reference, damaged), update_distortion(reference, restored)
    den = damage["absolute_l2"]
    result = dict(vector_recovery_fraction=None if den <= tolerance else (den-repair["absolute_l2"])/den,
                  damage_l2=den, restored_l2=repair["absolute_l2"], utility_recovery_fraction=None,
                  utility_headroom=None, utility_status="not_provided")
    utilities = (reference_utility, damaged_utility, restored_utility)
    if any(v is not None for v in utilities):
        if any(v is None or not math.isfinite(v) for v in utilities):
            raise ValueError("Provide all three finite utility scores")
        headroom = float(reference_utility-damaged_utility)
        result.update(utility_headroom=headroom,
                      utility_recovery_fraction=None if headroom <= tolerance else (restored_utility-damaged_utility)/headroom,
                      utility_status="nonpositive_or_tiny_headroom" if headroom <= tolerance else "defined")
    return result


def transmission_rate(framed_bytes: int, elapsed_seconds: float, *, scope: str) -> dict[str, Any]:
    if not isinstance(framed_bytes, (int, np.integer)) or framed_bytes < 0 or not math.isfinite(elapsed_seconds) or elapsed_seconds <= 0 or not scope.strip():
        raise ValueError("Nonnegative bytes, positive measured duration and explicit timing scope required")
    return dict(application_bits_per_second=8*int(framed_bytes)/elapsed_seconds,
                framed_bytes=int(framed_bytes), elapsed_seconds=float(elapsed_seconds), timing_scope=scope)


def evaluate16(*, labels: Any | None = None, probabilities: Any | None = None,
               observations: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    measured = dict(observations or {})
    if (labels is None) != (probabilities is None):
        raise ValueError("Labels and probabilities must be supplied together")
    if labels is not None:
        measured.update(classification_metrics(labels, probabilities))
        info = information_metrics(labels, probabilities)
        measured.setdefault("data_entropy_bits", info["data_entropy_bits"])
        measured.setdefault("joint_information", info)
    return [{**definition, "value": measured.get(definition["field"]),
             "status": "measured" if definition["field"] in measured and measured[definition["field"]] is not None else "not_measured"}
            for definition in metric_definitions()]
