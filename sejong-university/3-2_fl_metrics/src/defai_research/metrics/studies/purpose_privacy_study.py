from __future__ import annotations

import os
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"
os.environ["FLWR_TELEMETRY_ENABLED"] = "0"

import csv
import hashlib
import importlib.metadata
import io
import json
import math
from pathlib import Path
import secrets
import shutil
import sys
import time
import zipfile

from ._paths import ROOT, source_path, relative_path
import numpy as np
from sklearn.datasets import load_digits
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from scipy.stats import t as student_t
from flwr.common import FitIns, ndarrays_to_parameters, parameters_to_ndarrays
from . import _helpers as core

OUT = ROOT / "results/purpose_privacy"
PRIVATE = OUT / "research_private"
RELEASE = OUT / "release_candidate"
PLAN = {
    "schema": 1, "stage": "frozen_before_training", "dataset": "sklearn Digits",
    "target_cohorts": 5, "shadow_cohorts": 1, "sigma_values": [0., 2., 8., 32.],
    "rounds": 20, "client_slots": 12, "clip_norm": .2, "delta": 1e-5,
    "local_epochs": 2, "batch_size": 32, "base_learning_rate": .14,
    "learning_rate_schedule": "0.14/sqrt(1+(round-1)/20)", "l2": .0001,
    "public_initial_model_seed": 290071, "public_client_shuffle_seed": 290073,
    "public_shadow_target_pool_split_seed": 290077,
    "preprocessing": "Fixed public pixel division by16; no fitted global scaler, graph, filter, hash, deduplication, sample-count weighting or private hyperparameter selection.",
    "partition": "First disjoint shadow/target record pools. Within each pool, private stratified60% train/40% nonmember split. Fixed class-round-robin12 client containers are established before the protected mechanism. Adjacency is replacement of one such fixed client container, with all others fixed.",
    "participation": "All12 fixed slots every round; an empty replacement contributes zero; no client sampling or variable public participation list.",
    "mechanism": "Per-client vector delta clipped to L2 C, uniform mean over12, independent Gaussian coordinates std=sigma*2C/12; previous released model is the adaptive public context.",
    "checkpoint": "Fixed round20 only for utility/attack. All20 round models in the selected release transcript. No validation selection, early stop, retrial or utility-based release choice.",
    "accountant": "Perround rho=1/(2sigma^2), rho_total=T/(2sigma^2), epsilon=rho+2sqrt(rho*ln(1/delta)). This accounts for the entire20-model adaptive transcript, not only its final model.",
    "release_choice": {"target_cohort_index": 0, "sigma": 32., "rounds_released": 20,
                       "epsilon_budget": 1., "delta": 1e-5,
                       "selection": "Specified before training, independent of all utility and attack values."},
    "release_allowlist": "public_config.json plus model_round001.npz through model_round020.npz, fixed array names/shapes only. Deterministic ZIP timestamps. Everything else stays research_private.",
    "trusted_boundary": "A logical trusted central process/private local artifacts. Raw updates are in RAM only. No hardware enclave, remote adversarial server protection or access-control certification is implemented.",
    "noise": "Fresh secrets.SystemRandom; each coordinate uses (normalvariate(0,1)+normalvariate(0,1))/sqrt(2), following the two-normal mitigation described by IBM diffprivlib. No noise seed/state/vector is saved. Formal accounting is for the ideal Gaussian mechanism; production finite-precision security certification is not established.",
    "membership_probe": {"score": "Negative final-model true-label cross-entropy; larger predicts member.",
        "probe_size": "20 member and20 nonmember records per class, class matched;200+200 per target cohort.",
        "development": "One disjoint-record shadow cohort per sigma; choose a score threshold maximizing shadow TPR-FPR, ties choose highest threshold. Freeze thresholds before any target model is trained.",
        "evaluation": "Target AUC, TPR, FPR, fixed-threshold advantage=TPR-FPR and balanced accuracy. No target-score threshold optimization or direction flipping.",
        "scope": "A record-membership empirical probe under this restricted attacker, not a client-participation attack, worst-case attacker or DP proof."},
    "research_diagnostics": "Raw clipping, membership, utility, private split identifiers/seeds, shadow models and all non-selected/no-noise models are private research artifacts outside the protected public transcript. Publishing this private archive/report would need a separate privacy analysis.",
    "hypothetical_all_release": "Compute composition stress cases only; do not release all sigma/seed models. Any release of the no-noise private-data baseline has no finite DP bound here.",
    "primary_sources": [
        "https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf",
        "https://arxiv.org/abs/1605.02065",
        "https://github.com/IBM/differential-privacy-library/blob/main/diffprivlib/mechanisms/gaussian.py",
        "https://arxiv.org/abs/2107.10138",
        "https://arxiv.org/abs/1709.01604",
    ],
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def jsave(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def csave(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def accountant(rounds: int, sigma: float, delta: float = 1e-5) -> dict:
    if sigma == 0:
        return {"rho": None, "epsilon": None, "delta": delta,
                "PPI_original_1_minus_epsilon": None, "status": "no_noise_no_DP"}
    rho = rounds / (2 * sigma * sigma)
    eps = rho + 2 * math.sqrt(rho * math.log(1/delta))
    return {"rho": rho, "epsilon": eps, "delta": delta,
            "PPI_original_1_minus_epsilon": 1-eps,
            "status": "ideal_Gaussian_full_transcript_zCDP_bound"}


def clip(v, bound: float):
    v = np.asarray(v, np.float64)
    norm = float(np.linalg.norm(v))
    if not np.isfinite(norm):
        raise ValueError("Non-finite private local update; abort rather than release an unaccounted outcome")
    return v * min(1., bound / max(norm, 1e-300)), norm


def prepare_protocol() -> dict:
    if OUT.exists():
        raise FileExistsError("Use preserved completed study; secure-noise trials must never be silently rerun/overwritten")
    PRIVATE.mkdir(parents=True)
    noise = secrets.SystemRandom()
    plan = dict(PLAN)
    plan["private_target_data_seeds"] = noise.sample(range(100000, 2000000000), 5)
    plan["private_shadow_data_seed"] = noise.randrange(2000000001, 3000000000)
    plan["private_seeds_are_not_noise_seeds"] = True
    plan["source_sha256"] = {"purpose_privacy_study.py": digest(Path(__file__)),
                              "_helpers.py": digest(Path(core.__file__)), "learning.py":digest(source_path("learning.py")), "api.py":digest(source_path("api.py"))}
    plan["versions"] = {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "scikit-learn", "flwr")}
    plan["private_protocol_sha256_basis"] = "SHA256 of frozen protocol.json file, recorded alongside all run summaries. Seeds control only partitioning; no Gaussian random state is seeded or retained."
    jsave(PRIVATE / "protocol.json", plan)
    return plan


def validate_theory(plan):
    rng = np.random.default_rng(490071)
    C, K = plan["clip_norm"], plan["client_slots"]
    maximum = 0.
    for _ in range(300):
        vectors = rng.normal(size=(K, 31))
        replacement = vectors.copy()
        replacement[3] = rng.normal(size=31)
        a = np.asarray([clip(v, C)[0] for v in vectors]).mean(0)
        b = np.asarray([clip(v, C)[0] for v in replacement]).mean(0)
        observed = float(np.linalg.norm(a-b))
        assert observed <= 2*C/K + 1e-14
        maximum = max(maximum, observed)
    v = np.zeros(31); v[0] = C
    a, b = np.zeros((K, 31)), np.zeros((K, 31))
    a[3], b[3] = v, -v
    boundary = float(np.linalg.norm(a.mean(0)-b.mean(0)))
    assert abs(boundary-2*C/K) < 1e-15
    assert np.linalg.norm(clip(np.zeros(31), C)[0]) == 0
    accounting = [accountant(plan["rounds"], sigma) for sigma in (2, 8, 32)]
    assert accounting[0]["epsilon"] > accounting[1]["epsilon"] > accounting[2]["epsilon"]
    assert accounting[-1]["epsilon"] <= plan["release_choice"]["epsilon_budget"]
    per_cohort_rho = sum(r["rho"] for r in accounting)
    archive_rho = plan["target_cohorts"] * per_cohort_rho
    eps = lambda rho: rho + 2*math.sqrt(rho*math.log(1/plan["delta"]))
    return {"status": "passed", "replace_one_client_trials": 300,
        "maximum_random_observed_sensitivity": maximum, "exact_antipodal_boundary": boundary,
        "theoretical_sensitivity_2C_over_K": 2*C/K, "empty_client_delta_zero_checked": True,
        "proof": "Conditional on the same previous public transcript, only one of12 client vectors changes; each lies in the radius-C L2 ball. Triangle inequality bounds uniform mean difference by2C/12. Gaussian std=sigma*(2C/12) gives rho=1/(2sigma^2) per round. Adaptive composition adds all20 round rhos. Model serialization/rounding is postprocessing in the ideal mechanism analysis.",
        "release_candidate_accounting": accountant(plan["rounds"], 32),
        "release_epsilon_budget_met": True,
        "hypothetical_not_released": {
            "all_positive_sigmas_one_cohort_rho": per_cohort_rho,
            "all_positive_sigmas_one_cohort_epsilon": eps(per_cohort_rho),
            "all_positive_sigmas_five_cohorts_300models_rho": archive_rho,
            "all_positive_sigmas_five_cohorts_300models_epsilon": eps(archive_rho),
            "adjacency_for_conservative_five_cohort_stress": "Replace one entire fixed client container in each of the five cohorts; other containers fixed; sequentially compose all their outputs. This is a stress scenario, not the released candidate.",
            "including_sigma0": "No finite DP guarantee for the archive if no-noise private-data models are included."},
        "precision_limitation": "Ideal Gaussian accounting and boundary tests do not certify every floating-point/CSPRNG/OS side channel. No certified DP production service or hardware enclave is claimed."}


def make_dataset(seed: int, pool, x, y):
    member, nonmember = train_test_split(pool, train_size=.6, stratify=y[pool], random_state=seed)
    rng = np.random.default_rng(seed)
    clients = [[] for _ in range(12)]
    for label in range(10):
        ids = rng.permutation(member[y[member] == label])
        for index, sample in enumerate(ids):
            clients[(index+label) % 12].append(int(sample))
    clients = [np.asarray(ids, np.int64) for ids in clients]
    assert set(np.concatenate(clients)) == set(member)
    assert not (set(member) & set(nonmember))
    assert min(map(len, clients)) > 0
    probes = []
    for source in (member, nonmember):
        probes.append(np.concatenate([rng.choice(source[y[source] == label], 20, replace=False) for label in range(10)]))
    return {"member": member, "nonmember": nonmember, "clients": clients,
            "probe_ids": np.concatenate(probes), "probe_membership": np.repeat([1, 0], 200)}


def train_one(role, index, sigma, data, x, y, plan, protocol_sha):
    run_id = f"{role}_{index:02d}_sigma{int(sigma)}"
    dest = PRIVATE / "runs" / run_id
    dest.mkdir(parents=True, exist_ok=False)
    arrays = core.initial_parameters(plan["public_initial_model_seed"])
    noise_rng = secrets.SystemRandom()
    std = sigma * 2 * plan["clip_norm"] / plan["client_slots"]
    clients = [core.DigitsClient(cid, x[ids], y[ids], plan["public_client_shuffle_seed"]).to_client()
               for cid, ids in enumerate(data["clients"])]
    records, checkpoints = [], []
    fit_calls = 0
    for rnd in range(1, plan["rounds"]+1):
        previous = core.pack(arrays).astype(np.float64)
        parameters = ndarrays_to_parameters(arrays)
        clipped, norms = [], []
        for client in clients:
            response = client.fit(FitIns(parameters, {"round": rnd, "local_epochs": plan["local_epochs"],
                "learning_rate": plan["base_learning_rate"] / math.sqrt(1+(rnd-1)/20),
                "batch_size": plan["batch_size"], "l2": plan["l2"]}))
            local_delta = core.pack(parameters_to_ndarrays(response.parameters)).astype(np.float64)-previous
            bounded, norm = clip(local_delta, plan["clip_norm"])
            assert np.linalg.norm(bounded) <= plan["clip_norm"]+1e-12
            clipped.append(bounded); norms.append(norm); fit_calls += 1
        mean = np.stack(clipped).mean(axis=0)
        if sigma:
            noise = np.fromiter(((noise_rng.normalvariate(0., 1.)+noise_rng.normalvariate(0., 1.)) / math.sqrt(2)
                                 for _ in range(len(previous))), dtype=np.float64, count=len(previous))
            released = previous + mean + std*noise
        else:
            released = previous + mean
        arrays = core.unpack(released)
        if not all(np.isfinite(a).all() for a in arrays):
            raise FloatingPointError("Non-finite model; never retry or select a different trial")
        path = dest / f"model_round{rnd:03d}.npz"
        np.savez_compressed(path, **{f"array_{i}": a for i, a in enumerate(arrays)})
        checkpoints.append(path)
        records.append({"round": rnd, "preclip_norm_mean": float(np.mean(norms)),
                        "preclip_norm_max": max(norms), "clip_fraction": float(np.mean(np.asarray(norms) > plan["clip_norm"])),
                        "noise_std_public_constant": std, "max_postclip_norm": float(max(np.linalg.norm(v) for v in clipped))})
    p = core.predict(arrays, x[data["nonmember"]])
    utility = core.metric_bundle(y[data["nonmember"]], p)
    probe_p = core.predict(arrays, x[data["probe_ids"]]).astype(np.float64)
    losses = -np.log(np.clip(probe_p[np.arange(len(probe_p)), y[data["probe_ids"]]], 1e-30, 1.))
    scores = -losses
    np.savez_compressed(dest / "private_membership_scores.npz", score=scores,
                        membership=data["probe_membership"], source_ids=data["probe_ids"])
    csave(dest / "private_clip_rounds.csv", records)
    result = {"run_id": run_id, "role": role, "cohort_index": index, "sigma": sigma,
        "fixed_final_round": plan["rounds"], "Flower_NumPyClient_fit_calls": fit_calls,
        "private_train_records": len(data["member"]), "private_nonmember_records": len(data["nonmember"]),
        "clip_fraction_mean": float(np.mean([r["clip_fraction"] for r in records])),
        "preclip_norm_mean": float(np.mean([r["preclip_norm_mean"] for r in records])),
        "noise_std": std, "no_noise_seed_or_state_saved": True,
        "raw_model_updates_saved": False, "graph_hash_filter_computed_or_released": False,
        "protocol_sha256": protocol_sha, **accountant(plan["rounds"], sigma),
        **{f"test_{k}": v for k, v in utility.items()}}
    jsave(dest / "private_final.json", result)
    return result, scores, checkpoints


def develop_threshold(labels, scores):
    fpr, tpr, thresholds = roc_curve(labels, scores, drop_intermediate=False)
    index = int(np.argmax(tpr-fpr))
    threshold = float(thresholds[index])
    if not math.isfinite(threshold):
        threshold = float(np.max(scores)+1.)
    return threshold, float(roc_auc_score(labels, scores))


def probe_attack(labels, scores, threshold):
    predicted = scores >= threshold
    tpr = float(predicted[labels == 1].mean())
    fpr = float(predicted[labels == 0].mean())
    return {"membership_AUC": float(roc_auc_score(labels, scores)),
        "membership_TPR": tpr, "membership_FPR": fpr,
        "membership_fixed_threshold_advantage": tpr-fpr,
        "membership_balanced_accuracy": .5*(tpr+1-fpr),
        "shadow_frozen_loss_score_threshold": threshold,
        "target_members": int(sum(labels == 1)), "target_nonmembers": int(sum(labels == 0))}


def export_candidate(checkpoints, plan, protocol_sha):
    RELEASE.mkdir()
    public = {"schema": 1, "mechanism": "uniform-client L2-clipped Gaussian global-model transcript",
        "client_slots": plan["client_slots"], "full_participation": True,
        "rounds": plan["rounds"], "clip_norm": plan["clip_norm"], "sigma": 32.,
        "sensitivity": 2*plan["clip_norm"]/plan["client_slots"],
        "noise_std": 32*2*plan["clip_norm"]/plan["client_slots"],
        "epsilon": accountant(plan["rounds"], 32)["epsilon"], "delta": plan["delta"],
        "rho": accountant(plan["rounds"], 32)["rho"],
        "accounting": "All20 adaptively released round models; ideal Gaussian zCDP composition.",
        "adjacency": "Replace all records of one of12 fixed client containers; all other containers fixed; empty replacement contributes zero.",
        "training": {"model_shapes": [list(s) for s in core.SHAPES], "local_epochs": 2,
                     "batch_size": 32, "base_learning_rate": .14, "l2": .0001,
                     "public_initial_model_seed": plan["public_initial_model_seed"],
                     "public_shuffle_seed": plan["public_client_shuffle_seed"],
                     "preprocessing": "pixel/16 fixed public transform"},
        "release_selection": "cohort index0, sigma32, fixed before training; no utility-dependent checkpoint or selection",
        "trusted_boundary": "Logical trusted central process. Raw updates/data-dependent graph/hash/filter and private research diagnostics are not released.",
        "limitation": "Research numerical implementation; no certified finite-precision security, hardware enclave or physical side-channel guarantee. Utility/membership/clipping reports outside this folder are private diagnostics outside this protected transcript."}
    jsave(RELEASE / "public_config.json", public)
    for path in checkpoints:
        shutil.copyfile(path, RELEASE / path.name)
    fixed_epoch = 946684800
    for path in RELEASE.iterdir():
        os.utime(path, (fixed_epoch, fixed_epoch))
    expected = {"public_config.json"} | {f"model_round{r:03d}.npz" for r in range(1,21)}
    actual = {p.name for p in RELEASE.iterdir()}
    assert actual == expected and all(p.is_file() for p in RELEASE.iterdir())
    seen = []
    for path in sorted(RELEASE.glob("*.npz")):
        with np.load(path, allow_pickle=False) as archive:
            assert archive.files == [f"array_{i}" for i in range(4)]
            for index, shape in enumerate(core.SHAPES):
                a = archive[f"array_{index}"]
                assert a.shape == shape and a.dtype == np.float32 and np.isfinite(a).all()
        seen.append({"path": str(path.relative_to(OUT)), "sha256": digest(path)})
    archive_path = OUT / "release_candidate.zip"
    with zipfile.ZipFile(archive_path, "w") as z:
        for path in sorted(RELEASE.iterdir()):
            info = zipfile.ZipInfo(path.name, date_time=(2000,1,1,0,0,0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            z.writestr(info, path.read_bytes())
    with zipfile.ZipFile(archive_path) as z:
        assert set(z.namelist()) == expected
        assert all(i.date_time == (2000,1,1,0,0,0) for i in z.infolist())
        assert all(z.read(p.name) == p.read_bytes() for p in RELEASE.iterdir())
    audit = {"status": "passed", "files": 21, "noisy_model_files": 20,
        "allowlist_exact": True, "NPZ_array_schema_checked": True,
        "only_public_constant_config_and_noisy_models": True,
        "no_private_seed_utility_attack_clip_hash_filter_raw_update_or_noise_state_files": True,
        "no_data_dependent_checkpoint_selection": True, "normalized_timestamps": True,
        "zip_allowlist_and_contents_equal": True,
        "zip_sha256": digest(archive_path), "model_files": seen,
        "public_config_sha256": digest(RELEASE / "public_config.json"),
        "protocol_sha256_private_audit_only": protocol_sha,
        "excluded_scope": "All private research outputs, reports, runtime/OS side channels and any future release outside the allowlist."}
    jsave(PRIVATE / "release_allowlist_audit.json", audit)
    return audit


def summarize(rows):
    groups = []
    for sigma in PLAN["sigma_values"]:
        group = [r for r in rows if r["sigma"] == sigma]
        record = {"sigma": sigma, "n_independent_target_cohorts": len(group),
                  **accountant(20, sigma)}
        for field in ("test_accuracy", "test_macro_f1", "clip_fraction_mean", "membership_AUC",
                      "membership_fixed_threshold_advantage", "membership_balanced_accuracy"):
            a = np.asarray([r[field] for r in group], float)
            half = float(student_t.ppf(.975, len(a)-1)*a.std(ddof=1)/np.sqrt(len(a)))
            record[field] = {"mean": float(a.mean()), "sd": float(a.std(ddof=1)),
                             "ci95_descriptive": [float(a.mean()-half), float(a.mean()+half)]}
        groups.append(record)
    return groups


def main():
    started = time.perf_counter()
    plan = prepare_protocol()
    protocol_sha = digest(PRIVATE / "protocol.json")
    theory = validate_theory(plan)
    jsave(PRIVATE / "sensitivity_and_accounting_validation.json", theory)
    ds = load_digits()
    x, y = (ds.data/16).astype(np.float32), ds.target.astype(np.int64)
    ids = np.arange(len(y))
    shadow_pool, target_pool = train_test_split(ids, train_size=.4, stratify=y,
                                               random_state=plan["public_shadow_target_pool_split_seed"])
    assert not (set(shadow_pool) & set(target_pool))
    shadow = make_dataset(plan["private_shadow_data_seed"], shadow_pool, x, y)
    jsave(PRIVATE / "private_split_audit.json", {"shadow_pool_ids": shadow_pool.tolist(),
          "target_pool_ids": target_pool.tolist(), "pool_disjoint": True,
          "shadow_member_ids": shadow["member"].tolist(), "shadow_nonmember_ids": shadow["nonmember"].tolist(),
          "seeds_are_data_assignment_only": True, "no_noise_seed_present": True})
    thresholds, shadow_rows = {}, []
    for sigma in plan["sigma_values"]:
        result, scores, _ = train_one("shadow", 0, sigma, shadow, x, y, plan, protocol_sha)
        threshold, auc = develop_threshold(shadow["probe_membership"], scores)
        thresholds[sigma] = threshold
        shadow_rows.append({"sigma": sigma, "threshold": threshold, "shadow_development_AUC": auc})
        print(f"shadow development {len(shadow_rows)}/4 complete; no target scores accessed", flush=True)
    jsave(PRIVATE / "frozen_attack_thresholds.json", {"status": "frozen_before_target_training",
        "protocol_sha256": protocol_sha, "thresholds": shadow_rows,
        "shadow_target_records_disjoint": True, "target_threshold_optimization": False})
    target_rows, candidate_paths = [], None
    for index, seed in enumerate(plan["private_target_data_seeds"]):
        data = make_dataset(seed, target_pool, x, y)
        jsave(PRIVATE / f"private_cohort{index:02d}_ids.json", {"member": data["member"].tolist(),
              "nonmember": data["nonmember"].tolist(), "client_ids": [v.tolist() for v in data["clients"]],
              "train_nonmember_disjoint": True, "shadow_target_disjoint": True})
        for sigma in plan["sigma_values"]:
            result, scores, paths = train_one("target", index, sigma, data, x, y, plan, protocol_sha)
            result.update(probe_attack(data["probe_membership"], scores, thresholds[sigma]))
            result["is_prespecified_release_candidate"] = index == 0 and sigma == 32
            target_rows.append(result)
            if result["is_prespecified_release_candidate"]:
                candidate_paths = paths
            csave(PRIVATE / "final_metrics.csv", target_rows)
            print(f"target training {len(target_rows)}/20 complete; fixed protocol; sigma={sigma:g}", flush=True)
    assert len(target_rows) == 20 and candidate_paths is not None
    assert digest(PRIVATE / "protocol.json") == protocol_sha
    for filename, expected in plan["source_sha256"].items():
        path = source_path(filename)
        assert digest(path) == expected, "Source changed during private study"
    release_audit = export_candidate(candidate_paths, plan, protocol_sha)
    candidate = next(r for r in target_rows if r["is_prespecified_release_candidate"])
    result = {"status": "completed", "scope": "PRIVATE research assessment. Only release_candidate/ and its deterministic ZIP are protected-output candidates; this diagnostic JSON/report is not a DP release.",
        "target_runs": 20, "shadow_runs": 4, "rounds_per_run": 20,
        "Flower_fit_calls": sum(r["Flower_NumPyClient_fit_calls"] for r in target_rows)+4*20*12,
        "fixed_protocol_sha256": protocol_sha, "source_hashes_unchanged": True,
        "no_noise_seed_state_or_vectors_saved": True, "summary_by_sigma": summarize(target_rows),
        "prespecified_single_release_candidate_private_diagnostics": candidate,
        "protected_release_candidate_epsilon": accountant(20, 32)["epsilon"],
        "protected_release_candidate_delta": 1e-5, "epsilon_budget_1_met": True,
        "protected_outputs": "Exactly20 noisy model files and one public constant config; not graph/hash/filter/rawupdates/utility/attack/clip/private seeds.",
        "allowlist_audit_passed": release_audit["status"] == "passed",
        "accounting_and_boundary_checks": theory,
        "PPI_replacement": "Use (epsilon,delta,adjacency,release scope,accountant) instead of1-epsilon. The original PPI remains a traceable contrast, not a probability.",
        "attack_interpretation": "Fixed-shadow-threshold record-membership probes provide empirical diagnostics only. Low/negative advantage or AUC near0.5 is not evidence proving DP, nor a test of all client-participation attacks.",
        "privacy_utility_interpretation": "Evaluate the observed utility cost, even if severe; no noise budget, checkpoint, release seed or threshold was chosen using target results.",
        "production_limitations": "Logical central trust boundary only; no hardware enclave, malicious server resistance, certified finite-precision Gaussian proof, deployment network protection or DP protection for private research diagnostics.",
        "elapsed_seconds": time.perf_counter()-started}
    jsave(PRIVATE / "summary.json", result)
    (OUT / "README.txt").write_text(
        "LOCAL RESEARCH ARTIFACTS — NO EXTERNAL UPLOAD\n"
        "release_candidate/ and release_candidate.zip contain only the prespecified sigma32/cohort0 full20-model transcript and public constants.\n"
        "research_private/ includes non-DP baselines, shadow/other DP models, private membership/split/utility/clip diagnostics. DO NOT publish it as a protected transcript.\n"
        "The original1-epsilon is retained for comparison, never interpreted as a probability.\n"
        "Formal accounting is for the ideal Gaussian mechanism in a logical trusted central process. This research implementation is not a certified finite-precision DP service or hardware enclave.\n",
        encoding="utf-8")
    print(json.dumps({"status": "completed", "target_runs": 20, "shadow_runs": 4,
        "candidate_epsilon": candidate["epsilon"], "delta": 1e-5, "allowlist": "passed",
        "output": str(OUT)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    import argparse
    _parser = argparse.ArgumentParser(description=__doc__)
    _parser.parse_args()
    main()
