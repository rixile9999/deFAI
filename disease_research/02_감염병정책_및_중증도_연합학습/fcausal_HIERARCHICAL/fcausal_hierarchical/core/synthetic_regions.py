"""Scalable version of the Poisson/thinned-case generator used by the demo.

Fictional region IDs, explicit province-level assignment, paired potential
observations and aggregated daily CSVs. No new real-world regions are implied.
"""
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd

from core.policy_stats import AnalysisConfig


@dataclass(frozen=True)
class RegionGeneration:
    n_provinces: int = 32
    n_treated: int = 16
    districts_per_province: int = 3
    unequal_districts: bool = False
    policy_ratio: float = .7
    ramp_days: int = 7
    base_cases: float = 20.0
    baseline_sigma: float = .35
    serial_correlation: float = .8
    province_shock_sd: float = .15
    differential_trend: float = 0.0
    reporting_ratio: float = 1.0
    seed: int = 42

    def __post_init__(self):
        if self.n_provinces < 2 or not 0 < self.n_treated < self.n_provinces:
            raise ValueError("Need both treated and control provinces")
        if self.districts_per_province < 1 or self.ramp_days < 0 or self.base_cases <= 0:
            raise ValueError("Invalid district count, ramp or baseline")
        if not 0 < self.policy_ratio <= 1 or not 0 < self.reporting_ratio <= 1:
            raise ValueError("This thinning generator supports ratios in (0, 1]")
        if not 0 <= self.serial_correlation < 1 or min(self.baseline_sigma, self.province_shock_sd) < 0:
            raise ValueError("Invalid heterogeneity or serial correlation")


def generate_regions(settings, config=None):
    """Return ordinary analyzer records plus simulation-only potential outcomes.

    A daily count is marginally equivalent to individual Poisson cases followed
    by independent detection. CSVs avoid writing hundreds of thousands of rows
    with unused individual attributes. The analyzer never consumes oracle fields.
    """
    config = config or AnalysisConfig()
    rng = np.random.default_rng(settings.seed)
    days = len(config.calendar)
    pre_days = int((config.calendar < pd.Timestamp(config.policy)).sum())
    arms = np.zeros(settings.n_provinces, dtype=int)
    arms[rng.choice(settings.n_provinces, settings.n_treated, replace=False)] = 1
    sizes = (rng.integers(1, 2*settings.districts_per_province+1, settings.n_provinces)
             if settings.unequal_districts else np.full(settings.n_provinces, settings.districts_per_province))
    province_index = np.repeat(np.arange(settings.n_provinces), sizes)
    n_districts = len(province_index)
    treated = arms[province_index]
    rho, sd = settings.serial_correlation, settings.province_shock_sd
    shocks = np.empty((settings.n_provinces, days))
    shocks[:, 0] = rng.normal(0, sd, settings.n_provinces)
    innovations = rng.normal(0, sd*np.sqrt(1-rho*rho), (settings.n_provinces, days))
    for day in range(1, days):
        shocks[:, day] = rho*shocks[:, day-1]+innovations[:, day]
    t = np.arange(days)
    baseline = settings.base_cases*np.exp(rng.normal(0, settings.baseline_sigma, n_districts))
    # Same growth/weekly mechanism as data_gen_hierarchical.py, plus province shocks.
    means = (baseline[:, None]*np.exp(.015*t+shocks[province_index] +
             settings.differential_trend*treated[:, None]*t)*(1+.15*np.sin(2*np.pi*t/7)))
    untreated_true = rng.poisson(means)
    # Original severity-anchored observation mechanism, marginalized over severity.
    mild_per_severe = rng.uniform(12, 49, n_districts)
    detection = .02 + .98*np.minimum(mild_per_severe*.02/.98, 1)
    untreated_observed = rng.binomial(untreated_true, detection[:, None])
    ramp = ((t >= pre_days).astype(float) if settings.ramp_days == 0 else
            np.clip((t-pre_days)/settings.ramp_days, 0, 1))
    ratios = 1-treated[:, None]*(1-settings.policy_ratio)*ramp
    policy_observed = rng.binomial(untreated_observed, ratios)
    policy_true = policy_observed+rng.binomial(untreated_true-untreated_observed, ratios)
    reporting = 1-treated[:, None]*(1-settings.reporting_ratio)*(t >= pre_days)
    observed = rng.binomial(policy_observed, reporting)
    diagnoses = np.zeros((n_districts, days+5), dtype=np.int64)
    delays = rng.multinomial(observed, np.full(5, .2))
    for delay in range(1, 6):
        diagnoses[:, delay:delay+days] += delays[:, :, delay-1]
    records = []
    district_numbers = {}
    for i, province in enumerate(province_index):
        district_numbers[province] = district_numbers.get(province, 0)+1
        records.append({"province": f"Region_{province+1:03d}",
                        "district": f"District_{province+1:03d}_{district_numbers[province]:02d}",
                        "treated": int(treated[i]), "counts": observed[i],
                        "diagnosed_counts": diagnoses[i], "true_counts": policy_true[i],
                        "counterfactual_counts": untreated_observed[i], "policy_only_counts": policy_observed[i]})
    post = t >= pre_days
    oracle = (np.log1p(policy_observed[treated == 1][:, post]) -
              np.log1p(untreated_observed[treated == 1][:, post])).mean()
    return records, {"oracle_policy_effect_log1p": float(oracle),
                     "oracle_definition": "realized_treated_district_post_mean_log1p_policy_minus_no_policy_under_stable_reporting",
                     "n_districts": n_districts, "n_provinces": settings.n_provinces,
                     "n_treated_provinces": settings.n_treated,
                     "latent_policy_multiplier_after_ramp": settings.policy_ratio,
                     "reporting_ratio_after_policy": settings.reporting_ratio}


def export_regions(output, settings, config, records, truth):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)  # Never overwrite an earlier dataset.
    dates = pd.date_range(config.start, periods=len(config.calendar)+5).strftime("%Y-%m-%d")
    manifest = {"stage": "research_only_synthetic", "settings": asdict(settings), "config": asdict(config),
                "truth": truth, "regions": {}, "files": {}}
    for record in records:
        frame = pd.DataFrame({"date": dates, "record_format": "daily_counts_v1",
                              "시도": record["province"], "시군구": record["district"], "treated": record["treated"],
                              "cases": np.pad(record["counts"], (0, 5)), "diagnosed_cases": record["diagnosed_counts"],
                              "true_cases": np.pad(record["true_counts"], (0, 5)),
                              "counterfactual_no_policy_cases": np.pad(record["counterfactual_counts"], (0, 5)),
                              "policy_only_cases": np.pad(record["policy_only_counts"], (0, 5))})
        path = output / f"{record['district']}.csv"
        frame.to_csv(path, index=False, encoding="utf-8")
        manifest["files"][path.name] = sha256(path.read_bytes()).hexdigest()
        manifest["regions"][record["province"]] = record["treated"]
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest
