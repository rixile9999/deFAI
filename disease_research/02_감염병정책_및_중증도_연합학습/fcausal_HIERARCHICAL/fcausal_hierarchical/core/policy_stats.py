"""Additive statistics for a balanced district-day difference-in-differences panel.

Outcome: log(1 + observed daily cases). Every district has the same calendar.
The estimand gives equal weight to districts, not to patients or provinces.
Inference clusters district changes at the province (policy assignment) level.
"""
from dataclasses import asdict, dataclass
from functools import cached_property
import json

import numpy as np
import pandas as pd
from scipy.stats import t as student_t


SCHEMA = "district_log1p_moments_v2"
VECTOR_SIZE = 10  # districts, then (n, sum(y), sum(y*y)) for early-pre/late-pre/post


@dataclass(frozen=True)
class AnalysisConfig:
    start: str = "2024-01-01"
    policy: str = "2024-01-31"
    end: str = "2024-04-09"
    date_column: str = "발병일"

    def __post_init__(self):
        start, policy, end = map(pd.Timestamp, (self.start, self.policy, self.end))
        if not start < policy <= end or (policy - start).days < 2:
            raise ValueError("Need at least two pre days and one post day")
        if any(d != d.normalize() for d in (start, policy, end)):
            raise ValueError("Calendar boundaries must be whole dates")

    @property
    def signature(self):
        return json.dumps(asdict(self), ensure_ascii=False, sort_keys=True)

    @cached_property
    def calendar(self):
        return pd.date_range(self.start, self.end, freq="D")

    @cached_property
    def masks(self):
        dates = self.calendar
        pre = dates < pd.Timestamp(self.policy)
        split = int(pre.sum()) // 2
        return (np.arange(len(dates)) < split,
                pre & (np.arange(len(dates)) >= split), ~pre)


def add_analysis_arguments(parser):
    parser.add_argument("--start", default=AnalysisConfig.start)
    parser.add_argument("--policy", default=AnalysisConfig.policy)
    parser.add_argument("--end", default=AnalysisConfig.end)
    parser.add_argument("--date-column", default=AnalysisConfig.date_column)


def config_from_args(args):
    return AnalysisConfig(args.start, args.policy, args.end, args.date_column)


def daily_counts(frame, config):
    """A case line list requires a declared complete reporting calendar.

    Absence of a row is treated as zero cases, NOT as missing surveillance.
    Caller must check that this is appropriate before using real records.
    """
    if "record_format" in frame:
        if frame["record_format"].isna().any() or set(frame["record_format"]) != {"daily_counts_v1"}:
            raise ValueError("Unknown or mixed daily panel format")
        column = {"발병일": "cases", "진단일": "diagnosed_cases"}.get(config.date_column)
        if column is None:
            raise ValueError("Daily panels support onset or diagnosis counts")
        dates = pd.to_datetime(frame["date"], errors="raise").dt.normalize()
        if dates.isna().any() or dates.duplicated().any():
            raise ValueError("Daily panel dates must be unique and nonmissing")
        values = frame[column].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all() or (values < 0).any() or not np.equal(values, np.floor(values)).all():
            raise ValueError("Daily panel counts must be nonnegative integers")
        selected = pd.Series(values, index=dates).reindex(config.calendar)
        if selected.isna().any():
            raise ValueError("Daily panel must explicitly include every calendar day")
        return selected.to_numpy(dtype=np.float64)
    dates = pd.to_datetime(frame[config.date_column], errors="raise")
    if dates.isna().any():
        raise ValueError("Missing event dates cannot be treated as zero cases")
    counts = dates.dt.normalize().value_counts().reindex(config.calendar, fill_value=0)
    return counts.to_numpy(dtype=np.float64)


def read_district_file(path):
    return pd.read_csv(path) if str(path).lower().endswith(".csv") else pd.read_excel(path)


def district_treatment(frame, province):
    if "treated" not in frame:
        return int("서울" in province)  # Legacy synthetic workbooks only.
    if frame["treated"].isna().any() or frame["treated"].nunique() != 1 or frame["treated"].iloc[0] not in (0, 1):
        raise ValueError("District treatment must be one explicit 0/1 value")
    return int(frame["treated"].iloc[0])


def district_identity(frame):
    province_col = "시도" if "시도" in frame else "거주지시도"
    if frame.empty or frame[province_col].isna().any() or frame["시군구"].isna().any():
        raise ValueError("A district file must contain one nonempty province/district")
    if frame[province_col].nunique() != 1 or frame["시군구"].nunique() != 1:
        raise ValueError("Mixed provinces or districts in one district file")
    return str(frame[province_col].iloc[0]), str(frame["시군구"].iloc[0])


def district_vector(counts, config):
    counts = np.asarray(counts, dtype=np.float64)
    if counts.shape != (len(config.calendar),) or not np.isfinite(counts).all() or (counts < 0).any():
        raise ValueError("Counts must be finite, nonnegative and match the calendar")
    y = np.log1p(counts)
    vector = [1.0]
    for mask in config.masks:
        values = y[mask]
        vector.extend((len(values), values.sum(), values @ values))
    return np.asarray(vector, dtype=np.float64)


def validate_vector(vector, config, expected_districts=None):
    vector = np.asarray(vector, dtype=np.float64)
    if vector.shape != (VECTOR_SIZE,) or not np.isfinite(vector).all():
        raise ValueError("Invalid v2 statistic vector; legacy mean/std payloads are rejected")
    districts = int(round(float(vector[0])))
    if districts < 1 or abs(vector[0] - districts) > 1e-3:
        raise ValueError("Invalid district count (or incompatible encryption keys)")
    if expected_districts is not None and districts != expected_districts:
        raise ValueError("District count does not match metadata")
    for j, mask in enumerate(config.masks):
        n, total, squared = vector[1 + 3*j:4 + 3*j]
        expected = districts * int(mask.sum())
        if abs(n - expected) > 1e-3:
            raise ValueError("Calendar/sample-count mismatch")
        if total < -1e-5 or squared < -1e-5 or squared - total*total/expected < -1e-5 * max(1, squared):
            raise ValueError("Inconsistent moments (or incompatible encryption keys)")
    return vector


def moments(vector, period):
    blocks = vector[1:].reshape(3, 3)
    n, total, squared = blocks[:2].sum(axis=0) if period == "pre" else blocks[2]
    return {"n": int(round(float(n))), "mean": float(total/n),
            "variance": float(max(0, (squared-total*total/n)/(n-1))) if n > 1 else None}


def province_summary(vector, region, treated, config, expected_districts=None):
    if treated not in (0, 1):
        raise ValueError("Treatment must be 0 or 1")
    vector = validate_vector(vector, config, expected_districts)
    pre, post = moments(vector, "pre"), moments(vector, "post")
    blocks = vector[1:].reshape(3, 3)
    placebo = blocks[1, 1]/blocks[1, 0] - blocks[0, 1]/blocks[0, 0]
    return {"region": region, "treated": int(treated), "n_districts": int(round(vector[0])),
            "pre": pre, "post": post, "change": post["mean"]-pre["mean"],
            "pre_placebo_change": float(placebo)}


def _contrast(provinces, field):
    groups = [[p for p in provinces if p["treated"] == g] for g in (0, 1)]
    means = [sum(p["n_districts"]*p[field] for p in group) /
             sum(p["n_districts"] for p in group) for group in groups]
    return means[1]-means[0], means


def analyze(provinces):
    if len({p["region"] for p in provinces}) != len(provinces):
        raise ValueError("Duplicate province received")
    if {p["treated"] for p in provinces} != {0, 1}:
        raise ValueError("Both treatment and control provinces are required")
    effect, means = _contrast(provinces, "change")
    placebo, _ = _contrast(provinces, "pre_placebo_change")
    counts = [sum(p["treated"] == g for p in provinces) for g in (0, 1)]
    result = {"stage": "research_only", "schema": SCHEMA,
              "estimand": "equal_district_mean_change_in_log1p_observed_daily_cases",
              "effect": float(effect), "transformed_ratio_percent": float(100*np.expm1(effect)),
              "control_change": float(means[0]), "treated_change": float(means[1]),
              "pre_placebo_effect": float(placebo), "n_control_provinces": counts[0],
              "n_treated_provinces": counts[1], "provinces": provinces,
              "inference": {"status": "withheld", "se": None, "p_value": None, "ci95": None},
              "warnings": ["Causal interpretation requires parallel untreated trends, no spillovers, "
                           "no anticipation and stable reporting; a pre-placebo cannot prove these.",
                           "expm1(effect) concerns geometric means of (cases+1), not a raw case reduction."]}
    # The independent units are provinces; days and districts cannot replace them.
    if min(counts) < 2:
        result["inference"]["reason"] = "At least two provinces per arm are needed for cluster inference"
        return result
    g = len(provinces)
    n = sum(p["n_districts"] for p in provinces)
    totals = [sum(p["n_districts"] for p in provinces if p["treated"] == arm) for arm in (0, 1)]
    meat = sum((p["n_districts"]*(p["change"]-means[p["treated"]])/totals[p["treated"]])**2
               for p in provinces)
    # CR1 on district-level change regression [intercept, treated], clustered by province.
    correction = g/(g-1) * (n-1)/(n-2)
    se = float(np.sqrt(correction*meat))
    if se <= 1e-12:
        result["inference"]["reason"] = "No estimated between-province variation; inference is degenerate"
        return result
    df = g-1
    critical = float(student_t.ppf(.975, df))
    result["inference"] = {"status": "exploratory", "method": "province_cluster_CR1",
                           "df": df, "se": se, "p_value": float(2*student_t.sf(abs(effect/se), df)),
                           "ci95": [float(effect-critical*se), float(effect+critical*se)]}
    if min(counts) < 10:
        result["warnings"].append("Few provinces in at least one arm: cluster-t inference can be poorly calibrated")
    return result


def print_analysis(result):
    print("\n[RESEARCH ONLY] Hierarchical difference-in-differences")
    print("Outcome: log1p(observed daily cases); equal district weights")
    print(f"Effect: {result['effect']:+.6f} log units")
    print(f"Pre-period placebo contrast: {result['pre_placebo_effect']:+.6f} log units")
    inf = result["inference"]
    if inf["status"] == "withheld":
        print("Inference WITHHELD: " + inf["reason"])
    else:
        print(f"Exploratory province-cluster SE={inf['se']:.6f}, p={inf['p_value']:.6g}, 95% CI={inf['ci95']}")
    for warning in result["warnings"]:
        print("LIMITATION: " + warning)
