from types import SimpleNamespace

import flwr as fl
import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
import tenseal as ts
from statsmodels.stats.sandwich_covariance import cov_cluster

from core.policy_stats import (AnalysisConfig, analyze, daily_counts, district_identity,
                               district_vector, moments, province_summary, validate_vector)
from core.policy_transport import decode, encode, metadata, validate_metadata
from run_tier0 import CentralStrategy
from run_tier1 import ProvinceStrategy, ProvinceClient


@pytest.fixture
def config():
    return AnalysisConfig("2024-01-01", "2024-01-05", "2024-01-10")


@pytest.fixture(scope="module")
def contexts():
    secret = ts.context(ts.SCHEME_TYPE.CKKS, 8192, coeff_mod_bit_sizes=[60, 40, 40, 60])
    secret.global_scale = 2**40
    public = ts.context_from(secret.serialize(save_secret_key=False))
    return secret, public


def response(arrays, info):
    return (None, SimpleNamespace(parameters=fl.common.ndarrays_to_parameters(arrays), metrics=info))


def test_zero_days_and_explicit_calendar(config):
    frame = pd.DataFrame({"发病": ["2024-01-01", "2024-01-01", "2024-01-10", "2024-02-01"]})
    other = AnalysisConfig(config.start, config.policy, config.end, "发病")
    counts = daily_counts(frame, other)
    np.testing.assert_array_equal(counts, [2, 0, 0, 0, 0, 0, 0, 0, 0, 1])
    frame.loc[0, "发病"] = None
    with pytest.raises(ValueError, match="Missing"):
        daily_counts(frame, other)


def test_moments_match_pooled_data_with_unequal_district_values(config):
    arrays = [np.arange(10), np.full(10, 30), np.zeros(10)]
    total = sum(district_vector(x, config) for x in arrays)
    validate_vector(total, config, 3)
    y = np.log1p(np.stack(arrays))
    for period, mask in [("pre", config.calendar < config.policy), ("post", config.calendar >= config.policy)]:
        flat = y[:, mask].ravel()
        actual = moments(total, period)
        assert actual["n"] == flat.size
        assert actual["mean"] == pytest.approx(flat.mean())
        assert actual["variance"] == pytest.approx(flat.var(ddof=1))


def test_district_identity_rejects_mixed_regions():
    frame = pd.DataFrame({"시도": ["서울", "경기"], "시군구": ["A", "A"]})
    with pytest.raises(ValueError, match="Mixed"):
        district_identity(frame)


@pytest.mark.parametrize("bad", [np.zeros(7), np.full(10, np.nan), np.zeros(10), np.ones(10)])
def test_invalid_stats_rejected(config, bad):
    with pytest.raises(ValueError):
        validate_vector(bad, config)


def test_config_and_key_mismatches_rejected(config):
    good = metadata(config, True, "key1")
    for key, value in [("schema", "old"), ("encrypted", False), ("key_id", "other"),
                       ("analysis_config", AnalysisConfig().signature)]:
        with pytest.raises(ValueError, match="Protocol mismatch"):
            validate_metadata({**good, key: value}, config, True, "key1")


def test_federated_effect_and_cluster_se_match_independent_ols(config):
    rng = np.random.default_rng(7)
    provinces, changes, treatment, clusters = [], [], [], []
    for p, size in enumerate([1, 2, 5, 3, 2, 4, 1, 6]):
        treated = p < 4
        counts = rng.poisson(10+p, size=(size, 10))
        counts[:, 4:] += int(treated)*4
        provinces.append(province_summary(sum(district_vector(x, config) for x in counts), str(p), treated, config))
        y = np.log1p(counts)
        changes.extend((y[:, 4:].mean(1)-y[:, :4].mean(1)).tolist())
        treatment.extend([int(treated)]*size)
        clusters.extend([p]*size)
    reference = sm.OLS(changes, sm.add_constant(treatment)).fit()
    covariance = cov_cluster(reference, clusters, use_correction=True)
    actual = analyze(provinces)
    assert actual["effect"] == pytest.approx(reference.params[1], abs=1e-12)
    assert actual["inference"]["se"] == pytest.approx(np.sqrt(covariance[1, 1]), abs=1e-12)
    assert actual["inference"]["df"] == 7


def test_one_treated_province_does_not_create_significance(config):
    provinces = [province_summary(district_vector(np.arange(10)+p, config), str(p), p == 0, config)
                 for p in range(3)]
    result = analyze(provinces)
    assert result["inference"]["status"] == "withheld"
    assert result["inference"]["p_value"] is None
    with pytest.raises(ValueError, match="Duplicate"):
        analyze(provinces + provinces[:1])
    with pytest.raises(ValueError, match="Both"):
        analyze([provinces[0]])


@pytest.mark.parametrize("encrypted", [False, True])
def test_actual_tier_strategies_and_encrypted_addition(config, contexts, encrypted):
    secret, public = contexts
    province_context, central_context = (public, secret) if encrypted else (None, None)
    key_id = "shared" if encrypted else "plaintext"
    central_results, pooled = [], []
    for region, treated in [("서울", 1), ("경기", 0)]:
        strategy = ProvinceStrategy(province_context, region, config, treated, key_id,
                                    min_fit_clients=2, min_available_clients=2)
        responses, vectors = [], []
        for d in range(2):
            vector = district_vector(np.arange(10)+d+treated, config)
            info = {**metadata(config, encrypted, key_id), "province": region,
                    "district": str(d), "treated": treated}
            responses.append(response(encode(vector, province_context), info))
            vectors.append(vector)
        strategy.aggregate_fit(1, responses, [])
        relay = ProvinceClient(strategy.aggregated_stats, region, 2, treated, config, encrypted, key_id)
        arrays, _, info = relay.fit([], {})
        central_results.append(response(arrays, info))
        pooled.append(province_summary(sum(vectors), region, treated, config))
        if encrypted:
            with pytest.raises(ValueError):
                strategy.aggregated_stats.decrypt()
    central = CentralStrategy(central_context, config, key_id, min_fit_clients=2, min_available_clients=2)
    central.aggregate_fit(1, central_results, [])
    assert central.result["effect"] == pytest.approx(analyze(pooled)["effect"], abs=1e-6)
    assert central.result["inference"]["status"] == "withheld"


def test_missing_duplicate_and_mixed_districts_rejected(config):
    strategy = ProvinceStrategy(None, "서울", config, True, min_fit_clients=2, min_available_clients=2)
    vector = district_vector(np.arange(10), config)
    info = {**metadata(config, False, "plaintext"), "province": "서울", "district": "A", "treated": 1}
    r = response([vector], info)
    with pytest.raises(ValueError, match="Incomplete"):
        strategy.aggregate_fit(1, [r], [])
    with pytest.raises(ValueError, match="Duplicate"):
        strategy.aggregate_fit(1, [r, r], [])
    other = response([vector], {**info, "district": "B", "treated": 0})
    with pytest.raises(ValueError, match="Mixed"):
        strategy.aggregate_fit(1, [r, other], [])


def test_secret_key_rejected_at_province(contexts):
    secret, _ = contexts
    with pytest.raises(ValueError, match="secret key"):
        ProvinceStrategy(secret, "서울")


def test_known_effect_survives_district_duplication(config):
    # The old protocol multiplied a contrast by the number of districts.
    # A valid merge must leave the same contrast when identical districts repeat.
    def calculate(copies):
        provinces = []
        for region, arm in [("treated", 1), ("control", 0)]:
            y = np.full(10, 3.0)
            y[4:] += .1 - .4*arm
            vector = copies*district_vector(np.expm1(y), config)
            provinces.append(province_summary(vector, region, arm, config))
        return analyze(provinces)["effect"]
    assert calculate(1) == pytest.approx(-.4)
    assert calculate(7) == pytest.approx(-.4)
