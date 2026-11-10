import numpy as np

from scenario_test.runner import adaptability, run_scenario
from scenario_test.scenarios import ScenarioPlugin


class _Server:
    n = 10


def test_participation_state():
    p = ScenarioPlugin({"scenario": "participation"})
    p.rng = np.random.default_rng(0)
    state = p.round_state(_Server(), 1, {"participating": set(range(10)), "delay_factor": 1.0, "spikes": None,
                                          "down_edges": set(), "hang_edges": set(), "drop": set()})
    assert 1 <= len(state["participating"]) <= 10


def test_edge_failure_scenario(tmp_path, small):
    base = {"backend-config": small, "num-clients": 6, "num-edges": 2, "global-rounds": 2, "edge-rounds": 1}
    ref = run_scenario("ref", base, 0, str(tmp_path))
    r = run_scenario("edge_failure", base, 0, str(tmp_path), {"edge": 1, "from": 1, "to": 1})
    assert r["completed"] and r["alerts"] >= 1
    assert adaptability([ref, r])["scenarios_completed"] == ["edge_failure"]
