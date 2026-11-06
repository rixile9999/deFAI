import time

import numpy as np

from fl_core.records import delta, flatten, float_keys, to_arrays, to_record, unflatten
from fl_core.strategies import StrategyWrapper
from security_sim import detectors
from security_sim.keys import node_principal, reported_pid, verify

AUTH = ("cert", "hmac", "mfa")
STATISTICAL = ("stat", "data_dist", "ocsvm", "kmeans", "monitor")


class DefendedStrategy(StrategyWrapper):
    def __init__(self, inner, defenses, keys, prefix="client", owner=None, malicious=(), clip_factor=1.5, seed=0):
        super().__init__(inner)
        self.defenses = [d for d in defenses if d != "none"]
        self.keys, self.prefix, self.owner = keys, prefix, owner
        self.malicious = set(malicious)
        self.clip_factor, self.seed = clip_factor, seed
        self.ocsvm = detectors.OCSVMDetector()
        self.global_arrays = None
        self.log = []

    def configure_train(self, server_round, arrays, config, grid):
        self.global_arrays = to_arrays(arrays)
        return self.inner.configure_train(server_round, arrays, config, grid)

    def principal(self, reply):
        if self.prefix == "client":
            return node_principal(reply.metadata.src_node_id)
        return f"{self.prefix}{int(reply.metadata.src_node_id) - 1}"

    def label(self, reply):
        if self.prefix == "client":
            return f"client{reported_pid(reply)}"
        return self.principal(reply)

    def _alert(self, level, message):
        if self.owner is not None and hasattr(self.owner, "alert"):
            self.owner.alert(level, message)

    def aggregate_train(self, server_round, replies):
        replies = list(replies)
        errors = [r for r in replies if r.has_error()]
        valid = [r for r in replies if not r.has_error()]
        if not valid:
            return self.inner.aggregate_train(server_round, replies)
        if not self.defenses:
            population = [self.label(r) for r in valid]
            self.log.append({"round": server_round, "population": population, "flagged": [], "reasons": {},
                             "seconds": 0.0, **detectors.precision_recall([], self.malicious, population)})
            return self.inner.aggregate_train(server_round, replies)
        if self.owner is not None and getattr(self.owner, "edge_id", 0) is None:
            self.owner.round = server_round
        start = time.perf_counter()
        flags = {}
        checks = [d for d in self.defenses if d in AUTH]
        if checks:
            for k, r in enumerate(valid):
                problems = verify(r.content, self.principal(r), server_round, self.keys, checks)
                if problems:
                    flags[k] = problems
        rest = [k for k in range(len(valid)) if k not in flags]
        keys = float_keys(self.global_arrays)
        if rest and any(d in STATISTICAL or d == "clip" for d in self.defenses):
            U = np.stack([flatten(delta(to_arrays(valid[k].content["arrays"]), self.global_arrays), keys)
                          for k in rest])
            found = {}
            if "stat" in self.defenses and len(rest) >= 3:
                found.update(detectors.statistical(U))
            if "data_dist" in self.defenses:
                stats = [[valid[k].content["metrics"].get(m, 0.0) for m in ("data-mean", "data-std", "label-entropy")]
                         for k in rest]
                for i, why in detectors.data_distribution(stats).items():
                    found.setdefault(i, []).extend(why)
            if "ocsvm" in self.defenses and len(rest) >= 3:
                for i, why in self.ocsvm(U).items():
                    found.setdefault(i, []).extend(why)
            if "kmeans" in self.defenses:
                for i, why in detectors.kmeans_collusion(U, seed=self.seed).items():
                    found.setdefault(i, []).extend(why)
            if "monitor" in self.defenses and len(rest) >= 2:
                for i, why in detectors.insider_monitor(U).items():
                    found.setdefault(i, []).extend(why)
            for i, why in found.items():
                flags[rest[i]] = why
            keep_local = [i for i in range(len(rest)) if rest[i] not in flags]
            if "ocsvm" in self.defenses:
                self.ocsvm.accept(U, keep_local)
            if "clip" in self.defenses and keep_local:
                norms = np.linalg.norm(U, axis=1)
                bound = self.clip_factor * float(np.median(norms[keep_local]))
                for i in keep_local:
                    if norms[i] > bound:
                        r = valid[rest[i]]
                        arrays = to_arrays(r.content["arrays"])
                        g = {k: self.global_arrays[k] for k in keys}
                        arrays.update(unflatten(flatten(g, keys) + U[i] * (bound / norms[i]), g))
                        r.content["arrays"] = to_record(arrays)
        kept = [r for k, r in enumerate(valid) if k not in flags]
        population = [self.label(r) for r in valid]
        flagged = [population[k] for k in flags]
        for k, why in flags.items():
            self._alert("alert", f"rejected {population[k]}: {'; '.join(why)}")
        entry = {"round": server_round, "population": population, "flagged": flagged,
                 "reasons": {population[k]: why for k, why in flags.items()},
                 "seconds": time.perf_counter() - start,
                 **detectors.precision_recall(flagged, self.malicious, population)}
        self.log.append(entry)
        if not kept:
            self._alert("alert", f"round {server_round}: every update rejected, model kept")
            return None, None
        return self.inner.aggregate_train(server_round, kept + errors)
