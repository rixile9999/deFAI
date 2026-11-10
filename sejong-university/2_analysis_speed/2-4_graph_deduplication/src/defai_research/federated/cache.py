from __future__ import annotations

from collections import defaultdict
from hashlib import sha256
from pathlib import Path
import struct
import time
from typing import Any

import numpy as np

from .graph import GraphDeduplicator
from .snapshots import (
    ImmutableTrainingSnapshot,
    deserialize_digest_packet,
    deserialize_ownership_packet,
    serialize_digest_packet,
    serialize_ownership_packet,
)


class _PreparedIdentityGraphDeduplicator(GraphDeduplicator):
    def prepare_snapshot(self, snapshot):
        self.prepared_snapshot = snapshot
        self._identity_cursor = 0

    def _fingerprints(self, x, y):
        cid = self._identity_cursor
        snapshot = self.prepared_snapshot
        if (cid >= len(snapshot.x) or x is not snapshot.x[cid]
                or (snapshot.y is not None and y is not snapshot.y[cid])):
            raise AssertionError("Prepared SHA256 identities are not bound to these immutable arrays")
        self._identity_cursor += 1
        return list(snapshot.fingerprints[cid])


class GraphSAGECachedDeduplicator:
    def __init__(
        self, seed=42, dynamic=True, refresh_interval=4, *,
        refresh_policy="event_only", audit_dir: str | Path | None = None,
        **graph_options,
    ):
        if refresh_policy not in {"periodic4", "event_only"}:
            raise ValueError("refresh_policy must be periodic4 or event_only")
        if refresh_interval < 1:
            raise ValueError("refresh_interval must be positive")
        if graph_options.get("mode", "graphsage") != "graphsage":
            raise ValueError("Actual learned GraphSAGE is required")
        if int(graph_options.get("epochs", 100)) < 1:
            raise ValueError("Actual GraphSAGE training requires positive epochs")
        graph_options["mode"] = "graphsage"
        self.graph = _PreparedIdentityGraphDeduplicator(
            seed=seed, dynamic=False, refresh_interval=refresh_interval,
            audit_dir=audit_dir, **graph_options,
        )
        self.seed, self.dynamic = int(seed), bool(dynamic)
        self.refresh_interval, self.refresh_policy = int(refresh_interval), refresh_policy
        self.history = self.graph.history
        self.last_packets: list[dict[str, Any]] = []
        self._last_round = None
        self._snapshot = None
        self._seen_snapshots = set()
        self._client_versions = {}
        self._received_digests = {}
        self._received_summaries = {}
        self._graph_monitor_state = None
        self._client_masks = {}
        self._cached_masks = None
        self._cached_original = None
        self._cache_key = None
        self._delivered_adjacency = None

    @property
    def candidate_adjacency(self):
        return self.graph.candidate_adjacency

    @property
    def candidate_components(self):
        return self.graph.candidate_components

    @property
    def last_retained_multiplicities(self):
        return self.graph.last_retained_multiplicities

    def _packet(self, kind, payload, *, bootstrap=False):
        if not isinstance(payload, bytes):
            raise TypeError("Control payloads must be serialized bytes")
        self.last_packets.append({"kind": kind, "payload": payload, "bootstrap": bool(bootstrap)})

    def _summary_packet(self, snapshot, cid, round_id, summary, *, bootstrap):
        values = np.asarray(summary, dtype="<f8")
        payload = (struct.pack("<QQQII", cid, round_id, len(snapshot.x[cid]),
                               len(values), snapshot.x[cid].shape[1])
                   + snapshot.ordered_versions[cid] + values.tobytes())
        who, rnd, rows, dimension, features = struct.unpack_from("<QQQII", payload)
        decoded = np.frombuffer(payload, dtype="<f8", offset=64).copy()
        if (who != cid or rnd != round_id or rows != len(snapshot.x[cid])
                or features != snapshot.x[cid].shape[1] or dimension != len(values)
                or payload[32:64] != snapshot.ordered_versions[cid]
                or not np.array_equal(values, decoded)):
            raise AssertionError("Summary upload round trip failed")
        self._packet("summary_upload", payload, bootstrap=bootstrap)
        self._received_summaries[cid] = (rows, decoded)

    def _graph_packets(self, round_id, *, bootstrap):
        adjacency = self.graph.candidate_adjacency
        graph_version = sha256(adjacency.tobytes()).digest()
        components = {cid: comp for comp in self.graph.candidate_components for cid in comp}
        for cid in range(self.graph.n_clients):
            neighbors = np.flatnonzero(adjacency[cid]).astype("<u4")
            members = np.asarray(components[cid], dtype="<u4")
            payload = (struct.pack("<QQII", cid, round_id, len(neighbors), len(members))
                       + graph_version + neighbors.tobytes() + members.tobytes())
            who, rnd, degree, count = struct.unpack_from("<QQII", payload)
            decoded = np.frombuffer(payload, dtype="<u4", offset=56)
            if (who != cid or rnd != round_id or degree != len(neighbors)
                    or count != len(members) or payload[24:56] != graph_version
                    or not np.array_equal(decoded[:degree], neighbors)
                    or not np.array_equal(decoded[degree:], members)):
                raise AssertionError("Candidate graph download round trip failed")
            self._packet("graph_download", payload, bootstrap=bootstrap)
        self._delivered_adjacency = adjacency.tobytes()

    def apply(self, snapshot, round_id, *, active=None, force_refresh=False):
        started_cpu, started_wall = time.process_time(), time.perf_counter()
        if not isinstance(snapshot, ImmutableTrainingSnapshot):
            raise TypeError("Use ImmutableTrainingSnapshot.from_arrays for training input")
        snapshot.validate_readonly()
        rnd = int(round_id)
        if rnd < 1 or (self._last_round is not None and rnd <= self._last_round):
            raise ValueError("Apply once per strictly increasing positive round")
        all_clients = list(range(len(snapshot.x)))
        active = all_clients if active is None else sorted(set(int(cid) for cid in active))
        if active != all_clients:
            raise ValueError("Original GraphSAGE component matching requires every client active")
        if force_refresh and not self.dynamic:
            raise ValueError("Static GraphSAGE cannot be force refreshed")
        if self.graph.fitted and (len(snapshot.x) != self.graph.n_clients
                                 or snapshot.x[0].shape[1] != self.graph.feature_dim):
            raise ValueError("Snapshot client count or feature dimension changed")
        if self._snapshot is not None and ((snapshot.y is None) != (self._snapshot.y is None)):
            raise ValueError("Training label identity mode changed")

        self.last_packets = []
        replaced = snapshot is not self._snapshot
        changed_versions = []
        for cid in active:
            version = snapshot.ordered_versions[cid]
            payload = struct.pack("<QQ", cid, rnd) + version
            who, version_round = struct.unpack_from("<QQ", payload)
            if who != cid or version_round != rnd or payload[16:] != version:
                raise AssertionError("Training version round trip failed")
            self._packet("control_version", payload)
            if self._client_versions.get(cid) != version:
                changed_versions.append(cid)
                self._received_digests[cid] = {}
                self._client_masks[cid] = np.ones(len(snapshot.x[cid]), dtype=bool)
                self._client_versions[cid] = version

        before_history = len(self.history)
        bootstrap = not self.graph.fitted
        monitoring_cpu, monitoring_wall = time.process_time(), time.perf_counter()
        if bootstrap:
            self.graph.fit(list(snapshot.x), round_id=0)
            summaries = np.asarray(self.history[-1]["summaries"], dtype=np.float64)
        else:
            summaries = self.graph._summarize(list(snapshot.x))
        monitoring_cpu = time.process_time() - monitoring_cpu
        monitoring_wall = time.perf_counter() - monitoring_wall
        monitor_state = tuple((len(snapshot.x[cid]), np.asarray(summary, dtype="<f8").tobytes())
                              for cid, summary in enumerate(summaries))
        changed_summaries = []
        for cid, summary in enumerate(summaries):
            previous = self._received_summaries.get(cid)
            if (previous is None or previous[0] != len(snapshot.x[cid])
                    or not np.array_equal(previous[1], summary)):
                changed_summaries.append(cid)
                self._summary_packet(snapshot, cid, rnd, summary, bootstrap=bootstrap)
        summary_drift = (self._graph_monitor_state is not None
                         and monitor_state != self._graph_monitor_state)
        if bootstrap:
            self._graph_monitor_state = monitor_state
        scheduled = self.refresh_policy == "periodic4" and rnd - self.graph.last_refresh_round >= self.refresh_interval
        event = self.refresh_policy == "event_only" and summary_drift
        refit = bool(not bootstrap and self.dynamic and (force_refresh or scheduled or event))
        if refit:
            received_summaries = np.asarray([self._received_summaries[cid][1] for cid in active])
            self.graph._refresh(list(snapshot.x), rnd, summaries=received_summaries)
            self._graph_monitor_state = monitor_state
        new_audits = self.history[before_history:]
        adjacency = self.graph.candidate_adjacency.tobytes()
        topology_changed = self._delivered_adjacency != adjacency
        if topology_changed:
            self._graph_packets(rnd, bootstrap=bootstrap)

        cache_key = (tuple(snapshot.ordered_versions), adjacency)
        mask_cache_hit = self._cached_masks is not None and cache_key == self._cache_key
        if mask_cache_hit:
            masks = [mask.copy() for mask in self._cached_masks]
            original = dict(self._cached_original)
            current_verification_pairs = 0
        else:
            self.graph.prepare_snapshot(snapshot)
            masks, original = self.graph.apply(list(snapshot.x), rnd, client_y=None if snapshot.y is None else list(snapshot.y))
            if self.graph._identity_cursor != len(snapshot.x):
                raise AssertionError("Original exact search did not consume the expected immutable identities")
            current_verification_pairs = original["exact_verification_pairs"]
            self._cached_masks = [mask.copy() for mask in masks]
            self._cached_original = dict(original)
            self._cache_key = cache_key

        searched = [cid for comp in self.graph.candidate_components if len(comp) > 1 for cid in comp]
        uncached_digest_bytes = 0
        new_digest_rows = 0
        for cid in searched:
            values = snapshot.fingerprints[cid]
            uncached_digest_bytes += 52 + 40 * len(values)
            received = self._received_digests[cid]
            missing = [(index, digest) for index, digest in enumerate(values) if index not in received]
            if missing:
                payload = serialize_digest_packet(cid, rnd, snapshot.ordered_versions[cid], missing)
                who, version_round, version, decoded = deserialize_digest_packet(payload)
                if (who != cid or version_round != rnd or version != snapshot.ordered_versions[cid]
                        or decoded != missing):
                    raise AssertionError("Full SHA256 upload round trip failed")
                self._packet("digest_upload", payload)
                received.update(decoded)
                new_digest_rows += len(decoded)
            if len(received) != len(values) or any(received[index] != digest for index, digest in enumerate(values)):
                raise AssertionError("Candidate component lacks received full SHA256 identities")

        uncached_ownership_bytes = 0
        ownership_changed_rows = 0
        for cid, mask in enumerate(masks):
            prior = self._client_masks[cid]
            uncached_ownership_bytes += 52 + 9 * int((~mask).sum())
            changed = np.flatnonzero(prior != mask)
            if len(changed):
                changes = [(int(index), bool(mask[index])) for index in changed]
                payload = serialize_ownership_packet(cid, rnd, snapshot.ordered_versions[cid], changes)
                who, version_round, version, decoded = deserialize_ownership_packet(payload)
                delivered = prior.copy()
                for index, keep in decoded:
                    delivered[index] = keep
                if (who != cid or version_round != rnd or version != snapshot.ordered_versions[cid]
                        or not np.array_equal(delivered, mask)):
                    raise AssertionError("Ownership delta did not reconstruct the exact local mask")
                self._packet("ownership_delta", payload)
                ownership_changed_rows += len(changed)
                self._client_masks[cid] = delivered
            elif not np.array_equal(prior, mask):
                raise AssertionError("Unchanged ownership delivery disagrees with exact mask")

        sizes = defaultdict(int)
        bootstrap_bytes = 0
        for packet in self.last_packets:
            sizes[packet["kind"]] += len(packet["payload"])
            if packet["bootstrap"]:
                bootstrap_bytes += len(packet["payload"])
        prep_cpu = prep_wall = prep_hash = copied = 0
        if snapshot not in self._seen_snapshots:
            self._seen_snapshots.add(snapshot)
            prep_cpu, prep_wall = snapshot.construction_cpu_seconds, snapshot.construction_wall_seconds
            prep_hash, copied = snapshot.construction_hash_cpu_seconds, snapshot.copied_bytes
        audit = {
            **original, **self.graph.last_graph_metadata,
            "round": rnd, "mode": "graphsage", "dynamic": self.dynamic,
            "refresh_policy": self.refresh_policy, "graph_refresh": bool(new_audits),
            "graph_refresh_round": self.graph.last_refresh_round,
            "bootstrap_graph_training": bootstrap, "actual_graphsage_fits_this_round": len(new_audits),
            "actual_graphsage_total_fits": len(self.history), "graph_topology_changed": topology_changed,
            "changed_summary_clients": changed_summaries, "changed_ordered_version_clients": changed_versions,
            "training_summary_drift_since_graph_fit": bool(summary_drift),
            "summary_monitoring_clients": len(active), "summary_monitoring_cpu_seconds": monitoring_cpu,
            "summary_monitoring_wall_seconds": monitoring_wall,
            "summary_monitoring_timing_scope": "initial call includes the initial fit; later calls contain local summary computation only",
            "active_clients": active, "snapshot_replaced": replaced,
            "snapshot_preparation_cpu_seconds_charged": prep_cpu,
            "snapshot_preparation_wall_seconds_charged": prep_wall,
            "snapshot_copied_bytes_charged": copied, "snapshot_hash_cpu_seconds_charged": prep_hash,
            "hash_cpu_seconds": prep_hash, "hash_cpu_seconds_current_apply": 0.,
            "hashing_scope": "full SHA256 actually computed once during immutable snapshot construction; same ordered client version reuses those exact identities",
            "matching_scope": "original GraphDeduplicator connected components plus exact feature and label array collision guard",
            "mask_cache_hit": mask_cache_hit, "exact_verification_pairs": current_verification_pairs,
            "searched_clients": searched, "new_exact_digest_rows_sent": new_digest_rows,
            "ownership_changed_rows_sent": ownership_changed_rows,
            "summary_upload_bytes": sizes["summary_upload"], "summary_bytes_this_round": sizes["summary_upload"],
            "graph_download_bytes": sizes["graph_download"], "control_version_bytes": sizes["control_version"],
            "digest_upload_bytes": sizes["digest_upload"], "ownership_reply_bytes": sizes["ownership_delta"],
            "bootstrap_metadata_bytes_charged": bootstrap_bytes,
            "bootstrap_bytes_subset_of_categories": True,
            "metadata_bytes": sum(len(p["payload"]) for p in self.last_packets),
            "packet_count": len(self.last_packets),
            "uncached_full_digest_bytes_same_round": uncached_digest_bytes,
            "digest_bytes_avoided_by_version_cache": uncached_digest_bytes - sizes["digest_upload"],
            "uncached_full_ownership_bytes_same_round": uncached_ownership_bytes,
            "ownership_bytes_avoided_by_delta_delivery": uncached_ownership_bytes - sizes["ownership_delta"],
            "cache_scope": "full digest by client ordered version; mask by all ordered versions and actual candidate adjacency; graph download only on changed actual adjacency",
            "graph_training_seconds_this_round": sum(float(h["graph_training_seconds"]) for h in new_audits),
            "policy_cpu_seconds": time.process_time() - started_cpu,
            "policy_seconds": time.perf_counter() - started_wall,
            "wire_scope": "actual serialized application payload buffers in central-coordinator simulation; not observed physical network traffic",
            "exact_array_guard_scope": "in-process original collision guard, no raw-record transmission charged",
            "privacy_guarantee": "none; summaries, versions and hashes may reveal membership and linkability",
        }
        self._snapshot, self._last_round, self.last_audit = snapshot, rnd, audit
        return [mask.copy() for mask in masks], audit
