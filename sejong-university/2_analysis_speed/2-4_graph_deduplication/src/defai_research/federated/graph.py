
from __future__ import annotations

from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
import time
from typing import Any

import numpy as np


def _cosine(x: np.ndarray) -> np.ndarray:
    z = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    return np.clip(z @ z.T, -1.0, 1.0)


def _components(adjacency: np.ndarray) -> list[list[int]]:
    remaining = set(range(len(adjacency)))
    result: list[list[int]] = []
    while remaining:
        stack = [min(remaining)]
        remaining.remove(stack[0])
        component: list[int] = []
        while stack:
            node = stack.pop()
            component.append(node)
            neighbors = [int(j) for j in np.flatnonzero(adjacency[node]) if int(j) in remaining]
            remaining.difference_update(neighbors)
            stack.extend(neighbors)
        result.append(sorted(component))
    return result


def _kmeans(x: np.ndarray, k: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    k = min(k, len(x))
    first = int(np.random.default_rng(seed).integers(len(x)))
    chosen = [first]
    for _ in range(1, k):
        distances = np.min(np.sum((x[:, None, :] - x[chosen][None, :, :]) ** 2, axis=2), axis=1)
        distances[chosen] = -1
        chosen.append(int(np.argmax(distances)))
    centers = x[chosen].copy()
    assignment = np.zeros(len(x), dtype=int)
    for _ in range(40):
        new_assignment = np.argmin(np.sum((x[:, None, :] - centers[None, :, :]) ** 2, axis=2), axis=1)
        new_centers = centers.copy()
        for j in range(k):
            if np.any(new_assignment == j):
                new_centers[j] = x[new_assignment == j].mean(axis=0)
        if np.array_equal(new_assignment, assignment) and np.allclose(new_centers, centers):
            break
        assignment, centers = new_assignment, new_centers
    return assignment, centers


class GraphDeduplicator:
    def __init__(
        self,
        seed: int = 42,
        dynamic: bool = True,
        refresh_interval: int = 4,
        *,
        regions: int = 4,
        candidate_k: int = 2,
        candidate_threshold: float = 0.20,
        epochs: int = 100,
        fanout: int = 3,
        hidden_dim: int = 12,
        embedding_dim: int = 8,
        mode: str = "graphsage",
        audit_dir: str | Path | None = None,
    ) -> None:
        if mode not in {"graphsage", "raw_summary"}:
            raise ValueError("mode must be graphsage or raw_summary")
        if any(isinstance(v, bool) or not isinstance(v, (int, np.integer)) or v < 1
               for v in (refresh_interval, candidate_k, regions, epochs, fanout, hidden_dim, embedding_dim)):
            raise ValueError("Graph counts and dimensions must be positive integers")
        if not np.isfinite(candidate_threshold) or not 0 <= candidate_threshold <= 1:
            raise ValueError("candidate_threshold must be finite in [0, 1]")
        self.seed = int(seed)
        self.dynamic = bool(dynamic)
        self.refresh_interval = int(refresh_interval)
        self.regions = int(regions)
        self.candidate_k = int(candidate_k)
        self.candidate_threshold = float(candidate_threshold)
        self.epochs = int(epochs)
        self.fanout = int(fanout)
        self.hidden_dim = int(hidden_dim)
        self.embedding_dim = int(embedding_dim)
        self.mode = mode
        self.audit_dir = Path(audit_dir) if audit_dir else None
        self.fitted = False
        self._bootstrap_pending = False
        self.history: list[dict[str, Any]] = []

    @staticmethod
    def _validate(client_X: list[np.ndarray]) -> list[np.ndarray]:
        arrays = [np.asarray(x) for x in client_X]
        if not arrays or any(x.ndim != 2 or len(x) == 0 or x.shape[1] == 0 for x in arrays):
            raise ValueError("Each client must provide a non-empty 2D training array with at least one feature")
        if any(not (np.issubdtype(x.dtype, np.integer) or np.issubdtype(x.dtype, np.floating)
                    or np.issubdtype(x.dtype, np.bool_)) for x in arrays):
            raise ValueError("Training features must have real numeric or boolean dtype")
        if len({x.shape[1] for x in arrays}) != 1:
            raise ValueError("Client feature dimensions must match")
        if any(not np.all(np.isfinite(x)) for x in arrays):
            raise ValueError("Non-finite features are unsupported")
        return arrays

    def _summarize(self, arrays: list[np.ndarray]) -> np.ndarray:
        rows = []
        for x in arrays:
            x64 = x.astype(np.float64, copy=False)
            mean_sketch = x64.mean(axis=0) @ self.moment_projection
            std_sketch = x64.std(axis=0) @ self.moment_projection
            projected = x64 @ self.prototype_projection
            distances = np.sum((projected[:, None, :] - self.prototypes[None, :, :]) ** 2, axis=2)
            logits = -distances / 0.35
            logits -= logits.max(axis=1, keepdims=True)
            soft_assignments = np.exp(logits)
            soft_assignments /= soft_assignments.sum(axis=1, keepdims=True)
            occupancy = soft_assignments.mean(axis=0)
            rows.append(np.concatenate([mean_sketch, std_sketch, occupancy]))
        return np.asarray(rows)

    def fit(self, client_X: list[np.ndarray], round_id: int = 0) -> "GraphDeduplicator":
        arrays = self._validate(client_X)
        self.n_clients = len(arrays)
        self.feature_dim = arrays[0].shape[1]
        rng = np.random.default_rng(self.seed)
        self.moment_projection = rng.normal(size=(self.feature_dim, 8)) / np.sqrt(self.feature_dim)
        self.prototype_projection = rng.normal(size=(self.feature_dim, 2)) / np.sqrt(self.feature_dim)
        angles = np.arange(8) * (2 * np.pi / 8)
        self.prototypes = 0.7 * np.stack([np.cos(angles), np.sin(angles)], axis=1)
        summaries = self._summarize(arrays)
        self.summary_mean = summaries.mean(axis=0)
        self.summary_scale = np.maximum(summaries.std(axis=0), 0.025)
        standardized = (summaries - self.summary_mean) / self.summary_scale
        _, _, vt = np.linalg.svd(standardized, full_matrices=False)
        self.pca_basis = vt[:2].T
        self.fitted = True
        self._refresh(arrays, int(round_id), summaries=summaries)
        self.bootstrap_metadata_bytes = int(self.last_graph_metadata["summary_upload_bytes"])
        self._bootstrap_compute_s = self.last_graph_metadata["graph_training_seconds"]
        self._bootstrap_pending = True
        return self

    def _training_graph(self, standardized: np.ndarray, sizes: np.ndarray, round_id: int):
        region_assignment, centers = _kmeans(standardized, self.regions, self.seed)
        features = np.concatenate([standardized, centers], axis=0)
        n, r = len(standardized), len(centers)
        weighted = np.zeros((n + r, n + r), dtype=np.float64)
        positions = standardized @ self.pca_basis
        distances = np.sum((positions[:, None, :] - positions[None, :, :]) ** 2, axis=2)
        positive_distances = distances[distances > 1e-12]
        sigma2 = max(float(np.median(positive_distances)) if len(positive_distances) else 1.0, 1e-6)
        shared_similarity = np.exp(-distances / sigma2)
        for i in range(n):
            region = int(region_assignment[i])
            region_total = max(float(sizes[region_assignment == region].sum()), 1.0)
            weight = 0.35 + 0.65 * float(sizes[i]) / region_total
            weighted[i, n + region] = weighted[n + region, i] = weight
            neighbors = np.argsort(distances[i] + np.eye(n)[i] * 1e30)[:min(3, n - 1)]
            for j in neighbors:
                weighted[i, j] = weighted[j, i] = max(weighted[i, j], float(shared_similarity[i, j]))
        return features, weighted, region_assignment, positions, shared_similarity

    def _train_graphsage(self, x: np.ndarray, graph: np.ndarray, round_id: int):
        rng = np.random.default_rng(self.seed + 104729)
        n, p = x.shape
        h, e = self.hidden_dim, self.embedding_dim
        params = {
            "pool_W": rng.normal(0, 0.18 / np.sqrt(p), (p, h)),
            "pool_b": np.full(h, 0.10),
            "out_W": rng.normal(0, 0.35 / np.sqrt(p + h), (p + h, e)),
            "out_b": np.zeros(e),
        }
        positives = np.argwhere(np.triu(graph > 0, 1))
        negatives = np.argwhere(np.triu(graph == 0, 1))
        if len(positives) == 0 or len(negatives) == 0:
            raise ValueError("Graph link training requires positive and negative pairs")
        moments = {key: np.zeros_like(value) for key, value in params.items()}
        variances = {key: np.zeros_like(value) for key, value in params.items()}
        losses: list[float] = []
        samples_per_epoch: list[list[list[int]]] = []

        def forward(sample: bool):
            pre = x @ params["pool_W"] + params["pool_b"]
            activated = np.maximum(pre, 0)
            pooled = np.zeros((n, h))
            argmax = np.zeros((n, h), dtype=int)
            edge_scale = np.zeros((n, h))
            sampled_neighbors = []
            for i in range(n):
                neighbors = np.flatnonzero(graph[i] > 0)
                if len(neighbors) == 0:
                    neighbors = np.array([i])
                    weights = np.ones(1)
                else:
                    weights = graph[i, neighbors].copy()
                if sample and len(neighbors) > self.fanout:
                    chosen = rng.choice(len(neighbors), self.fanout, replace=False, p=weights / weights.sum())
                    neighbors, weights = neighbors[chosen], weights[chosen]
                values = activated[neighbors] * weights[:, None]
                local_argmax = values.argmax(axis=0)
                pooled[i] = values[local_argmax, np.arange(h)]
                argmax[i] = neighbors[local_argmax]
                edge_scale[i] = weights[local_argmax]
                sampled_neighbors.append(neighbors.tolist())
            combined = np.concatenate([x, pooled], axis=1)
            z = np.tanh(combined @ params["out_W"] + params["out_b"])
            return z, (pre, combined, argmax, edge_scale), sampled_neighbors

        for epoch in range(1, self.epochs + 1):
            z, cache, sampled_neighbors = forward(True)
            sampled_negatives = negatives[rng.choice(len(negatives), len(positives), replace=len(negatives) < len(positives))]
            pairs = np.concatenate([positives, sampled_negatives], axis=0)
            labels = np.concatenate([np.ones(len(positives)), np.zeros(len(sampled_negatives))])
            logits = np.sum(z[pairs[:, 0]] * z[pairs[:, 1]], axis=1) / np.sqrt(e)
            loss = np.mean(np.logaddexp(0, logits) - labels * logits)
            losses.append(float(loss))
            probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))
            gradient_logit = (probabilities - labels) / (len(pairs) * np.sqrt(e))
            gradient_z = np.zeros_like(z)
            np.add.at(gradient_z, pairs[:, 0], gradient_logit[:, None] * z[pairs[:, 1]])
            np.add.at(gradient_z, pairs[:, 1], gradient_logit[:, None] * z[pairs[:, 0]])
            pre, combined, argmax, edge_scale = cache
            gradient_output = gradient_z * (1 - z ** 2)
            gradients = {
                "out_W": combined.T @ gradient_output,
                "out_b": gradient_output.sum(axis=0),
            }
            gradient_pool = (gradient_output @ params["out_W"].T)[:, p:]
            gradient_activation = np.zeros((n, h))
            for i in range(n):
                np.add.at(gradient_activation, (argmax[i], np.arange(h)), gradient_pool[i] * edge_scale[i])
            gradient_pre = gradient_activation * (pre > 0)
            gradients["pool_W"] = x.T @ gradient_pre
            gradients["pool_b"] = gradient_pre.sum(axis=0)
            for key in params:
                gradient = gradients[key] + (1e-4 * params[key] if key.endswith("W") else 0)
                moments[key] = 0.9 * moments[key] + 0.1 * gradient
                variances[key] = 0.999 * variances[key] + 0.001 * gradient * gradient
                params[key] -= 0.025 * (moments[key] / (1 - 0.9 ** epoch)) / (np.sqrt(variances[key] / (1 - 0.999 ** epoch)) + 1e-8)
            if epoch in {1, self.epochs}:
                samples_per_epoch.append(sampled_neighbors)
        embeddings, _, _ = forward(False)
        return embeddings, losses, params, samples_per_epoch

    def _refresh(self, arrays: list[np.ndarray], round_id: int, summaries: np.ndarray | None = None):
        started = time.perf_counter()
        summaries = self._summarize(arrays) if summaries is None else summaries
        standardized = np.clip((summaries - self.summary_mean) / self.summary_scale, -8, 8)
        sizes = np.asarray([len(x) for x in arrays])
        node_features, graph, assignments, positions, shared_similarity = self._training_graph(standardized, sizes, round_id)
        if self.mode == "graphsage":
            embeddings, losses, params, sampled = self._train_graphsage(node_features, graph, round_id)
            embedding_similarity = (_cosine(embeddings[:self.n_clients]) + 1) / 2
            combined_similarity = 0.65 * embedding_similarity + 0.35 * shared_similarity
        else:
            embeddings, losses, params, sampled = standardized, [], {}, []
            embedding_similarity = (_cosine(standardized) + 1) / 2
            combined_similarity = shared_similarity
        n = self.n_clients
        ranked = np.argsort(-(combined_similarity - np.eye(n) * 1e6), axis=1)[:, :min(self.candidate_k, n - 1)]
        candidates = np.zeros((n, n), dtype=bool)
        for i in range(n):
            for j in ranked[i]:
                if i in ranked[j] and assignments[i] == assignments[j] and combined_similarity[i, j] >= self.candidate_threshold:
                    candidates[i, j] = candidates[j, i] = True
        self.candidate_adjacency = candidates
        self.candidate_components = _components(candidates)
        self.last_refresh_round = round_id
        elapsed = time.perf_counter() - started
        upload_bytes = int(n * (summaries.shape[1] * 8 + 8))
        graph_metadata: dict[str, Any] = {
            "round": round_id,
            "mode": self.mode,
            "summary_upload_bytes": upload_bytes,
            "graph_training_seconds": elapsed,
            "edge_count": int(candidates.sum() // 2),
            "training_edge_count": int((graph > 0).sum() // 2),
            "client_nodes": n,
            "region_nodes": int(len(node_features) - n),
            "components": self.candidate_components,
            "region_assignment": assignments.tolist(),
            "candidate_threshold": self.candidate_threshold,
            "candidate_k": self.candidate_k,
            "embedding_loss_first": losses[0] if losses else None,
            "embedding_loss_last": losses[-1] if losses else None,
            "embedding_loss_best": min(losses) if losses else None,
            "embedding_epochs": len(losses),
            "learned_parameter_count": int(sum(value.size for value in params.values())),
            "summary_dimension": int(summaries.shape[1]),
            "weighted_sampling_fanout": self.fanout,
            "link_negative_positive_ratio": 1,
            "aggregation": "edge_weighted_relu_maxpool_concat_tanh",
            "train_only_unlabeled": True,
        }
        self.last_graph_metadata = graph_metadata
        audit = dict(graph_metadata)
        audit.update({
            "loss_curve": losses,
            "summaries": summaries.tolist(),
            "summary_pca_2d": positions.tolist(),
            "embedding": embeddings.tolist(),
            "training_weighted_adjacency": graph.tolist(),
            "candidate_adjacency": candidates.astype(int).tolist(),
            "combined_candidate_similarity": combined_similarity.tolist(),
            "first_last_sampled_neighbors": sampled,
        })
        self.history.append(audit)
        if self.audit_dir:
            self.audit_dir.mkdir(parents=True, exist_ok=True)
            (self.audit_dir / f"graph_round_{round_id:03d}.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
            if params:
                np.savez_compressed(self.audit_dir / f"graph_weights_round_{round_id:03d}.npz", **params)

    @staticmethod
    def _fingerprints(x: np.ndarray, y: np.ndarray | None) -> list[bytes]:
        if y is not None and len(y) != len(x):
            raise ValueError("Label and feature lengths must match")
        header = f"{x.dtype.str}:{x.shape[1]}:".encode("ascii")
        result = []
        for index, row in enumerate(x):
            digest = sha256(header)
            digest.update(np.ascontiguousarray(row).tobytes())
            if y is not None:
                label = np.asarray(y[index])
                digest.update(f"|y:{label.dtype.str}:{label.shape}:".encode("ascii"))
                digest.update(np.ascontiguousarray(label).tobytes())
            result.append(digest.digest())
        return result

    def apply(
        self,
        client_X: list[np.ndarray],
        round_id: int,
        force_refresh: bool = False,
        *,
        client_y: list[np.ndarray] | None = None,
    ) -> tuple[list[np.ndarray], dict[str, Any]]:
        started = time.perf_counter()
        arrays = self._validate(client_X)
        if not self.fitted:
            self.fit(arrays, round_id=0)
        if len(arrays) != self.n_clients:
            raise ValueError("Client count must remain fixed")
        if any(x.shape[1] != self.feature_dim for x in arrays):
            raise ValueError("Feature dimension changed")
        if client_y is not None and len(client_y) != len(arrays):
            raise ValueError("Client label arrays must match client count")
        refresh = bool(force_refresh or (self.dynamic and round_id - self.last_refresh_round >= self.refresh_interval))
        if refresh:
            self._refresh(arrays, int(round_id))
        hashes = [self._fingerprints(x, None if client_y is None else np.asarray(client_y[i])) for i, x in enumerate(arrays)]
        masks = [np.ones(len(x), dtype=bool) for x in arrays]
        hash_equal_groups = 0
        cross_client_removed = 0
        verification_pairs = 0
        retained_multiplicities = [np.ones(len(x), dtype=np.int64) for x in arrays]
        for component in self.candidate_components:
            buckets: dict[bytes, list[tuple[int, int]]] = defaultdict(list)
            for client in component:
                for index, digest in enumerate(hashes[client]):
                    buckets[digest].append((client, index))
            for members in buckets.values():
                if len(members) <= 1:
                    continue
                members.sort()
                owner_client, owner_index = members[0]
                retained_multiplicities[owner_client][owner_index] = len(members)
                hash_equal_groups += 1
                for client, index in members[1:]:
                    same_x = arrays[owner_client][owner_index].dtype == arrays[client][index].dtype and np.array_equal(arrays[owner_client][owner_index], arrays[client][index])
                    same_y = client_y is None or np.array_equal(client_y[owner_client][owner_index], client_y[client][index])
                    verification_pairs += 1
                    if same_x and same_y:
                        masks[client][index] = False
                        cross_client_removed += int(client != owner_client)
        searched_clients = [i for comp in self.candidate_components if len(comp) > 1 for i in comp]
        digest_bytes = sum(len(arrays[i]) * (32 + 8) + 16 for i in searched_clients)
        summary_bytes = int(self.last_graph_metadata["summary_upload_bytes"]) if refresh else 0
        bootstrap_bytes = self.bootstrap_metadata_bytes if self._bootstrap_pending else 0
        bootstrap_seconds = self._bootstrap_compute_s if self._bootstrap_pending else 0.0
        self._bootstrap_pending = False
        ownership_reply_bytes = sum(int((~mask).sum()) * 8 + 16 for mask in masks)
        removed = sum(int((~mask).sum()) for mask in masks)
        candidate_pairs = sum(len(comp) * (len(comp) - 1) // 2 for comp in self.candidate_components)
        metadata = {
            **self.last_graph_metadata,
            "round": int(round_id),
            "graph_refresh": refresh,
            "graph_refresh_round": self.last_refresh_round,
            "duplicate_removed": removed,
            "cross_client_removed": cross_client_removed,
            "retained_records": sum(int(mask.sum()) for mask in masks),
            "input_records": sum(len(x) for x in arrays),
            "candidate_client_pairs": candidate_pairs,
            "all_client_pairs": self.n_clients * (self.n_clients - 1) // 2,
            "candidate_pair_reduction": 1 - candidate_pairs / max(self.n_clients * (self.n_clients - 1) / 2, 1),
            "hash_scope": "features_and_labels" if client_y is not None else "features_only",
            "hash_algorithm": "SHA256_then_exact_array_equality",
            "exact_array_guard_scope": "local_simulation_only; modeled distributed wire carries digests, not raw rows",
            "singleton_component_deduplication": "local_only; no digest upload charged",
            "hash_equal_groups": hash_equal_groups,
            "exact_verification_pairs": verification_pairs,
            "digest_upload_bytes": int(digest_bytes),
            "ownership_reply_bytes": int(ownership_reply_bytes),
            "summary_bytes_this_round": summary_bytes + bootstrap_bytes,
            "bootstrap_metadata_bytes_charged": bootstrap_bytes,
            "metadata_bytes": int(digest_bytes + summary_bytes + bootstrap_bytes + ownership_reply_bytes),
            "graph_training_seconds_this_round": (self.last_graph_metadata["graph_training_seconds"] if refresh else 0) + bootstrap_seconds,
            "policy_seconds": time.perf_counter() - started + bootstrap_seconds,
            "owner_rule": "lexicographic_minimum_client_id_then_local_row_index",
            "objective": "unique_exact_records_within_discovered_candidate_component",
            "multiplicity_preserved": False,
            "privacy_guarantee": "none; summaries and deterministic hashes may reveal membership/linkability",
        }
        self.last_retained_multiplicities = retained_multiplicities
        return masks, metadata

