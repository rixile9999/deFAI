"""SecureBoost-style binary histogram boosting; local protocol simulation.

Only label owner holds the private key. Feature owners aggregate encrypted
derivatives and retain threshold lookups. Fixed-point sums are approximate;
this is not a claim of bit-identical compatibility with XGBoost or FATE.
"""
from dataclasses import dataclass
import math
import numpy as np
from .paillier import keygen


@dataclass(frozen=True)
class Candidate:
    owner: str
    record: int
    count: int
    gradient: object
    hessian: object


class HistogramParty:
    def __init__(self, name, features, *, bins=16):
        self.name, self.x = name, np.asarray(features, dtype=float)
        if not name or self.x.ndim != 2 or not self.x.size or not np.isfinite(self.x).all():
            raise ValueError('Finite nonempty local feature matrix required')
        if isinstance(bins, bool) or not isinstance(bins, int) or bins < 2:
            raise ValueError('At least two histogram bins required')
        self._lookup = {}
        for feature in range(self.x.shape[1]):
            values = np.unique(self.x[:, feature])
            if len(values) > bins:
                thresholds = np.unique(np.quantile(values, np.linspace(0, 1, bins+1)[1:-1]))
            else:
                thresholds = values[:-1] / 2 + values[1:] / 2
            for threshold in thresholds:
                self._lookup[len(self._lookup)] = (feature, threshold)

    def candidates(self, rows, gradients, hessians, public=None):
        # No labels or private key are available to this endpoint.
        result = []
        for record in self._lookup:
            left = self.route(record, rows)
            gs, hs = [gradients[i] for i in left], [hessians[i] for i in left]
            g, h = (public.sum(gs), public.sum(hs)) if public else (sum(gs), sum(hs))
            result.append(Candidate(self.name, record, len(left), g, h))
        return result

    def route(self, record, rows):
        if record not in self._lookup:
            raise ValueError('Unknown local split record')
        feature, threshold = self._lookup[record]
        rows = np.asarray(rows, dtype=int)
        return rows[self.x[rows, feature] <= threshold]

    def with_features(self, features):
        """A local inference view retaining TRAINING thresholds, not refitted bins."""
        x = np.asarray(features, dtype=float)
        if x.ndim != 2 or x.shape[1] != self.x.shape[1] or not len(x) or not np.isfinite(x).all():
            raise ValueError('Inference columns must match the training owner schema')
        view = object.__new__(type(self))
        view.name, view.x, view._lookup = self.name, x, self._lookup.copy()
        return view


@dataclass
class Node:
    weight: float
    owner: str | None = None
    record: int | None = None
    left: object = None
    right: object = None


class SecureBoostClassifier:
    def __init__(self, *, trees=5, depth=2, lr=.3, l2=1., gamma=0., min_leaf=2,
                 key_bits=2048, insecure_test_key=False, encrypted=True):
        if any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in (trees, depth, min_leaf)):
            raise ValueError('Positive integer tree/depth/leaf counts required')
        if any(not math.isfinite(v) or v < 0 for v in (lr, l2, gamma)) or lr == 0 or l2 == 0:
            raise ValueError('Positive learning rate/L2 and nonnegative gamma required')
        self.trees, self.depth, self.lr, self.l2, self.gamma, self.min_leaf = trees, depth, lr, l2, gamma, min_leaf
        self.key_bits, self.insecure_test_key, self.encrypted = key_bits, insecure_test_key, encrypted
        self.forest = []
        self.audit = []

    def fit(self, parties, labels):
        parties = list(parties)
        y = np.asarray(labels, dtype=float)
        if len(parties) < 2 or len({p.name for p in parties}) != len(parties):
            raise ValueError('At least two uniquely named feature owners required')
        if y.ndim != 1 or not len(y) or not np.isin(y, [0, 1]).all() or any(len(p.x) != len(y) for p in parties):
            raise ValueError('Aligned binary labels required')
        public, private = keygen(self.key_bits, insecure_test_key=self.insecure_test_key) if self.encrypted else (None, None)
        self.forest, self.audit = [], []
        score = np.zeros(len(y))
        for tree in range(self.trees):
            probability = np.exp(-np.logaddexp(0, -score))
            g, h = probability - y, np.maximum(probability * (1-probability), 1e-8)
            gradients = [public.encrypt(v) for v in g] if public else g
            hessians = [public.encrypt(v) for v in h] if public else h

            def grow(rows, level):
                total_g, total_h = g[rows].sum(), h[rows].sum()
                node = Node(float(-total_g / (total_h + self.l2)))
                if level >= self.depth or len(rows) < 2*self.min_leaf:
                    return node
                best_gain, best = 0., None
                for party in parties:
                    for candidate in party.candidates(rows, gradients, hessians, public):
                        if candidate.count < self.min_leaf or len(rows)-candidate.count < self.min_leaf:
                            continue
                        gl = private.decrypt(candidate.gradient) if private else candidate.gradient
                        hl = private.decrypt(candidate.hessian) if private else candidate.hessian
                        gr, hr = total_g-gl, max(0., total_h-hl)
                        gain = .5*(gl*gl/(hl+self.l2) + gr*gr/(hr+self.l2) - total_g*total_g/(total_h+self.l2)) - self.gamma
                        if gain > best_gain + 1e-10:
                            best_gain, best = gain, (party, candidate)
                if best is None:
                    return node
                party, candidate = best
                left = party.route(candidate.record, rows)
                right = rows[~np.isin(rows, left)]
                node.owner, node.record = party.name, candidate.record
                self.audit.append({'tree': tree, 'owner': party.name, 'record': candidate.record,
                                   'gain': float(best_gain), 'left_count': len(left), 'encrypted': self.encrypted})
                node.left, node.right = grow(left, level+1), grow(right, level+1)
                return node

            root = grow(np.arange(len(y)), 0)
            self.forest.append(root)
            score += self.lr * self._tree_predict(root, {p.name: p for p in parties}, len(y))
        return self

    @staticmethod
    def _tree_predict(root, parties, size):
        result = np.zeros(size)

        def visit(node, rows):
            if node.owner is None:
                result[rows] = node.weight
                return
            if node.owner not in parties:
                raise ValueError('All split owners must participate in inference')
            left = parties[node.owner].route(node.record, rows)
            visit(node.left, left)
            visit(node.right, rows[~np.isin(rows, left)])
        visit(root, np.arange(size))
        return result

    def predict_proba(self, parties):
        parties = list(parties)
        if not self.forest or len(parties) < 2 or len({p.name for p in parties}) != len(parties):
            raise ValueError('Fit first and supply every aligned feature owner')
        n = len(parties[0].x)
        if any(len(p.x) != n for p in parties):
            raise ValueError('Inference sample rows must be aligned')
        score = sum(self.lr*self._tree_predict(tree, {p.name: p for p in parties}, n) for tree in self.forest)
        p = np.exp(-np.logaddexp(0, -score))
        return np.column_stack((1-p, p))
