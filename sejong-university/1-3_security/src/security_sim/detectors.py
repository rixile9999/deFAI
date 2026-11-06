import numpy as np


def robust_z(values):
    v = np.asarray(values, dtype=float)
    med = np.median(v)
    mad = np.median(np.abs(v - med)) * 1.4826
    if mad < 1e-12:
        mad = max(np.std(v), 1e-12)
    return (v - med) / mad


def cosine_to(u, ref):
    return u @ ref / (np.linalg.norm(u, axis=1) * np.linalg.norm(ref) + 1e-12)


def features(U):
    norms = np.log(np.linalg.norm(U, axis=1) + 1e-12)
    ref = np.median(U, axis=0)
    cos = cosine_to(U, ref)
    spread = np.log(U.std(1) + 1e-12)
    peak = np.log(np.abs(U).max(1) + 1e-12)
    return np.column_stack([norms - np.median(norms), cos, spread - np.median(spread), peak - np.median(peak)])


def statistical(U, threshold=3.0):
    norms = np.linalg.norm(U, axis=1)
    zn = robust_z(np.log(norms + 1e-12))
    cos = cosine_to(U, np.median(U, axis=0))
    zc = robust_z(cos)
    out = {}
    for k in range(len(U)):
        reasons = []
        if abs(zn[k]) > threshold:
            reasons.append(f"update-norm z={zn[k]:.1f}")
        if zc[k] < -threshold:
            reasons.append(f"direction z={zc[k]:.1f}")
        if reasons:
            out[k] = reasons
    return out


def data_distribution(stats, threshold=3.0):
    out = {}
    if len(stats) < 3:
        return out
    arr = np.asarray(stats, dtype=float)
    for col, name in enumerate(("data-mean", "data-std", "label-entropy")):
        z = robust_z(arr[:, col])
        for k in np.nonzero(np.abs(z) > threshold)[0]:
            out.setdefault(int(k), []).append(f"{name} z={z[k]:.1f}")
    return out


class OCSVMDetector:
    def __init__(self, nu=0.1, gamma="scale", min_history=8, max_history=200):
        self.nu, self.gamma = nu, gamma
        self.min_history, self.max_history = min_history, max_history
        self.history = []

    def __call__(self, U):
        from sklearn.preprocessing import RobustScaler
        from sklearn.svm import OneClassSVM

        X = features(U)
        z = np.abs(np.column_stack([robust_z(X[:, j]) for j in range(X.shape[1])]))
        inliers = np.nonzero(z.max(1) < 3.0)[0]
        train = X[inliers] if len(inliers) >= 3 else X
        if len(self.history) >= self.min_history:
            train = np.vstack([np.asarray(self.history), train])
        scaler = RobustScaler().fit(train)
        model = OneClassSVM(nu=self.nu, gamma=self.gamma).fit(scaler.transform(train))
        score = model.decision_function(scaler.transform(X))
        ref = model.decision_function(scaler.transform(train))
        threshold = min(0.0, ref.min() - (np.median(ref) - ref.min()))
        flagged = np.nonzero(score < threshold - 1e-9)[0]
        return {int(k): [f"OCSVM outlier (score={score[k]:.2f})"] for k in flagged}

    def accept(self, U, keep):
        if len(keep):
            self.history.extend(features(U)[keep].tolist())
            self.history = self.history[-self.max_history:]


def kmeans_collusion(U, dup_threshold=0.95, seed=0):
    n = len(U)
    if n < 3:
        return {}
    from sklearn.cluster import KMeans

    Un = U / (np.linalg.norm(U, axis=1, keepdims=True) + 1e-12)
    best = {}
    for k in range(2, min(4, n - 1) + 1):
        labels = KMeans(n_clusters=k, n_init=5, random_state=seed).fit_predict(Un)
        for c in range(k):
            idx = np.nonzero(labels == c)[0]
            if len(idx) < 2 or len(idx) > n - 1:
                continue
            S = Un[idx] @ Un[idx].T
            tight = S[np.triu_indices(len(idx), 1)].mean()
            rest = np.setdiff1d(np.arange(n), idx)
            outside = (Un[idx] @ Un[rest].T).mean() if len(rest) else 0.0
            if tight > dup_threshold and tight - outside > 0.2:
                for i in idx:
                    best[int(i)] = [f"K-means collusion cluster (k={k}, cos={tight:.3f})"]
    return best


def insider_monitor(U, dup_threshold=0.98, cos_threshold=0.0, ratio_threshold=3.0):
    n = len(U)
    norms = np.linalg.norm(U, axis=1)
    Un = U / (norms[:, None] + 1e-12)
    S = Un @ Un.T
    np.fill_diagonal(S, 0.0)
    dup = (S > dup_threshold).any(1)
    base = [k for k in range(n) if not dup[k]] or list(range(n))
    ref = np.median(U[base], axis=0)
    ref_norm = float(np.median(norms[base]))
    cos = cosine_to(U, ref)
    out = {}
    for k in range(n):
        reasons = []
        if dup[k]:
            reasons.append("near-duplicate update (coordinated insiders)")
        if cos[k] < cos_threshold:
            reasons.append(f"direction against reference (cos={cos[k]:.2f})")
        if norms[k] > ratio_threshold * ref_norm:
            reasons.append(f"magnitude {norms[k] / (ref_norm + 1e-12):.1f}x reference")
        if reasons:
            out[k] = reasons
    return out


def precision_recall(flagged, malicious, population):
    flagged, malicious = set(flagged), set(malicious) & set(population)
    tp = len(flagged & malicious)
    fp = len(flagged - malicious)
    fn = len(malicious - flagged)
    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None}
