import numpy as np
from scipy.spatial import cKDTree


def estimate_noise_std_np(points, sample_size=2048, k=24, quantile=0.5, seed=2026, clip=(0.003, 0.030)):
    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    n = pts.shape[0]
    if n < 8:
        return float(clip[0])
    sample_n = min(int(sample_size), n)
    rng = np.random.default_rng(seed)
    if sample_n < n:
        sample_idx = rng.choice(n, size=sample_n, replace=False)
    else:
        sample_idx = np.arange(n)
    nn_k = min(int(k) + 1, n)
    _, idx = cKDTree(pts).query(pts[sample_idx], k=nn_k)
    if idx.ndim == 1:
        idx = idx[:, None]
    idx = idx[:, 1:] if idx.shape[1] > 1 else idx

    # Batch the independent 3x3 covariance/eigenvalue calculations.  This is
    # mathematically identical to the former Python loop but makes calibration
    # over many 50k-point clouds practical.
    neigh = pts[idx]
    centered = neigh - neigh.mean(axis=1, keepdims=True)
    denom = max(1, neigh.shape[1] - 1)
    cov = np.einsum("nki,nkj->nij", centered, centered) / float(denom)
    eig = np.linalg.eigvalsh(cov)
    vals = np.maximum(eig[:, 0], 0.0).astype(np.float32)
    sigma = float(np.sqrt(np.quantile(vals, float(quantile))))
    lo, hi = float(clip[0]), float(clip[1])
    return float(np.clip(sigma, lo, hi))
