#!/usr/bin/env python3
"""GT-free local-surface postprocessing for fixed-size point clouds.

The local surface is estimated from the input prediction with PCA/MLS-style
neighborhoods.  This script deliberately never reads clean/GT points while
computing the output; they are only copied for the downstream official scorer.
"""
import argparse
import shutil
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


def local_geometry(x, k):
    tree = cKDTree(x)
    d, idx = tree.query(x, k=min(int(k) + 1, len(x)))
    idx = np.asarray(idx[:, 1:], dtype=np.int64)
    d = np.asarray(d[:, 1:], dtype=np.float32)
    nb = x[idx]
    center = nb.mean(axis=1)
    z = nb - center[:, None, :]
    cov = np.einsum("nki,nkj->nij", z, z) / max(1, nb.shape[1])
    vals, vecs = np.linalg.eigh(cov)
    normal = vecs[:, :, 0].astype(np.float32)
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-8)
    tangent = nb - x[:, None, :]
    return idx, d, center.astype(np.float32), normal, tangent.astype(np.float32)


def project(x, center, normal, strength):
    off = np.sum((x - center) * normal, axis=1, keepdims=True) * normal
    return x - float(strength) * off


def bilateral(x, idx, d, center, normal, alpha, sigma_scale, tangent_only):
    # Robust local bandwidth: median neighbor radius per point.  The range
    # term prevents averaging across sharp local changes in the point cloud.
    h = np.maximum(np.median(d, axis=1, keepdims=True) * float(sigma_scale), 1e-6)
    w = np.exp(-0.5 * (d / h) ** 2).astype(np.float32)
    nb = x[idx]
    mean = np.sum(w[:, :, None] * nb, axis=1) / np.maximum(np.sum(w, axis=1, keepdims=True), 1e-8)
    delta = mean - x
    if tangent_only:
        delta = delta - np.sum(delta * normal, axis=1, keepdims=True) * normal
    return x + float(alpha) * delta


def wlop_step(x, idx, d, center, normal, alpha, repulsion):
    # A conservative WLOP-like MLS attraction plus tangent repulsion.  Both
    # terms are local and are clipped by the local spacing to avoid clusters
    # jumping across thin structures.
    h = np.maximum(np.median(d, axis=1, keepdims=True), 1e-6)
    w = np.exp(-(d / h) ** 2).astype(np.float32)
    nb = x[idx]
    attraction = np.sum(w[:, :, None] * (nb - x[:, None, :]), axis=1) / np.maximum(np.sum(w, axis=1, keepdims=True), 1e-8)
    rel = x[:, None, :] - nb
    unit = rel / np.maximum(d[:, :, None], 1e-6)
    repel = np.sum(unit * np.exp(-0.5 * (d / h) ** 2)[:, :, None], axis=1)
    repel = repel / max(1, idx.shape[1])
    attraction = attraction - np.sum(attraction * normal, axis=1, keepdims=True) * normal
    repel = repel - np.sum(repel * normal, axis=1, keepdims=True) * normal
    radius = np.median(d, axis=1, keepdims=True)
    delta = float(alpha) * attraction + float(repulsion) * radius * repel
    limit = 0.35 * radius
    norm = np.linalg.norm(delta, axis=1, keepdims=True)
    delta *= np.minimum(1.0, limit / np.maximum(norm, 1e-8))
    return x + delta


def upsample_downsample(x, idx, d, normal, jitter_scale, keep_ratio):
    # Lightweight fixed-count resampling: create one tangent jittered copy,
    # then retain one point per voxel first and fill to N by deterministic
    # farthest-in-voxel representatives. This avoids O(N^2) FPS.
    n = len(x)
    radius = np.maximum(np.median(d, axis=1, keepdims=True), 1e-6)
    tangent_noise = np.random.default_rng(20260727).normal(size=x.shape).astype(np.float32)
    tangent_noise -= np.sum(tangent_noise * normal, axis=1, keepdims=True) * normal
    tangent_noise /= np.maximum(np.linalg.norm(tangent_noise, axis=1, keepdims=True), 1e-8)
    jitter = x + tangent_noise * radius * float(jitter_scale)
    allp = np.concatenate([x, jitter], axis=0)
    # voxel size chosen from the median spacing; select one representative per
    # occupied voxel, then deterministic evenly spaced points if overfull.
    voxel = max(float(np.median(radius)) * float(keep_ratio), 1e-6)
    keys = np.floor(allp / voxel).astype(np.int64)
    _, first = np.unique(keys, axis=0, return_index=True)
    selected = np.sort(first)
    if len(selected) > n:
        selected = selected[np.linspace(0, len(selected) - 1, n).astype(np.int64)]
    elif len(selected) < n:
        remaining = np.setdiff1d(np.arange(len(allp)), selected, assume_unique=False)
        need = n - len(selected)
        selected = np.concatenate([selected, remaining[:need]])
    return allp[selected[:n]]


def process(x, args, geometry=None):
    # Geometry may be estimated from the noisy input.  This is still test-time
    # only and often gives a less collapsed neighborhood graph than the raw
    # prediction when the network output contains dense clusters.
    geom = x if geometry is None else geometry
    idx, d, center, normal, _ = local_geometry(geom, args.k)
    y = x.copy()
    if args.mode in ("project", "project_bilateral", "project_wlop"):
        y = project(y, center, normal, args.project_strength)
    if args.mode == "bilateral" or args.mode == "project_bilateral":
        y = bilateral(y, idx, d, center, normal, args.alpha, args.sigma_scale, args.tangent_only)
    elif args.mode == "wlop" or args.mode == "project_wlop":
        for _ in range(args.iters):
            y = wlop_step(y, idx, d, center, normal, args.alpha, args.repulsion)
    elif args.mode == "upsample":
        y = upsample_downsample(y, idx, d, normal, args.jitter_scale, args.keep_ratio)
    return y.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_root", required=True)
    ap.add_argument("--output_root", required=True)
    ap.add_argument("--mode", choices=["project", "bilateral", "project_bilateral", "wlop", "project_wlop", "upsample"], required=True)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--project_strength", type=float, default=0.10)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--sigma_scale", type=float, default=1.0)
    ap.add_argument("--tangent_only", action="store_true")
    ap.add_argument("--repulsion", type=float, default=0.05)
    ap.add_argument("--iters", type=int, default=1)
    ap.add_argument("--jitter_scale", type=float, default=0.15)
    ap.add_argument("--keep_ratio", type=float, default=0.8)
    ap.add_argument("--geometry_source", choices=["pred", "noisy"], default="pred")
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--shard_index", type=int, default=0)
    args = ap.parse_args()
    source, target = Path(args.input_root), Path(args.output_root)
    target.mkdir(parents=True, exist_ok=True)
    paths = sorted((source / "pred").rglob("denoised.npy"))
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("invalid shard")
    count = 0
    for p in paths[args.shard_index::args.num_shards]:
        rel = p.relative_to(source / "pred")
        x = np.load(p).astype(np.float32)
        noisy = None
        if args.geometry_source == "noisy":
            noisy = np.load(source / "noisy" / rel.parent / "noisy.npy").astype(np.float32)
        y = process(x, args, noisy)
        dst = target / "pred" / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        np.save(dst, y)
        for sub, fn in (("gt", "clean.npy"), ("noisy", "noisy.npy")):
            src = source / sub / rel.parent / fn
            out = target / sub / rel.parent / fn
            out.parent.mkdir(parents=True, exist_ok=True)
            if not out.exists():
                shutil.copy2(src, out)
        count += 1
    print("processed={} shard={}/{}".format(count, args.shard_index, args.num_shards))


if __name__ == "__main__":
    main()
