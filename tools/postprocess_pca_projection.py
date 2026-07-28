#!/usr/bin/env python3
"""Jittor local-PCA surface projection for existing denoised point clouds."""
import argparse
import shutil
from pathlib import Path

import jittor as jt
import numpy as np
from scipy.spatial import cKDTree


def plane_frames(points, k):
    _, indices = cKDTree(points).query(points, k=min(int(k), len(points)))
    neighbours = points[np.asarray(indices, dtype=np.int64)]
    centers = neighbours.mean(axis=1)
    centered = neighbours - centers[:, None, :]
    covariance = np.einsum("nki,nkj->nij", centered, centered) / float(neighbours.shape[1])
    _, vectors = np.linalg.eigh(covariance)
    normals = vectors[:, :, 0]
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-8)
    return centers.astype(np.float32), normals.astype(np.float32)


def project(points, k, strength):
    centers, normals = plane_frames(points, k)
    p = jt.array(points.astype(np.float32))
    c = jt.array(centers)
    n = jt.array(normals)
    normal_offset = jt.sum((p - c) * n, dim=-1, keepdims=True) * n
    return (p - float(strength) * normal_offset).numpy().astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_root", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--strength", type=float, required=True)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    args = parser.parse_args()
    jt.flags.use_cuda = 1
    # Each invocation only handles a small, independent shard.  Jittor's
    # default parallel op compiler may recursively spawn compiler processes
    # when several shards start together, so keep compilation local here.
    jt.flags.use_parallel_op_compiler = 0
    source, target = Path(args.input_root), Path(args.output_root)
    target.mkdir(parents=True, exist_ok=True)
    count = 0
    pred_paths = sorted((source / "pred").rglob("denoised.npy"))
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("invalid shard specification")
    for pred_path in pred_paths[args.shard_index::args.num_shards]:
        relative = pred_path.relative_to(source / "pred")
        pred = np.load(pred_path).astype(np.float32)
        output = project(pred, args.k, args.strength)
        dst = target / "pred" / relative
        dst.parent.mkdir(parents=True, exist_ok=True)
        np.save(dst, output)
        for subdir, filename in (("gt", "clean.npy"), ("noisy", "noisy.npy")):
            src = source / subdir / relative.parent / filename
            copied = target / subdir / relative.parent / filename
            copied.parent.mkdir(parents=True, exist_ok=True)
            if not copied.exists():
                shutil.copy2(src, copied)
        count += 1
    print("processed={} shard={}/{}".format(count, args.shard_index, args.num_shards))


if __name__ == "__main__":
    # See the companion tangent-repulsion postprocessor: avoid an unrelated
    # Jittor CUDA shutdown double-free after successful output serialization.
    main()
    import os
    import sys
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
