#!/usr/bin/env python3
"""No-GT, seam-aware displacement consensus for PGD outputs.

PGD's overlap patches predict several locally consistent displacement fields,
but hard patch selection can leave a discontinuity at a seed Voronoi boundary.
This postprocess never averages 3-D *positions*.  Instead, it filters the
one-to-one displacement ``denoised - noisy`` on the noisy-cloud KNN graph.
Spatial, normal and displacement-residual weights keep the consensus within a
single surface sheet and reject neighbours belonging to a different patch
mode.  It is therefore complementary to PCA (normal projection) and tangent
repulsion (minimum-spacing correction).
"""
import argparse
import os
import shutil
import sys
from pathlib import Path

import jittor as jt
import numpy as np
from scipy.spatial import cKDTree


def normals_and_graph(points, k):
    distances, indices = cKDTree(points).query(points, k=min(int(k) + 1, len(points)))
    indices = np.asarray(indices[:, 1:], dtype=np.int64)
    distances = np.asarray(distances[:, 1:], dtype=np.float32)
    neighbours = points[indices]
    centered = neighbours - neighbours.mean(axis=1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", centered, centered) / float(neighbours.shape[1])
    _, vectors = np.linalg.eigh(cov)
    normals = vectors[:, :, 0]
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-8)
    # local spacing makes the filter scale independent across ShapeNet classes
    spacing = np.median(distances, axis=1).astype(np.float32)
    return indices, distances, normals.astype(np.float32), spacing


def consensus(pred, noisy, k, spatial_scale, disp_scale, mix, normal_power):
    indices, distances, normals, spacing = normals_and_graph(noisy, k)
    p = jt.array(pred.astype(np.float32))
    x = jt.array(noisy.astype(np.float32))
    ind = jt.array(indices).int64()
    n = jt.array(normals)
    local = jt.array(np.maximum(spacing, 1e-7)).unsqueeze(1)
    disp = p - x
    nbr_disp = disp[ind]
    # Do not connect two opposing / unrelated local sheets near thin parts.
    normal_dot = jt.abs(jt.sum(n[ind] * n.unsqueeze(1), dim=-1))
    normal_w = normal_dot ** float(normal_power)
    dist = jt.array(distances)
    spatial_w = jt.exp(-0.5 * (dist / (float(spatial_scale) * local)) ** 2)
    residual = jt.sqrt(jt.maximum(jt.sum((nbr_disp - disp.unsqueeze(1)) ** 2, dim=-1), 1e-14))
    residual_w = jt.exp(-0.5 * (residual / (float(disp_scale) * local)) ** 2)
    weights = spatial_w * normal_w * residual_w
    # The self term makes the operation conservative where all neighbours are
    # rejected, and is intentionally much stronger than a plain KNN blur.
    self_weight = jt.ones((len(noisy), 1), dtype=jt.float32)
    filtered = (jt.sum(nbr_disp * weights.unsqueeze(-1), dim=1) + self_weight * disp) / \
        (jt.sum(weights, dim=1, keepdims=True) + self_weight)
    out = x + (1.0 - float(mix)) * disp + float(mix) * filtered
    return out.numpy().astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_root", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--k", type=int, default=12)
    parser.add_argument("--spatial_scale", type=float, default=2.5)
    parser.add_argument("--disp_scale", type=float, default=1.5,
                        help="residual bandwidth in local noisy-point spacings")
    parser.add_argument("--mix", type=float, required=True, help="0 leaves predictions unchanged")
    parser.add_argument("--normal_power", type=float, default=2.0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    args = parser.parse_args()
    if not 0.0 <= args.mix <= 1.0:
        raise ValueError("mix must be in [0, 1]")
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("invalid shard specification")
    jt.flags.use_cuda = 1
    jt.flags.use_parallel_op_compiler = 0
    source, target = Path(args.input_root), Path(args.output_root)
    processed = 0
    for pred_path in sorted((source / "pred").rglob("denoised.npy"))[args.shard_index::args.num_shards]:
        rel = pred_path.relative_to(source / "pred")
        pred = np.load(pred_path).astype(np.float32)
        noisy = np.load(source / "noisy" / rel.parent / "noisy.npy").astype(np.float32)
        result = consensus(pred, noisy, args.k, args.spatial_scale, args.disp_scale, args.mix, args.normal_power)
        dst = target / "pred" / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        np.save(dst, result)
        for subdir, filename in (("gt", "clean.npy"), ("noisy", "noisy.npy")):
            src, copied = source / subdir / rel.parent / filename, target / subdir / rel.parent / filename
            copied.parent.mkdir(parents=True, exist_ok=True)
            if not copied.exists():
                shutil.copy2(src, copied)
        processed += 1
    print("processed={} shard={}/{}".format(processed, args.shard_index, args.num_shards))


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    sys.stderr.flush()
    # Matches the other Jittor postprocessors: CUDA teardown has intermittently
    # double-freed after work had safely serialized.
    os._exit(0)
