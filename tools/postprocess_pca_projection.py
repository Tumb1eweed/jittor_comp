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
    indices = np.asarray(indices, dtype=np.int64)
    neighbours = points[np.asarray(indices, dtype=np.int64)]
    centers = neighbours.mean(axis=1)
    centered = neighbours - centers[:, None, :]
    covariance = np.einsum("nki,nkj->nij", centered, centered) / float(neighbours.shape[1])
    values, vectors = np.linalg.eigh(covariance)
    normals = vectors[:, :, 0]
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-8)
    variation = values[:, 0] / np.maximum(values.sum(axis=1), 1e-12)
    consistency = np.abs(
        np.sum(normals[:, None, :] * normals[indices], axis=-1)
    ).mean(axis=1)
    return (
        centers.astype(np.float32),
        normals.astype(np.float32),
        variation.astype(np.float32),
        consistency.astype(np.float32),
    )


def surface_confidence(variation, consistency, mode, variation_tau, consistency_floor):
    if mode == "none":
        return np.ones_like(variation, dtype=np.float32)
    variation_gate = np.exp(
        -variation / max(float(variation_tau), 1e-8)
    ).astype(np.float32)
    consistency_gate = np.clip(
        (consistency - float(consistency_floor))
        / max(1.0 - float(consistency_floor), 1e-8),
        0.0,
        1.0,
    ).astype(np.float32)
    if mode == "variation":
        return variation_gate
    if mode == "consistency":
        return consistency_gate
    if mode == "hybrid":
        return variation_gate * consistency_gate
    raise ValueError("unknown surface gate: {}".format(mode))


def project(
    points,
    k,
    strength,
    surface_gate="none",
    variation_tau=0.02,
    consistency_floor=0.75,
):
    centers, normals, variation, consistency = plane_frames(points, k)
    confidence = surface_confidence(
        variation,
        consistency,
        surface_gate,
        variation_tau,
        consistency_floor,
    )
    p = jt.array(points.astype(np.float32))
    c = jt.array(centers)
    n = jt.array(normals)
    gate = jt.array(confidence).unsqueeze(-1)
    normal_offset = jt.sum((p - c) * n, dim=-1, keepdims=True) * n
    output = p - float(strength) * gate * normal_offset
    return output.numpy().astype(np.float32), float(confidence.mean())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_root", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--strength", type=float, required=True)
    parser.add_argument(
        "--surface_gate",
        choices=["none", "variation", "consistency", "hybrid"],
        default="none",
        help="attenuate projection at high-curvature or normal-inconsistent points",
    )
    parser.add_argument("--variation_tau", type=float, default=0.02)
    parser.add_argument("--consistency_floor", type=float, default=0.75)
    parser.add_argument(
        "--include_list",
        default="",
        help="optional list of relative sample keys to process (for fixed holdout screens)",
    )
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
    confidences = []
    pred_paths = sorted((source / "pred").rglob("denoised.npy"))
    if args.include_list:
        include = {
            line.strip().strip("/")
            for line in Path(args.include_list).read_text().splitlines()
            if line.strip()
        }
        pred_paths = [
            path
            for path in pred_paths
            if str(path.parent.relative_to(source / "pred")) in include
        ]
        missing = include - {
            str(path.parent.relative_to(source / "pred")) for path in pred_paths
        }
        if missing:
            raise FileNotFoundError(
                "include_list contains {} samples absent from input_root (first: {})".format(
                    len(missing), sorted(missing)[0]
                )
            )
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("invalid shard specification")
    for pred_path in pred_paths[args.shard_index::args.num_shards]:
        relative = pred_path.relative_to(source / "pred")
        pred = np.load(pred_path).astype(np.float32)
        output, confidence = project(
            pred,
            args.k,
            args.strength,
            args.surface_gate,
            args.variation_tau,
            args.consistency_floor,
        )
        if output.shape != pred.shape:
            raise RuntimeError(
                "{}: projected shape {} != input shape {}".format(
                    pred_path, output.shape, pred.shape
                )
            )
        dst = target / "pred" / relative
        dst.parent.mkdir(parents=True, exist_ok=True)
        np.save(dst, output)
        for subdir, filename in (("gt", "clean.npy"), ("noisy", "noisy.npy")):
            src = source / subdir / relative.parent / filename
            copied = target / subdir / relative.parent / filename
            if src.exists() and not copied.exists():
                copied.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, copied)
        count += 1
        confidences.append(confidence)
    mean_confidence = float(np.mean(confidences)) if confidences else float("nan")
    print(
        "processed={} shard={}/{}, mean_confidence={:.6f}".format(
            count, args.shard_index, args.num_shards, mean_confidence
        )
    )


if __name__ == "__main__":
    # See the companion tangent-repulsion postprocessor: avoid an unrelated
    # Jittor CUDA shutdown double-free after successful output serialization.
    main()
    import os
    import sys
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
