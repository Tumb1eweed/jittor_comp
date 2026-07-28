#!/usr/bin/env python3
"""Jittor tangent-only anti-collapse postprocess for denoised point clouds.

The noisy cloud supplies a stable local graph.  For every graph edge that has
collapsed more than a global denoising-scale factor allows, a short tangent
repulsion is applied.  Normal components are explicitly removed, preserving
the predicted surface placement as much as possible.
"""
import argparse
import shutil
from pathlib import Path

import jittor as jt
import numpy as np
from scipy.spatial import cKDTree


def local_frames(points, indices):
    neighbours = points[indices]
    centered = neighbours - neighbours.mean(axis=1, keepdims=True)
    covariance = np.einsum("nki,nkj->nij", centered, centered) / float(indices.shape[1])
    values, vectors = np.linalg.eigh(covariance)
    normals = vectors[:, :, 0]
    normals = normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-8)
    variation = values[:, 0] / np.maximum(values.sum(axis=1), 1e-12)
    consistency = np.abs(
        np.sum(normals[:, None, :] * normals[indices], axis=-1)
    ).mean(axis=1)
    return (
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


def repel(
    pred,
    noisy,
    k,
    margin,
    step,
    iterations,
    normal_source,
    surface_gate="none",
    variation_tau=0.02,
    consistency_floor=0.75,
):
    tree = cKDTree(noisy)
    _, indices = tree.query(noisy, k=min(int(k) + 1, len(noisy)))
    indices = np.asarray(indices[:, 1:], dtype=np.int64)
    normal_points = pred if normal_source == "pred" else noisy
    normals, variation, consistency = local_frames(normal_points, indices)
    confidence = surface_confidence(
        variation,
        consistency,
        surface_gate,
        variation_tau,
        consistency_floor,
    )
    noisy_edge = np.linalg.norm(noisy[indices] - noisy[:, None, :], axis=-1)
    pred_edge = np.linalg.norm(pred[indices] - pred[:, None, :], axis=-1)
    # Preserve the observed global denoising scale while correcting only
    # locally excessive contraction.
    scale = float(np.median(pred_edge) / max(np.median(noisy_edge), 1e-8))
    target = jt.array((scale * noisy_edge).astype(np.float32))
    index = jt.array(indices).int64()
    normal = jt.array(normals)
    gate = jt.array(confidence).unsqueeze(-1)
    out = jt.array(pred.astype(np.float32))
    for _ in range(int(iterations)):
        rel = out[index] - out.unsqueeze(1)
        tangent = rel - jt.sum(rel * normal.unsqueeze(1), dim=-1, keepdims=True) * normal.unsqueeze(1)
        length = jt.sqrt(jt.maximum(jt.sum(tangent * tangent, dim=-1), 1e-12))
        shortfall = jt.maximum(float(margin) * target - length, 0.0)
        force = -jt.sum(tangent / (length.unsqueeze(-1) + 1e-8) * shortfall.unsqueeze(-1), dim=1)
        out = out + float(step) * gate * force
    return out.numpy().astype(np.float32), scale, float(confidence.mean())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_root", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--margin", type=float, default=0.85)
    parser.add_argument("--step", type=float, required=True)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--normal_source", choices=["pred", "noisy"], default="pred",
                        help="surface used to estimate tangent planes; pred avoids noisy normal drift")
    parser.add_argument(
        "--surface_gate",
        choices=["none", "variation", "consistency", "hybrid"],
        default="none",
        help="attenuate tangent transport where the estimated local surface is unreliable",
    )
    parser.add_argument("--variation_tau", type=float, default=0.02)
    parser.add_argument("--consistency_floor", type=float, default=0.75)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    args = parser.parse_args()
    jt.flags.use_cuda = 1
    # Avoid recursive compiler workers when the screen launches shards.
    jt.flags.use_parallel_op_compiler = 0
    source = Path(args.input_root)
    target = Path(args.output_root)
    target.mkdir(parents=True, exist_ok=True)
    scales = []
    confidences = []
    pred_paths = sorted((source / "pred").rglob("denoised.npy"))
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("invalid shard specification")
    for pred_path in pred_paths[args.shard_index::args.num_shards]:
        relative = pred_path.relative_to(source / "pred")
        pred = np.load(pred_path).astype(np.float32)
        noisy_path = source / "noisy" / relative.parent / "noisy.npy"
        noisy = np.load(noisy_path).astype(np.float32)
        corrected, scale, confidence = repel(
            pred,
            noisy,
            args.k,
            args.margin,
            args.step,
            args.iterations,
            args.normal_source,
            args.surface_gate,
            args.variation_tau,
            args.consistency_floor,
        )
        if corrected.shape != pred.shape or corrected.shape != noisy.shape:
            raise RuntimeError(
                "{}: corrected {}, pred {}, noisy {}".format(
                    pred_path, corrected.shape, pred.shape, noisy.shape
                )
            )
        destination = target / "pred" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.save(destination, corrected)
        for subdir, filename in (("gt", "clean.npy"), ("noisy", "noisy.npy")):
            src = source / subdir / relative.parent / filename
            dst = target / subdir / relative.parent / filename
            if src.exists() and not dst.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
        scales.append(scale)
        confidences.append(confidence)
    scale_name = "scales.npy" if args.num_shards == 1 else "scales_shard{:02d}.npy".format(args.shard_index)
    np.save(target / scale_name, np.asarray(scales, dtype=np.float32))
    median = float(np.median(scales)) if scales else float("nan")
    mean_confidence = float(np.mean(confidences)) if confidences else float("nan")
    print("processed={}, shard={}/{}, median_scale={:.6f}, mean_confidence={:.6f}".format(
        len(scales), args.shard_index, args.num_shards, median, mean_confidence
    ))


if __name__ == "__main__":
    # Jittor/CUDA teardown occasionally double-frees after all arrays have
    # already been written.  Exit after explicitly flushing the completed job.
    main()
    import os
    import sys
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
