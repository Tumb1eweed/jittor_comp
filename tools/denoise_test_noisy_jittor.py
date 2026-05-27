#!/usr/bin/env python
import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("nvcc_path", "")

import jittor as jt
import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.pgd import PGDModel
from models.utils import farthest_point_sampling, knn_points_np
from utils.transforms import NormalizeUnitSphere


def iter_noisy_files(input_root):
    root = Path(input_root)
    yield from sorted(root.glob("shapenet/*/*/noisy.npy"))


def relative_output_path(input_root, noisy_path):
    rel = Path(noisy_path).relative_to(input_root)
    return Path(rel).parent / "denoised.npy"


def denoise_one(model, noisy, args):
    noisy_norm, center, scale = NormalizeUnitSphere.normalize_np(noisy.astype(np.float32))
    pred = noisy_norm.astype(np.float32)
    for _ in range(args.niters):
        pred = patch_based_denoise_lowmem(model, pred, args)
    denoised = pred * scale + center
    return denoised.astype(np.float32)


def patch_based_denoise_lowmem(model, pcl_np, args):
    n = pcl_np.shape[0]
    patch_size = args.patch_size
    num_patches = int(args.seed_k * n / patch_size)
    seed_np, _ = farthest_point_sampling(pcl_np[None, :, :], num_patches)
    seed_np = seed_np[0]
    dists, idx, patches_np = knn_points_np(seed_np[None, :, :], pcl_np[None, :, :], k=patch_size, return_nn=True)
    patch_dists = dists[0]
    point_idxs = idx[0].astype(np.int64)
    patches_np = patches_np[0]
    patches_centered = patches_np - seed_np[:, None, :]

    denom = np.maximum(patch_dists[:, -1:], 1e-12)
    patch_dists = patch_dists / denom
    all_dists = np.full((num_patches, n), np.inf, dtype=np.float32)
    for pi in range(num_patches):
        all_dists[pi, point_idxs[pi]] = patch_dists[pi]
    best_patch = np.argmin(all_dists, axis=0).astype(np.int64)

    patch_step = int(n / (args.seed_k_alpha * patch_size))
    patch_step = max(patch_step, int(args.patch_batch_size))
    if patch_step <= 0:
        raise ValueError("seed_k_alpha needs to be decreased to increase patch_step")

    patches_denoised = []
    with jt.no_grad():
        for i in range(0, num_patches, patch_step):
            curr = jt.array(patches_centered[i:i + patch_step].astype(np.float32))
            den = model.denoise_langevin_dynamics(curr).numpy()
            patches_denoised.append(den)
            jt.clean()
    patches_denoised = np.concatenate(patches_denoised, axis=0) + seed_np[:, None, :]

    local_for_point = np.zeros((num_patches, n), dtype=np.int32)
    local_ids = np.arange(patch_size, dtype=np.int32)
    for pi in range(num_patches):
        local_for_point[pi, point_idxs[pi]] = local_ids
    selected_local = local_for_point[best_patch, np.arange(n)]
    out = patches_denoised[best_patch, selected_local]
    if out.shape != pcl_np.shape:
        raise RuntimeError(f"lowmem output shape {out.shape} != input shape {pcl_np.shape}")
    return out.astype(np.float32)


def verify_output(input_root, output_root, noisy_paths):
    missing = []
    mismatches = []
    nonfinite = []
    for noisy_path in noisy_paths:
        out_path = Path(output_root) / relative_output_path(input_root, noisy_path)
        if not out_path.exists():
            missing.append(str(out_path))
            continue
        noisy = np.load(noisy_path, mmap_mode="r")
        pred = np.load(out_path, mmap_mode="r")
        if tuple(pred.shape) != tuple(noisy.shape):
            mismatches.append({
                "input": str(noisy_path),
                "output": str(out_path),
                "input_shape": list(noisy.shape),
                "output_shape": list(pred.shape),
            })
        if pred.ndim != 2 or pred.shape[-1] != 3 or not np.isfinite(pred).all():
            nonfinite.append(str(out_path))
    return missing, mismatches, nonfinite


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", default="/home/chenrui/workspace/dataset_test_noisy")
    parser.add_argument("--output-root", default="/home/chenrui/workspace/test_noisy")
    parser.add_argument("--weights", default=str(ROOT / "weights/pgd-epoch19-val_loss0.00024044.npz"))
    parser.add_argument("--patch-size", type=int, default=1200)
    parser.add_argument("--seed-k", type=int, default=6)
    parser.add_argument("--seed-k-alpha", type=float, default=5)
    parser.add_argument("--patch-batch-size", type=int, default=8)
    parser.add_argument("--niters", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--use-cuda", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    args = parser.parse_args()

    np.random.seed(args.seed)
    jt.set_global_seed(args.seed)
    if args.use_cuda:
        jt.flags.use_cuda = 1

    input_root = Path(args.input_root).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    noisy_paths = list(iter_noisy_files(input_root))
    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if not (0 <= args.shard_index < args.num_shards):
        raise ValueError("--shard-index must satisfy 0 <= shard-index < num-shards")
    noisy_paths = noisy_paths[args.shard_index::args.num_shards]
    if args.max_samples:
        noisy_paths = noisy_paths[:args.max_samples]
    if not noisy_paths:
        raise FileNotFoundError(f"No noisy.npy files under {input_root}")

    model = PGDModel.load_from_npz(args.weights)
    metadata = vars(args).copy()
    metadata["input_root"] = str(input_root)
    metadata["output_root"] = str(output_root)
    metadata["num_files"] = len(noisy_paths)
    metadata["framework"] = "jittor"
    metadata["load_report"] = getattr(model, "_load_report", {})
    with open(output_root / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    for noisy_path in tqdm(noisy_paths, desc="Denoise test noisy"):
        out_path = output_root / relative_output_path(input_root, noisy_path)
        if args.skip_existing and out_path.exists():
            continue
        noisy = np.load(noisy_path).astype(np.float32)
        if noisy.ndim != 2 or noisy.shape[1] != 3:
            raise ValueError(f"{noisy_path}: expected (N, 3), got {noisy.shape}")
        denoised = denoise_one(model, noisy, args)
        if denoised.shape != noisy.shape:
            raise RuntimeError(f"{noisy_path}: output shape {denoised.shape} != input shape {noisy.shape}")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, denoised)

    missing, mismatches, nonfinite = verify_output(input_root, output_root, noisy_paths)
    report = {
        "num_inputs": len(noisy_paths),
        "missing": missing,
        "shape_mismatches": mismatches,
        "nonfinite_or_invalid": nonfinite,
    }
    with open(output_root / "count_check.json", "w") as f:
        json.dump(report, f, indent=2)
    if missing or mismatches or nonfinite:
        raise RuntimeError(f"Output verification failed; see {output_root / 'count_check.json'}")


if __name__ == "__main__":
    main()
