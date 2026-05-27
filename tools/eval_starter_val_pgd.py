#!/usr/bin/env python
import argparse
import json
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("nvcc_path", "")
import numpy as np
import subprocess
from scipy.spatial import cKDTree
from tqdm import tqdm
import jittor as jt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.pgd import PGDModel
from utils.transforms import NormalizeUnitSphere


def read_split(path):
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def xyz_name(entry):
    return entry.replace("/", "__") + ".xyz"


def ensure_point_count(pred, noisy, entry):
    target_n = noisy.shape[0]
    if pred.shape == noisy.shape:
        return pred, None
    if pred.ndim != 2 or pred.shape[1] != 3:
        raise ValueError(f"{entry}: invalid predicted shape {pred.shape}")
    if pred.shape[0] == target_n:
        return pred, None
    _, idx = cKDTree(pred).query(noisy, k=1)
    fixed = pred[idx]
    return fixed, {"entry": entry, "pred_shape": list(pred.shape), "input_shape": list(noisy.shape), "fixed_shape": list(fixed.shape)}


def save_sample(out_root, entry, clean, noisy, pred):
    out_root = Path(out_root)
    pred_dir = out_root / "pred" / entry
    gt_dir = out_root / "gt" / entry
    noisy_dir = out_root / "noisy" / entry
    for path in (pred_dir, gt_dir, noisy_dir):
        path.mkdir(parents=True, exist_ok=True)
    np.save(pred_dir / "denoised.npy", pred.astype(np.float32))
    np.save(gt_dir / "clean.npy", clean.astype(np.float32))
    np.save(noisy_dir / "noisy.npy", noisy.astype(np.float32))


def verify_saved(out_root, entries):
    mismatches = []
    nonfinite = []
    for entry in entries:
        pred_path = Path(out_root) / "pred" / entry / "denoised.npy"
        noisy_path = Path(out_root) / "noisy" / entry / "noisy.npy"
        if not pred_path.exists() or not noisy_path.exists():
            continue
        pred = np.load(pred_path, mmap_mode="r")
        noisy = np.load(noisy_path, mmap_mode="r")
        if pred.shape != noisy.shape:
            mismatches.append({"entry": entry, "pred_shape": list(pred.shape), "noisy_shape": list(noisy.shape)})
        if not np.isfinite(pred).all():
            nonfinite.append({"entry": entry, "nan": int(np.isnan(pred).sum()), "inf": int(np.isinf(pred).sum())})
    return mismatches, nonfinite


def run_evaluate(args):
    cmd = [
        sys.executable,
        str(Path(args.starter_root) / "evaluate.py"),
        "--pred_dir", str(Path(args.output_root) / "pred"),
        "--gt_dir", str(Path(args.output_root) / "gt"),
        "--noisy_dir", str(Path(args.output_root) / "noisy"),
        "--mesh_dir", str(args.mesh_root),
        "--pred_filename", "denoised.npy",
        "--gt_filename", "clean.npy",
        "--noisy_filename", "noisy.npy",
        "--workers", str(args.workers),
        "--verbose",
    ]
    log_path = Path(args.output_root) / "evaluate.log"
    with open(log_path, "w") as f:
        proc = subprocess.run(cmd, cwd=args.starter_root, stdout=f, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        raise RuntimeError(f"starter evaluate failed with code {proc.returncode}; see {log_path}")


def denoise(model, noisy_norm, args):
    pred = jt.array(noisy_norm.astype(np.float32))
    for _ in range(args.niters):
        pred = model.patch_based_denoise(
            pred,
            patch_size=args.patch_size,
            seed_k=args.seed_k,
            seed_k_alpha=args.seed_k_alpha,
            patch_batch_size=args.patch_batch_size,
        )
    return pred.numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default=str(ROOT / "weights/pgd-epoch19-val_loss0.00024044.npz"))
    parser.add_argument("--starter-root", default="/home/chenrui/workspace/starter_code")
    parser.add_argument("--mesh-root", default="/home/chenrui/workspace/dataset_train")
    parser.add_argument("--val-list", default="/home/chenrui/workspace/starter_code/datalist/validate.txt")
    parser.add_argument("--xyz-dir", default="/home/chenrui/workspace/StraightPCF/data/StarterShapeNet/pointclouds/test/10000_poisson")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--noise-std", type=float, default=0.015)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--patch-size", type=int, default=1200)
    parser.add_argument("--seed-k", type=int, default=6)
    parser.add_argument("--seed-k-alpha", type=float, default=5)
    parser.add_argument("--patch-batch-size", type=int, default=8)
    parser.add_argument("--niters", type=int, default=2)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--run-evaluate", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--use-cuda", action="store_true")
    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)
    if args.use_cuda:
        jt.flags.use_cuda = 1
    out_root = Path(args.output_root).resolve()
    args.output_root = str(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    model = PGDModel.load_from_npz(args.weights)
    entries = read_split(args.val_list)
    if args.max_samples is not None:
        entries = entries[:args.max_samples]
    meta = vars(args).copy()
    meta["num_entries"] = len(entries)
    meta["framework"] = "jittor"
    meta["load_report"] = getattr(model, "_load_report", {})
    with open(out_root / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    mismatches = []
    missing = []
    for i, entry in enumerate(tqdm(entries, desc="PGD Jittor starter val predict")):
        pred_path = out_root / "pred" / entry / "denoised.npy"
        if args.skip_existing and pred_path.exists():
            continue
        clean_path = Path(args.xyz_dir) / xyz_name(entry)
        if not clean_path.exists():
            missing.append({"entry": entry, "path": str(clean_path)})
            continue
        clean_np = np.loadtxt(clean_path, dtype=np.float32)
        clean_norm, center, scale = NormalizeUnitSphere.normalize_np(clean_np)
        rng = np.random.default_rng(args.seed + i)
        noisy_norm = clean_norm + rng.normal(size=clean_norm.shape).astype(np.float32) * args.noise_std
        pred_norm = denoise(model, noisy_norm, args)
        noisy = noisy_norm * scale + center
        pred = pred_norm * scale + center
        pred, mismatch = ensure_point_count(pred, noisy, entry)
        if mismatch is not None:
            mismatches.append(mismatch)
        save_sample(out_root, entry, clean_np, noisy, pred)

    saved_mismatches, saved_nonfinite = verify_saved(out_root, entries)
    with open(out_root / "count_check.json", "w") as f:
        json.dump({"mismatches_fixed": mismatches, "saved_count_mismatches": saved_mismatches, "saved_nonfinite": saved_nonfinite, "missing": missing}, f, indent=2)
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} val xyz files; see {out_root / 'count_check.json'}")
    if saved_mismatches:
        raise RuntimeError(f"Saved point-count mismatch in {len(saved_mismatches)} samples")
    if saved_nonfinite:
        raise RuntimeError(f"Saved non-finite predictions in {len(saved_nonfinite)} samples")
    if args.run_evaluate:
        run_evaluate(args)


if __name__ == "__main__":
    main()
