#!/usr/bin/env python
import argparse
import json
import os
import random
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("nvcc_path", "")

import jittor as jt
import numpy as np
from scipy.spatial import cKDTree
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.asdn import ASDNModel
from models.pgd import PGDModel
from tools.train_shapenet_one_epoch import build_category_vocab, extract_synset_id, load_obj_mesh, normalize_unit_sphere, read_split, sample_mesh
from utils.noise_estimate import estimate_noise_std_np


def normalize_with_params(points):
    p_max = points.max(axis=0, keepdims=True)
    p_min = points.min(axis=0, keepdims=True)
    center = (p_max + p_min) * 0.5
    centered = points - center
    scale = np.sqrt((centered * centered).sum(axis=1, keepdims=True)).max(axis=0, keepdims=True)
    if float(scale.reshape(-1)[0]) <= 0:
        scale = np.ones((1, 1), dtype=np.float32)
    return (centered / scale).astype(np.float32), center.astype(np.float32), scale.astype(np.float32)


def load_reference_eval_sample(reference_eval_root, entry):
    root = Path(reference_eval_root)
    clean_path = root / "gt" / entry / "clean.npy"
    noisy_path = root / "noisy" / entry / "noisy.npy"
    if not clean_path.exists() or not noisy_path.exists():
        raise FileNotFoundError("missing reference clean/noisy for {}".format(entry))
    clean = np.load(clean_path).astype(np.float32)
    noisy = np.load(noisy_path).astype(np.float32)
    _, center, scale = normalize_with_params(clean)
    noisy_norm = ((noisy - center) / scale).astype(np.float32)
    return clean, noisy, noisy_norm, center.astype(np.float32), scale.astype(np.float32)


def ensure_point_count(pred, noisy):
    if pred.shape == noisy.shape:
        return pred, None
    _, idx = cKDTree(pred).query(noisy, k=1)
    return pred[idx], {"pred_shape": list(pred.shape), "noisy_shape": list(noisy.shape)}


def directional_cd_terms(pred, clean):
    """Competition-normalized directed squared-CD terms.

    The starter scorer reports only their sum.  Keeping both directions makes
    density/coverage failures diagnosable: pred->GT measures off-surface or
    duplicated predictions, while GT->pred measures missing coverage.
    """
    clean_norm, center, scale = normalize_with_params(clean)
    pred_norm = ((pred - center) / scale).astype(np.float32)
    pred_to_gt = float(np.mean(cKDTree(clean_norm).query(pred_norm, k=1)[0] ** 2))
    gt_to_pred = float(np.mean(cKDTree(pred_norm).query(clean_norm, k=1)[0] ** 2))
    return pred_to_gt, gt_to_pred


def save_sample(out_root, entry, clean, noisy, pred):
    for subdir, filename, arr in (
        ("gt", "clean.npy", clean),
        ("noisy", "noisy.npy", noisy),
        ("pred", "denoised.npy", pred),
    ):
        path = Path(out_root) / subdir / entry
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / filename, arr.astype(np.float32))


def rotation_matrices(num_rotations):
    mats = [np.eye(3, dtype=np.float32)]
    if num_rotations >= 4:
        for angle in (0.5 * np.pi, np.pi, 1.5 * np.pi):
            c = np.cos(angle)
            s = np.sin(angle)
            mats.append(np.asarray([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32))
    if num_rotations >= 8:
        mats.extend([
            np.asarray([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=np.float32),
            np.asarray([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]], dtype=np.float32),
            np.asarray([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]], dtype=np.float32),
            np.asarray([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]], dtype=np.float32),
        ])
    return mats[:max(1, int(num_rotations))]


def denoise(models, noisy_norm, args, category_id=0, noise_std=None):
    pred_np = noisy_norm.astype(np.float32)
    rotations = rotation_matrices(args.tta_rotations)
    with jt.no_grad():
        for _ in range(args.niters):
            disp_sum = np.zeros_like(pred_np, dtype=np.float32)
            count = 0
            for model in models:
                for rot in rotations:
                    rot_pred_np = pred_np @ rot
                    rot_pred = jt.array(rot_pred_np.astype(np.float32))
                    if args.model == "asdn":
                        den = model.patch_based_denoise(
                            rot_pred,
                            patch_size=args.patch_size,
                            seed_k=args.seed_k,
                            seed_k_alpha=args.seed_k_alpha,
                            patch_batch_size=args.patch_batch_size,
                            noise_std=noise_std,
                            category_id=category_id,
                            fusion=args.patch_fusion,
                        )
                    else:
                        den = model.patch_based_denoise(
                            rot_pred,
                            patch_size=args.patch_size,
                            context_patch_size=(
                                args.context_patch_size
                                if args.context_patch_size > 0
                                else None
                            ),
                            seed_k=args.seed_k,
                            seed_k_alpha=args.seed_k_alpha,
                            patch_batch_size=args.patch_batch_size,
                            fusion=args.patch_fusion,
                            noise_std=noise_std,
                            category_id=category_id,
                        )
                    unrot = den.numpy().astype(np.float32) @ rot.T
                    disp_sum += unrot - pred_np
                    count += 1
            pred_np = pred_np + float(args.pred_weight) * (disp_sum / float(max(1, count)))
    return pred_np


def build_model(args, weights):
    if args.model == "asdn":
        return ASDNModel.load_from_npz(weights, args=args)
    return PGDModel.load_from_npz(weights, args=args)


def build_models(args):
    weights = [w.strip() for w in args.weights_list.split(",") if w.strip()]
    if not weights:
        weights = [args.weights]
    return [build_model(args, w) for w in weights], weights


def run_evaluate(args):
    cmd = [
        sys.executable,
        str(Path(args.starter_root) / "evaluate.py"),
        "--pred_dir", str(Path(args.output_root) / "pred"),
        "--gt_dir", str(Path(args.output_root) / "gt"),
        "--noisy_dir", str(Path(args.output_root) / "noisy"),
        "--mesh_dir", str(args.dataset_root),
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
        raise RuntimeError("starter evaluate failed with code {}; see {}".format(proc.returncode, log_path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--weights_list", default="")
    parser.add_argument("--model", choices=["pgd", "asdn"], default="pgd")
    parser.add_argument("--dataset_root", default="/home/dataset_train")
    parser.add_argument("--datalist_dir", default="/home/PGD/datalist")
    parser.add_argument("--val_list", default="/home/PGD/datalist/validate.txt")
    parser.add_argument("--data_name", default="models/model_normalized.obj")
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--reference_eval_root", default="")
    parser.add_argument("--precomputed_points_dir", default="",
                        help="Optional normalized clean .npy root for fast screening")
    parser.add_argument("--starter_root", default="/home/starter_code")
    parser.add_argument("--sample_points", type=int, default=10000)
    parser.add_argument("--noise_std_min", type=float, default=0.005)
    parser.add_argument("--noise_std_max", type=float, default=0.020)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--patch_size", type=int, default=1000)
    parser.add_argument("--context_patch_size", type=int, default=0,
                        help="larger PGD network context; only nearest patch_size points are fused")
    parser.add_argument("--seed_k", type=int, default=5)
    parser.add_argument("--seed_k_alpha", type=float, default=10)
    parser.add_argument("--patch_batch_size", type=int, default=8)
    parser.add_argument("--patch_fusion", choices=["select", "weighted"], default="select")
    parser.add_argument("--pgd_use_noise_gate", action="store_true")
    parser.add_argument("--pgd_use_cond_gate", action="store_true")
    parser.add_argument("--pgd_gate_low_noise", type=float, default=0.005)
    parser.add_argument("--pgd_gate_high_noise", type=float, default=0.020)
    parser.add_argument("--pgd_gate_min", type=float, default=0.35)
    parser.add_argument("--pgd_gate_max", type=float, default=1.0)
    parser.add_argument("--pgd_gate_noise_source", choices=["known", "estimate"], default="known")
    parser.add_argument("--pgd_cond_gate_scale", type=float, default=0.25)
    parser.add_argument("--pgd_cond_hidden_dim", type=int, default=16)
    parser.add_argument("--pgd_cond_use_noise", action="store_true")
    parser.add_argument("--pgd_use_noise_conditioning", action="store_true")
    parser.add_argument("--pgd_noise_condition_hidden_dim", type=int, default=16)
    parser.add_argument("--pgd_noise_condition_scale", type=float, default=0.50)
    parser.add_argument("--pgd_noise_condition_min", type=float, default=0.005)
    parser.add_argument("--pgd_noise_condition_max", type=float, default=0.020)
    parser.add_argument("--pgd_noise_estimate_scale", type=float, default=1.0)
    parser.add_argument("--pgd_noise_estimate_bias", type=float, default=0.0)
    parser.add_argument("--pgd_use_surface_flow", action="store_true")
    parser.add_argument("--pgd_surface_flow_hidden_dim", type=int, default=32)
    parser.add_argument("--pgd_surface_flow_log_scale_min", type=float, default=-2.0)
    parser.add_argument("--pgd_surface_flow_log_scale_max", type=float, default=0.4)
    parser.add_argument("--pgd_use_surface_head", action="store_true")
    parser.add_argument("--pgd_surface_head_hidden_dim", type=int, default=64)
    parser.add_argument("--pgd_surface_head_max_distance", type=float, default=0.02)
    parser.add_argument("--pgd_use_surface_vector_head", action="store_true")
    parser.add_argument("--pgd_surface_vector_hidden_dim", type=int, default=64)
    parser.add_argument("--pgd_surface_vector_max_distance", type=float, default=0.02)
    parser.add_argument("--pgd_surface_vector_unit_slope", action="store_true")
    parser.add_argument("--pgd_two_stage", action="store_true")
    parser.add_argument("--pgd_use_separate_stage2", action="store_true")
    parser.add_argument("--pgd_second_stage_scale", type=float, default=1.0)
    parser.add_argument("--pgd_second_stage_tangent_only", action="store_true",
                        help="decompose stage-2 displacement into tangent/normal PCA components")
    parser.add_argument("--pgd_second_stage_tangent_scale", type=float, default=1.0)
    parser.add_argument("--pgd_second_stage_normal_scale", type=float, default=0.15)
    parser.add_argument("--pgd_second_stage_surface_k", type=int, default=16)
    parser.add_argument("--pgd_use_stage2_dual_gate", action="store_true")
    parser.add_argument("--pgd_stage2_dual_gate_scale", type=float, default=0.90)
    parser.add_argument("--pgd_use_local_geom_feature", action="store_true")
    parser.add_argument("--pgd_local_geom_k", type=int, default=16)
    parser.add_argument("--pgd_use_separate_refiner", action="store_true")
    parser.add_argument("--pgd_refiner_use_disp_feature", action="store_true")
    parser.add_argument("--pgd_use_refine_gate", action="store_true")
    parser.add_argument("--pgd_use_local_refine_gate", action="store_true")
    parser.add_argument("--pgd_use_global_refine_gate", action="store_true")
    parser.add_argument("--pgd_use_context_residual", action="store_true")
    parser.add_argument("--pgd_context_residual_scale", type=float, default=0.10)
    parser.add_argument("--pgd_use_equivariant_local_residual", action="store_true")
    parser.add_argument("--pgd_equivariant_local_k", type=int, default=16)
    parser.add_argument("--pgd_equivariant_residual_scale", type=float, default=0.10)
    parser.add_argument("--pgd_refine_gate_scale", type=float, default=0.25)
    parser.add_argument("--niters", type=int, default=1)
    parser.add_argument("--pred_weight", type=float, default=1.0)
    parser.add_argument("--tta_rotations", type=int, default=1)
    parser.add_argument("--category_embed_dim", type=int, default=16)
    parser.add_argument("--noise_embed_dim", type=int, default=16)
    parser.add_argument("--asdn_max_disp", type=float, default=1.0)
    parser.add_argument("--asdn_use_codebook", action="store_true")
    parser.add_argument("--asdn_stage3", action="store_true")
    parser.add_argument("--asdn_stage3_noise_threshold", type=float, default=0.018)
    parser.add_argument("--asdn_stage3_conf_threshold", type=float, default=0.45)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--use_cuda", action="store_true")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--run_evaluate", action="store_true")
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    args = parser.parse_args()

    if args.use_cuda:
        jt.flags.use_cuda = 1
    np.random.seed(args.seed)
    random.seed(args.seed)
    jt.set_global_seed(args.seed)

    out_root = Path(args.output_root).resolve()
    args.output_root = str(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    category_to_id = build_category_vocab(args.datalist_dir)
    args.num_categories = len(category_to_id) + 1
    models, weights = build_models(args)
    entries = read_split(args.val_list)
    if args.num_shards < 1:
        raise ValueError("--num_shards must be >= 1")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("--shard_index must be in [0, num_shards)")
    all_entries = entries
    entries = all_entries[args.shard_index::args.num_shards]
    metadata = vars(args).copy()
    metadata["num_entries"] = len(all_entries)
    metadata["num_shard_entries"] = len(entries)
    metadata["framework"] = "jittor"
    metadata["weights_list_resolved"] = weights
    metadata["category_to_id"] = category_to_id
    metadata["load_report"] = [getattr(model, "_load_report", {}) for model in models]
    metadata_name = "metadata.json" if args.num_shards == 1 else "metadata_shard{:02d}.json".format(args.shard_index)
    with open(out_root / metadata_name, "w") as f:
        json.dump(metadata, f, indent=2)

    missing = []
    fixed = []
    nonfinite = []
    directional_cd = []
    for local_i, entry in enumerate(tqdm(entries, desc="ShapeNet mesh val predict")):
        i = args.shard_index + local_i * args.num_shards
        pred_path = out_root / "pred" / entry / "denoised.npy"
        if args.skip_existing and pred_path.exists():
            continue
        if args.reference_eval_root:
            try:
                clean, noisy, noisy_norm, center, scale = load_reference_eval_sample(args.reference_eval_root, entry)
            except FileNotFoundError as exc:
                missing.append({"entry": entry, "path": str(exc)})
                continue
            noise_std = None
        else:
            mesh_path = Path(args.dataset_root) / entry / args.data_name
            if not mesh_path.exists():
                missing.append({"entry": entry, "path": str(mesh_path)})
                continue
            precomputed = None
            if args.precomputed_points_dir:
                precomputed = Path(args.precomputed_points_dir) / (entry + ".npy")
            if precomputed is not None and precomputed.exists():
                clean_norm = np.load(precomputed).astype(np.float32)
                if 0 < int(args.sample_points) < clean_norm.shape[0]:
                    # Fast candidate screening: keep a deterministic subset
                    # of the pre-sampled cloud.  Final scoring still uses the
                    # official 50k-point mesh path.
                    sub_rng = np.random.default_rng(args.seed + i)
                    keep = sub_rng.choice(clean_norm.shape[0], int(args.sample_points), replace=False)
                    clean_norm = clean_norm[np.sort(keep)]
                clean = clean_norm.copy()
                center = np.zeros((1, 3), dtype=np.float32)
                scale = np.ones((1, 1), dtype=np.float32)
            else:
                rng_state = np.random.get_state()
                np.random.seed(args.seed + i)
                vertices, faces = load_obj_mesh(mesh_path)
                clean = sample_mesh(vertices, faces, args.sample_points)
                np.random.set_state(rng_state)
                clean_norm, center, scale = normalize_with_params(clean)
            rng = np.random.default_rng(args.seed + i)
            noise_std = float(rng.uniform(args.noise_std_min, args.noise_std_max))
            noisy_norm = clean_norm + rng.normal(0.0, noise_std, size=clean_norm.shape).astype(np.float32)
            noisy = noisy_norm * scale + center
        gate_noise_std = noise_std
        uses_noise = args.pgd_use_noise_gate or args.pgd_use_noise_conditioning
        if args.reference_eval_root and uses_noise:
            gate_noise_std = estimate_noise_std_np(noisy_norm)
        elif uses_noise and args.pgd_gate_noise_source == "estimate":
            gate_noise_std = estimate_noise_std_np(noisy_norm)
        if uses_noise and gate_noise_std is not None:
            gate_noise_std = float(
                np.clip(
                    args.pgd_noise_estimate_scale * gate_noise_std
                    + args.pgd_noise_estimate_bias,
                    args.noise_std_min,
                    args.noise_std_max,
                )
            )
        category_id = category_to_id.get(extract_synset_id(entry), 0)
        pred_norm = denoise(models, noisy_norm, args, category_id=category_id, noise_std=gate_noise_std)
        pred_norm, mismatch = ensure_point_count(pred_norm, noisy_norm)
        if mismatch is not None:
            mismatch["entry"] = entry
            fixed.append(mismatch)

        pred = pred_norm * scale + center
        if not np.isfinite(pred).all():
            nonfinite.append(entry)
        pred_to_gt, gt_to_pred = directional_cd_terms(pred, clean)
        noisy_to_gt, gt_to_noisy = directional_cd_terms(noisy, clean)
        directional_cd.append({
            "entry": entry,
            "pred_to_gt": pred_to_gt,
            "gt_to_pred": gt_to_pred,
            "pred_cd": pred_to_gt + gt_to_pred,
            "noisy_to_gt": noisy_to_gt,
            "gt_to_noisy": gt_to_noisy,
            "noisy_cd": noisy_to_gt + gt_to_noisy,
        })
        save_sample(out_root, entry, clean, noisy, pred)

    if directional_cd:
        mean_directional = {
            key: float(np.mean([row[key] for row in directional_cd]))
            for key in ("pred_to_gt", "gt_to_pred", "pred_cd", "noisy_to_gt", "gt_to_noisy", "noisy_cd")
        }
    else:
        mean_directional = {}
    report = {
        "missing": missing,
        "mismatches_fixed": fixed,
        "saved_nonfinite": nonfinite,
        "directional_cd_mean": mean_directional,
        "directional_cd_per_sample": directional_cd,
    }
    count_name = "count_check.json" if args.num_shards == 1 else "count_check_shard{:02d}.json".format(args.shard_index)
    with open(out_root / count_name, "w") as f:
        json.dump(report, f, indent=2)
    if missing or nonfinite:
        raise RuntimeError("invalid eval outputs; see {}".format(out_root / "count_check.json"))
    if args.run_evaluate:
        run_evaluate(args)


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
