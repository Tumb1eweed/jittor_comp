#!/usr/bin/env python3.7
"""Create colored before/after PLY overlays for ShapeNet mesh validation.

The default model settings intentionally reproduce the PGD competition
submission stored in ``experiments/test_submission_best7911/result``.
"""

import argparse
import json
import os
import random
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

from models.pgd import PGDModel
from tools.eval_shapenet_mesh_val import (
    build_category_vocab,
    denoise,
    extract_synset_id,
    load_obj_mesh,
    normalize_with_params,
    read_split,
    sample_mesh,
)

RED = np.asarray([255, 48, 48], dtype=np.uint8)
BLUE = np.asarray([48, 96, 255], dtype=np.uint8)


def write_colored_overlay(path, red_points, blue_points):
    """Write one ASCII PLY with red points followed by blue reference points."""
    red_points = np.asarray(red_points, dtype=np.float32)
    blue_points = np.asarray(blue_points, dtype=np.float32)
    for name, points in (("red", red_points), ("blue", blue_points)):
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("{} points must have shape (N, 3), got {}".format(name, points.shape))
        if not np.isfinite(points).all():
            raise ValueError("{} points contain non-finite coordinates".format(name))
    points = np.concatenate([red_points, blue_points], axis=0)
    colors = np.concatenate([
        np.broadcast_to(RED, (red_points.shape[0], 3)),
        np.broadcast_to(BLUE, (blue_points.shape[0], 3)),
    ], axis=0)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write("comment red: noisy (before) or denoised (after)\n")
        f.write("comment blue: clean ground truth\n")
        f.write("element vertex {}\n".format(points.shape[0]))
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for point, color in zip(points, colors):
            f.write("{:.8f} {:.8f} {:.8f} {} {} {}\n".format(
                point[0], point[1], point[2], int(color[0]), int(color[1]), int(color[2])
            ))


def directional_cd(pred, clean):
    """Return scorer-compatible squared CD directions after GT normalization."""
    clean_norm, center, scale = normalize_with_params(clean)
    pred_norm = (np.asarray(pred, dtype=np.float32) - center) / scale
    pred_to_clean = cKDTree(clean_norm).query(pred_norm, k=1)[0]
    clean_to_pred = cKDTree(pred_norm).query(clean_norm, k=1)[0]
    return float(np.mean(pred_to_clean ** 2)), float(np.mean(clean_to_pred ** 2))


def select_entries(entries, count, seed):
    if count < 1:
        raise ValueError("--num_samples must be at least one")
    if count > len(entries):
        raise ValueError("requested {} samples but split has {} entries".format(count, len(entries)))
    return random.Random(seed).sample(entries, count)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["pgd"], default="pgd")
    parser.add_argument("--weights", default=str(ROOT / "experiments/pgd_normalcorr_continue400_from7908/pgd-shapenet-epoch00-loss4.53743574.npz"))
    parser.add_argument("--dataset_root", default="/home/dataset_train")
    parser.add_argument("--datalist_dir", default=str(ROOT / "datalist"))
    parser.add_argument("--val_list", default=str(ROOT / "datalist/validate.txt"))
    parser.add_argument("--data_name", default="models/model_normalized.obj")
    parser.add_argument("--output_root", default=str(ROOT / "experiments/visualize_best7911"))
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--sample_points", type=int, default=50000)
    parser.add_argument("--noise_std_min", type=float, default=0.005)
    parser.add_argument("--noise_std_max", type=float, default=0.020)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--patch_size", type=int, default=1500)
    parser.add_argument("--seed_k", type=int, default=7)
    parser.add_argument("--seed_k_alpha", type=float, default=10.0)
    parser.add_argument("--patch_batch_size", type=int, default=8)
    parser.add_argument("--niters", type=int, default=1)
    parser.add_argument("--pred_weight", type=float, default=1.0)
    parser.add_argument("--category_embed_dim", type=int, default=16)
    parser.add_argument("--noise_embed_dim", type=int, default=16)
    parser.add_argument("--pgd_two_stage", action="store_true", default=True)
    parser.add_argument("--pgd_second_stage_scale", type=float, default=0.5)
    parser.add_argument("--pgd_use_refine_gate", action="store_true", default=True)
    parser.add_argument("--pgd_refine_gate_scale", type=float, default=0.25)
    parser.add_argument("--tta_rotations", type=int, default=1)
    parser.add_argument("--patch_fusion", choices=["select"], default="select")
    parser.add_argument("--use_cuda", action="store_true")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument(
        "--eval_root", default="",
        help="Existing eval_shapenet_mesh_val.py output with gt/noisy/pred; when set, only create PLY overlays.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.use_cuda:
        jt.flags.use_cuda = 1
    np.random.seed(args.seed)
    random.seed(args.seed)
    jt.set_global_seed(args.seed)

    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    entries = read_split(args.val_list)
    selected = select_entries(entries, args.num_samples, args.seed)
    (output_root / "selected_validate.txt").write_text("\n".join(selected) + "\n")

    category_to_id = build_category_vocab(args.datalist_dir)
    args.num_categories = len(category_to_id) + 1
    eval_root = Path(args.eval_root).resolve() if args.eval_root else None
    model = None if eval_root else PGDModel.load_from_npz(args.weights, args=args)
    metadata = vars(args).copy()
    metadata.update({
        "framework": "jittor",
        "selected_entries": selected,
        "category_to_id": category_to_id,
        "load_report": getattr(model, "_load_report", {}),
        "colors": {"red": "noisy before / denoised after", "blue": "clean ground truth"},
    })
    with (output_root / "metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2)

    manifest = []
    for index, entry in enumerate(tqdm(selected, desc="Visualize ShapeNet validation")):
        sample_root = output_root / entry
        pred_path = sample_root / "denoised.npy"
        clean_path = sample_root / "clean.npy"
        noisy_path = sample_root / "noisy.npy"
        if eval_root:
            clean = np.load(eval_root / "gt" / entry / "clean.npy").astype(np.float32)
            noisy = np.load(eval_root / "noisy" / entry / "noisy.npy").astype(np.float32)
            pred = np.load(eval_root / "pred" / entry / "denoised.npy").astype(np.float32)
            noise_std = None
            sample_root.mkdir(parents=True, exist_ok=True)
            np.save(clean_path, clean)
            np.save(noisy_path, noisy)
            np.save(pred_path, pred)
        elif args.skip_existing and all(path.exists() for path in (pred_path, clean_path, noisy_path)):
            clean = np.load(clean_path)
            noisy = np.load(noisy_path)
            pred = np.load(pred_path)
            noise_std = None
        else:
            source_index = entries.index(entry)
            mesh_path = Path(args.dataset_root) / entry / args.data_name
            if not mesh_path.exists():
                raise FileNotFoundError(mesh_path)
            state = np.random.get_state()
            np.random.seed(args.seed + source_index)
            vertices, faces = load_obj_mesh(mesh_path)
            clean = sample_mesh(vertices, faces, args.sample_points)
            np.random.set_state(state)
            clean_norm, center, scale = normalize_with_params(clean)
            rng = np.random.default_rng(args.seed + source_index)
            noise_std = float(rng.uniform(args.noise_std_min, args.noise_std_max))
            noisy_norm = clean_norm + rng.normal(0.0, noise_std, size=clean_norm.shape).astype(np.float32)
            noisy = noisy_norm * scale + center
            category_id = category_to_id.get(extract_synset_id(entry), 0)
            pred_norm = denoise([model], noisy_norm, args, category_id=category_id, noise_std=noise_std)
            pred = pred_norm * scale + center
            sample_root.mkdir(parents=True, exist_ok=True)
            np.save(clean_path, clean.astype(np.float32))
            np.save(noisy_path, noisy.astype(np.float32))
            np.save(pred_path, pred.astype(np.float32))

        before_p2c, before_c2p = directional_cd(noisy, clean)
        after_p2c, after_c2p = directional_cd(pred, clean)
        write_colored_overlay(sample_root / "before.ply", noisy, clean)
        write_colored_overlay(sample_root / "after.ply", pred, clean)
        sample_metadata = {
            "index": index,
            "entry": entry,
            "noise_std": noise_std,
            "before": {"pred_to_clean": before_p2c, "clean_to_pred": before_c2p, "cd": before_p2c + before_c2p},
            "after": {"pred_to_clean": after_p2c, "clean_to_pred": after_c2p, "cd": after_p2c + after_c2p},
        }
        with (sample_root / "metadata.json").open("w") as f:
            json.dump(sample_metadata, f, indent=2)
        manifest.append(sample_metadata)
        jt.clean()

    with (output_root / "manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)


if __name__ == "__main__":
    main()
