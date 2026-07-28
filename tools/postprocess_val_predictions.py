#!/usr/bin/env python
import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.noise_estimate import estimate_noise_std_np


def iter_pred_files(source_root):
    pred_root = Path(source_root) / "pred"
    yield from sorted(pred_root.glob("**/denoised.npy"))


def relative_entry(pred_path, source_root):
    return pred_path.relative_to(Path(source_root) / "pred").parent


def parse_eval_log(log_path):
    result = {}
    if not Path(log_path).exists():
        return result
    text = Path(log_path).read_text(errors="ignore")
    patterns = {
        "cd_score": r"CD\s+得分:\s*([0-9.]+)",
        "p2s_score": r"P2S\s+得分:\s*([0-9.]+)",
        "final_score": r"最终得分.*?:\s*([0-9.]+)",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            result[key] = float(match.group(1))
    return result


def save_array(root, subdir, entry, filename, arr):
    path = Path(root) / subdir / entry
    path.mkdir(parents=True, exist_ok=True)
    np.save(path / filename, arr.astype(np.float32))


def parse_float_list(value):
    return [float(v.strip()) for v in value.split(",") if v.strip()]


def normalize_weights(weights, count):
    if weights is None:
        weights = [1.0] * count
    if len(weights) != count:
        raise ValueError("weights count {} does not match source roots {}".format(len(weights), count))
    total = float(sum(weights))
    if total == 0.0:
        raise ValueError("blend weights must not sum to zero")
    return [float(w) / total for w in weights]


def entry_synset_id(entry):
    parts = Path(entry).parts
    if len(parts) >= 2 and parts[0] == "shapenet":
        return parts[1]
    return parts[0] if parts else ""


def parse_category_blend_weights(value, count):
    result = {}
    if not value:
        return result
    for item in value.split(";"):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError("category blend item must be synset=w1,w2: {}".format(item))
        synset, raw_weights = item.split("=", 1)
        synset = synset.strip()
        if not synset:
            raise ValueError("category blend synset must not be empty")
        result[synset] = normalize_weights(parse_float_list(raw_weights), count)
    return result


def weights_for_entry(entry, default_weights, category_blend_weights):
    synset = entry_synset_id(entry)
    if synset in category_blend_weights:
        return category_blend_weights[synset]
    return default_weights


def noise_gate_value(noise_std, low=0.005, high=0.020, gate_min=0.35, gate_max=1.0):
    denom = max(float(high) - float(low), 1e-6)
    alpha = (float(noise_std) - float(low)) / denom
    alpha = min(max(alpha, 0.0), 1.0)
    return float(gate_min) + (float(gate_max) - float(gate_min)) * alpha


def build_entry_index(split_path):
    if not split_path:
        return {}
    entries = [line.strip() for line in Path(split_path).read_text().splitlines() if line.strip()]
    return {entry: i for i, entry in enumerate(entries)}


def known_noise_std_for_entry(entry, entry_to_index, seed, noise_std_min, noise_std_max):
    key = str(entry)
    if key not in entry_to_index:
        raise KeyError("entry {} is not in noise gate val list".format(key))
    rng = np.random.default_rng(int(seed) + int(entry_to_index[key]))
    return float(rng.uniform(float(noise_std_min), float(noise_std_max)))


def blend_entry_from_roots(source_roots, entry, weights=None):
    roots = [Path(root) for root in source_roots]
    weights = normalize_weights(weights, len(roots))
    noisy = None
    clean = None
    disp = None
    for root, weight in zip(roots, weights):
        pred = np.load(root / "pred" / entry / "denoised.npy").astype(np.float32)
        root_noisy = np.load(root / "noisy" / entry / "noisy.npy").astype(np.float32)
        root_clean = np.load(root / "gt" / entry / "clean.npy").astype(np.float32)
        if noisy is None:
            noisy = root_noisy
            clean = root_clean
            disp = np.zeros_like(pred, dtype=np.float32)
        elif not np.allclose(root_noisy, noisy, rtol=1e-6, atol=1e-6):
            raise ValueError("source roots do not share noisy input for {}".format(entry))
        elif not np.allclose(root_clean, clean, rtol=1e-6, atol=1e-6):
            raise ValueError("source roots do not share clean target for {}".format(entry))
        if pred.shape != noisy.shape:
            raise ValueError("prediction shape does not match noisy for {}".format(entry))
        disp += float(weight) * (pred - noisy)
    return noisy + disp, noisy, clean


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
    return parse_eval_log(log_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_root", default="")
    parser.add_argument("--source_roots", default="")
    parser.add_argument("--blend_weights", default="")
    parser.add_argument(
        "--category_blend_weights",
        default="",
        help="optional per-synset source weights, e.g. 03642806=0,1;04074963=0.75,0.25",
    )
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--dataset_root", default="/home/dataset_train")
    parser.add_argument("--starter_root", default="/home/starter_code")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--noise_gate", action="store_true")
    parser.add_argument("--noise_gate_source", choices=["known", "estimate"], default="known")
    parser.add_argument("--noise_gate_val_list", default="")
    parser.add_argument("--noise_gate_seed", type=int, default=2026)
    parser.add_argument("--noise_gate_std_min", type=float, default=0.005)
    parser.add_argument("--noise_gate_std_max", type=float, default=0.020)
    parser.add_argument("--noise_gate_low", type=float, default=0.005)
    parser.add_argument("--noise_gate_high", type=float, default=0.020)
    parser.add_argument("--noise_gate_min", type=float, default=0.35)
    parser.add_argument("--noise_gate_max", type=float, default=1.0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--run_evaluate", action="store_true")
    args = parser.parse_args()

    source_root = Path(args.source_root).resolve() if args.source_root else None
    source_roots = [Path(v.strip()).resolve() for v in args.source_roots.split(",") if v.strip()]
    if source_roots and source_root is not None:
        raise ValueError("use either --source_root or --source_roots, not both")
    if not source_roots:
        if source_root is None:
            raise ValueError("one of --source_root or --source_roots is required")
        source_roots = [source_root]
    blend_weights = parse_float_list(args.blend_weights) if args.blend_weights else None
    default_blend_weights = normalize_weights(blend_weights, len(source_roots))
    category_blend_weights = parse_category_blend_weights(args.category_blend_weights, len(source_roots))
    entry_to_index = build_entry_index(args.noise_gate_val_list) if args.noise_gate else {}
    out_root = Path(args.output_root).resolve()
    args.output_root = str(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    entries = []
    nonfinite = []
    mismatches = []
    for pred_path in tqdm(list(iter_pred_files(source_roots[0])), desc="Blend val predictions"):
        entry = relative_entry(pred_path, source_roots[0])
        entry_weights = weights_for_entry(entry, default_blend_weights, category_blend_weights)
        try:
            blended, noisy, clean = blend_entry_from_roots(source_roots, entry, weights=entry_weights)
        except ValueError as exc:
            mismatches.append({"entry": str(entry), "error": str(exc)})
            continue
        blended = noisy + float(args.alpha) * (blended - noisy)
        if args.noise_gate:
            if args.noise_gate_source == "known":
                noise_std = known_noise_std_for_entry(
                    entry,
                    entry_to_index,
                    args.noise_gate_seed,
                    args.noise_gate_std_min,
                    args.noise_gate_std_max,
                )
            else:
                noise_std = estimate_noise_std_np(noisy)
            gate = noise_gate_value(
                noise_std,
                low=args.noise_gate_low,
                high=args.noise_gate_high,
                gate_min=args.noise_gate_min,
                gate_max=args.noise_gate_max,
            )
            blended = noisy + gate * (blended - noisy)
        if not np.isfinite(blended).all():
            nonfinite.append(str(entry))
        save_array(out_root, "pred", entry, "denoised.npy", blended)
        save_array(out_root, "noisy", entry, "noisy.npy", noisy)
        save_array(out_root, "gt", entry, "clean.npy", clean)
        entries.append(str(entry))

    metadata = vars(args).copy()
    metadata.update({
        "source_root": str(source_root) if source_root is not None else "",
        "source_roots_resolved": [str(root) for root in source_roots],
        "blend_weights_normalized": default_blend_weights,
        "category_blend_weights_normalized": category_blend_weights,
        "noise_gate_enabled": bool(args.noise_gate),
        "output_root": str(out_root),
        "num_entries": len(entries),
        "mismatches": mismatches,
        "nonfinite": nonfinite,
    })
    if args.run_evaluate and not mismatches and not nonfinite:
        metadata["eval"] = run_evaluate(args)
    with open(out_root / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    if mismatches or nonfinite:
        raise RuntimeError("invalid blended outputs; see {}".format(out_root / "metadata.json"))
    if "eval" in metadata:
        print(json.dumps(metadata["eval"], indent=2))


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
