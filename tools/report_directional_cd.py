#!/usr/bin/env python3
"""Report scorer-consistent pred→GT and GT→pred squared-CD components."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


def normalize_reference(points):
    center = (points.max(axis=0) + points.min(axis=0)) * 0.5
    centered = points - center
    scale = np.sqrt(np.sum(centered * centered, axis=1)).max()
    if scale < 1e-12:
        raise ValueError("degenerate reference point cloud")
    return centered / scale, center, scale


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred_dir", required=True)
    parser.add_argument("--gt_dir", required=True)
    parser.add_argument("--pred_filename", default="denoised.npy")
    parser.add_argument("--gt_filename", default="clean.npy")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    pred_root, gt_root = Path(args.pred_dir), Path(args.gt_dir)
    records = []
    for path in sorted(pred_root.rglob(args.pred_filename)):
        rel = path.parent.relative_to(pred_root)
        gt_path = gt_root / rel / args.gt_filename
        if not gt_path.exists():
            raise FileNotFoundError(str(gt_path))
        pred = np.load(path).astype(np.float64)
        gt = np.load(gt_path).astype(np.float64)
        gt_norm, center, scale = normalize_reference(gt)
        pred_norm = (pred - center) / scale
        p2g = np.mean(cKDTree(gt_norm).query(pred_norm, k=1)[0] ** 2)
        g2p = np.mean(cKDTree(pred_norm).query(gt_norm, k=1)[0] ** 2)
        records.append({"key": str(rel), "pred_to_gt": float(p2g), "gt_to_pred": float(g2p), "cd": float(p2g + g2p)})
    if not records:
        raise RuntimeError("no prediction files found")
    payload = {
        "count": len(records),
        "mean_pred_to_gt": float(np.mean([x["pred_to_gt"] for x in records])),
        "mean_gt_to_pred": float(np.mean([x["gt_to_pred"] for x in records])),
        "mean_cd": float(np.mean([x["cd"] for x in records])),
        "samples": records,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({k: payload[k] for k in payload if k != "samples"}, indent=2))


if __name__ == "__main__":
    main()
