#!/usr/bin/env python
"""Diagnostic oracle: move predictions toward their exact closest mesh points.

This intentionally uses the training-holdout mesh and is never a deployable
postprocess.  Its purpose is to test whether the current prediction has a
nearby normal-direction update that improves both official CD and P2S.  The
closest-point query uses the same PCU geometry backend as the official scorer;
the actual interpolation is executed with Jittor (optionally CUDA).
"""

import argparse
from pathlib import Path

import jittor as jt
import numpy as np
import point_cloud_utils as pcu


def read_list(path):
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def closest_mesh_points(points, mesh_path):
    vertices, faces = pcu.load_mesh_vf(str(mesh_path))
    vertices = vertices.astype(np.float32)
    faces = faces.astype(np.int32)
    _, face_ids, bary = pcu.closest_points_on_mesh(
        points.astype(np.float32), vertices, faces
    )
    triangles = vertices[faces[face_ids]]
    closest = np.sum(triangles * bary.astype(np.float32)[:, :, None], axis=1)
    # Some ShapeNet OBJs contain degenerate triangles. PCU still returns a
    # finite distance for them, but their barycentric coordinates can be
    # non-finite for a handful of points. Keeping those points unchanged is
    # conservative and prevents a diagnostic oracle from manufacturing NaNs.
    valid = np.isfinite(closest).all(axis=1)
    closest[~valid] = points[~valid]
    return closest


def save(root, subdir, entry, filename, value):
    path = root / subdir / entry / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, value.astype(np.float32))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_root", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--mesh_root", default="/home/dataset_train")
    parser.add_argument("--val_list", required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--use_cuda", action="store_true")
    args = parser.parse_args()

    if args.use_cuda:
        jt.flags.use_cuda = 1
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    mesh_root = Path(args.mesh_root)
    rows = []
    for entry in read_list(args.val_list):
        pred_path = input_root / "pred" / entry / "denoised.npy"
        gt_path = input_root / "gt" / entry / "clean.npy"
        noisy_path = input_root / "noisy" / entry / "noisy.npy"
        mesh_path = mesh_root / entry / "models/model_normalized.obj"
        pred = np.load(pred_path).astype(np.float32)
        gt = np.load(gt_path).astype(np.float32)
        noisy = np.load(noisy_path).astype(np.float32)
        closest = closest_mesh_points(pred, mesh_path)
        with jt.no_grad():
            pred_jt = jt.array(pred)
            closest_jt = jt.array(closest)
            projected = (
                pred_jt + float(args.alpha) * (closest_jt - pred_jt)
            ).numpy()
        save(output_root, "pred", entry, "denoised.npy", projected)
        save(output_root, "gt", entry, "clean.npy", gt)
        save(output_root, "noisy", entry, "noisy.npy", noisy)
        rows.append(
            [
                entry,
                float(np.mean(np.sum((closest - pred) ** 2, axis=1))),
                float(np.max(np.linalg.norm(closest - pred, axis=1))),
            ]
        )
    output_root.mkdir(parents=True, exist_ok=True)
    np.save(output_root / "oracle_projection_stats.npy", np.asarray(rows, dtype=object))
    print("saved {} samples to {} (alpha={})".format(len(rows), output_root, args.alpha))


if __name__ == "__main__":
    main()
