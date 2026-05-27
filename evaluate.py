import argparse
import json
from pathlib import Path

import numpy as np
import point_cloud_utils as pcu
import trimesh
from scipy.spatial import cKDTree


def load_xyz_dir(path):
    out = {}
    for fn in sorted(Path(path).glob("*.xyz")):
        out[fn.stem] = np.loadtxt(fn, dtype=np.float64)
    return out


def normalize_to_unit_sphere(pc):
    center = (pc.max(axis=0) + pc.min(axis=0)) / 2.0
    centered = pc - center
    scale = np.sqrt((centered ** 2).sum(axis=1)).max()
    return centered / scale, center, scale


def chamfer_distance(pc_a, pc_b):
    pc_b, center, scale = normalize_to_unit_sphere(pc_b)
    pc_a = (pc_a - center) / scale
    dist_a, _ = cKDTree(pc_b).query(pc_a, k=1)
    dist_b, _ = cKDTree(pc_a).query(pc_b, k=1)
    return float((dist_a ** 2).mean() + (dist_b ** 2).mean())


def load_mesh(path):
    try:
        v, f = pcu.load_mesh_vf(str(path))
        return v.astype(np.float64), f.astype(np.int32)
    except Exception:
        mesh = trimesh.load(path, process=False)
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
        return np.asarray(mesh.vertices, dtype=np.float64), np.asarray(mesh.faces, dtype=np.int32)


def point_to_surface(pc, verts, faces, ref_pc):
    _, center, scale = normalize_to_unit_sphere(ref_pc)
    pc = (pc - center) / scale
    verts = (verts - center) / scale
    dists, _, _ = pcu.closest_points_on_mesh(pc.astype(np.float32), verts.astype(np.float32), faces)
    return float((dists ** 2).mean())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_dir", required=True)
    parser.add_argument("--gt_dir", required=True)
    parser.add_argument("--mesh_dir", required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    pred = load_xyz_dir(args.pred_dir)
    gt = load_xyz_dir(args.gt_dir)
    rows = {}
    for name, pred_pc in pred.items():
        if name not in gt:
            continue
        mesh_path = Path(args.mesh_dir) / f"{name}.off"
        if not mesh_path.exists():
            continue
        verts, faces = load_mesh(mesh_path)
        rows[name] = {
            "cd_sph": chamfer_distance(pred_pc, gt[name]) * 10000,
            "p2f": point_to_surface(pred_pc, verts, faces, gt[name]) * 10000,
        }
    mean = {k: float(np.mean([r[k] for r in rows.values()])) for k in ["cd_sph", "p2f"]} if rows else {}
    result = {"per_shape": rows, "mean": mean}
    print(json.dumps(result, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
