#!/usr/bin/env python3
"""Export standalone, uncoloured denoised point clouds from a visualization run."""
from __future__ import print_function

import argparse
import json
import shutil
from pathlib import Path

import numpy as np


def write_xyz_ply(path, points):
    """Write an ASCII PLY containing only x/y/z vertex properties."""
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Expected an (N, 3) point cloud, got {}".format(points.shape))
    with path.open("w") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write("element vertex {}\n".format(len(points)))
        handle.write("property float x\nproperty float y\nproperty float z\n")
        handle.write("end_header\n")
        np.savetxt(handle, points, fmt="%.8f %.8f %.8f")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path,
                        default=Path("experiments/visualize_best7911/shapenet"))
    parser.add_argument("--output", type=Path,
                        default=Path("experiments/visualize_best7911/denoised_only"))
    args = parser.parse_args()

    source = args.source.resolve()
    output = args.output.resolve()
    if not source.is_dir():
        raise ValueError("Source directory does not exist: {}".format(source))
    if output.exists():
        raise ValueError("Refusing to overwrite existing export: {}".format(output))

    clouds = sorted(source.rglob("denoised.npy"))
    if not clouds:
        raise ValueError("No denoised.npy files found under {}".format(source))

    records = []
    for cloud_path in clouds:
        rel = cloud_path.relative_to(source).parent
        destination = output / rel
        destination.mkdir(parents=True, exist_ok=False)
        npy_output = destination / "denoised.npy"
        shutil.copy2(str(cloud_path), str(npy_output))
        points = np.load(str(cloud_path))
        write_xyz_ply(destination / "denoised.ply", points)
        records.append({"sample": str(rel), "points": int(points.shape[0])})

    (output / "manifest.json").write_text(json.dumps({
        "description": "Denoised point clouds only: no GT, noisy input, or colour fields.",
        "source": str(source),
        "samples": records,
    }, indent=2) + "\n")
    print("Exported {} clouds to {}".format(len(records), output))


if __name__ == "__main__":
    main()
