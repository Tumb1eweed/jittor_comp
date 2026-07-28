#!/usr/bin/env python3
"""Export a NumPy ``(N, 3)`` point cloud as an ASCII vertex-only PLY."""

import argparse
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_npy")
    parser.add_argument("output_ply")
    args = parser.parse_args()

    points = np.asarray(np.load(args.input_npy), dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("expected an (N, 3) point cloud, got {}".format(points.shape))
    if not np.isfinite(points).all():
        raise ValueError("point cloud contains non-finite coordinates")

    output = Path(args.output_ply)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write("element vertex {}\n".format(points.shape[0]))
        f.write("property float x\nproperty float y\nproperty float z\nend_header\n")
        np.savetxt(f, points, fmt="%.8f %.8f %.8f")


if __name__ == "__main__":
    main()
