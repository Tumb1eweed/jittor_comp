import os
import sys
from pathlib import Path

import numpy as np
import jittor as jt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.train_shapenet_one_epoch import tangent_spacing_consistency_loss


def test_tangent_spacing_penalizes_planar_point_collapse():
    clean = jt.array(np.asarray([[
        [0.0, 0.0, 0.0], [0.1, 0.0, 0.0],
        [0.0, 0.1, 0.0], [0.1, 0.1, 0.0],
    ]], dtype=np.float32))
    normals = jt.array(np.asarray([[[0.0, 0.0, 1.0]] * 4], dtype=np.float32))
    matched = clean.copy()
    collapsed = jt.zeros_like(clean)
    matched_loss = float(tangent_spacing_consistency_loss(
        matched, clean, normals, max_points=4, k=2
    ).numpy())
    collapsed_loss = float(tangent_spacing_consistency_loss(
        collapsed, clean, normals, max_points=4, k=2
    ).numpy())
    assert np.isfinite(matched_loss)
    assert np.isfinite(collapsed_loss)
    assert collapsed_loss > matched_loss + 1e-4


if __name__ == "__main__":
    test_tangent_spacing_penalizes_planar_point_collapse()
    os._exit(0)
