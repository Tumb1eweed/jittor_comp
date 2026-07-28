import os
import sys
from pathlib import Path

import numpy as np
import jittor as jt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.train_shapenet_one_epoch import sinkhorn_coverage_cost, sinkhorn_coverage_relative_loss


def test_balanced_coverage_penalizes_a_cluster_that_chamfer_can_underweight():
    # Four nearby surface samples.  A collapsed prediction has a finite
    # clean-to-pred nearest-neighbour loss, but cannot satisfy balanced OT's
    # equal prediction mass constraint.
    clean = jt.array(np.asarray([[
        [0.0, 0.0, 0.0], [0.1, 0.0, 0.0],
        [0.0, 0.1, 0.0], [0.1, 0.1, 0.0],
    ]], dtype=np.float32))
    matched = clean.copy()
    collapsed = jt.zeros_like(clean)
    matched_cost = float(sinkhorn_coverage_cost(matched, clean, max_points=4, iterations=8, blur=1.5).numpy())
    collapsed_cost = float(sinkhorn_coverage_cost(collapsed, clean, max_points=4, iterations=8, blur=1.5).numpy())
    assert np.isfinite(matched_cost)
    assert np.isfinite(collapsed_cost)
    assert collapsed_cost > matched_cost + 1e-4


def test_balanced_coverage_relative_loss_is_finite():
    clean = jt.array(np.asarray([[[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0]]], dtype=np.float32))
    noisy = clean + 0.01
    value = sinkhorn_coverage_relative_loss(clean, clean, noisy, max_points=3, iterations=4, blur=2.0)
    assert np.isfinite(float(value.numpy()))


if __name__ == "__main__":
    test_balanced_coverage_penalizes_a_cluster_that_chamfer_can_underweight()
    test_balanced_coverage_relative_loss_is_finite()
    os._exit(0)
