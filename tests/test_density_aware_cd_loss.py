import os
import sys
from pathlib import Path

import numpy as np
import jittor as jt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.train_shapenet_one_epoch import density_aware_chamfer_relative_loss


def test_dcd_multiplicity_penalizes_many_to_one_cluster():
    clean = jt.array(np.asarray([[
        [0.0, 0.0, 0.0], [0.1, 0.0, 0.0],
        [0.0, 0.1, 0.0], [0.1, 0.1, 0.0],
    ]], dtype=np.float32))
    noisy = clean + jt.array(np.asarray([0.025, -0.012, 0.0], dtype=np.float32))
    matched = clean.copy()
    collapsed = jt.zeros_like(clean)
    matched_loss = float(density_aware_chamfer_relative_loss(
        matched, clean, noisy, max_points=4, alpha=1.0, n_lambda=1.0
    ).numpy())
    collapsed_loss = float(density_aware_chamfer_relative_loss(
        collapsed, clean, noisy, max_points=4, alpha=1.0, n_lambda=1.0
    ).numpy())
    assert np.isfinite(matched_loss)
    assert np.isfinite(collapsed_loss)
    assert collapsed_loss > matched_loss + 1e-4


if __name__ == "__main__":
    test_dcd_multiplicity_penalizes_many_to_one_cluster()
    os._exit(0)
