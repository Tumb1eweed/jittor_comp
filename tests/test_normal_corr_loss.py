import os
import sys
from pathlib import Path

import numpy as np
import jittor as jt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.train_shapenet_one_epoch import estimate_patch_normals_np, normal_tangent_corr_loss


def test_estimate_patch_normals_np_recovers_planar_normals():
    xs, ys = np.meshgrid(np.linspace(-1.0, 1.0, 8), np.linspace(-1.0, 1.0, 8))
    points = np.stack([xs.reshape(-1), ys.reshape(-1), np.zeros(xs.size)], axis=1).astype(np.float32)

    normals = estimate_patch_normals_np(points, k=8)

    assert normals.shape == points.shape
    np.testing.assert_allclose(np.abs(normals[:, 2]), 1.0, atol=1e-4)
    np.testing.assert_allclose(normals[:, :2], 0.0, atol=1e-4)


def test_normal_tangent_corr_loss_weights_normal_errors_more():
    pred_normal = jt.array(np.asarray([[[0.0, 0.0, 1.0]]], dtype=np.float32))
    pred_tangent = jt.array(np.asarray([[[1.0, 0.0, 0.0]]], dtype=np.float32))
    clean = jt.zeros((1, 1, 3), dtype=jt.float32)
    normals = jt.array(np.asarray([[[0.0, 0.0, 1.0]]], dtype=np.float32))

    normal_loss = float(normal_tangent_corr_loss(pred_normal, clean, normals).numpy())
    tangent_loss = float(normal_tangent_corr_loss(pred_tangent, clean, normals).numpy())

    assert normal_loss > tangent_loss
    np.testing.assert_allclose(normal_loss, 2.0, atol=1e-6)
    np.testing.assert_allclose(tangent_loss, 1.0, atol=1e-6)


if __name__ == "__main__":
    test_estimate_patch_normals_np_recovers_planar_normals()
    test_normal_tangent_corr_loss_weights_normal_errors_more()
    os._exit(0)
