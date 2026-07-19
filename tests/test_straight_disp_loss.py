import os
import sys
from pathlib import Path

import numpy as np
import jittor as jt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.train_shapenet_one_epoch import straight_displacement_loss


def test_straight_displacement_loss_is_zero_for_matching_displacement():
    noisy = jt.zeros((1, 1, 3), dtype=jt.float32)
    clean = jt.array(np.asarray([[[1.0, 0.0, 0.0]]], dtype=np.float32))
    pred = clean

    loss, direction, distance = straight_displacement_loss(pred, noisy, clean)

    np.testing.assert_allclose(float(loss.numpy()), 0.0, atol=1e-6)
    np.testing.assert_allclose(float(direction.numpy()), 0.0, atol=1e-6)
    np.testing.assert_allclose(float(distance.numpy()), 0.0, atol=1e-6)


def test_straight_displacement_loss_penalizes_wrong_direction_more_than_short_step():
    noisy = jt.zeros((1, 1, 3), dtype=jt.float32)
    clean = jt.array(np.asarray([[[1.0, 0.0, 0.0]]], dtype=np.float32))
    wrong_direction = jt.array(np.asarray([[[-1.0, 0.0, 0.0]]], dtype=np.float32))
    short_step = jt.array(np.asarray([[[0.5, 0.0, 0.0]]], dtype=np.float32))

    wrong_loss, wrong_direction_term, _ = straight_displacement_loss(wrong_direction, noisy, clean)
    short_loss, short_direction_term, short_distance_term = straight_displacement_loss(short_step, noisy, clean)

    assert float(wrong_loss.numpy()) > float(short_loss.numpy())
    assert float(wrong_direction_term.numpy()) > float(short_direction_term.numpy())
    assert float(short_distance_term.numpy()) > 0.0


if __name__ == "__main__":
    test_straight_displacement_loss_is_zero_for_matching_displacement()
    test_straight_displacement_loss_penalizes_wrong_direction_more_than_short_step()
    os._exit(0)
