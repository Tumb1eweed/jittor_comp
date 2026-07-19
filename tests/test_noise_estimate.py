import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.noise_estimate import estimate_noise_std_np


def make_noisy_plane(std, seed=0):
    rng = np.random.default_rng(seed)
    xy = rng.uniform(-0.5, 0.5, size=(4096, 2)).astype(np.float32)
    z = rng.normal(0.0, std, size=(4096, 1)).astype(np.float32)
    return np.concatenate([xy, z], axis=1)


def test_local_pca_noise_estimate_tracks_noise_level():
    low = estimate_noise_std_np(make_noisy_plane(0.005, seed=1), sample_size=1024, k=24)
    high = estimate_noise_std_np(make_noisy_plane(0.020, seed=2), sample_size=1024, k=24)

    assert 0.0025 <= low <= 0.010
    assert 0.012 <= high <= 0.030
    assert high > low * 2.0


if __name__ == "__main__":
    test_local_pca_noise_estimate_tracks_noise_level()
    os._exit(0)
