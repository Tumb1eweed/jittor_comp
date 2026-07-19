import os
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.eval_shapenet_mesh_val import load_reference_eval_sample, normalize_with_params


def test_reference_eval_sample_reconstructs_normalized_noisy():
    entry = "shapenet/00000000/example"
    clean = np.asarray(
        [
            [-1.0, -1.0, 0.0],
            [1.0, -1.0, 0.0],
            [1.0, 1.0, 0.0],
            [-1.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    clean_norm, center, scale = normalize_with_params(clean)
    noisy_norm_expected = clean_norm + np.asarray(
        [[0.01, 0.0, 0.0], [0.0, -0.02, 0.0], [0.0, 0.0, 0.03], [-0.01, 0.0, 0.0]],
        dtype=np.float32,
    )
    noisy = noisy_norm_expected * scale + center

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "gt" / entry).mkdir(parents=True)
        (root / "noisy" / entry).mkdir(parents=True)
        np.save(root / "gt" / entry / "clean.npy", clean)
        np.save(root / "noisy" / entry / "noisy.npy", noisy.astype(np.float32))

        loaded_clean, loaded_noisy, loaded_noisy_norm, loaded_center, loaded_scale = load_reference_eval_sample(root, entry)

    np.testing.assert_allclose(loaded_clean, clean)
    np.testing.assert_allclose(loaded_noisy, noisy, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(loaded_noisy_norm, noisy_norm_expected, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(loaded_center, center)
    np.testing.assert_allclose(loaded_scale, scale)


if __name__ == "__main__":
    test_reference_eval_sample_reconstructs_normalized_noisy()
    os._exit(0)
