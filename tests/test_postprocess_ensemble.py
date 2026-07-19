import os
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.postprocess_val_predictions import (
    blend_entry_from_roots,
    noise_gate_value,
    parse_category_blend_weights,
    weights_for_entry,
)


def _save(root, subdir, entry, filename, arr):
    path = Path(root) / subdir / entry
    path.mkdir(parents=True)
    np.save(path / filename, arr.astype(np.float32))


def test_blend_entry_from_roots_averages_displacements_from_shared_noisy():
    entry = Path("shapenet/00000000/example")
    noisy = np.asarray([[1.0, 2.0, 3.0], [2.0, 0.0, -1.0]], dtype=np.float32)
    clean = noisy - 0.5
    pred_a = noisy + np.asarray([[0.2, 0.0, 0.0], [0.0, 0.2, 0.0]], dtype=np.float32)
    pred_b = noisy + np.asarray([[0.0, 0.4, 0.0], [0.0, 0.0, 0.4]], dtype=np.float32)

    with tempfile.TemporaryDirectory() as tmp:
        root_a = Path(tmp) / "a"
        root_b = Path(tmp) / "b"
        for root, pred in ((root_a, pred_a), (root_b, pred_b)):
            _save(root, "pred", entry, "denoised.npy", pred)
            _save(root, "noisy", entry, "noisy.npy", noisy)
            _save(root, "gt", entry, "clean.npy", clean)

        blended, loaded_noisy, loaded_clean = blend_entry_from_roots([root_a, root_b], entry, weights=[0.25, 0.75])

    expected = noisy + 0.25 * (pred_a - noisy) + 0.75 * (pred_b - noisy)
    np.testing.assert_allclose(blended, expected, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(loaded_noisy, noisy)
    np.testing.assert_allclose(loaded_clean, clean)


def test_noise_gate_value_scales_between_min_and_max():
    assert noise_gate_value(0.003, low=0.005, high=0.020, gate_min=0.35, gate_max=1.0) == 0.35
    assert noise_gate_value(0.020, low=0.005, high=0.020, gate_min=0.35, gate_max=1.0) == 1.0
    mid = noise_gate_value(0.0125, low=0.005, high=0.020, gate_min=0.35, gate_max=1.0)
    assert abs(mid - 0.675) < 1e-6


def test_category_blend_weights_override_default_by_synset():
    category_weights = parse_category_blend_weights("03642806=0,2;04074963=3,1", 2)

    assert category_weights["03642806"] == [0.0, 1.0]
    assert category_weights["04074963"] == [0.75, 0.25]
    assert weights_for_entry(Path("shapenet/03642806/example"), [0.5, 0.5], category_weights) == [0.0, 1.0]
    assert weights_for_entry(Path("shapenet/99999999/example"), [0.5, 0.5], category_weights) == [0.5, 0.5]


if __name__ == "__main__":
    test_blend_entry_from_roots_averages_displacements_from_shared_noisy()
    test_noise_gate_value_scales_between_min_and_max()
    test_category_blend_weights_override_default_by_synset()
    os._exit(0)
