#!/usr/bin/env python3
"""Regression tests for GT-free geometry confidence gates."""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.postprocess_pca_projection import surface_confidence as projection_confidence
from tools.postprocess_tangent_repulsion import surface_confidence as tangent_confidence


def _check_confidence_function(function):
    variation = np.asarray([0.0, 0.02, 0.10], dtype=np.float32)
    consistency = np.asarray([1.0, 0.875, 0.50], dtype=np.float32)

    np.testing.assert_array_equal(
        function(variation, consistency, "none", 0.02, 0.75),
        np.ones(3, dtype=np.float32),
    )

    variation_gate = function(
        variation, consistency, "variation", 0.02, 0.75
    )
    assert 1.0 >= variation_gate[0] > variation_gate[1] > variation_gate[2] >= 0.0

    consistency_gate = function(
        variation, consistency, "consistency", 0.02, 0.75
    )
    np.testing.assert_allclose(
        consistency_gate,
        np.asarray([1.0, 0.5, 0.0], dtype=np.float32),
        atol=1e-7,
    )

    hybrid_gate = function(
        variation, consistency, "hybrid", 0.02, 0.75
    )
    np.testing.assert_allclose(
        hybrid_gate,
        variation_gate * consistency_gate,
        atol=1e-7,
    )


def test_projection_confidence_is_bounded_and_monotonic():
    _check_confidence_function(projection_confidence)


def test_tangent_confidence_is_bounded_and_monotonic():
    _check_confidence_function(tangent_confidence)


if __name__ == "__main__":
    test_projection_confidence_is_bounded_and_monotonic()
    test_tangent_confidence_is_bounded_and_monotonic()
    # This machine's Jittor/CUDA teardown can double-free after a successful
    # import-only test; match the production postprocessors' clean exit.
    import os
    import sys

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
