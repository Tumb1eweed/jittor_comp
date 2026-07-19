import os
import sys
from pathlib import Path

import numpy as np
import jittor as jt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.pgd import PGDModel
from tools.train_shapenet_one_epoch import (
    directional_chamfer_terms,
    score_aligned_relative_cd_loss,
    select_trainable_parameters,
)


def test_directional_chamfer_terms_are_zero_for_identical_clouds():
    points = jt.array(np.asarray([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]], dtype=np.float32))

    pred_to_clean, clean_to_pred = directional_chamfer_terms(points, points)

    np.testing.assert_allclose(float(pred_to_clean.numpy()), 0.0, atol=1e-7)
    np.testing.assert_allclose(float(clean_to_pred.numpy()), 0.0, atol=1e-7)


def test_directional_chamfer_terms_keep_pred_to_clean_signal_separate():
    clean = jt.array(np.asarray([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]], dtype=np.float32))
    pred = jt.array(np.asarray([[[0.2, 0.0, 0.0], [1.2, 0.0, 0.0]]], dtype=np.float32))

    pred_to_clean, clean_to_pred = directional_chamfer_terms(pred, clean)

    assert float(pred_to_clean.numpy()) > 0.0
    assert float(clean_to_pred.numpy()) > 0.0
    np.testing.assert_allclose(
        float(pred_to_clean.numpy()), float(clean_to_pred.numpy()), atol=1e-7
    )


def test_score_aligned_relative_cd_averages_per_sample_ratios():
    # The official metric averages ratios per sample.  This differs from a
    # ratio over a batch aggregate when noisy baselines differ in magnitude.
    clean = jt.zeros((2, 1, 3), dtype=jt.float32)
    noisy = jt.array(np.asarray([[[1.0, 0.0, 0.0]], [[3.0, 0.0, 0.0]]], dtype=np.float32))
    pred = jt.array(np.asarray([[[0.5, 0.0, 0.0]], [[1.5, 0.0, 0.0]]], dtype=np.float32))

    loss = score_aligned_relative_cd_loss(pred, clean, noisy)

    np.testing.assert_allclose(float(loss.numpy()), 0.25, atol=1e-6)


class DecoderLastEncoderArgs:
    pgd_train_cond_gate_only = False
    pgd_train_head_only = False
    pgd_train_decoder_head_only = False
    pgd_train_decoder_last_encoder = True
    pgd_use_noise_gate = False
    pgd_use_cond_gate = False
    pgd_gate_low_noise = 0.005
    pgd_gate_high_noise = 0.020
    pgd_gate_min = 0.35
    pgd_gate_max = 1.0
    pgd_cond_gate_scale = 0.25
    pgd_cond_hidden_dim = 8
    category_embed_dim = 0
    num_categories = 1


def test_select_trainable_parameters_can_add_only_last_encoder_block():
    model = PGDModel(args=DecoderLastEncoderArgs())

    params, names = select_trainable_parameters(model, DecoderLastEncoderArgs())

    assert len(params) > 5
    assert any(name.startswith("feature_nets.decoder_blocks.") for name in names)
    assert any(name.startswith("feature_nets.encoder_blocks.4.") for name in names)
    assert not any(name.startswith("feature_nets.encoder_blocks.0.") for name in names)
    assert not any(name.startswith("feature_nets.encoder_blocks.3.") for name in names)


if __name__ == "__main__":
    test_directional_chamfer_terms_are_zero_for_identical_clouds()
    test_directional_chamfer_terms_keep_pred_to_clean_signal_separate()
    test_score_aligned_relative_cd_averages_per_sample_ratios()
    test_select_trainable_parameters_can_add_only_last_encoder_block()
    os._exit(0)
