import os
import sys
from pathlib import Path

import numpy as np
import jittor as jt
from jittor import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.pgd import PGDModel
from tools.train_shapenet_one_epoch import pgd_training_loss


class ConstantFeature(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = float(value)
        self.calls = 0

    def execute(self, pcl, feat=None, offset=None):
        self.calls += 1
        return jt.ones_like(pcl) * self.value


class OneStageArgs:
    pgd_use_noise_gate = False
    pgd_use_cond_gate = False
    pgd_two_stage = False
    pgd_second_stage_scale = 1.0
    pgd_gate_low_noise = 0.005
    pgd_gate_high_noise = 0.020
    pgd_gate_min = 0.35
    pgd_gate_max = 1.0
    pgd_cond_gate_scale = 0.25
    pgd_cond_hidden_dim = 8
    category_embed_dim = 0
    num_categories = 1


class TwoStageArgs(OneStageArgs):
    pgd_two_stage = True
    pgd_second_stage_scale = 0.5


class LossArgs:
    pgd_use_normal_corr_loss = False
    corr_huber_delta = 0.01
    relative_eps = 1e-8
    loss_corr_weight = 0.0
    loss_relative_weight = 0.0
    loss_infocd_weight = 0.0
    loss_uniform_weight = 0.0
    loss_straight_weight = 0.0
    loss_stage_weight = 1.0
    pgd_loss_disp_weight = 0.0
    pgd_loss_low_noise_disp_weight = 0.0
    pgd_gate_low_noise = 0.005
    pgd_gate_high_noise = 0.020


class StageOnlyModel:
    def __call__(self, noisy, noise_std=None, category_id=None, return_dict=False):
        x1 = noisy + 0.5
        disp = jt.zeros_like(noisy)
        return {"disp": disp, "x1": x1}


def test_pgd_default_single_stage_contract_is_unchanged():
    model = PGDModel(args=OneStageArgs())
    model.feature_nets = ConstantFeature(0.1)
    noisy = jt.zeros((1, 4, 3), dtype=jt.float32)

    out = model(noisy, return_dict=True)

    assert model.feature_nets.calls == 1
    assert "x1" not in out
    np.testing.assert_allclose(out["disp"].numpy(), np.full((1, 4, 3), 0.1, dtype=np.float32), atol=1e-6)


def test_pgd_two_stage_recomputes_features_and_returns_stage_outputs():
    model = PGDModel(args=TwoStageArgs())
    model.feature_nets = ConstantFeature(0.1)
    noisy = jt.zeros((1, 4, 3), dtype=jt.float32)

    out = model(noisy, return_dict=True)

    assert model.feature_nets.calls == 2
    assert {"disp", "disp1", "disp2", "x1", "final"}.issubset(set(out.keys()))
    np.testing.assert_allclose(out["disp1"].numpy(), np.full((1, 4, 3), 0.1, dtype=np.float32), atol=1e-6)
    np.testing.assert_allclose(out["disp2"].numpy(), np.full((1, 4, 3), 0.05, dtype=np.float32), atol=1e-6)
    np.testing.assert_allclose(out["x1"].numpy(), np.full((1, 4, 3), 0.1, dtype=np.float32), atol=1e-6)
    np.testing.assert_allclose(out["final"].numpy(), np.full((1, 4, 3), 0.15, dtype=np.float32), atol=1e-6)
    np.testing.assert_allclose(out["disp"].numpy(), np.full((1, 4, 3), 0.15, dtype=np.float32), atol=1e-6)


def test_pgd_training_loss_uses_stage_output_when_available():
    noisy = jt.zeros((1, 2, 3), dtype=jt.float32)
    clean = jt.zeros((1, 2, 3), dtype=jt.float32)
    noise_std = jt.array(np.asarray([[0.01]], dtype=np.float32))
    category_id = jt.array(np.asarray([0], dtype=np.int32))

    loss, metrics = pgd_training_loss(StageOnlyModel(), noisy, clean, noise_std, category_id, None, LossArgs())

    assert float(metrics["stage"].numpy()) > 0.0
    np.testing.assert_allclose(float(loss.numpy()), float(metrics["stage"].numpy()), atol=1e-6)


if __name__ == "__main__":
    test_pgd_default_single_stage_contract_is_unchanged()
    test_pgd_two_stage_recomputes_features_and_returns_stage_outputs()
    test_pgd_training_loss_uses_stage_output_when_available()
    os._exit(0)
