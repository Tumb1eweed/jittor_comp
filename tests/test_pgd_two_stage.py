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
from tools.train_shapenet_one_epoch import anti_cluster_knn_loss, pgd_training_loss


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


class SeparateStage2Args(TwoStageArgs):
    pgd_use_separate_stage2 = True


class DualGateArgs(TwoStageArgs):
    pgd_use_stage2_dual_gate = True
    pgd_stage2_dual_gate_scale = 0.9
    pgd_second_stage_surface_k = 3


class NoiseConditionArgs(TwoStageArgs):
    pgd_use_noise_conditioning = True
    pgd_noise_condition_hidden_dim = 2
    pgd_noise_condition_scale = 0.5
    pgd_noise_condition_min = 0.005
    pgd_noise_condition_max = 0.020


class SurfaceFlowArgs(TwoStageArgs):
    pgd_use_surface_flow = True
    pgd_surface_flow_hidden_dim = 4
    pgd_surface_flow_log_scale_min = -2.0
    pgd_surface_flow_log_scale_max = 0.4


class SurfaceHeadArgs(TwoStageArgs):
    pgd_use_surface_head = True
    pgd_surface_head_hidden_dim = 4
    pgd_surface_head_max_distance = 0.02


class SurfaceVectorArgs(TwoStageArgs):
    pgd_use_surface_vector_head = True
    pgd_surface_vector_hidden_dim = 4
    pgd_surface_vector_max_distance = 0.02


class ConstantSurfaceFeature(ConstantFeature):
    feature_dim = 32

    def execute(self, pcl, feat=None, offset=None, return_features=None):
        self.calls += 1
        if return_features:
            return jt.ones((pcl.shape[0], pcl.shape[1], self.feature_dim), dtype=jt.float32)
        return jt.ones_like(pcl) * self.value

    def project_features(self, features):
        return jt.ones((features.shape[0], features.shape[1], 3), dtype=jt.float32) * self.value


class LossArgs:
    pgd_use_normal_corr_loss = False
    corr_huber_delta = 0.01
    relative_eps = 1e-8
    loss_corr_weight = 0.0
    loss_relative_weight = 0.0
    loss_infocd_weight = 0.0
    loss_uniform_weight = 0.0
    loss_density_weight = 0.0
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


def test_pgd_separate_stage2_uses_the_nonshared_backbone():
    model = PGDModel(args=SeparateStage2Args())
    model.feature_nets = ConstantFeature(0.1)
    model.feature_nets_stage2 = ConstantFeature(0.2)
    noisy = jt.zeros((1, 4, 3), dtype=jt.float32)

    out = model(noisy, return_dict=True)

    assert model.feature_nets.calls == 1
    assert model.feature_nets_stage2.calls == 1
    np.testing.assert_allclose(
        out["disp"].numpy(),
        np.full((1, 4, 3), 0.2, dtype=np.float32),
        atol=1e-6,
    )


def test_stage2_dual_gate_is_neutral_at_initialization():
    model = PGDModel(args=DualGateArgs())
    model.feature_nets = ConstantFeature(0.1)
    noisy = jt.array(np.asarray([[
        [0.0, 0.0, 0.0],
        [0.1, 0.0, 0.0],
        [0.0, 0.1, 0.0],
        [0.1, 0.1, 0.0],
    ]], dtype=np.float32))

    out = model(noisy, return_dict=True)

    np.testing.assert_allclose(
        out["stage2_normal_gate"].numpy(), np.ones((1, 4, 1), dtype=np.float32), atol=1e-6
    )
    np.testing.assert_allclose(
        out["stage2_tangent_gate"].numpy(), np.ones((1, 4, 1), dtype=np.float32), atol=1e-6
    )
    np.testing.assert_allclose(
        out["disp2"].numpy(), np.full((1, 4, 3), 0.05, dtype=np.float32), atol=1e-5
    )


def test_noise_conditioning_is_neutral_at_initialization():
    model = PGDModel(args=NoiseConditionArgs())
    model.feature_nets = ConstantFeature(0.1)
    noisy = jt.zeros((2, 4, 3), dtype=jt.float32)
    sigma = jt.array(np.asarray([[0.005], [0.020]], dtype=np.float32))

    out = model(noisy, noise_std=sigma, return_dict=True)

    np.testing.assert_allclose(
        out["stage1_noise_gate"].numpy(), np.ones((2, 1, 1), dtype=np.float32), atol=1e-6
    )
    np.testing.assert_allclose(
        out["stage2_noise_gate"].numpy(), np.ones((2, 1, 1), dtype=np.float32), atol=1e-6
    )
    np.testing.assert_allclose(
        out["disp"].numpy(), np.full((2, 4, 3), 0.15, dtype=np.float32), atol=1e-6
    )


def test_noise_conditioning_can_learn_distinct_stage_gates():
    model = PGDModel(args=NoiseConditionArgs())
    model.noise_condition_fc1.weight.assign(
        jt.array(np.asarray([[1.0, 0.0], [-1.0, 0.0]], dtype=np.float32))
    )
    model.noise_condition_fc1.bias.assign(jt.zeros_like(model.noise_condition_fc1.bias))
    model.noise_condition_fc2.weight.assign(
        jt.array(np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))
    )
    model.noise_condition_fc2.bias.assign(jt.zeros_like(model.noise_condition_fc2.bias))
    sigma = jt.array(np.asarray([[0.005], [0.020]], dtype=np.float32))

    stage1, stage2 = model._noise_condition_gates(sigma, 2)
    stage1 = stage1.numpy().reshape(-1)
    stage2 = stage2.numpy().reshape(-1)

    assert stage1[0] < stage1[1]
    assert stage2[0] > stage2[1]


def test_surface_flow_is_neutral_at_initialization():
    model = PGDModel(args=SurfaceFlowArgs())
    model.feature_nets = ConstantSurfaceFeature(0.1)
    noisy = jt.zeros((2, 4, 3), dtype=jt.float32)

    out = model(noisy, return_dict=True)

    np.testing.assert_allclose(
        out["flow_distance"].numpy(), np.ones((2, 1, 1), dtype=np.float32), atol=1e-6
    )
    np.testing.assert_allclose(
        out["flow_velocity"].numpy(), np.full((2, 4, 3), 0.15, dtype=np.float32), atol=1e-6
    )
    np.testing.assert_allclose(
        out["disp"].numpy(), out["flow_velocity"].numpy(), atol=1e-6
    )


def test_surface_flow_distance_scales_velocity():
    model = PGDModel(args=SurfaceFlowArgs())
    model.feature_nets = ConstantSurfaceFeature(0.1)
    model.surface_flow_distance_fc2.weight.assign(
        jt.zeros_like(model.surface_flow_distance_fc2.weight)
    )
    model.surface_flow_distance_fc2.bias.assign(
        jt.ones_like(model.surface_flow_distance_fc2.bias) * np.log(0.5)
    )
    noisy = jt.zeros((1, 4, 3), dtype=jt.float32)

    out = model(noisy, return_dict=True)

    np.testing.assert_allclose(
        out["flow_distance"].numpy(), np.full((1, 1, 1), 0.5, dtype=np.float32), atol=1e-6
    )
    np.testing.assert_allclose(
        out["disp"].numpy(), np.full((1, 4, 3), 0.075, dtype=np.float32), atol=1e-6
    )


def test_surface_head_is_neutral_at_initialization():
    model = PGDModel(args=SurfaceHeadArgs())
    model.feature_nets = ConstantSurfaceFeature(0.1)
    noisy = jt.zeros((1, 4, 3), dtype=jt.float32)

    out = model(noisy, return_dict=True)

    np.testing.assert_allclose(
        out["surface_head_distance"].numpy(),
        np.zeros((1, 4, 1), dtype=np.float32),
        atol=1e-7,
    )
    np.testing.assert_allclose(
        out["surface_head_correction"].numpy(),
        np.zeros((1, 4, 3), dtype=np.float32),
        atol=1e-7,
    )
    np.testing.assert_allclose(
        out["disp"].numpy(),
        np.full((1, 4, 3), 0.15, dtype=np.float32),
        atol=1e-6,
    )


def test_surface_vector_head_is_neutral_at_initialization():
    model = PGDModel(args=SurfaceVectorArgs())
    model.feature_nets = ConstantSurfaceFeature(0.1)
    noisy = jt.zeros((1, 4, 3), dtype=jt.float32)

    out = model(noisy, return_dict=True)

    np.testing.assert_allclose(
        out["disp"].numpy(),
        np.full((1, 4, 3), 0.15, dtype=np.float32),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        out["surface_vector_correction"].numpy(),
        np.zeros((1, 4, 3), dtype=np.float32),
        atol=1e-7,
    )
    np.testing.assert_allclose(
        out["surface_base_disp"].numpy(),
        np.full((1, 4, 3), 0.15, dtype=np.float32),
        atol=1e-6,
    )


def test_pgd_training_loss_uses_stage_output_when_available():
    noisy = jt.zeros((1, 2, 3), dtype=jt.float32)
    clean = jt.zeros((1, 2, 3), dtype=jt.float32)
    noise_std = jt.array(np.asarray([[0.01]], dtype=np.float32))
    category_id = jt.array(np.asarray([0], dtype=np.int32))

    loss, metrics = pgd_training_loss(StageOnlyModel(), noisy, clean, noise_std, category_id, None, LossArgs())

    assert float(metrics["stage"].numpy()) > 0.0
    np.testing.assert_allclose(float(loss.numpy()), float(metrics["stage"].numpy()), atol=1e-6)


def test_pgd_density_term_is_finite_and_penalizes_collapsed_points():
    class DensityArgs(LossArgs):
        loss_stage_weight = 0.0
        loss_density_weight = 1.0

    noisy = jt.zeros((1, 4, 3), dtype=jt.float32)
    clean = jt.array(np.asarray([[[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.1, 0.1, 0.0]]], dtype=np.float32))
    noise_std = jt.array(np.asarray([[0.01]], dtype=np.float32))
    category_id = jt.array(np.asarray([0], dtype=np.int32))
    loss, metrics = pgd_training_loss(StageOnlyModel(), noisy, clean, noise_std, category_id, None, DensityArgs())

    assert np.isfinite(float(loss.numpy()))
    assert float(metrics["density"].numpy()) > 0.0


def test_pgd_weighted_patch_fusion_preserves_cardinality_and_is_finite():
    model = PGDModel(args=OneStageArgs())
    model.feature_nets = ConstantFeature(0.1)
    pcl = jt.array(np.asarray([
        [0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0], [0.3, 0.0, 0.0],
        [0.4, 0.0, 0.0], [0.5, 0.0, 0.0], [0.6, 0.0, 0.0], [0.7, 0.0, 0.0],
    ], dtype=np.float32))
    out = model.patch_based_denoise(pcl, patch_size=4, seed_k=1, seed_k_alpha=2, fusion="weighted")
    assert tuple(out.shape) == tuple(pcl.shape)
    assert np.isfinite(out.numpy()).all()


def test_pgd_core_context_fuses_only_core_and_preserves_cardinality():
    model = PGDModel(args=OneStageArgs())
    model.feature_nets = ConstantFeature(0.1)
    pcl = jt.array(np.asarray([
        [0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0], [0.3, 0.0, 0.0],
        [0.4, 0.0, 0.0], [0.5, 0.0, 0.0], [0.6, 0.0, 0.0], [0.7, 0.0, 0.0],
    ], dtype=np.float32))

    out = model.patch_based_denoise(
        pcl,
        patch_size=4,
        context_patch_size=6,
        seed_k=1,
        seed_k_alpha=2,
        patch_batch_size=1,
        fusion="select",
    )

    assert tuple(out.shape) == tuple(pcl.shape)
    assert np.isfinite(out.numpy()).all()
    np.testing.assert_allclose(
        out.numpy(), pcl.numpy() + 0.1, atol=1e-6
    )


def test_anti_cluster_knn_loss_penalizes_collapsed_spacing():
    clean = jt.array(np.asarray([[[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0], [0.3, 0.0, 0.0]]], dtype=np.float32))
    collapsed = jt.zeros_like(clean)
    matched = clean.copy()
    collapsed_loss = float(anti_cluster_knn_loss(collapsed, clean, k=2, margin=0.85).numpy())
    matched_loss = float(anti_cluster_knn_loss(matched, clean, k=2, margin=0.85).numpy())
    assert np.isfinite(collapsed_loss)
    assert collapsed_loss > matched_loss + 1e-6


if __name__ == "__main__":
    test_pgd_default_single_stage_contract_is_unchanged()
    test_pgd_two_stage_recomputes_features_and_returns_stage_outputs()
    test_stage2_dual_gate_is_neutral_at_initialization()
    test_noise_conditioning_is_neutral_at_initialization()
    test_noise_conditioning_can_learn_distinct_stage_gates()
    test_surface_flow_is_neutral_at_initialization()
    test_surface_flow_distance_scales_velocity()
    test_surface_head_is_neutral_at_initialization()
    test_surface_vector_head_is_neutral_at_initialization()
    test_pgd_training_loss_uses_stage_output_when_available()
    test_pgd_density_term_is_finite_and_penalizes_collapsed_points()
    test_pgd_weighted_patch_fusion_preserves_cardinality_and_is_finite()
    test_pgd_core_context_fuses_only_core_and_preserves_cardinality()
    test_anti_cluster_knn_loss_penalizes_collapsed_spacing()
    os._exit(0)
