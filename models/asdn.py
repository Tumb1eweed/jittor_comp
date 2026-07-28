import os
from pathlib import Path

os.environ.setdefault("nvcc_path", "")

import numpy as np
import jittor as jt
from jittor import nn

from models.feature import FeatureExtraction
from models.utils import farthest_point_sampling_jt, knn_points


def _as_var(x, dtype=np.float32):
    if isinstance(x, jt.Var):
        return x
    return jt.array(np.asarray(x, dtype=dtype))


class ASDNStage(nn.Module):
    def __init__(self, cond_dim, max_disp=1.0, use_codebook=False):
        super().__init__()
        self.max_disp = float(max_disp)
        self.backbone = FeatureExtraction(cond_dim=cond_dim, return_features=True, use_codebook=use_codebook)
        feat_dim = self.backbone.feature_dim
        self.disp_head = nn.Sequential(
            nn.Linear(feat_dim, 128, bias=False),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
        )
        self.gate_head = nn.Sequential(
            nn.Linear(feat_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.local_noise_head = nn.Sequential(
            nn.Linear(feat_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self._init_pgd_compatible_heads()

    def _init_pgd_compatible_heads(self):
        state = self.state_dict()
        if "backbone.condition_film.film.2.weight" in state:
            state["backbone.condition_film.film.2.weight"].assign(jt.zeros_like(state["backbone.condition_film.film.2.weight"]))
        if "backbone.condition_film.film.2.bias" in state:
            state["backbone.condition_film.film.2.bias"].assign(jt.zeros_like(state["backbone.condition_film.film.2.bias"]))
        if "local_noise_head.2.weight" in state:
            state["local_noise_head.2.weight"].assign(jt.zeros_like(state["local_noise_head.2.weight"]))
        if "local_noise_head.2.bias" in state:
            bias = np.log(0.015 / (0.04 - 0.015))
            state["local_noise_head.2.bias"].assign(jt.full_like(state["local_noise_head.2.bias"], float(bias)))

    def execute(self, points, cond):
        b, n, _ = points.shape
        offset = jt.array(np.array([(i + 1) * n for i in range(b)], dtype=np.int32))
        feat = self.backbone(points, None, offset, cond=cond, return_features=True)
        flat = feat.reshape(b * n, -1)
        disp = jt.tanh(self.disp_head(flat)).reshape(b, n, 3) * self.max_disp
        gate = jt.sigmoid(self.gate_head(flat)).reshape(b, n, 1)
        local_sigma = jt.sigmoid(self.local_noise_head(flat)).reshape(b, n, 1) * 0.04
        return disp, gate, local_sigma, feat


class ASDNModel(nn.Module):
    def __init__(self, args=None):
        super().__init__()
        self.args = args
        self.num_categories = int(getattr(args, "num_categories", 1) if args is not None else 1)
        self.category_dim = int(getattr(args, "category_embed_dim", 16) if args is not None else 16)
        self.noise_dim = int(getattr(args, "noise_embed_dim", 16) if args is not None else 16)
        self.max_disp = float(getattr(args, "asdn_max_disp", 1.0) if args is not None else 1.0)
        self.use_stage3 = bool(getattr(args, "asdn_stage3", False) if args is not None else False)
        self.use_codebook = bool(getattr(args, "asdn_use_codebook", False) if args is not None else False)
        self.stage3_noise_threshold = float(getattr(args, "asdn_stage3_noise_threshold", 0.018) if args is not None else 0.018)
        self.stage3_conf_threshold = float(getattr(args, "asdn_stage3_conf_threshold", 0.45) if args is not None else 0.45)

        self.noise_encoder = nn.Sequential(
            nn.Linear(1, self.noise_dim),
            nn.ReLU(),
            nn.Linear(self.noise_dim, self.noise_dim),
        )
        self.category_embed = nn.Embedding(max(1, self.num_categories), self.category_dim)
        cond_dim = self.noise_dim + self.category_dim
        self.stage1 = ASDNStage(cond_dim=cond_dim, max_disp=self.max_disp, use_codebook=self.use_codebook)
        self.stage2 = ASDNStage(cond_dim=cond_dim, max_disp=self.max_disp, use_codebook=self.use_codebook)
        self.stage3 = ASDNStage(cond_dim=cond_dim, max_disp=self.max_disp, use_codebook=self.use_codebook) if self.use_stage3 else None

    @classmethod
    def load_from_npz(cls, path, args=None):
        model = cls(args=args)
        model.load_npz(path)
        model.eval()
        return model

    def _default_noise(self, points):
        return jt.full((points.shape[0], 1), 0.015, dtype=jt.float32)

    def _default_category(self, points):
        return jt.zeros((points.shape[0],), dtype=jt.int32)

    def _make_condition(self, points, noise_std=None, category_id=None):
        if noise_std is None:
            noise_std = self._default_noise(points)
        else:
            noise_std = _as_var(noise_std)
            if len(noise_std.shape) == 0:
                noise_std = noise_std.reshape(1, 1).broadcast((points.shape[0], 1))
            elif len(noise_std.shape) == 1:
                noise_std = noise_std.reshape(-1, 1)
            if noise_std.shape[0] == 1 and points.shape[0] > 1:
                noise_std = noise_std.broadcast((points.shape[0], 1))
        if category_id is None:
            category_id = self._default_category(points)
        else:
            category_id = _as_var(category_id, dtype=np.int32).int32()
            if len(category_id.shape) == 0:
                category_id = category_id.reshape(1).broadcast((points.shape[0],))
            elif category_id.shape[0] == 1 and points.shape[0] > 1:
                category_id = category_id.broadcast((points.shape[0],))
        category_id = jt.minimum(jt.maximum(category_id, 0), self.num_categories - 1).int32()
        z_noise = self.noise_encoder(noise_std.float32())
        z_category = self.category_embed(category_id)
        return jt.concat([z_noise, z_category], dim=1)

    def _stage3_mask(self, sigma_global, gate1, gate2):
        conf = (gate1.mean(dim=1) + gate2.mean(dim=1)) * 0.5
        high_noise = (sigma_global > self.stage3_noise_threshold).float32()
        low_conf = (conf < self.stage3_conf_threshold).float32()
        return jt.maximum(high_noise, low_conf).reshape(-1, 1, 1)

    def execute(self, pcl_noisy, noise_std=None, category_id=None, return_dict=False):
        cond0 = self._make_condition(pcl_noisy, noise_std=noise_std, category_id=category_id)
        disp1, gate1, local_sigma1, _ = self.stage1(pcl_noisy, cond0)
        delta1 = gate1 * disp1
        x1 = pcl_noisy + delta1

        sigma1_global = local_sigma1.mean(dim=1)
        cond1 = self._make_condition(x1, noise_std=sigma1_global.detach(), category_id=category_id)
        disp2, gate2, local_sigma2, _ = self.stage2(x1, cond1)
        delta2 = gate2 * disp2
        x2 = x1 + delta2
        sigma2_global = local_sigma2.mean(dim=1)

        result = {
            "x0": pcl_noisy,
            "x1": x1,
            "x2": x2,
            "disp1": disp1,
            "disp2": disp2,
            "delta1": delta1,
            "delta2": delta2,
            "gate1": gate1,
            "gate2": gate2,
            "sigma1_global": sigma1_global,
            "sigma2_global": sigma2_global,
            "sigma_local": local_sigma2,
            "sigma_global": sigma2_global,
        }
        final = x2
        if self.stage3 is not None:
            cond2 = self._make_condition(x2, noise_std=sigma2_global.detach(), category_id=category_id)
            disp3, gate3, local_sigma3, _ = self.stage3(x2, cond2)
            stage3_mask = self._stage3_mask(sigma2_global, gate1, gate2)
            delta3 = stage3_mask * gate3 * disp3
            final = x2 + delta3
            sigma3_global = local_sigma3.mean(dim=1)
            result.update({
                "x3": final,
                "disp3": disp3,
                "delta3": delta3,
                "gate3": gate3,
                "stage3_mask": stage3_mask,
                "sigma_local": local_sigma3,
                "sigma3_global": sigma3_global,
                "sigma_global": sigma3_global,
            })
        result["final"] = final
        result["disp"] = final - pcl_noisy
        if return_dict:
            return result
        return result["disp"]

    def denoise_langevin_dynamics(self, pcl_noisy, noise_std=None, category_id=None):
        return pcl_noisy + self(pcl_noisy, noise_std=noise_std, category_id=category_id)

    def save_npz(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **{k: v.numpy() for k, v in self.state_dict().items()})

    def load_npz(self, path):
        params = np.load(path)
        state = self.state_dict()
        is_pgd_checkpoint = "linear0_1.weight" in params and "stage1.disp_head.0.weight" not in params
        loaded = 0
        missing = []
        for name, var in state.items():
            key = name
            if key not in params and ".backbone." in name:
                key = name.split(".backbone.", 1)[1]
            if key not in params:
                missing.append(name)
                continue
            arr = params[key]
            if tuple(arr.shape) != tuple(var.shape):
                missing.append(name)
                continue
            var.assign(jt.array(arr))
            loaded += 1
        if is_pgd_checkpoint:
            loaded += self._load_pgd_output_head(params)
            self._init_pgd_warm_start_gates()
        self._load_report = {"path": str(path), "loaded": loaded, "missing": missing}
        return self._load_report

    def _assign_if_shape_matches(self, name, arr):
        state = self.state_dict()
        if name not in state or tuple(state[name].shape) != tuple(arr.shape):
            return 0
        state[name].assign(jt.array(arr))
        return 1

    def _load_pgd_output_head(self, params):
        disp_scale = 1.0 / max(float(self.max_disp), 1e-6)
        mapping = {
            "stage1.disp_head.0.weight": ("linear0_1.weight", 1.0),
            "stage1.disp_head.2.weight": ("linear0_2.weight", 1.0),
            "stage1.disp_head.2.bias": ("linear0_2.bias", 1.0),
            "stage1.disp_head.4.weight": ("linear0_3.weight", disp_scale),
            "stage1.disp_head.4.bias": ("linear0_3.bias", disp_scale),
        }
        loaded = 0
        for target, (source, scale) in mapping.items():
            if source in params:
                loaded += self._assign_if_shape_matches(target, params[source] * scale)
        return loaded

    def _set_gate_head(self, stage_name, bias_value, zero_weight=True):
        state = self.state_dict()
        weight_name = "{}.gate_head.2.weight".format(stage_name)
        bias_name = "{}.gate_head.2.bias".format(stage_name)
        if zero_weight and weight_name in state:
            state[weight_name].assign(jt.zeros_like(state[weight_name]))
        if bias_name in state:
            state[bias_name].assign(jt.full_like(state[bias_name], float(bias_value)))

    def _init_pgd_warm_start_gates(self):
        self._set_gate_head("stage1", 8.0)
        self._set_gate_head("stage2", -5.0)
        if self.stage3 is not None:
            self._set_gate_head("stage3", -5.0)

    def patch_based_denoise(
        self,
        pcl_noisy,
        patch_size=1000,
        seed_k=6,
        seed_k_alpha=10,
        patch_batch_size=None,
        noise_std=None,
        category_id=None,
        fusion="select",
    ):
        pcl = pcl_noisy if isinstance(pcl_noisy, jt.Var) else jt.array(np.asarray(pcl_noisy, dtype=np.float32))
        assert len(pcl.shape) == 2 and pcl.shape[1] == 3
        n = pcl.shape[0]
        num_patches = max(1, int(seed_k * n / patch_size))
        seed = farthest_point_sampling_jt(pcl.unsqueeze(0), num_patches)[0]
        dists, idx, patches = knn_points(seed.unsqueeze(0), pcl.unsqueeze(0), k=patch_size, return_nn=True)
        patch_dists = dists[0]
        point_idxs_jt = idx[0].int64()
        point_idxs = point_idxs_jt.numpy().astype(np.int64)
        patches_centered = patches[0] - seed[:, None, :]
        denom = jt.maximum(patch_dists[:, -1:], jt.array(1e-12, dtype=jt.float32))
        patch_dists_norm = patch_dists / denom

        patch_step = int(n / (seed_k_alpha * patch_size))
        if patch_batch_size is not None:
            patch_step = max(patch_step, int(patch_batch_size))
        if patch_step <= 0:
            raise ValueError("Seed_k_alpha needs to be decreased to increase patch_step")

        if fusion == "select":
            all_dists = jt.full((num_patches, n), 1e10, dtype=jt.float32).scatter(1, point_idxs_jt, patch_dists_norm)
            best_patch = jt.argmax(-all_dists, dim=0)[0].int64()
            patches_denoised = []
            for i in range(0, num_patches, patch_step):
                curr = patches_centered[i:i + patch_step]
                out = self(curr, noise_std=noise_std, category_id=category_id, return_dict=True)
                patches_denoised.append(out["final"])
            patches_denoised = jt.concat(patches_denoised, dim=0) + seed[:, None, :]
            local_ids = jt.arange(patch_size).reshape(1, patch_size).broadcast((num_patches, patch_size)).int64()
            local_for_point = jt.zeros((num_patches, n), dtype=jt.int64).scatter(1, point_idxs_jt, local_ids)
            point_ids = jt.arange(n).int64()
            selected_local = local_for_point.reshape(-1)[best_patch * n + point_ids]
            out = patches_denoised.reshape(num_patches * patch_size, 3)[best_patch * patch_size + selected_local, :]
            return out.float32()
        if fusion != "weighted":
            raise ValueError("unsupported patch fusion: {}".format(fusion))

        center_weight = (1.0 - patch_dists_norm).numpy().astype(np.float32)
        center_weight = np.maximum(center_weight, 0.0) ** 2
        disp_sum = np.zeros((n, 3), dtype=np.float32)
        weight_sum = np.zeros((n, 1), dtype=np.float32)
        for i in range(0, num_patches, patch_step):
            curr = patches_centered[i:i + patch_step]
            out = self(curr, noise_std=noise_std, category_id=category_id, return_dict=True)
            disp = out["disp"].numpy().astype(np.float32)
            move_gate = out["gate2"].numpy().astype(np.float32)
            conf = np.clip(1.0 - move_gate, 0.05, 1.0)
            weight = center_weight[i:i + patch_step, :, None] * conf
            idx_np = point_idxs[i:i + patch_step]
            for bi in range(disp.shape[0]):
                np.add.at(disp_sum, idx_np[bi], disp[bi] * weight[bi])
                np.add.at(weight_sum, idx_np[bi], weight[bi])
        weight_sum = np.maximum(weight_sum, 1e-8)
        out = pcl.numpy().astype(np.float32) + disp_sum / weight_sum
        return jt.array(out).float32()
