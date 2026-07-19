import os
from pathlib import Path

os.environ.setdefault("nvcc_path", "")
import numpy as np
import jittor as jt
from jittor import nn

from models.feature import FeatureExtraction
from models.utils import farthest_point_sampling_jt, knn_points


class PGDModel(nn.Module):
    def __init__(self, args=None):
        super().__init__()
        self.args = args
        self.feature_nets = FeatureExtraction(d_in=0, use_codebook=True)
        self.two_stage = bool(getattr(args, "pgd_two_stage", False) if args is not None else False)
        self.second_stage_scale = float(getattr(args, "pgd_second_stage_scale", 1.0) if args is not None else 1.0)
        self.use_refine_gate = bool(getattr(args, "pgd_use_refine_gate", False) if args is not None else False)
        self.refine_gate_scale = float(getattr(args, "pgd_refine_gate_scale", 0.25) if args is not None else 0.25)
        self.detach_second_stage_backbone = bool(
            getattr(args, "pgd_train_detach_second_stage_backbone", False) if args is not None else False
        )
        if self.use_refine_gate:
            self.refine_gate_fc1 = nn.Linear(6, 32)
            self.refine_gate_fc2 = nn.Linear(32, 1)
            self.refine_gate_fc2.weight.assign(jt.zeros_like(self.refine_gate_fc2.weight))
            if self.refine_gate_fc2.bias is not None:
                self.refine_gate_fc2.bias.assign(jt.zeros_like(self.refine_gate_fc2.bias))

    @classmethod
    def load_from_npz(cls, path, args=None):
        model = cls(args=args)
        model.load_npz(path)
        model.eval()
        return model

    def load_npz(self, path):
        params = np.load(path)
        state = self.state_dict()
        loaded = 0
        missing = []
        for name, var in state.items():
            candidates = [name]
            if name.startswith("feature_nets."):
                candidates.append(name[len("feature_nets."):])
            source_name = None
            for candidate in candidates:
                if candidate in params:
                    source_name = candidate
                    break
            if source_name is None:
                missing.append(name)
                continue
            arr = params[source_name]
            if tuple(arr.shape) != tuple(var.shape):
                raise ValueError(f"shape mismatch for {name}: npz {arr.shape}, model {tuple(var.shape)}")
            var.assign(jt.array(arr))
            loaded += 1
        self._load_report = {"path": str(path), "loaded": loaded, "missing": missing}
        return self._load_report

    def save_npz(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {k: v.numpy() for k, v in self.feature_nets.state_dict().items()}
        for name, var in self.state_dict().items():
            if name.startswith("feature_nets."):
                continue
            arrays[name] = var.numpy()
        np.savez_compressed(path, **arrays)

    def _refinement_gate(self, disp1, raw_disp2):
        gate = jt.ones((disp1.shape[0], disp1.shape[1], 1), dtype=jt.float32)
        if self.use_refine_gate:
            features = jt.concat([disp1, raw_disp2], dim=-1)
            raw_gate = self.refine_gate_fc2(nn.relu(self.refine_gate_fc1(features)))
            gate = gate * (1.0 + self.refine_gate_scale * jt.tanh(raw_gate))
        return gate

    def execute(self, pcl_noisy, noise_std=None, category_id=None, return_dict=False):
        b, n, _ = pcl_noisy.shape
        offset = jt.array(np.array([(i + 1) * n for i in range(b)], dtype=np.int32))
        raw_disp = self.feature_nets(pcl_noisy, None, offset)
        disp = raw_disp
        if self.two_stage:
            x1 = pcl_noisy + disp
            if self.detach_second_stage_backbone:
                with jt.no_grad():
                    raw_disp2 = self.feature_nets(x1.detach(), None, offset)
            else:
                raw_disp2 = self.feature_nets(x1, None, offset)
            refine_gate = self._refinement_gate(disp, raw_disp2)
            disp2 = raw_disp2 * refine_gate * self.second_stage_scale
            total_disp = disp + disp2
            final = pcl_noisy + total_disp
            if return_dict:
                return {
                    "disp": total_disp,
                    "raw_disp": raw_disp,
                    "raw_disp1": raw_disp,
                    "raw_disp2": raw_disp2,
                    "disp1": disp,
                    "disp2": disp2,
                    "x1": x1,
                    "final": final,
                    "refine_gate": refine_gate,
                }
            return total_disp
        if return_dict:
            return {
                "disp": disp,
                "raw_disp": raw_disp,
            }
        return disp

    def denoise_langevin_dynamics(self, pcl_noisy, noise_std=None, category_id=None):
        pred_disp = self(pcl_noisy, noise_std=noise_std, category_id=category_id)
        return pcl_noisy + pred_disp

    def patch_based_denoise(self, pcl_noisy, patch_size=1000, seed_k=5, seed_k_alpha=10, patch_batch_size=None, fusion="select", noise_std=None, category_id=None):
        pcl = pcl_noisy if isinstance(pcl_noisy, jt.Var) else jt.array(np.asarray(pcl_noisy, dtype=np.float32))
        assert len(pcl.shape) == 2 and pcl.shape[1] == 3
        n = pcl.shape[0]
        num_patches = int(seed_k * n / patch_size)
        seed = farthest_point_sampling_jt(pcl.unsqueeze(0), num_patches)[0]
        dists, idx, patches = knn_points(seed.unsqueeze(0), pcl.unsqueeze(0), k=patch_size, return_nn=True)
        patch_dists = dists[0]
        point_idxs = idx[0].int64()
        patches_centered = patches[0] - seed[:, None, :]
        denom = jt.maximum(patch_dists[:, -1:], jt.array(1e-12, dtype=jt.float32))
        patch_dists = patch_dists / denom
        all_dists = jt.full((num_patches, n), 1e10, dtype=jt.float32).scatter(1, point_idxs, patch_dists)
        best_patch = jt.argmax(-all_dists, dim=0)[0].int64()

        patches_denoised = []
        i = 0
        patch_step = int(n / (seed_k_alpha * patch_size))
        if patch_batch_size is not None:
            patch_step = max(patch_step, int(patch_batch_size))
        if patch_step <= 0:
            raise ValueError("Seed_k_alpha needs to be decreased to increase patch_step")
        if fusion != "select":
            raise ValueError("unsupported patch fusion: {}".format(fusion))
        while i < num_patches:
            curr = patches_centered[i:i + patch_step]
            den = self.denoise_langevin_dynamics(curr, noise_std=noise_std, category_id=category_id)
            patches_denoised.append(den)
            i += patch_step
        patches_denoised = jt.concat(patches_denoised, dim=0) + seed[:, None, :]
        local_ids = jt.arange(patch_size).reshape(1, patch_size).broadcast((num_patches, patch_size)).int64()
        local_for_point = jt.zeros((num_patches, n), dtype=jt.int64).scatter(1, point_idxs, local_ids)
        point_ids = jt.arange(n).int64()
        selected_local = local_for_point.reshape(-1)[best_patch * n + point_ids]
        out = patches_denoised.reshape(num_patches * patch_size, 3)[best_patch * patch_size + selected_local, :]
        return out.float32()
