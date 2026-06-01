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
        self.feature_nets = FeatureExtraction()

    @classmethod
    def load_from_npz(cls, path, args=None):
        model = cls(args=args)
        model.load_npz(path)
        model.eval()
        return model

    def load_npz(self, path):
        params = np.load(path)
        state = self.feature_nets.state_dict()
        loaded = 0
        missing = []
        for name, var in state.items():
            if name not in params:
                missing.append(name)
                continue
            arr = params[name]
            if tuple(arr.shape) != tuple(var.shape):
                raise ValueError(f"shape mismatch for {name}: npz {arr.shape}, model {tuple(var.shape)}")
            var.assign(jt.array(arr))
            loaded += 1
        self._load_report = {"path": str(path), "loaded": loaded, "missing": missing}
        return self._load_report

    def save_npz(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **{k: v.numpy() for k, v in self.feature_nets.state_dict().items()})

    def execute(self, pcl_noisy):
        b, n, _ = pcl_noisy.shape
        feat = None
        offset = jt.array(np.array([(i + 1) * n for i in range(b)], dtype=np.int32))
        return self.feature_nets(pcl_noisy, feat, offset)

    def denoise_langevin_dynamics(self, pcl_noisy):
        pred_disp = self(pcl_noisy)
        return pcl_noisy + pred_disp

    def patch_based_denoise(self, pcl_noisy, patch_size=1000, seed_k=5, seed_k_alpha=10, patch_batch_size=None):
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
        while i < num_patches:
            curr = patches_centered[i:i + patch_step]
            den = self.denoise_langevin_dynamics(curr)
            patches_denoised.append(den)
            i += patch_step
        patches_denoised = jt.concat(patches_denoised, dim=0) + seed[:, None, :]
        local_ids = jt.arange(patch_size).reshape(1, patch_size).broadcast((num_patches, patch_size)).int64()
        local_for_point = jt.zeros((num_patches, n), dtype=jt.int64).scatter(1, point_idxs, local_ids)
        point_ids = jt.arange(n).int64()
        selected_local = local_for_point.reshape(-1)[best_patch * n + point_ids]
        out = patches_denoised.reshape(num_patches * patch_size, 3)[best_patch * patch_size + selected_local, :]
        return out.float32()
