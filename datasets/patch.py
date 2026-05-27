import random
import numpy as np

from models.utils import knn_points_np


def _to_numpy(x):
    if hasattr(x, "numpy"):
        return x.numpy()
    return np.asarray(x, dtype=np.float32)


def make_patches_for_pcl_pair(pcl_A, pcl_B, patch_size, num_patches, ratio):
    pcl_A = _to_numpy(pcl_A).astype(np.float32)
    pcl_B = _to_numpy(pcl_B).astype(np.float32)
    n = pcl_A.shape[0]
    seed_idx = np.random.permutation(n)[:num_patches]
    seed_pnts = pcl_A[seed_idx][None, :, :]
    _, idx_A, pat_A = knn_points_np(seed_pnts, pcl_A[None, :, :], k=patch_size, return_nn=True)
    idx_A = idx_A[0]
    pat_A = pat_A[0]
    pat_B = pcl_B[idx_A]
    return pat_A.astype(np.float32), pat_B.astype(np.float32), seed_pnts.astype(np.float32), seed_idx


class PairedPatchDataset:
    def __init__(self, datasets, split="train", patch_size=1000, num_patches=1000, patch_ratio=1.0, on_the_fly=True, transform=None):
        self.datasets = datasets
        self.split = split
        self.len_datasets = sum(len(dset) for dset in datasets)
        self.patch_ratio = patch_ratio
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.transform = transform

    def __len__(self):
        return self.len_datasets * self.num_patches

    def __getitem__(self, idx):
        pcl_dset = random.choice(self.datasets)
        pcl_data = pcl_dset[idx % len(pcl_dset)]
        pat_noisy, pat_clean, seed_pts, seed_idx = make_patches_for_pcl_pair(
            pcl_data["pcl_noisy"],
            pcl_data["pcl_clean"],
            patch_size=self.patch_size,
            num_patches=1,
            ratio=self.patch_ratio,
        )
        data = {
            "pcl_noisy": pat_noisy[0],
            "pcl_clean": pat_clean[0],
            "seed_pnts": seed_pts[0],
            "pcl_std": pcl_data.get("noise_std", 0.015),
        }
        if self.transform is not None:
            data = self.transform(data)
        return data
