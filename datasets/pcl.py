import os
import numpy as np
from tqdm.auto import tqdm


class PointCloudDataset:
    def __init__(self, root, dataset, split, resolution, transform=None, max_shapes=0):
        self.transform = transform
        self.pointclouds = []
        self.pointcloud_names = []
        self.resolution = resolution
        self.split = split
        self.pcl_dir = os.path.join(root, dataset, "pointclouds", split, resolution)
        for fn in tqdm(os.listdir(self.pcl_dir), desc="Loading"):
            if not fn.endswith(".xyz"):
                continue
            pcl_path = os.path.join(self.pcl_dir, fn)
            if not os.path.exists(pcl_path):
                raise FileNotFoundError(f"File not found: {pcl_path}")
            self.pointclouds.append(np.loadtxt(pcl_path, dtype=np.float32))
            self.pointcloud_names.append(fn[:-4])
            if max_shapes and len(self.pointclouds) >= max_shapes:
                break

    def __len__(self):
        return len(self.pointclouds)

    def __getitem__(self, idx):
        data = {"pcl_clean": self.pointclouds[idx].copy(), "name": self.pointcloud_names[idx]}
        if self.transform is not None:
            data = self.transform(data)
        return data
