import math
import random
import numbers
import os
os.environ.setdefault("nvcc_path", "")
import numpy as np
import jittor as jt

from utils.noise import DEFAULT_NOISE_TYPES, add_jittor_noise


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data):
        for transform in self.transforms:
            data = transform(data)
        return data


class NormalizeUnitSphere:
    @staticmethod
    def normalize_np(pcl, center=None, scale=None):
        pcl = np.asarray(pcl, dtype=np.float32)
        if center is None:
            center = (pcl.max(axis=0, keepdims=True) + pcl.min(axis=0, keepdims=True)) / 2
        pcl = pcl - center
        if scale is None:
            scale = np.sqrt(np.sum(pcl * pcl, axis=1, keepdims=True)).max(axis=0, keepdims=True)
        return (pcl / scale).astype(np.float32), center.astype(np.float32), scale.astype(np.float32)

    @staticmethod
    def normalize(pcl, center=None, scale=None):
        arr = pcl.numpy() if isinstance(pcl, jt.Var) else np.asarray(pcl, dtype=np.float32)
        out, center, scale = NormalizeUnitSphere.normalize_np(arr, center=center, scale=scale)
        return jt.array(out), jt.array(center), jt.array(scale)

    def __call__(self, data):
        assert "pcl_noisy" not in data
        data["pcl_clean"], center, scale = self.normalize(data["pcl_clean"])
        data["center"] = center
        data["scale"] = scale
        return data


class AddNoise:
    def __init__(self, noise_std_min, noise_std_max, noise_types=DEFAULT_NOISE_TYPES):
        self.noise_std_min = noise_std_min
        self.noise_std_max = noise_std_max
        self.noise_types = noise_types

    def __call__(self, data):
        noise_std = random.uniform(self.noise_std_min, self.noise_std_max)
        data["pcl_noisy"], noise_type = add_jittor_noise(data["pcl_clean"], noise_std, self.noise_types)
        data["noise_std"] = noise_std
        data["noise_type"] = noise_type
        return data


class RandomScale:
    def __init__(self, scales):
        assert isinstance(scales, (tuple, list)) and len(scales) == 2
        self.scales = scales

    def __call__(self, data):
        scale = random.uniform(*self.scales)
        data["pcl_clean"] = data["pcl_clean"] * scale
        if "pcl_noisy" in data:
            data["pcl_noisy"] = data["pcl_noisy"] * scale
        return data


class RandomRotate:
    def __init__(self, degrees=180.0, axis=0):
        if isinstance(degrees, numbers.Number):
            degrees = (-abs(degrees), abs(degrees))
        assert isinstance(degrees, (tuple, list)) and len(degrees) == 2
        self.degrees = degrees
        self.axis = axis

    def __call__(self, data):
        degree = math.pi * random.uniform(*self.degrees) / 180.0
        sin, cos = math.sin(degree), math.cos(degree)
        if self.axis == 0:
            matrix = [[1, 0, 0], [0, cos, sin], [0, -sin, cos]]
        elif self.axis == 1:
            matrix = [[cos, 0, -sin], [0, 1, 0], [sin, 0, cos]]
        else:
            matrix = [[cos, sin, 0], [-sin, cos, 0], [0, 0, 1]]
        matrix = jt.array(np.asarray(matrix, dtype=np.float32))
        data["pcl_clean"] = jt.matmul(data["pcl_clean"], matrix)
        if "pcl_noisy" in data:
            data["pcl_noisy"] = jt.matmul(data["pcl_noisy"], matrix)
        return data


def standard_train_transforms(noise_std_min, noise_std_max, rotate=True, scale_d=0, noise_types=DEFAULT_NOISE_TYPES):
    transforms = [
        NormalizeUnitSphere(),
        AddNoise(noise_std_min=noise_std_min, noise_std_max=noise_std_max, noise_types=noise_types),
        RandomScale([1 - scale_d, 1 + scale_d]),
    ]
    if rotate:
        transforms += [RandomRotate(axis=0), RandomRotate(axis=1), RandomRotate(axis=2)]
    return Compose(transforms)
