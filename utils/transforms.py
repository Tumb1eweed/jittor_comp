import math
import random
import numbers
import torch
from torchvision.transforms import Compose


class NormalizeUnitSphere(object):

    def __init__(self):
        super().__init__()

    @staticmethod
    def normalize(pcl, center=None, scale=None):
        """
        Args:
            pcl:  The point cloud to be normalized, (N, 3)
        """
        if center is None:
            p_max = pcl.max(dim=0, keepdim=True)[0]
            p_min = pcl.min(dim=0, keepdim=True)[0]
            center = (p_max + p_min) / 2    # (1, 3)
        pcl = pcl - center
        if scale is None:
            scale = (pcl ** 2).sum(dim=1, keepdim=True).sqrt().max(dim=0, keepdim=True)[0]  # (1, 1)
        pcl = pcl / scale
        return pcl, center, scale

    def __call__(self, data):
        assert 'pcl_noisy' not in data, 'Point clouds must be normalized before applying noise perturbation.'
        data['pcl_clean'], center, scale = self.normalize(data['pcl_clean'])
        data['center'] = center
        data['scale'] = scale
        return data


class AddNoise(object):

    def __init__(self, noise_std_min, noise_std_max):
        super().__init__()
        self.noise_std_min = noise_std_min
        self.noise_std_max = noise_std_max

    def __call__(self, data):
        noise_std = random.uniform(self.noise_std_min, self.noise_std_max)
        data['pcl_noisy'] = data['pcl_clean'] + torch.randn_like(data['pcl_clean']) * noise_std
        data['noise_std'] = noise_std
        return data


 


class RandomScale(object):

    def __init__(self, scales):
        assert isinstance(scales, (tuple, list)) and len(scales) == 2
        self.scales = scales

    def __call__(self, data):
        scale = random.uniform(*self.scales)
        data['pcl_clean'] = data['pcl_clean'] * scale
        if 'pcl_noisy' in data:
            data['pcl_noisy'] = data['pcl_noisy'] * scale
        return data


class RandomRotate(object):

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
        matrix = torch.tensor(matrix)

        data['pcl_clean'] = torch.matmul(data['pcl_clean'], matrix)
        if 'pcl_noisy' in data:
            data['pcl_noisy'] = torch.matmul(data['pcl_noisy'], matrix)

        return data


def standard_train_transforms(noise_std_min, noise_std_max, rotate=True, scale_d=0):
    transforms = [
        NormalizeUnitSphere(),
        AddNoise(noise_std_min=noise_std_min, noise_std_max=noise_std_max),
        RandomScale([1 - scale_d, 1 + scale_d]),
    ]
    if rotate:
        transforms += [
            RandomRotate(axis=0),
            RandomRotate(axis=1),
            RandomRotate(axis=2),
        ]
    return Compose(transforms)

