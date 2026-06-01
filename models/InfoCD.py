import os

os.environ.setdefault("nvcc_path", "")

import jittor as jt


def pairwise_squared_distance(p1, p2):
    return jt.sum((p1[:, :, None, :] - p2[:, None, :, :]) ** 2, dim=-1)


def calc_cd_like_InfoV2(p1, p2):
    dist = pairwise_squared_distance(p1, p2)
    dist1 = jt.maximum(dist.min(dim=2), 1e-9)
    dist2 = jt.maximum(dist.min(dim=1), 1e-9)
    d1 = jt.sqrt(dist1)
    d2 = jt.sqrt(dist2)

    weight1 = jt.exp(-0.5 * d1)
    weight2 = jt.exp(-0.5 * d2)
    norm1 = jt.sum(weight1 + 1e-7, dim=-1).unsqueeze(-1)
    norm2 = jt.sum(weight2 + 1e-7, dim=-1).unsqueeze(-1)
    distances1 = -jt.log(weight1 / (norm1 ** 1e-7))
    distances2 = -jt.log(weight2 / (norm2 ** 1e-7))
    return (jt.sum(distances1) + jt.sum(distances2)) / (2 * p1.shape[0])
