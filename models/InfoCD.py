import jittor as jt


def calc_cd_like_InfoV2(p1, p2):
    dist = jt.sum((p1[:, :, None, :] - p2[:, None, :, :]) ** 2, dim=-1)
    dist1 = jt.maximum(dist.min(dim=2), 1e-9)
    dist2 = jt.maximum(dist.min(dim=1), 1e-9)
    d1 = jt.sqrt(dist1)
    d2 = jt.sqrt(dist2)
    distances1 = -jt.log(jt.exp(-0.5 * d1) / (jt.sum(jt.exp(-0.5 * d1) + 1e-7, dim=-1).unsqueeze(-1)) ** 1e-7)
    distances2 = -jt.log(jt.exp(-0.5 * d2) / (jt.sum(jt.exp(-0.5 * d2) + 1e-7, dim=-1).unsqueeze(-1)) ** 1e-7)
    return (jt.sum(distances1) + jt.sum(distances2)) / (2 * p1.shape[0])
