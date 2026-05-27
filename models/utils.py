import numpy as np
import os
os.environ.setdefault("nvcc_path", "")
import jittor as jt


def normalize_sphere(pc, radius=1.0):
    p_max = pc.max(dim=-2, keepdims=True)
    p_min = pc.min(dim=-2, keepdims=True)
    center = (p_max + p_min) / 2
    pc = pc - center
    scale = jt.sqrt(jt.sum(pc ** 2, dim=-1, keepdims=True)).max(dim=-2, keepdims=True) / radius
    return pc / scale, center, scale


def normalize_pcl(pc, center, scale):
    return (pc - center) / scale


def denormalize_pcl(pc, center, scale):
    return pc * scale + center


def knn_points(p1, p2, k=1, return_nn=False):
    p1 = p1 if isinstance(p1, jt.Var) else jt.array(np.asarray(p1, dtype=np.float32))
    p2 = p2 if isinstance(p2, jt.Var) else jt.array(np.asarray(p2, dtype=np.float32))
    kk = min(k, p2.shape[1])
    all_d = []
    all_idx = []
    all_nn = []
    for b in range(p1.shape[0]):
        diff = p1[b:b + 1].transpose(0, 1) - p2[b:b + 1]
        dist = jt.sum(diff * diff, dim=-1)
        vals, idx = jt.topk(-dist, kk, dim=1)
        vals = -vals
        if kk < k:
            pad_idx = idx[:, -1:].broadcast((idx.shape[0], k - kk))
            pad_vals = vals[:, -1:].broadcast((vals.shape[0], k - kk))
            idx = jt.concat([idx, pad_idx], dim=1)
            vals = jt.concat([vals, pad_vals], dim=1)
        all_d.append(vals.unsqueeze(0))
        all_idx.append(idx.int64().unsqueeze(0))
        if return_nn:
            nn = p2[b][idx.reshape(-1).int64(), :].reshape(idx.shape[0], idx.shape[1], p2.shape[-1])
            all_nn.append(nn.unsqueeze(0))
    d = jt.concat(all_d, dim=0)
    idx = jt.concat(all_idx, dim=0)
    if return_nn:
        return d, idx, jt.concat(all_nn, dim=0)
    return d, idx


def knn_points_np(p1, p2, k=1, return_nn=False):
    d, idx, *rest = knn_points(p1, p2, k=k, return_nn=return_nn)
    d_np = d.numpy().astype(np.float32)
    idx_np = idx.numpy().astype(np.int64)
    if return_nn:
        return d_np, idx_np, rest[0].numpy().astype(np.float32)
    return d_np, idx_np


def _farthest_point_sampling_one(pts, num_pnts):
    n = pts.shape[0]
    m = min(int(num_pnts), n)
    selected = []
    dist = jt.full((n,), 1e10, dtype=jt.float32)
    farthest = 0
    for _ in range(m):
        selected.append(farthest)
        delta = pts - pts[farthest:farthest + 1]
        dist = jt.minimum(dist, jt.sum(delta * delta, dim=1))
        farthest = int(jt.argmax(dist, dim=0)[0].item())
    return jt.array(np.asarray(selected, dtype=np.int64)).int64()


def farthest_point_sampling_jt(pcls, num_pnts, return_index=False):
    pcls = pcls if isinstance(pcls, jt.Var) else jt.array(np.asarray(pcls, dtype=np.float32))
    sampled = []
    indices = []
    for b in range(pcls.shape[0]):
        idx = _farthest_point_sampling_one(pcls[b], num_pnts)
        sampled.append(pcls[b][idx, :].unsqueeze(0))
        indices.append(idx.unsqueeze(0))
    sampled = jt.concat(sampled, dim=0)
    indices = jt.concat(indices, dim=0)
    if return_index:
        return sampled, indices
    return sampled


def knn_points_np_legacy(p1, p2, k=1, return_nn=False):
    from scipy.spatial import cKDTree
    kk = min(k, p2.shape[1])
    all_idx = []
    all_d = []
    for b in range(p1.shape[0]):
        d, idx = cKDTree(p2[b]).query(p1[b], k=kk)
        if kk == 1:
            d = d[:, None]
            idx = idx[:, None]
        all_idx.append(idx.astype(np.int64))
        all_d.append((d * d).astype(np.float32))
    idx = np.stack(all_idx, axis=0)
    d = np.stack(all_d, axis=0)
    if kk < k:
        idx = np.concatenate([idx, np.repeat(idx[:, :, -1:], k - kk, axis=2)], axis=2)
        d = np.concatenate([d, np.repeat(d[:, :, -1:], k - kk, axis=2)], axis=2)
    if return_nn:
        nn = np.stack([p2[b][idx[b]] for b in range(p1.shape[0])], axis=0)
        return d.astype(np.float32), idx.astype(np.int64), nn.astype(np.float32)
    return d.astype(np.float32), idx.astype(np.int64)


def chamfer_distance_unit_sphere(gen, ref, batch_reduction="mean", point_reduction="mean"):
    gen_np = gen.numpy() if isinstance(gen, jt.Var) else np.asarray(gen)
    ref_np = ref.numpy() if isinstance(ref, jt.Var) else np.asarray(ref)
    ref_jt, center, scale = normalize_sphere(jt.array(ref_np.astype(np.float32)))
    gen_jt = normalize_pcl(jt.array(gen_np.astype(np.float32)), center, scale)
    gen_np = gen_jt.numpy()
    ref_np = ref_jt.numpy()
    d1, _ = knn_points_np(gen_np, ref_np, k=1)
    d2, _ = knn_points_np(ref_np, gen_np, k=1)
    cd = d1[:, :, 0].mean(axis=1) + d2[:, :, 0].mean(axis=1)
    if batch_reduction == "mean":
        return jt.array(cd.mean()), None
    return jt.array(cd), None


def farthest_point_sampling(pcls, num_pnts):
    sampled, idx = farthest_point_sampling_jt(pcls, num_pnts, return_index=True)
    return sampled.numpy().astype(np.float32), [v for v in idx.numpy().astype(np.int64)]
