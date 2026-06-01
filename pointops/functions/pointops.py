import numpy as np
import os
os.environ.setdefault("nvcc_path", "")
import jittor as jt


def _to_numpy(x):
    if isinstance(x, jt.Var):
        return x.numpy()
    return np.asarray(x)


def _offset_ranges(offset):
    off = _to_numpy(offset).astype(np.int64).reshape(-1)
    start = 0
    for end in off:
        yield start, int(end)
        start = int(end)


def furthestsampling(xyz, offset, new_offset):
    new_off = _to_numpy(new_offset).astype(np.int64).reshape(-1)
    out = []
    prev_new = 0
    for (start, end), curr_new in zip(_offset_ranges(offset), new_off):
        pts = xyz[start:end]
        m = int(curr_new - prev_new)
        prev_new = int(curr_new)
        if m <= 0:
            continue
        if pts.shape[0] == 0:
            continue
        chosen = []
        dist = jt.full((pts.shape[0],), 1e10, dtype=jt.float32)
        farthest = 0
        for _ in range(m):
            chosen.append(start + farthest)
            delta = pts - pts[farthest:farthest + 1]
            dist = jt.minimum(dist, jt.sum(delta * delta, dim=1))
            farthest = int(jt.argmax(dist, dim=0)[0].item())
        out.append(jt.array(np.asarray(chosen, dtype=np.int32)))
    if not out:
        return jt.array(np.zeros((0,), dtype=np.int32))
    return jt.concat(out, dim=0).int32()


def knnquery(nsample, xyz, new_xyz, offset, new_offset):
    if new_xyz is None:
        new_xyz = xyz
    idx_all = []
    dist_all = []
    for (start, end), (new_start, new_end) in zip(_offset_ranges(offset), _offset_ranges(new_offset)):
        ref = xyz[start:end]
        query = new_xyz[new_start:new_end]
        k = min(nsample, ref.shape[0])
        diff = query[:, None, :] - ref[None, :, :]
        dist = jt.sum(diff * diff, dim=-1)
        vals, idx_local = jt.topk(-dist, k, dim=1)
        vals = jt.sqrt(-vals + 1e-12)
        if k < nsample:
            pad = idx_local[:, -1:].broadcast((idx_local.shape[0], nsample - k))
            pad_d = vals[:, -1:].broadcast((vals.shape[0], nsample - k))
            idx_local = jt.concat([idx_local, pad], dim=1)
            vals = jt.concat([vals, pad_d], dim=1)
        idx_all.append((idx_local + start).int32())
        dist_all.append(vals.float32())
    return jt.concat(idx_all, dim=0), jt.concat(dist_all, dim=0)


def queryandgroup(nsample, xyz, new_xyz, feat, idx, offset, new_offset, use_xyz=True, return_index=False):
    if new_xyz is None:
        new_xyz = xyz
    if idx is None:
        idx, _ = knnquery(nsample, xyz, new_xyz, offset, new_offset)
    idx = idx.int64()
    grouped_xyz = xyz[idx.reshape(-1), :].reshape(idx.shape[0], idx.shape[1], xyz.shape[1]) - new_xyz[:, None, :]
    grouped_feat = feat[idx.reshape(-1), :].reshape(idx.shape[0], idx.shape[1], feat.shape[1])
    if use_xyz:
        grouped_feat = jt.concat([grouped_xyz, grouped_feat], dim=-1)
    out = grouped_feat
    if return_index:
        return out, idx.reshape(-1).int32()
    return out


def interpolation(xyz, new_xyz, feat, offset, new_offset, k=3):
    idx, dist = knnquery(k, xyz, new_xyz, offset, new_offset)
    idx = idx.int64()
    recip = 1.0 / (dist + 1e-8)
    weight = recip / jt.sum(recip, dim=1, keepdims=True)
    gathered = feat[idx.reshape(-1), :].reshape(idx.shape[0], idx.shape[1], feat.shape[1])
    return jt.sum(gathered * weight[:, :, None], dim=1)
