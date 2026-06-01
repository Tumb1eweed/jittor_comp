import numpy as np
import os
os.environ.setdefault("nvcc_path", "")
import jittor as jt
from jittor import nn

from pointops.functions import pointops


def block_decider(name):
    if name == "startblock":
        return StartBlock
    if name == "upsample":
        return Upsampling
    if name == "downsample":
        return Downsampling
    raise ValueError(name)


class StartBlock(nn.Module):
    def __init__(self, d_in, d_out, nsample, stride):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_in + 3, d_out),
            nn.BatchNorm1d(d_out),
            nn.LeakyReLU(0.2),
        )

    def execute(self, p, x, o):
        return p, self.mlp(p), o, None


class Downsampling(nn.Module):
    def __init__(self, d_in, d_out, nsample, stride):
        super().__init__()
        self.stride = stride
        self.mre = MRE(d_in, d_out)

    def execute(self, p, x, o):
        x = self.mre(p, x, o)
        count = int(o[0].item()) * self.stride // (self.stride + 1)
        n_o = jt.array(np.array([count * (i + 1) for i in range(o.shape[0])], dtype=np.int32))
        idx = pointops.furthestsampling(p, o, n_o)
        n_p = p[idx.int64(), :]
        n_x = x[idx.int64(), :]
        return n_p, n_x, n_o, idx


class RFE(nn.Module):
    def __init__(self, d_out):
        super().__init__()
        self.BiMLP = nn.Sequential(
            nn.Conv1d(10, 2 * d_out, 1),
            nn.BatchNorm1d(2 * d_out),
            nn.ReLU(),
            nn.Conv1d(2 * d_out, d_out, 1),
        )
        self.score_fn = nn.Sequential(nn.Linear(d_out * 2, d_out * 2, bias=False), nn.Softmax(dim=-2))
        self.mlp_out = nn.Sequential(nn.Linear(d_out * 2, d_out), nn.BatchNorm1d(d_out), nn.ReLU())

    def execute(self, p, x):
        extended = p[:, 0, :].unsqueeze(1).expand((-1, 16, -1))
        dist = jt.sqrt(jt.sum((extended - p) ** 2, dim=2, keepdims=True))
        concat = jt.concat([extended, p, extended - p, dist], dim=-1)
        p_c = self.BiMLP(concat.permute(0, 2, 1)).permute(0, 2, 1)
        p_x = jt.concat([p_c, x], dim=-1)
        scores = self.score_fn(p_x)
        features = jt.sum(scores * p_x, dim=1, keepdims=True)
        return self.mlp_out(features.squeeze(1))


class MRE(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.mlp0 = nn.Sequential(nn.Linear(d_in, d_out // 2), nn.BatchNorm1d(d_out // 2), nn.ReLU())
        self.mlp1 = nn.Sequential(nn.Linear(d_out, d_out), nn.BatchNorm1d(d_out), nn.ReLU())
        self.mlp01 = nn.Sequential(nn.Linear(d_in, d_out), nn.BatchNorm1d(d_out), nn.ReLU())
        self.Rfe_1 = RFE(d_out // 2)
        self.Rfe_2 = RFE(d_out // 2)

    def execute(self, p, x, o):
        x_start = x
        x = self.mlp0(x)
        xr, _ = pointops.queryandgroup(16, p, p, x, None, offset=o, new_offset=o, use_xyz=True, return_index=True)
        x = self.Rfe_1(xr[:, :, :3], xr[:, :, 3:])
        x_middle = x
        xr, _ = pointops.queryandgroup(16, p, p, x, None, offset=o, new_offset=o, use_xyz=True, return_index=True)
        x = self.Rfe_2(xr[:, :, :3], xr[:, :, 3:])
        x = jt.concat([x_middle, x], dim=1)
        return self.mlp01(x_start) + self.mlp1(x)


class Upsampling(nn.Module):
    def __init__(self, d_in_sparse_fusion, d_out, nsample, stride, idx_module=None):
        super().__init__()
        d_in_sparse, d_in_dense = d_in_sparse_fusion
        self.stride = stride
        self.CrossPT_func = CrossAttentionPointTransformerLayer(
            dim=d_in_sparse,
            dim_dense=d_in_dense,
            k_sample=self.stride,
            attn_mlp_hidden_mult=1,
            num_neighbors=16,
        )
        self.mlp = nn.Sequential(nn.Linear(d_in_sparse + d_in_dense, d_out), nn.BatchNorm1d(d_out), nn.ReLU())
        self.linear_upsample = nn.Linear(d_in_sparse + d_in_dense, d_in_sparse)

    def U_function(self, x1, x2_interpolated):
        return self.linear_upsample(jt.concat([x1, x2_interpolated], dim=-1))

    def execute(self, p1, x1, o1, idx, p2, x2, o2, batch_size=5, codebook=None):
        num_points = p1.shape[0] // batch_size
        num_points_sparse = p2.shape[0] // batch_size
        x2_interpolated = pointops.interpolation(p2, p1, x2, o2, o1, k=8)
        u_query = self.U_function(x1, x2_interpolated)
        if codebook is not None:
            x1 = codebook(x1)
        x1_enhance = self.CrossPT_func(
            u_query.reshape(batch_size, num_points, -1),
            x2.reshape(batch_size, num_points_sparse, -1),
            x2.reshape(batch_size, num_points_sparse, -1),
            p1.reshape(batch_size, num_points, -1),
            idx,
            x1.reshape(batch_size, num_points, -1),
        ).reshape(batch_size * num_points, -1)
        x = self.mlp(jt.concat([x1_enhance, x1], dim=-1))
        return p1, x, o1


class CrossAttentionPointTransformerLayer(nn.Module):
    def __init__(self, dim, dim_dense, k_sample, attn_mlp_hidden_mult=4, num_neighbors=None):
        super().__init__()
        self.num_neighbors = num_neighbors
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim * (k_sample + 1) // k_sample, bias=False)
        self.to_v = nn.Linear(dim, dim * (k_sample + 1) // k_sample, bias=False)
        self.film_layer = FiLMLayer(dim_dense, dim)
        self.attn_bimlp = nn.Sequential(
            nn.Linear(dim, dim * attn_mlp_hidden_mult),
            nn.BatchNorm1d(dim * attn_mlp_hidden_mult),
            nn.ReLU(),
            nn.Linear(dim * attn_mlp_hidden_mult, dim),
        )

    def execute(self, x_e, x_r, x_d, pos, idx, quantized_features=None):
        n = x_e.shape[1]
        q = self.to_q(x_e)
        k = self.to_k(x_r).reshape(q.shape[0], q.shape[1], -1)
        v = self.to_v(x_d).reshape(q.shape[0], q.shape[1], -1)
        if quantized_features is not None:
            gamma, beta = self.film_layer(quantized_features)
            k = gamma * k + beta
        if self.num_neighbors is not None and self.num_neighbors < n:
            rel = pos[:, :, None, :] - pos[:, None, :, :]
            dist = jt.norm(rel, dim=-1)
            _, indices = jt.topk(-dist, self.num_neighbors, dim=2)
            v = batched_index_select(v, indices)
            k = batched_index_select(k, indices)
            qk_rel = q[:, :, None] - k
            x_e = batched_index_select(x_e, indices)
        else:
            qk_rel = q[:, :, None] - k[:, None, :, :]
        v = v + x_e
        b, n, neigh_num, c = qk_rel.shape
        sim = self.attn_bimlp((qk_rel + x_e).reshape(b * n * neigh_num, c)).reshape(b, n, neigh_num, c)
        attn = nn.softmax(sim, dim=-2)
        return jt.sum(attn * v, dim=2)


def batched_index_select(values, indices):
    outs = []
    for b in range(values.shape[0]):
        flat = indices[b].reshape(-1).int64()
        gathered = values[b][flat, :].reshape(indices.shape[1], indices.shape[2], values.shape[-1])
        outs.append(gathered.unsqueeze(0))
    return jt.concat(outs, dim=0)


class CodebookModule(nn.Module):
    def __init__(self, feature_dim=48, codebook_size=128, temperature=0.1):
        super().__init__()
        self.feature_dim = feature_dim
        self.codebook_size = codebook_size
        self.temperature = temperature
        self.codebook = jt.randn((codebook_size, feature_dim))

    def soft_quantize(self, features):
        features_norm = features / (jt.norm(features, dim=1, keepdims=True) + 1e-8)
        codebook_norm = self.codebook / (jt.norm(self.codebook, dim=1, keepdims=True) + 1e-8)
        similarity = jt.matmul(features_norm, codebook_norm.transpose())
        weights = nn.softmax(similarity / self.temperature, dim=1)
        quantized = jt.matmul(weights, self.codebook)
        indices = jt.argmax(similarity, dim=1)
        return quantized, indices, weights

    def execute(self, features):
        quantized, _, _ = self.soft_quantize(features)
        return quantized


class FiLMLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.film = nn.Sequential(nn.Linear(input_dim, input_dim * 2), nn.LeakyReLU(0.2), nn.Linear(input_dim * 2, output_dim * 2))

    def execute(self, x):
        params = self.film(x)
        return jt.chunk(params, 2, dim=-1)
