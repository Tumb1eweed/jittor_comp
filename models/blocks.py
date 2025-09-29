import numpy as np
from pointops.functions import pointops
import torch.nn as nn
import torch
import torch.nn.functional as F


def block_decider(name):
    if name == 'startblock':
        return StartBlock
    if name == 'upsample':
        return Upsampling 
    if name == 'downsample':
        return Downsampling

class StartBlock(nn.Module):
    def __init__(self, d_in, d_out, nsample, stride):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_in+3, d_out),
            nn.BatchNorm1d(d_out),
            nn.LeakyReLU(0.2),
        )
    
    def forward(self, p, x, o):
        x = self.mlp(p)
        return p, x, o, None
    
class Downsampling(nn.Module):
    def __init__(self, d_in, d_out, nsample, stride):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.nsample = nsample
        self.stride = stride
        self.mre = MRE(d_in, d_out)
    def forward(self, p, x, o):
        # local feature aggregation
        x = self.mre(p, x, o)

        # farthest point sampling
        count = o[0].item() * self.stride // (self.stride + 1)
        n_o = [count * (i + 1) for i in range(o.shape[0])]
        n_o = torch.tensor(n_o, dtype=torch.int32, device='cuda')
        idx = pointops.furthestsampling(p, o, n_o)
        n_p = p[idx.long(), :]
        n_x = x[idx.long(), :]

        return n_p, n_x, n_o, idx


class RFE(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out

        self.BiMLP = nn.Sequential(
            nn.Conv1d(10, 2*d_out, 1),
            nn.BatchNorm1d(2*d_out),
            nn.ReLU(inplace=True),
            nn.Conv1d(2*d_out, d_out, 1),
        )
        
        self.score_fn = nn.Sequential(
            nn.Linear(d_out * 2, d_out * 2, bias=False), nn.Softmax(dim=-2)
        )
        self.mlp_out = nn.Sequential(
            nn.Linear(d_out * 2, d_out), nn.BatchNorm1d(d_out), nn.ReLU(inplace=True)
        )

    def forward(self, p, x):
        # positional encoding over local neighborhood
        extended_coords = p[:, 0, :].unsqueeze(1).expand(-1, 16, -1)
        neighbors = p
        dist = torch.sqrt(torch.sum((extended_coords - neighbors) ** 2, dim=2, keepdim=True))
        concat = torch.cat([extended_coords, neighbors, extended_coords - neighbors, dist], dim=-1)
        p_c = self.BiMLP(concat.permute(0, 2, 1)).permute(0, 2, 1).contiguous()

        # attention-weighted fusion with features
        p_x = torch.cat([p_c, x], dim=-1)
        scores = self.score_fn(p_x)
        features = torch.sum(scores * p_x, dim=1, keepdim=True)
        features = self.mlp_out(features.squeeze())
        return features


class MRE(nn.Module):
    def __init__(self, d_in, d_out,):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.mlp0 = nn.Sequential(
            nn.Linear(d_in, d_out // 2),
            nn.BatchNorm1d(d_out // 2),
            nn.ReLU(inplace=True),
        )
        self.mlp1 = nn.Sequential(
            nn.Linear(d_out , d_out), nn.BatchNorm1d(d_out), nn.ReLU(inplace=True)
        )
        self.mlp01 = nn.Sequential(
            nn.Linear(d_in, d_out), nn.BatchNorm1d(d_out), nn.ReLU(inplace=True)
        )
        self.Rfe_1 = RFE(d_out // 2, d_out // 2)
        self.Rfe_2 = RFE(d_out // 2, d_out // 2)

    def forward(self, p, x, o):
        x_start = x
        x = self.mlp0(x)
        xr, _ = pointops.queryandgroup(16, p, p, x, None, offset=o, new_offset=o, use_xyz=True, return_index=True)
        x = self.Rfe_1(xr[:, :, :3], xr[:, :, 3:])
        x_middle = x
        xr, _ = pointops.queryandgroup(16, p, p, x, None, offset=o, new_offset=o, use_xyz=True, return_index=True)
        x = self.Rfe_2(xr[:, :, :3], xr[:, :, 3:])
        x = torch.cat([x_middle, x], dim=1)
        x = self.mlp01(x_start) + self.mlp1(x)
        return x

class Upsampling(nn.Module):
    def __init__(self, d_in_sparse_fusion, d_out, nsample, stride, idx_module=None):
        super().__init__()
        d_in_sparse, d_in_dense = d_in_sparse_fusion
        self.nsample = nsample
        self.d_out = d_out
        self.stride = stride
        self.idx_module = idx_module
        self.CrossPT_func = CrossAttentionPointTransformerLayer(
            dim=d_in_sparse,
            dim_dense=d_in_dense,
            k_sample = self.stride,
            attn_mlp_hidden_mult=1,
            num_neighbors=16
        )

        self.mlp = nn.Sequential(
            nn.Linear(d_in_sparse + d_in_dense, d_out),
            nn.BatchNorm1d(d_out),
            nn.ReLU(inplace=True)
        )
        
        self.linear_upsample = nn.Linear(d_in_sparse + d_in_dense, d_in_sparse)

    def U_function(self, x1, x2_interpolated):
        # fuse dense and interpolated sparse features
        merged = torch.cat([x1, x2_interpolated], dim=-1)
        return self.linear_upsample(merged)
    
    def codebook_lookup(self, query_feats, codebook):
        diff = query_feats.unsqueeze(1) - codebook.unsqueeze(0)
        dist = torch.norm(diff, dim=-1)
        nearest_idx = torch.argmin(dist, dim=1)
        return codebook[nearest_idx]
    
    def soft_codebook_lookup(self, query_feats, codebook, temperature=0.1):
        diff = query_feats.unsqueeze(1) - codebook.unsqueeze(0)
        dist = torch.norm(diff, dim=-1)
        logits = -dist / temperature
        weights = torch.softmax(logits, dim=1)
        return torch.matmul(weights, codebook)
    
    def forward(self, p1, x1, o1, idx, p2, x2, o2, batch_size=5, codebook=None, calculate_commitment_loss_for_block=False):
        # p1/x1/o1: dense; p2/x2/o2: sparse
        num_points = p1.shape[0]//batch_size
        num_points_sparse = p2.shape[0]//batch_size

        x2_interpolated = pointops.interpolation(p2, p1, x2, o2, o1, k=8)
        U_query = self.U_function(x1, x2_interpolated)

        block_commitment_loss = torch.tensor(0.0, device=x1.device)

        if codebook is not None and isinstance(codebook, CodebookModule):
            x1, block_commitment_loss = codebook(x1, calculate_commitment_loss=calculate_commitment_loss_for_block)
        
        x1_enhance = self.CrossPT_func(
            U_query.view(batch_size, num_points, -1),
            x2.view(batch_size, num_points_sparse, -1),
            x2.view(batch_size, num_points_sparse, -1),
            p1.view(batch_size, num_points, -1),
            idx,
            x1.view(batch_size, num_points, -1)
        ).view(batch_size*num_points, -1)
        
        x = self.mlp(torch.cat([x1_enhance, x1], dim=-1))
        
        return p1, x, o1, block_commitment_loss


class CrossAttentionPointTransformerLayer(nn.Module):
    def __init__(self, dim, dim_dense, k_sample, attn_mlp_hidden_mult=4, num_neighbors=None):
        super().__init__()
        self.dim_dense = dim_dense
        self.num_neighbors = num_neighbors
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim * (k_sample + 1) // k_sample, bias=False)
        self.to_v = nn.Linear(dim, dim * (k_sample + 1) // k_sample, bias=False)
        
        # conditional modulation via FiLM
        self.film_layer = FiLMLayer(dim_dense, dim)
        
        self.attn_bimlp = nn.Sequential(
            nn.Linear(dim, dim * attn_mlp_hidden_mult),
            nn.BatchNorm1d(dim * attn_mlp_hidden_mult),
            nn.ReLU(inplace=True),
            nn.Linear(dim * attn_mlp_hidden_mult, dim),
        )

    def forward(self, x_e, x_r, x_d, pos, idx, quantized_features=None):
        n = x_e.shape[1]
        q = self.to_q(x_e)
        k = self.to_k(x_r).view(q.shape[0], q.shape[1], -1)
        v = self.to_v(x_d).view(q.shape[0], q.shape[1], -1)
        
        if quantized_features is not None:
            gamma, beta = self.film_layer(quantized_features)
            k = gamma * k + beta

        if self.num_neighbors is not None and self.num_neighbors < n:
            rel_pos = pos[:, :, None, :] - pos[:, None, :, :]
            rel_dist = rel_pos.norm(dim = -1)
            dist, indices = rel_dist.topk(self.num_neighbors, largest = False)

            v = batched_index_select(v, indices, dim = 1)
            k = batched_index_select(k, indices, dim = 1)
            qk_rel = q[:,:,None] - k
            x_e = batched_index_select(x_e, indices, dim = 1)
        else:
            qk_rel = q[:,:,None] - k[:,:,None,:]
            
        v = v + x_e
        
        B, N, neigh_num, C = qk_rel.shape
        sim = self.attn_bimlp((qk_rel + x_e).view(B*N*neigh_num, C)).view(B, N, neigh_num, C)
        attn = sim.softmax(dim=-2)
        agg = torch.einsum('bmnf,bmnf->bmf', attn, v)
        return agg


def batched_index_select(values, indices, dim = 1):
    value_dims = values.shape[(dim + 1):]
    values_shape, indices_shape = map(lambda t: list(t.shape), (values, indices))
    indices = indices[(..., *((None,) * len(value_dims)))]
    indices = indices.expand(*((-1,) * len(indices_shape)), *value_dims)
    value_expand_len = len(indices_shape) - (dim + 1)
    values = values[(*((slice(None),) * dim), *((None,) * value_expand_len), ...)]

    value_expand_shape = [-1] * len(values.shape)
    expand_slice = slice(dim, (dim + value_expand_len))
    value_expand_shape[expand_slice] = indices.shape[expand_slice]
    values = values.expand(*value_expand_shape)

    dim += value_expand_len
    return values.gather(dim, indices)

class CodebookModule(nn.Module):
    def __init__(self, feature_dim=48, codebook_size=128, momentum=0.99, 
                 commitment_cost=0, use_ema=True, temperature=0.1):
        super().__init__()
        self.feature_dim = feature_dim
        self.codebook_size = codebook_size
        self.momentum = momentum
        self.commitment_cost = commitment_cost
        self.use_ema = use_ema
        self.temperature = temperature
        
        codebook = torch.randn(codebook_size, feature_dim)
        codebook = F.normalize(codebook, dim=1)
        self.register_buffer('codebook', codebook)
        
        self.register_buffer('cluster_size', torch.zeros(codebook_size))
        self.register_buffer('cluster_sum', torch.zeros(codebook_size, feature_dim))
        
        self.register_buffer('usage_count', torch.zeros(codebook_size))
        self.register_buffer('last_usage', torch.zeros(codebook_size))
        self.register_buffer('step_counter', torch.zeros(1, dtype=torch.long))
    
    def get_codebook_features(self, indices=None, features=None):
        if indices is not None:
            return self.codebook[indices]
        elif features is not None:
            features_norm = F.normalize(features, dim=1)
            codebook_norm = F.normalize(self.codebook, dim=1)
            similarity = torch.matmul(features_norm, codebook_norm.t())
            indices = torch.argmax(similarity, dim=1)
            return self.codebook[indices], indices
        else:
            return self.codebook
    
    def soft_quantize(self, features):
        features_norm = F.normalize(features, dim=1)
        codebook_norm = F.normalize(self.codebook, dim=1)
        similarity = torch.matmul(features_norm, codebook_norm.t())
        logits = similarity / self.temperature
        weights = F.softmax(logits, dim=1)
        quantized = torch.matmul(weights, self.codebook)
        indices = torch.argmax(similarity, dim=1)
        return quantized, indices, weights
    
    def update_codebook(self, features, indices=None, weights=None):
        if not self.training or not self.use_ema:
            return
        with torch.no_grad():
            self.step_counter += 1
            if weights is not None:
                new_cluster_size = weights.sum(dim=0)
                new_cluster_sum = torch.matmul(weights.t(), features)
            else:
                if indices is None:
                    features_norm = F.normalize(features, dim=1)
                    codebook_norm = F.normalize(self.codebook, dim=1)
                    similarity = torch.matmul(features_norm, codebook_norm.t())
                    indices = torch.argmax(similarity, dim=1)
                encodings = F.one_hot(indices, self.codebook_size).float()
                new_cluster_size = encodings.sum(dim=0)
                new_cluster_sum = torch.matmul(encodings.t(), features)
            self.usage_count.data += new_cluster_size
            self.last_usage.masked_fill_(new_cluster_size > 0, self.step_counter.item())
            self.cluster_size.data = self.cluster_size * self.momentum + new_cluster_size * (1 - self.momentum)
            self.cluster_sum.data = self.cluster_sum * self.momentum + new_cluster_sum * (1 - self.momentum)
            updated_codebook = self.cluster_sum / (self.cluster_size.unsqueeze(1) + 1e-5)
            usage_mask = (self.cluster_size > 1e-5).float().unsqueeze(1)
            self.codebook.data = updated_codebook * usage_mask + self.codebook.data * (1 - usage_mask)
            if self.step_counter.item() % 1000 == 0:
                self._reset_dead_codebook_vectors()
    
    def _reset_dead_codebook_vectors(self, threshold=5000):
        with torch.no_grad():
            current_step = self.step_counter.item()
            unused_steps = current_step - self.last_usage
            dead_indices = torch.where(unused_steps > threshold)[0]
            if len(dead_indices) == 0:
                return
            _, most_used_indices = torch.topk(self.usage_count, len(dead_indices))
            for i, dead_idx in enumerate(dead_indices):
                most_used_idx = most_used_indices[i]
                new_vector = self.codebook[most_used_idx] + torch.randn_like(self.codebook[most_used_idx]) * 0.1
                new_vector = F.normalize(new_vector, dim=0)
                self.codebook.data[dead_idx] = new_vector
                self.cluster_size.data[dead_idx] = 0
                self.cluster_sum.data[dead_idx] = 0
                self.usage_count.data[dead_idx] = 0
                self.last_usage.data[dead_idx] = current_step
    
    def forward(self, features, calculate_commitment_loss=False):
        quantized, _, weights = self.soft_quantize(features)
        commitment_loss_value = torch.tensor(0.0, device=features.device)
        if self.training:
            self.update_codebook(features, weights=weights)
        if self.training and calculate_commitment_loss:
            commitment_loss_value = F.mse_loss(quantized.detach(), features) * self.commitment_cost
            output_features = features + (quantized - features).detach()
        else:
            quantized_res = quantized - features
            output_features = features + quantized_res.detach()
        return output_features, commitment_loss_value

class FiLMLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.film = nn.Sequential(
            nn.Linear(input_dim, input_dim*2),
            nn.LeakyReLU(0.2),
            nn.Linear(input_dim*2, output_dim*2)
        )
        
    def forward(self, x):
        params = self.film(x)
        gamma, beta = torch.chunk(params, 2, dim=-1)
        return gamma, beta

