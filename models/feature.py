import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module, Linear, ModuleList
from .utils import *
from models.blocks import *
from models.blocks import CodebookModule

class FeatureExtraction(Module):
    """Encoder-decoder point feature extractor with optional per-decoder-layer codebooks."""
    def __init__(self, d_in=0, d_out=32,
                 n_cls=3, nsample=16, stride_list=[4, 3, 2, 1],
                 architecture=None):  #
        super().__init__()
        architecture = ['startblock',
                        'downsample',
                        'downsample',
                        'downsample',
                        'downsample',
                        'upsample',
                        'upsample',
                        'upsample',
                        'upsample', ]
        d_in = d_in
        d_out = d_out
        n_cls = n_cls
        nsample = nsample
        stride_list = stride_list
        stride_dim_list = [1.5, 1.5, 1.5, 1.5]
        stride = 1
        stride_idx = 0
        d_prev = d_in

        # construct encoder
        self.encoder_blocks = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        self.encoder_skip_dims = [] 

        for block_name in architecture:
            if 'downsample' in block_name:
                self.encoder_skip_dims.append(d_prev)
                stride = stride_list[stride_idx]
                d_out = int(d_out * stride_dim_list[stride_idx])
                stride_idx += 1
                self.encoder_blocks.append(
                    block_decider(block_name)(d_prev, d_out, nsample, stride)
                )
            elif 'upsample' in block_name:
                stride_idx -= 1
                stride = stride_list[stride_idx]
                skip_dim = self.encoder_skip_dims.pop()
                d_out = skip_dim
                self.decoder_blocks.append(
                    block_decider(block_name)([d_prev, skip_dim], d_out, nsample, stride)
                )
            else:
                self.encoder_blocks.append(
                    block_decider(block_name)(d_prev, d_out, nsample, stride)
                )
            d_prev = d_out

        self.linear0_1 = nn.Linear(d_out, 128, bias=False)
        self.linear0_2 = nn.Linear(128 , 64)
        self.linear0_3 = nn.Linear(64, n_cls)

        # 初始化码本模块
        self.codebooks = nn.ModuleList()
        for i in range(len(architecture)):
            if i >= 5:
                decoder_idx = i - 5
                if decoder_idx == 0:
                    feature_dim = 108
                    current_codebook_size = 512
                elif decoder_idx == 1:
                    feature_dim = 72
                    current_codebook_size = 384
                elif decoder_idx == 2:
                    feature_dim = 48
                    current_codebook_size = 256
                elif decoder_idx == 3:
                    feature_dim = 32
                    current_codebook_size = 192
                else:
                    feature_dim = 0
                    current_codebook_size = 0

                if feature_dim > 0:
                    self.codebooks.append(CodebookModule(
                        feature_dim=feature_dim, 
                        codebook_size=current_codebook_size,
                        commitment_cost=0,
                        temperature=0.1
                    ))
                else:
                     self.codebooks.append(None)
            else:
                self.codebooks.append(None)

    def forward(self, p, x, o, calculate_commitment_losses=False):
        batch_size = p.size(0)
        p_from_encoder = []
        x_from_encoder = []
        o_from_encoder = []
        idx_from_encoder = []
        
        total_commitment_loss = torch.tensor(0.0, device=p.device)
        
        # encoder
        for block_i, block in enumerate(self.encoder_blocks):
            p, x, o, idx = block(p.view(-1, 3), x, o)
            p_from_encoder.append(p)
            x_from_encoder.append(x)
            o_from_encoder.append(o)
            idx_from_encoder.append(idx)
            
        x = x_from_encoder.pop()
        p = p_from_encoder.pop()
        o = o_from_encoder.pop()

        for block_i, block in enumerate(self.decoder_blocks):
            x_skip = x_from_encoder.pop()
            p_skip = p_from_encoder.pop()
            o_skip = o_from_encoder.pop()
            idx_skip = idx_from_encoder.pop()

            codebook_global_idx = block_i + len(self.encoder_blocks)
            codebook = self.codebooks[codebook_global_idx] if codebook_global_idx < len(self.codebooks) else None

            p_output, x_output, o_output, block_commitment_loss = block(
                p1=p_skip,
                x1=x_skip,
                o1=o_skip,
                idx=idx_skip,
                p2=p.view(-1,3),
                x2=x,
                o2=o,
                batch_size=o_skip.size(0) if (o_skip is not None and isinstance(o_skip, torch.Tensor) and o_skip.dim() > 0) else p.size(0),
                codebook=codebook,
                calculate_commitment_loss_for_block=calculate_commitment_losses
            )

            p, x, o = p_output, x_output, o_output
            total_commitment_loss += block_commitment_loss

        x = F.relu(self.linear0_1(x))
        x = F.relu(self.linear0_2(x))
        x_out = torch.tanh(self.linear0_3(x))

        num_points_in_final_x = x.shape[0]
        final_N_per_sample = num_points_in_final_x // batch_size
        x_out = x_out.view(batch_size, final_N_per_sample, -1)

        return x_out, total_commitment_loss

