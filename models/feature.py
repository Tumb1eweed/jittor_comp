import os
os.environ.setdefault("nvcc_path", "")
import jittor as jt
from jittor import nn

from models.blocks import block_decider, CodebookModule


class EmptyModule(nn.Module):
    def execute(self, *args, **kwargs):
        return None


class FeatureExtraction(nn.Module):
    def __init__(self, d_in=0, d_out=32, n_cls=3, nsample=16, stride_list=None):
        super().__init__()
        if stride_list is None:
            stride_list = [4, 3, 2, 1]
        architecture = [
            "startblock",
            "downsample",
            "downsample",
            "downsample",
            "downsample",
            "upsample",
            "upsample",
            "upsample",
            "upsample",
        ]
        stride_dim_list = [1.5, 1.5, 1.5, 1.5]
        stride = 1
        stride_idx = 0
        d_prev = d_in
        self.encoder_blocks = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        self.encoder_skip_dims = []

        for block_name in architecture:
            if "downsample" in block_name:
                self.encoder_skip_dims.append(d_prev)
                stride = stride_list[stride_idx]
                d_out = int(d_out * stride_dim_list[stride_idx])
                stride_idx += 1
                self.encoder_blocks.append(block_decider(block_name)(d_prev, d_out, nsample, stride))
            elif "upsample" in block_name:
                stride_idx -= 1
                stride = stride_list[stride_idx]
                skip_dim = self.encoder_skip_dims.pop()
                d_out = skip_dim
                self.decoder_blocks.append(block_decider(block_name)([d_prev, skip_dim], d_out, nsample, stride))
            else:
                self.encoder_blocks.append(block_decider(block_name)(d_prev, d_out, nsample, stride))
            d_prev = d_out

        self.linear0_1 = nn.Linear(d_out, 128, bias=False)
        self.linear0_2 = nn.Linear(128, 64)
        self.linear0_3 = nn.Linear(64, n_cls)
        self.codebooks = nn.ModuleList()
        for i in range(len(architecture)):
            if i < 5:
                self.codebooks.append(EmptyModule())
                continue
            decoder_idx = i - 5
            dims = [(108, 512), (72, 384), (48, 256), (32, 192)]
            if decoder_idx < len(dims):
                feature_dim, codebook_size = dims[decoder_idx]
                self.codebooks.append(CodebookModule(feature_dim=feature_dim, codebook_size=codebook_size, temperature=0.1))
            else:
                self.codebooks.append(EmptyModule())

    def execute(self, p, x, o):
        batch_size = p.shape[0]
        p_from_encoder, x_from_encoder, o_from_encoder, idx_from_encoder = [], [], [], []
        for block in self.encoder_blocks:
            p, x, o, idx = block(p.reshape(-1, 3), x, o)
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
            codebook_idx = block_i + len(self.encoder_blocks)
            codebook = self.codebooks[codebook_idx] if codebook_idx < len(self.codebooks) else None
            if isinstance(codebook, EmptyModule):
                codebook = None
            p, x, o = block(
                p1=p_skip,
                x1=x_skip,
                o1=o_skip,
                idx=idx_skip,
                p2=p.reshape(-1, 3),
                x2=x,
                o2=o,
                batch_size=o_skip.shape[0],
                codebook=codebook,
            )

        x = nn.relu(self.linear0_1(x))
        x = nn.relu(self.linear0_2(x))
        x_out = jt.tanh(self.linear0_3(x))
        final_n = x.shape[0] // batch_size
        return x_out.reshape(batch_size, final_n, -1)
