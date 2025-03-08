
import torch
import torch.nn as nn
import math

class PositionEmbeddingNerf(nn.Module):
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super(PositionEmbeddingNerf, self).__init__()
        self.num_pos_feats = num_pos_feats // 2
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32)
        dim_t = self.temperature ** (2 * (dim_t.div(2, rounding_mode="trunc")) / self.num_pos_feats)
        self.dim_t = nn.Parameter(dim_t, requires_grad=False)

    def forward(self, x):
        if x.dim() == 2:
            pos = x.unsqueeze(2) / self.dim_t.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 3:
            B, N, C = x.size()
            pos = x.view(-1, C, 1) / self.dim_t.unsqueeze(0).unsqueeze(0)
            pos = pos.view(B, N, C, self.num_pos_feats)
        pos_emb = torch.cat((pos.sin(), pos.cos()), dim=-1)

        return pos_emb