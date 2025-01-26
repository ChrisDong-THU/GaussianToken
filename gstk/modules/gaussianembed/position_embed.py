import math
import torch
from torch import nn


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, pos_feat_dim=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.pos_feat_dim = pos_feat_dim
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x):
        b, c, h, w = x.shape # [bs, c, h, w]
        mask = torch.ones((b, h, w), device=x.device) # [bs, h, w]

        y_embed = mask.cumsum(1, dtype=torch.float32)
        x_embed = mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = (y_embed - 0.5) / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = (x_embed - 0.5) / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.pos_feat_dim, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.pos_feat_dim)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3) # pos_feat_dim必须为偶数，embed_dim必须为4的倍数
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        
        return pos


class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """
    def __init__(self, num_embed=100, pos_feat_dim=256):
        super().__init__()
        self.row_embed = nn.Embedding(num_embed, pos_feat_dim)
        self.col_embed = nn.Embedding(num_embed, pos_feat_dim)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, x):
        h, w = x.shape[-2:]
        i = torch.arange(w, device=x.device)
        j = torch.arange(h, device=x.device)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(h, 1, 1),
            y_emb.unsqueeze(1).repeat(1, w, 1),
        ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
        return pos

    
def build_position_encoding(version, pe_dim):
    pos_feat_dim = pe_dim // 2 # 隐层维度的一半用于x，一半用于y
    if version in ('v2', 'sine'):
        position_embedding = PositionEmbeddingSine(pos_feat_dim, normalize=True)
    elif version in ('v3', 'learned'):
        position_embedding = PositionEmbeddingLearned(pos_feat_dim = pos_feat_dim)
    else:
        raise ValueError(f"not supported {version}")

    return position_embedding