import torch
import torch.nn as nn

from .position_embed import build_position_encoding
from .misc import safe_inverse_sigmoid, linear_relu_ln


class FeatMapLifter(nn.Module):
    def __init__(self, in_dim=128, pe_version='sine', out_dim=128):
        super().__init__()

        self.pos_embed = build_position_encoding(pe_version, out_dim)

        self.input_proj = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=1),
            nn.GroupNorm(32, out_dim),
        )
        
        self.init_param()
        
    def init_param(self):
        nn.init.xavier_uniform_(self.input_proj[0].weight, gain=1)
        nn.init.constant_(self.input_proj[0].bias, 0)
    
    def forward(self, feat_map, **kwargs):
        feat_map = self.input_proj(feat_map)
        pos = self.pos_embed(feat_map)

        src_flatten = feat_map.flatten(2).transpose(1, 2) # [bs, c, h, w] -> [bs, h*w, c]
        pos = pos.flatten(2).transpose(1, 2)
        
        return src_flatten, pos
    
    
class GaussianLifter(nn.Module):
    def __init__(
        self,
        num_anchor,
        embed_dim,
        anchor_grad=True,
        ins_feat_grad=True,
        feature_dim=None,
        include_opa=True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        
        xy = torch.rand(num_anchor, 2, dtype=torch.float)
        xy = safe_inverse_sigmoid(xy) # (-inf, inf)
            
        scale = torch.rand_like(xy)
        scale = safe_inverse_sigmoid(scale)

        rots = torch.zeros(num_anchor, 1, dtype=torch.float)

        if include_opa:
            opacity = safe_inverse_sigmoid(0.1 * torch.ones((num_anchor, 1), dtype=torch.float))
        else:
            opacity = torch.ones((num_anchor, 0), dtype=torch.float)

        feature = torch.randn(num_anchor, feature_dim, dtype=torch.float)

        anchor = torch.cat([xy, scale, rots, opacity, feature], dim=-1)

        self.num_anchor = num_anchor
        # 初始化锚点，即 Initial Properties
        self.anchor = nn.Parameter(
            anchor.float(),
            requires_grad=anchor_grad,
        )
        self.instance_feature = nn.Parameter(
            torch.zeros([num_anchor, self.embed_dim]),
            requires_grad=ins_feat_grad,
        )
        self.init_weight()

    def init_weight(self):
        if self.instance_feature.requires_grad:
            torch.nn.init.xavier_uniform_(self.instance_feature.data, gain=1) # 均匀分布初始化张量

    def forward(self, src_flatten, **kwargs):
        
        batch_size = src_flatten.shape[0]
        instance_feature = self.instance_feature.unsqueeze(0).repeat(batch_size, 1, 1)
        anchor = self.anchor.unsqueeze(0).repeat(batch_size, 1, 1)

        return instance_feature, anchor # [b, num_anchor, 5 + include_opa + feature_dim]


class AnchorEncoder(nn.Module):
    """
    encode 2d Gaussian parameters into a unified embedding space.

    """
    def __init__(
        self, 
        embed_dim: int = 256, 
        include_opa=True,
        feature_dim=None
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.include_opa = include_opa

        # 嵌入层，高斯参数输入得到统一的嵌入特征embeded_dims
        def embedding_layer(input_dim):
            return nn.Sequential(*linear_relu_ln(embed_dim, 1, 2, input_dim))

        self.xy_fc = embedding_layer(2)
        self.scale_fc = embedding_layer(2)
        self.rot_fc = embedding_layer(1)
        if include_opa:
            self.opacity_fc = embedding_layer(1)
            
        self.feature_fc = embedding_layer(feature_dim)
        self.feature_start = 5 + int(include_opa)

        self.feature_dim = feature_dim
        self.output_fc = embedding_layer(self.embed_dim)

    def forward(self, anchor: torch.Tensor, **kwargs):
        xy_feat = self.xy_fc(anchor[..., :2])
        rot_feat = self.rot_fc(anchor[..., 2:3])
        scale_feat = self.scale_fc(anchor[..., 3:5])
        
        if self.include_opa:
            opacity_feat = self.opacity_fc(anchor[..., 5:6])
        else:
            opacity_feat = 0.

        feature = self.feature_fc(anchor[..., self.feature_start: (self.feature_start + self.feature_dim)])

        # 特征融合
        output = xy_feat + scale_feat + rot_feat + opacity_feat + feature
        output = self.output_fc(output)
        
        return output
    
    
def build_fm_lifter(cfg):
    return FeatMapLifter(**cfg)

def build_gaussian_lifter(cfg):
    return GaussianLifter(**cfg)

def build_anchor_encoder(cfg):
    return AnchorEncoder(**cfg)
