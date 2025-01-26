import torch
import torch.nn as nn

from torch import Tensor
from dataclasses import dataclass

from .misc import Scale, safe_sigmoid, safe_tanh, linear_relu_ln


def _apply_range(tensor, scale_range):
    return scale_range[0] + (scale_range[1] - scale_range[0]) * tensor


def _unpack_tensor(x):
    return [torch.tensor(item, dtype=torch.float32) for item in x]


class RangeScheduler(nn.Module):
    def __init__(self, initial_range=[0.1, 5.0], final_range=[0.1, 0.5], decay_rate=1e-3, ema_alpha=0.01):
        super(RangeScheduler, self).__init__()
        
        self.range_min, self.range_max = _unpack_tensor(initial_range)
        self.final_range_min, self.final_range_max = _unpack_tensor(final_range)
        self.decay_rate = decay_rate
        self.ema_alpha = ema_alpha

        self.ema_x_mean = (self.range_min + self.range_max) / 2
        self.register_buffer('current_range_max', self.range_max)

    def update(self, x_mean):
        self.ema_x_mean = self.ema_alpha * x_mean + (1 - self.ema_alpha) * self.ema_x_mean
        
        if self.current_range_max > self.final_range_max:
            delta = max(self.ema_x_mean - (self.current_range_max + self.range_min) / 2, 1e-2)
            adjustment = self.decay_rate * delta
            self.current_range_max -= adjustment
            self.current_range_max = torch.max(self.current_range_max, self.final_range_max)

    @property
    def current_range(self):
        return [self.range_min.item(), self.current_range_max.item()]


@dataclass
class GaussianPrediction:
    means: Tensor
    scales: Tensor
    rotations: Tensor
    opacities: Tensor
    features: Tensor


class GaussianRefinementModule(nn.Module):
    def __init__(
        self,
        embed_dim=256,
        xy_range=None, # xy范围 [-1, 1]
        initial_scale_range=None, # 协方差缩放范围 [min, max]
        final_scale_range=None, # 最终协方差缩放范围 [min, max]
        refine_state=None, # [0, 1] 微调均值部分
        include_opa=True,
        dim_feature=3,
        z_channels=16,
    ):
        super(GaussianRefinementModule, self).__init__()
        self.embed_dim = embed_dim
        
        self.output_dim = 5 + int(include_opa) + dim_feature # 2 + 1 + 2 + 1 + dim_feature
        self.feature_start = 5 + int(include_opa)
        self.dim_feature = dim_feature
        self.include_opa = include_opa

        self.xy_range = xy_range
        self.range_scheduler = RangeScheduler(initial_range=initial_scale_range, final_range=final_scale_range)
        
        self.refine_state = refine_state
        assert all([self.refine_state[i] == i for i in range(len(self.refine_state))])

        self.layers = nn.Sequential(
            *linear_relu_ln(embed_dim, 2, 2),
            nn.Linear(self.embed_dim, self.output_dim),
            Scale([1.0] * self.output_dim))
        
        self.to_features = nn.Sequential(
            *linear_relu_ln(self.embed_dim, 2, 2),
            nn.Linear(self.embed_dim, z_channels),
            Scale([1.0] * z_channels)
        )

    def forward(
        self,
        instance_feature: torch.Tensor,
        anchor_embed: torch.Tensor,
        anchor: torch.Tensor,
        **kwargs
    ):
        '''由instance_feature和anchor_embed得到微调量

        '''
        output = self.layers(instance_feature+anchor_embed) # 实例特征 + 锚点特征 [bs, num_gs, embed_dim]
        
        # 微调部分，0,1对应均值m的部分
        if len(self.refine_state) > 0:
            refined_part_output = output[..., self.refine_state] + anchor[..., self.refine_state]
            output = torch.cat([refined_part_output, output[..., len(self.refine_state):]], dim=-1)
        
        return output
    
    def get_gaussian(self, output):
        xys = safe_sigmoid(output[..., :2])
        xys = _apply_range(xys, self.xy_range)
        rots = safe_sigmoid(output[..., 2:3]) * torch.pi # [0, pi] or [0, 2*pi]
        opacities = safe_sigmoid(output[..., 5: (5 + int(self.include_opa))])

        gs_scales = safe_sigmoid(output[..., 3:5]) # [0, 1]
        gs_scales = _apply_range(gs_scales, self.range_scheduler.current_range)
        if self.training:
            self.range_scheduler.update(gs_scales.mean().item())
        
        features = output[..., self.feature_start: (self.feature_start + self.dim_feature)]
        features = self.to_features(features)
        
        gaussian = GaussianPrediction(
            means=xys,
            scales=gs_scales,
            rotations=rots,
            opacities=opacities,
            features=features
        )
        
        return gaussian
        

def build_refine_module(cfg):
    return GaussianRefinementModule(**cfg)