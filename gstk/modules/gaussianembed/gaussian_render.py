import torch

from gsplat.project_gaussians_2d_scale_rot import project_gaussians_2d_scale_rot
from gsplat.rasterize_sum import rasterize_gaussians_sum

from dataclasses import dataclass


@dataclass
class RenderSet:
    img_H: int = 256
    img_W: int = 256
    block_H: int = 16
    block_W: int = 16
    tile_bounds: tuple = None
    background: torch.Tensor = None
    
    def post_set(self, feature_dim):
        self.tile_bounds = (
            (self.img_W + self.block_W - 1) // self.block_W,
            (self.img_H + self.block_H - 1) // self.block_H,
            1,
        )
        self.background = torch.zeros(feature_dim, dtype=torch.float).cuda()
        
        return self


def _gaussian2image(_xy, _scaling, _rotation, _feature, render_set: RenderSet, _opacity=None):
    _opacity = torch.ones(_xy.shape[0], 1, dtype=torch.float, device=_xy.device) if _opacity is None or _opacity.shape[-1]==0 else _opacity
    xys, depths, radii, conics, num_tiles_hit = project_gaussians_2d_scale_rot(_xy, _scaling, _rotation, render_set.img_H, render_set.img_W, render_set.tile_bounds)
    out_img = rasterize_gaussians_sum(xys, depths, radii, conics, num_tiles_hit, _feature, _opacity, render_set.img_H, render_set.img_W, render_set.block_H, render_set.block_W, background=render_set.background, return_alpha=False)
    out_img = out_img.permute(2, 0, 1).contiguous()
    
    return out_img


def render_gaussians(gaussians, render_set: RenderSet):
    img_batch = []
    for mean, scale, rot, feature, opacity in zip(gaussians.means, gaussians.scales, gaussians.rotations, gaussians.features, gaussians.opacities):
        img = _gaussian2image(mean, scale, rot, feature, render_set, opacity)
        img_batch.append(img)
        
    img_batch = torch.stack(img_batch)
    
    return img_batch
