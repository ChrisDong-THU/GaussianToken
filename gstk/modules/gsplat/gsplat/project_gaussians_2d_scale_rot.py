"""Python bindings for 3D gaussian projection"""

from typing import Tuple

from jaxtyping import Float
from torch import Tensor
from torch.autograd import Function

import gsplat.cuda as _C


# 缩放旋转形式GS渲染
def project_gaussians_2d_scale_rot(
    means2d: Float[Tensor, "*batch 2"],
    scales2d: Float[Tensor, "*batch 2"],
    rotation: Float[Tensor, "*batch 1"],
    img_height: int,
    img_width: int,
    tile_bounds: Tuple[int, int, int],
    clip_thresh: float = 0.01,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, int]:
    return _ProjectGaussians2dScaleRot.apply(
        means2d.contiguous(),
        scales2d.contiguous(),
        rotation.contiguous(),
        img_height,
        img_width,
        tile_bounds,
        clip_thresh,
    )

class _ProjectGaussians2dScaleRot(Function):
    """Project 3D gaussians to 2D."""

    @staticmethod
    def forward(
        ctx,
        means2d: Float[Tensor, "*batch 2"],
        scales2d: Float[Tensor, "*batch 2"],
        rotation: Float[Tensor, "*batch 1"],
        img_height: int,
        img_width: int,
        tile_bounds: Tuple[int, int, int],
        clip_thresh: float = 0.01,
    ):
        num_points = means2d.shape[-2]
        if num_points < 1 or means2d.shape[-1] != 2:
            raise ValueError(f"Invalid shape for means2d: {means2d.shape}")
        (
            xys,
            depths,
            radii,
            conics,
            num_tiles_hit,
        ) = _C.project_gaussians_2d_scale_rot_forward(
            num_points,
            means2d,
            scales2d,
            rotation,
            img_height,
            img_width,
            tile_bounds,
            clip_thresh,
        )

        # Save non-tensors.
        ctx.img_height = img_height
        ctx.img_width = img_width
        ctx.num_points = num_points

        # Save tensors.
        ctx.save_for_backward(
            means2d,
            scales2d,
            rotation,
            radii,
            conics,
        )

        return (xys, depths, radii, conics, num_tiles_hit)

    @staticmethod
    def backward(ctx, v_xys, v_depths, v_radii, v_conics, v_num_tiles_hit):
        (
            means2d,
            scales2d,
            rotation,
            radii,
            conics,
        ) = ctx.saved_tensors

        (v_cov2d, v_mean2d, v_scale, v_rot) = _C.project_gaussians_2d_scale_rot_backward(
            ctx.num_points,
            means2d,
            scales2d,
            rotation,
            ctx.img_height,
            ctx.img_width,
            radii,
            conics,
            v_xys,
            v_depths,
            v_conics,
        )

        # Return a gradient for each input.
        return (
            # means2d: Float[Tensor, "*batch 2"],
            v_mean2d,
            # scales: Float[Tensor, "*batch 2"],
            v_scale,
            #rotation: Float,
            v_rot,
            # img_height: int,
            None,
            # img_width: int,
            None,
            # tile_bounds: Tuple[int, int, int],
            None,
            # clip_thresh,
            None,
        )
