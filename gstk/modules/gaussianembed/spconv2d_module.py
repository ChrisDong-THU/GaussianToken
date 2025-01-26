import torch
import torch.nn as nn

from spconv.pytorch import SubMConv2d, SparseConvTensor
from .misc import safe_sigmoid


class SparseConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, fm_shape: tuple, use_out_proj=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.conv = SubMConv2d(in_channels, out_channels, kernel_size=kernel_size, padding=(kernel_size-1) // 2, bias=False)
        self.out_proj = nn.Linear(out_channels, out_channels) if use_out_proj else nn.Identity()
        
        grid_size = 1 / torch.tensor(fm_shape).float() # 网格分辨率
        self.register_buffer('grid_size', grid_size)
        
    
    def forward(self, instance_feature, anchor):
        b, g = instance_feature.shape[:2]
        
        anchor_xy = anchor[..., :2]
        anchor_xy = safe_sigmoid(anchor_xy).reshape(-1, 2)
        
        indices = (anchor_xy - anchor_xy.min(dim=0, keepdim=True)[0]) / self.grid_size # 自动广播
        indices = indices.to(torch.int32)
        batch_indices = torch.cat([torch.arange(b, device=indices.device, dtype=indices.dtype).reshape(b, 1, 1).expand(-1, g, -1).flatten(0, 1), 
                             indices], dim=-1)
        
        spatial_shape = indices.max(0)[0]
        
        instance_feature = instance_feature.to(torch.float32) # SubMConv2d only support float32
        input = SparseConvTensor(instance_feature.flatten(0, 1), indices=batch_indices, spatial_shape=spatial_shape, batch_size=b)
        output = self.conv(input)
        
        output = output.features.unflatten(0, (b, g))
        
        return self.out_proj(output) # WARNING: spconv2d is not stable, would cause different output with same input
    
    
def build_spconv2d(cfg):
    return SparseConv(**cfg)


if __name__ == "__main__":
    bs = 4
    num_gs = 256
    in_channels = 64
    out_channels = 64
    kernel_size = 5
    fm_shape = (32, 32)
    use_out_proj = False
    
    load = False
    
    import lightning as L
    L.seed_everything(0)
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    spconv2d = SparseConv(in_channels, out_channels, kernel_size, fm_shape, use_out_proj)
    
    if load:
        sd = torch.load('./tmp/spconv2d.pth')
        spconv2d.load_state_dict(sd)
    else:
        torch.save(spconv2d.state_dict(), './tmp/spconv2d.pth')
    
    spconv2d = spconv2d.cuda()
    spconv2d = spconv2d.eval()
    
    instance_feature = torch.randn(bs, num_gs, in_channels).cuda()
    anchor = torch.randn(bs, num_gs, 5 + in_channels).cuda()
    
    with torch.no_grad():
        output = spconv2d(instance_feature, anchor)
    
    pass