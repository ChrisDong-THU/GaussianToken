import torch
import torch.nn as nn


class Scale(nn.Module):
    def __init__(self, scale=1.0):
        super(Scale, self).__init__()
        # 初始化一个可学习的标量参数
        self.scale = nn.Parameter(torch.tensor(scale, dtype=torch.float32))

    def forward(self, x):
        # 将输入乘以可学习的标量参数
        return x * self.scale


def swish(x):
    # swish
    return x * torch.sigmoid(x)


def safe_sigmoid(x, eps = 9.21024):
    x = x.clamp(min=-eps, max=eps)
    return torch.sigmoid(x)


def safe_inverse_sigmoid(x, eps=0.9999):
    x = x.clamp(1-eps, eps)
    return torch.log(x / (1-x))


def safe_tanh(x, eps=10.0):
    x = x.clamp(min=-eps, max=eps)
    return torch.tanh(x)


# Relu + LayerNorm
def linear_relu_ln(embed_dim, in_loops, out_loops, input_dims=None):
    if input_dims is None:
        input_dims = embed_dim
    layers = []
    for _ in range(out_loops):
        for _ in range(in_loops):
            layers.append(nn.Linear(input_dims, embed_dim))
            layers.append(nn.ReLU())
            input_dims = embed_dim
        layers.append(nn.LayerNorm(embed_dim))
    return layers


def get_rotation_matrix(rots):
    bs, num = rots.shape[:2]
    
    cos_rot = torch.cos(rots)
    sin_rot = torch.sin(rots)
    
    rot_matrix = torch.stack([cos_rot, -sin_rot, sin_rot, cos_rot], dim=-1)
    
    return rot_matrix.view(bs, num, 2, 2)