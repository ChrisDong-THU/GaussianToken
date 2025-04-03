import torch
from torch import nn
from torch.nn import MultiheadAttention as MHA

from .ops.modules import MSDeformAttn as MSDA

import copy


def _with_pos_embed(tensor, pos):
    return tensor if pos is None else tensor + pos


def _to_device_tensor(x, device):
    return torch.tensor([x,], device=device)


def _layer_clone(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return nn.ReLU()
    if activation == "gelu":
        return nn.GELU()
    if activation == "glu":
        return nn.GLU()
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


def _get_mlp(d_model, d_ffn, dropout, activation):
    return nn.Sequential(
        nn.Linear(d_model, d_ffn),
        _get_activation_fn(activation),
        nn.Dropout(dropout),
        nn.Linear(d_ffn, d_model),
        nn.Dropout(dropout)
    )


class DAEncoderLayer(nn.Module):
    def __init__(self, embed_dim, dropout=0.1, activation='relu', n_levels=1, num_heads=8, n_points=4):
        super().__init__()
        self.self_attn = MSDA(embed_dim, n_levels, num_heads, n_points)
        self.droupout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        
        self.mlp = _get_mlp(embed_dim, embed_dim*4, dropout, activation)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDA):
                m._reset_parameters()
    
    def forward_ffn(self, src):
        src2 = self.mlp(src)
        src = self.norm2(src + src2)
        return src
    
    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index=[0,]): 
        with torch.autocast(device_type='cuda', dtype=torch.float32): # MSDA not support fp16
            src2 = self.self_attn(_with_pos_embed(src, pos), reference_points, src, spatial_shapes, level_start_index)
        
        src = src + self.droupout1(src2)
        src = self.norm1(src)
        
        src = self.forward_ffn(src)
        
        return src


class DAEncoder(nn.Module):
    def __init__(self, fm_shape, encoder_layer_cfg, num_layers):
        super().__init__()
        encoder_layer = DAEncoderLayer(**encoder_layer_cfg)
        self.layers = _layer_clone(encoder_layer, num_layers)
        self.fm_shape = fm_shape
    
    @staticmethod
    def get_reference_points(H, W, device):
        ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H - 0.5, H, dtype=torch.float32, device=device)/H,
                                        torch.linspace(0.5, W - 0.5, W, dtype=torch.float32, device=device)/W, indexing='ij')
        ref_y = ref_y.reshape(-1)[None] # [1, H*W]
        ref_x = ref_x.reshape(-1)[None]
        reference_points = torch.stack((ref_x, ref_y), -1) # [1, H*W, 2]
        reference_points = reference_points.unsqueeze(2) # [1, H*W, 1, 2]
        return reference_points
        
    def forward(self, src, pos=None):
        reference_points = self.get_reference_points(*self.fm_shape, src.device)
        
        output = src
        for _, layer in enumerate(self.layers):
            output = layer(output, pos, reference_points, \
                _to_device_tensor(self.fm_shape, src.device), _to_device_tensor(0, src.device))
        
        return output
    
    
class DADecoderLayer(nn.Module):
    def __init__(self, embed_dim, dropout=0.1, activation='relu', n_levels=1, num_heads=8, n_points=4):
        super().__init__()
        self.self_attn = MHA(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.droupout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        
        self.cross_attn = MSDA(embed_dim, n_levels, num_heads, n_points).to(torch.float32) # MSDA not support fp16
        self.droupout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        self.mlp = _get_mlp(embed_dim, embed_dim*4, dropout, activation)
        self.norm3 = nn.LayerNorm(embed_dim)
        
        self._reset_parameters()
        
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDA):
                m._reset_parameters()
                
    def forward_ffn(self, tgt):
        tgt2 = self.mlp(tgt)
        tgt = self.norm3(tgt + tgt2)
        return tgt
    
    def forward(self, tgt, query_pos, reference_points, src, spatial_shapes, level_start_index):
        qk = _with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(qk, qk, tgt)[0]
        tgt = tgt + self.droupout1(tgt2)
        tgt = self.norm1(tgt)

        with torch.autocast(device_type='cuda', dtype=torch.float32):
            tgt2 = self.cross_attn(_with_pos_embed(tgt, query_pos), reference_points, src, spatial_shapes, level_start_index)
        tgt = tgt + self.droupout2(tgt2)
        tgt = self.norm2(tgt)
        
        tgt = self.forward_ffn(tgt)
        
        return tgt
    
    
class DADecoder(nn.Module):
    def __init__(self, fm_shape, decoder_layer_cfg, num_layers, proj_drop=0.0, residual_mode="cat"):
        super().__init__()
        decoder_layer = DADecoderLayer(**decoder_layer_cfg)
        self.layers = _layer_clone(decoder_layer, num_layers)
        self.fm_shape = fm_shape
        
        self.proj_drop = nn.Dropout(proj_drop)
        self.residual_mode = residual_mode
    
    def forward(self, tgt, query_pos, src, reference_points):
        output = tgt
        for _, layer in enumerate(self.layers):
            output = layer(output, query_pos, reference_points, src, \
                _to_device_tensor(self.fm_shape, src.device), _to_device_tensor(0, src.device))
        
        output = self.proj_drop(output)
        output = torch.cat([output, tgt], dim=-1) if self.residual_mode=="cat" else \
            output + tgt

        return output
    

def build_attn_encoder(cfg):
    return DAEncoder(**cfg)

def build_attn_decoder(cfg):
    return DADecoder(**cfg)
