import torch
import torch.nn as nn


class AsymmetricFFN(nn.Module):
    """
    非对称前馈网络 (FFN) 模块，全连接 + 恒等映射

    """
    def __init__(
        self,
        in_channels=None,
        pre_norm=True,
        out_norm=True,
        embed_dim=256,
        num_fcs=2,
        ffn_drop=0.0,
        add_identity=True,
        **kwargs,
    ):
        super().__init__()
        assert num_fcs >= 2, (
            "num_fcs should be no less " f"than 2. got {num_fcs}."
        )
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.feedforward_channels = embed_dim * 4
        self.num_fcs = num_fcs

        layers = []
        if in_channels is None:
            in_channels = embed_dim

        self.pre_norm_layer = nn.LayerNorm(in_channels) if pre_norm else nn.Identity()
        self.out_norm_layer = nn.LayerNorm(embed_dim) if out_norm else nn.Identity()

        # 多个全连接层
        for _ in range(num_fcs - 1):
            layers.append(
                nn.Sequential(
                    nn.Linear(in_channels, self.feedforward_channels),
                    nn.ReLU(),
                    nn.Dropout(ffn_drop),
                )
            )
            in_channels = self.feedforward_channels
        
        # 将升维后的x输出到embed_dim
        layers.append(nn.Linear(self.feedforward_channels, embed_dim))
        layers.append(nn.Dropout(ffn_drop))
        self.layers = nn.Sequential(*layers)

        self.add_identity = add_identity
        
        # 将identity部分输出到embed_dim
        if self.add_identity:
            self.identity_fc = (
                torch.nn.Identity()
                if in_channels == embed_dim
                else nn.Linear(self.in_channels, embed_dim)
            )

    def forward(self, instance_feature, identity=None, **kwargs):
        x = instance_feature
        x = self.pre_norm_layer(x)
        out = self.layers(x)

        if self.add_identity:
            identity = x if identity is None else identity
            identity = self.identity_fc(identity)
            out = identity + out
        
        return self.out_norm_layer(out)


def build_ffn(cfg):
    return AsymmetricFFN(**cfg)