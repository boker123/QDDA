import torch
import torch.nn as nn
from torch.nn import functional as F
from timm.models.layers import trunc_normal_, DropPath
from thop import profile

from utils.AttentionFunc import Attn_QCS_SD


class CrossAttention(nn.Module):

    def __init__(self, embed_dim=768):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.proj = nn.Linear(embed_dim, embed_dim * 2)  #

        self.mlp = Mlp(in_features=embed_dim, hidden_features=int(embed_dim * 4), act_layer=nn.GELU, drop=0)

        self.theta = nn.Parameter(torch.tensor(0.), requires_grad=True)

    def forward(self, x_a, x_p, x_n, x_n2):
        # print(f'x_a.shape: {x_a.shape}')
        B, N, C = x_a.shape

        x_a = self.norm1(x_a)
        x_p = self.norm1(x_p)
        x_n = self.norm1(x_n)
        x_n2 = self.norm1(x_n2)

        qkv_a = self.proj(x_a).reshape(B, N, 2, C).permute(2, 0, 1, 3)  # shape it to QK same tensor and V
        QK1, V1 = qkv_a[0], qkv_a[1]
        # print(f1.shape, qkv1.shape)

        qkv_p = self.proj(x_p).reshape(B, N, 2, C).permute(2, 0, 1, 3)
        QK2, V2 = qkv_p[0], qkv_p[1]

        qkv_n = self.proj(x_n).reshape(B, N, 2, C).permute(2, 0, 1, 3)
        QK3, V3 = qkv_n[0], qkv_n[1]

        qkv_n2 = self.proj(x_n2).reshape(B, N, 2, C).permute(2, 0, 1, 3)
        QK4, V4 = qkv_n2[0], qkv_n2[1]

        ##############################################

        k = torch.tanh(self.theta)
        cross_map1, cross_map2, cross_map3, cross_map4 = Attn_QCS_SD(QK1, QK2, QK3, QK4, k) # # B WH WH # torch.Size([64, 49, 49])

        ############################################
        cross_map1 = torch.reshape(cross_map1, shape=(B, N, 1))
        attn_a = cross_map1 * V1  # B N C # torch.Size([64, 49, 768])
        x_a = x_a + attn_a

        cross_map2 = torch.reshape(cross_map2, shape=(B, N, 1))
        attn_p = cross_map2 * V2  # B N C
        x_p = x_p + attn_p

        cross_map3 = torch.reshape(cross_map3, shape=(B, N, 1))
        attn_n = cross_map3 * V3  # B N C
        x_n = x_n + attn_n

        cross_map4 = torch.reshape(cross_map4, shape=(B, N, 1))
        attn_n2 = cross_map4 * V4  # B N C
        x_n2 = x_n2 + attn_n2

        x_a = self.norm2(x_a)
        mlp_a = self.mlp(x_a)
        x_a = x_a + mlp_a

        x_p = self.norm2(x_p)
        mlp_p = self.mlp(x_p)
        x_p = x_p + mlp_p

        x_n = self.norm2(x_n)
        mlp_n = self.mlp(x_n)
        x_n = x_n + mlp_n

        x_n2 = self.norm2(x_n2)
        mlp_n2 = self.mlp(x_n2)
        x_n2 = x_n2 + mlp_n2

        return x_a, x_p, x_n, x_n2, k

class Mlp(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x