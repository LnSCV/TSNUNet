import torch
import torch.nn as nn
import numpy as np
from timm.models.layers import DropPath
from timm.models.helpers import load_pretrained
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_


class Mlp(nn.Module):
    def __init__(self, inc, hidden=None, outc=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        outc = outc or inc
        hidden = hidden or inc
        self.fc1 = nn.Linear(inc, hidden)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden, outc)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        # num_heads = 1, dim = embed_dim = 512, mlp_ratio = 3 qkv_bias=False, qk_scale=None, drop_rate=0.,attn_drop_rate=0.,drop_path=dpr[i]
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)  # [3,B,self.num_heads,N,C // self.num_heads]   [0,B,1,N,512],[1,B,1,N,512],[2,B,1,N,512]
        q, k, v = qkv[0], qkv[1], qkv[2] #[B,1,N,512],[B,1,N,512],[B,1,N,512]

        attn = (q @ k.transpose(-2, -1)) * self.scale #[B,1,N,N]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C) #[B,1,N,512] --> [B,N,1,512]--> [B,N,512]
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def get_sinusoid_encoding(n_position, d_hid):
    ''' Sinusoid position encoding table '''

    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # num_heads = 1, dim = embed_dim = 512, mlp_ratio = 3 qkv_bias=False, qk_scale=None, drop_rate=0.,attn_drop_rate=0.,drop_path=dpr[i]
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(inc=dim, hidden=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Transformer(nn.Module):
    def __init__(self, depth, num_heads, embed_dim, num_patches, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm):
        #(depth=2, num_heads=1,embed_dim=256, mlp_ratio=3, num_patches=196)
        # depth = 2, num_heads = 1, embed_dim = 512, mlp_ratio = 3, num_patches = 2304

        super(Transformer, self).__init__()

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            #num_heads = 1, embed_dim = 512, mlp_ratio = 3 qkv_bias=False, qk_scale=None, drop_rate=0.,attn_drop_rate=0.,drop_path=dpr[i]
            for i in range(depth)]) #depth=2

        self.norm = norm_layer(embed_dim) #embed_dim=256  512 320 128 64
        self.pos_embed = nn.Parameter(data=get_sinusoid_encoding(n_position=num_patches, d_hid=embed_dim),#embed_dim= 512,num_patches=2304
                                      requires_grad=False)
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, peb=True):
        # receive x in shape of B,HW,C
        if peb:
            x = x + self.pos_embed
            x = self.pos_drop(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        return x