import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Simplified DropPath for parameter counting (no trainable parameters)
class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor

# From Transformer.py
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
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

def get_sinusoid_encoding(n_position, d_hid):
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
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
        super(Transformer, self).__init__()

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

        self.norm = norm_layer(embed_dim)
        self.pos_embed = nn.Parameter(data=get_sinusoid_encoding(n_position=num_patches, d_hid=embed_dim),
                                      requires_grad=False)
        self.pos_drop = nn.Dropout(p=drop_rate)


    def forward(self, x, peb=True):
        if peb:
            x = x + self.pos_embed
            x = self.pos_drop(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        return x

# From u2net.py (Nested_Trans_U)
class Nested_Trans_U(nn.Module):
    def __init__(self, in_ch=128, mid_ch=64, num_patches=144):
        super(Nested_Trans_U,self).__init__()
        self.mid_ch = mid_ch
        self.rebnconvin  = nn.Sequential(nn.Conv2d(in_ch, 64, kernel_size=3, padding=1, bias=True),
                      nn.BatchNorm2d(64),
                      nn.LeakyReLU(inplace=True),
                                         )
        self.rebnconv1 = nn.Sequential(nn.Conv2d(256, 64, kernel_size=3, padding=1, bias=True),
                      nn.BatchNorm2d(64),
                      nn.LeakyReLU(inplace=True),
                                       )
        self.rebnconv2 = nn.Sequential(nn.Conv2d(256, 64, kernel_size=3, padding=1, bias=True),
                      nn.BatchNorm2d(64),
                      nn.LeakyReLU(inplace=True),
                                       )
        self.rebnconv3_1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True),
                      nn.BatchNorm2d(64),
                      nn.LeakyReLU(inplace=True),
                                         )
        self.global_attn = Transformer(depth=2, num_heads=1, embed_dim=self.mid_ch, mlp_ratio=3, num_patches=num_patches)

        self.rebnconv3_2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True),
                      nn.BatchNorm2d(64),
                      nn.LeakyReLU(inplace=True),
                                         )

        self.rebnconv2d = nn.Sequential(nn.Conv2d(mid_ch*2,mid_ch*4, kernel_size=3, padding=1, bias=True),
                      nn.BatchNorm2d(mid_ch*4),
                      nn.LeakyReLU(inplace=True),
                                        )

        self.rebnconv1d = nn.Sequential(nn.Conv2d(mid_ch*2,mid_ch*4, kernel_size=3, padding=1, bias=True),
                      nn.BatchNorm2d(mid_ch*4),
                      nn.LeakyReLU(inplace=True),
                                        )

        self.rebnconvout  = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True),
                      nn.BatchNorm2d(64),
                      nn.LeakyReLU(inplace=True))


    def forward(self,x):
        hxin = self.rebnconvin(x)

        hx1 = F.pixel_unshuffle(hxin,2)

        hx1 = self.rebnconv1(hx1)
        hx2 = F.pixel_unshuffle(hx1, 2)
        hx2 = self.rebnconv2(hx2)

        hx3 = self.rebnconv3_1(hx2)

        B, C, h, w = hx3.shape
        fx_g = hx3.reshape(B, self.mid_ch, -1).transpose(1, 2)
        x_g = self.global_attn(fx_g)
        x_g = x_g.transpose(1, 2).reshape(B, self.mid_ch, h, w)
        hx3 = self.rebnconv3_2(x_g)


        hx2d = self.rebnconv2d(torch.cat((hx3,hx2),1))
        hx2d = F.pixel_shuffle(hx2d, 2)
        hx1d = self.rebnconv1d(torch.cat((hx2d,hx1),1))
        hx1d = F.pixel_shuffle(hx1d, 2)
        hxout = self.rebnconvout(hx1d)
        return hxout + hxin

# Instantiate the model with default parameters
# These parameters (in_ch, mid_ch, num_patches) need to match how the module is actually used in your full model.
# Using default values from the __init__ method for demonstration.
model = Nested_Trans_U(in_ch=128, mid_ch=64, num_patches=576)

# Calculate the number of parameters
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters in Nested_Trans_U: {total_params / 1e6:.3f} M")
