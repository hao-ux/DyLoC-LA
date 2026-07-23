import einops
import torch
import torch.nn as nn
from functools import partial
from timm.models.layers import DropPath
import torch.nn.functional as F
from einops import rearrange, repeat


class ConvBlock(nn.Module):
    def __init__(self, dim, k=3, r=0.50):
        super().__init__()
        self.dw_dim = int(dim * r)
        self.dw = nn.Conv2d(self.dw_dim, self.dw_dim, k, padding=k // 2, groups=self.dw_dim, bias=False)
        self.fc1 = nn.Conv2d(dim - self.dw_dim, dim - self.dw_dim, 1, bias=False)
        
        self.bn_act = nn.Sequential(
            nn.BatchNorm2d(dim),
            nn.GELU(),
        )
        
        self.ffn = nn.Sequential(
            nn.Conv2d(dim, dim * 4, 1),
            nn.BatchNorm2d(dim * 4),
            nn.GELU(),
            nn.Conv2d(dim * 4, dim, 1),
            nn.BatchNorm2d(dim),
            nn.GELU(),
        )
        self.layer_scale1 = nn.Parameter(torch.ones((dim)), requires_grad=True)
        self.layer_scale2 = nn.Parameter(torch.ones((dim)), requires_grad=True)
        
    def forward(self, x):
        x1, x2 = torch.split(x, [self.dw_dim, x.shape[1] - self.dw_dim], dim=1)
        x1 = self.dw(x1)
        x2 = self.fc1(x2)
        out = torch.cat([x1, x2], dim=1)
        out = x + self.bn_act(out) * self.layer_scale1.view(1, -1, 1, 1)
        out = out + self.ffn(out) * self.layer_scale2.view(1, -1, 1, 1)
        return out
 

class PatchEmbedding(nn.Module):
    def __init__(self, in_ch=3, dims=[8, 16, 32], kernels=[3, 3, 7], depths=[1, 1, 3]):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_ch, dims[0], 3, padding=1),
            nn.BatchNorm2d(dims[0]),
            nn.ReLU(inplace=True),
        )

        self.layer2 = nn.Sequential(
            nn.Sequential(*[
                ConvBlock(dims[0], kernels[0]) for _ in range(depths[0])
            ]),
            nn.Conv2d(dims[0], dims[0], 3, padding=1),
            nn.BatchNorm2d(dims[0]),
            nn.ReLU(inplace=True),
        )

        self.layer3 = nn.Sequential(
            nn.Sequential(*[
                ConvBlock(dims[0], kernels[1]) for _ in range(depths[1])
            ]),
            nn.Conv2d(dims[0], dims[1], 3, padding=1),
            nn.BatchNorm2d(dims[1]),
            nn.ReLU(inplace=True),
        )
        
        self.layer4 = nn.Sequential(
            nn.Sequential(*[
                ConvBlock(dims[1], kernels[2]) for _ in range(depths[2])
            ]),
            nn.Conv2d(dims[1], dims[2], 3, padding=1),
            nn.BatchNorm2d(dims[2]),
            nn.ReLU(inplace=True),
        )
        
        self.down = nn.MaxPool2d(2, 2)

    def forward(self, x):
        out = []
        
        x = self.layer1(x)
        
        x = self.layer2(x)
        
        x = self.down(x)
        x = self.layer3(x)
        out.append(x)
        x = self.down(x)
        x = self.layer4(x)
        out.append(x)
        return x, out


class LocalBlock(nn.Module):
    def __init__(self, dim, kernel=7):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, 1)
        self.conv2 = nn.Conv2d(dim, dim, 1)

        self.dw = nn.Conv2d(dim, dim, kernel, padding=kernel // 2, groups=dim)
        self.norm = nn.BatchNorm2d(dim)

        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)

        x1 = self.norm(self.dw(x1))
        x2 = F.silu(x2)
        x = x1 * x2

        x = self.proj(x)

        return x
    


class GlocalAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.qkv = nn.Linear(dim, dim * 4)
        self.gate = nn.Sequential(nn.Linear(dim, dim), nn.SiLU())
        self.lepe = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.proj = nn.Linear(dim, dim)
        
    
    def forward(self, x):
        B, N, C = x.shape
        
        gate = self.gate(x)
        lepe = self.lepe(x.transpose(1, 2).reshape(B, C, int(N**0.5), int(N**0.5)))
        lepe = lepe.reshape(B, C, N).transpose(1, 2)
        qkv = self.qkv(x).reshape(B, N, 4, C).permute(2, 0, 1, 3)
        q1, q2, k, v = qkv[0], qkv[1], qkv[2], qkv[3]
        k = torch.nn.functional.normalize(k, dim=-1)
        q_g = torch.sum(q1 * q2.softmax(dim=1), dim=1, keepdim=True) 
        g = F.sigmoid(q_g * k)
        out = g * v + lepe
        out = out * gate
        out = self.proj(out)
        return out




class Block(nn.Module):
    def __init__(self, dim, kernel=7, e=4, dropout=0.0, drop_path=0.0):
        super().__init__()
        self.dw1 = nn.Conv2d(dim, dim, kernel_size=3, groups=dim, padding=1)
        self.dw2 = nn.Conv2d(dim, dim, kernel_size=3, groups=dim, padding=1)

        self.norm1 = nn.BatchNorm2d(dim)
        self.norm2 = nn.BatchNorm2d(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.norm4 = nn.LayerNorm(dim)

        mlp_dim = int(dim * e)
        self.attn_local = LocalBlock(dim, kernel)
        self.mlp1 = nn.Sequential(
            nn.Conv2d(dim, mlp_dim, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(mlp_dim, mlp_dim, kernel_size=3, groups=mlp_dim, padding=1),
            nn.GELU(),
            nn.Conv2d(mlp_dim, dim, kernel_size=1),
            nn.Dropout(dropout),
        )
        self.attn_global = GlocalAttention(dim)
        self.mlp2 = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        x = x + self.dw1(x)
        x = x + self.drop_path(self.attn_local(self.norm1(x)))
        x = x + self.drop_path(self.mlp1(self.norm2(x)))

        x = x + self.dw2(x)
        B, C, H, W = x.shape
        x = rearrange(x, "b c h w -> b (h w) c")
        x = x + self.drop_path(self.attn_global(self.norm3(x)))
        x = x + self.drop_path(self.mlp2(self.norm4(x)))
        x = rearrange(x, "b (h w) c -> b c h w", h=H)
        return x


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.up(x)
        return x


class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class DyLoC_LA(nn.Module):
    def __init__(
        self,
        input_channel=3,
        dims=[32, 64, 128],
        depths=[1, 1, 3, 5],
        kernels=[3, 3, 7],
        num_classes=1,
        drop_path=0.1
    ):
        super().__init__()
        self.stem = PatchEmbedding(input_channel, dims=dims, kernels=kernels, depths=depths)

        self.block = nn.Sequential(
            *[Block(dims[2], kernel=9, drop_path=drop_path) for _ in range(depths[3])]
        )
        self.down = nn.MaxPool2d(2, 2)

        self.Up2 = up_conv(dims[2], dims[2])
        self.Up_conv2 = conv_block(dims[2] * 2, dims[2])

        self.Up3 = up_conv(dims[2], dims[1])
        self.Up_conv3 = conv_block(dims[1] * 2, dims[1])
        self.Up4 = up_conv(dims[1], dims[0])
        self.Up_conv4 = conv_block(dims[0], dims[0])
        self.head = nn.Conv2d(dims[0], num_classes, kernel_size=1, stride=1, padding=0)
        

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        x, enc_output = self.stem(x)
        x = self.down(x)
        x = self.block(x)
        
        x = self.Up2(x)
        x = self.Up_conv2(torch.cat([x, enc_output[-1]], 1))

        x = self.Up3(x)
        x = self.Up_conv3(torch.cat([x, enc_output[-2]], 1))

        x = self.Up4(x)
        x = self.Up_conv4(x)
        x = self.head(x)
        return x


def DyLoC_LA_T(input_channel=3, num_classes=1):
    return DyLoC_LA(input_channel=input_channel, dims=[16, 32, 64], depths=[1, 1, 2, 1], drop_path=0.0, num_classes=num_classes)


def DyLoC_LA_S(input_channel=3, num_classes=1):
    return DyLoC_LA(input_channel=input_channel, dims=[32, 64, 128], depths=[1, 1, 3, 5], drop_path=0.1, num_classes=num_classes)


def DyLoC_LA_L(input_channel=3, num_classes=1):
    return DyLoC_LA(input_channel=input_channel, dims=[64, 128, 256], depths=[1, 1, 4, 6], drop_path=0.2, num_classes=num_classes)