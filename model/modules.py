import math
import torch
import einops
import torch.nn as nn
import torch.nn.functional as F
from model.ffc import FFC_BN_ACT
from typing import List, Tuple, Optional

def default_conv(in_channels, out_channels, kernel_size, bias=True, groups = 1):
    wn = lambda x:torch.nn.utils.weight_norm(x)
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, groups = groups)

class Scale(nn.Module):
    def __init__(self, value: Optional[float]=1e-3) -> None:
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor([value]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale

def default(val, d):
    if val is not None:
        return val
    return d() if callable(d) else d

class PixelShuffleUpsample(nn.Module):
    """
    code shared by @MalumaDev at DALLE2-pytorch for addressing checkboard artifacts
    https://arxiv.org/ftp/arxiv/papers/1707/1707.02937.pdf
    """
    def __init__(self, dim, dim_out = None, scale=2):
        super().__init__()
        self.scale = scale
        dim_out = default(dim_out, dim)
        hidden_dims = dim_out * 9 if scale == 3 else dim_out * 4
        conv = nn.Conv2d(dim, hidden_dims, 1)

        self.net = nn.Sequential(
            conv,
            nn.SiLU(),
            nn.PixelShuffle(3) if scale == 3 else nn.PixelShuffle(2) 
        )

        self.init_conv_(conv)

    def init_conv_(self, conv):
        o, i, h, w = conv.weight.shape
        conv_weight = torch.empty(o // (4 if self.scale == 2 else 9), i, h, w)
        nn.init.kaiming_uniform_(conv_weight)
        
        if self.scale == 3:
            conv_weight = einops.repeat(conv_weight, 'o ... -> (o 9) ...')
        else:
            conv_weight = einops.repeat(conv_weight, 'o ... -> (o 4) ...')
        
        conv.weight.data.copy_(conv_weight)
        nn.init.zeros_(conv.bias.data)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class one_conv(nn.Module):
    def __init__(self, in_channels, growth_rate, kernel_size = 3, relu = True):
        super(one_conv,self).__init__()
        self.conv = nn.Conv2d(in_channels, growth_rate, kernel_size=kernel_size, padding = kernel_size>>1, stride= 1)
        self.flag = relu
        self.conv1 = nn.Conv2d(growth_rate, in_channels, kernel_size=kernel_size, padding = kernel_size>>1, stride= 1)
        
        if relu:
            self.relu = nn.PReLU(growth_rate)
        
        self.w = Scale(1.)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.flag == False:
            output = self.w(self.conv1(self.conv(x)))
        else:
            output = self.w(self.conv1(self.relu(self.conv(x))))
        return output + x

class RFB(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()

        self.ffc = FFC_BN_ACT(in_channels // 2, out_channels // 2, 3, 0.5, 0.5, 1, 1, activation_layer=nn.ReLU, enable_lfu=True, efficient_conv=False)

        self.downsample = nn.AvgPool2d(2)
        extractor = one_conv(out_channels // 2, out_channels // 4, 3)
        self.extractor_body = nn.Sequential(
            *[
                extractor for _ in range(5)
            ]
        )

        self.scam = SCAM(out_channels // 2, True)
        self.conv1x1 = nn.Conv2d(out_channels // 2, out_channels // 2, 1, 1, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B,C,H,W = x.size()

        in_xl, in_xg = x[:,:C//2], x[:,C//2:]
        
        fourier_x = in_xl
        spatial_x = in_xg

        ffc_out = self.ffc(fourier_x) + fourier_x

        out = spatial_x
        downsample = self.downsample(spatial_x)
        downsample = F.interpolate(downsample, size=out.size()[-2:], mode='bilinear', align_corners=False)
        high = out - downsample
        high = self.extractor_body(high)
        out = high + downsample
        out = self.conv1x1(out) + spatial_x

        ffc_out, out = self.scam(ffc_out, out)

        out = torch.cat([ffc_out, out], 1)
        return out + x

class SCAM(nn.Module):
    def __init__(self, channels, bias=False) -> None:
        super().__init__()

        self.ln1 = nn.LayerNorm(channels)
        self.ln2 = nn.LayerNorm(channels)

        self.q = nn.Linear(channels, channels, bias=bias)
        self.k = nn.Linear(channels, channels, bias=bias)

        self.v1 = nn.Linear(channels, channels, bias=bias)
        self.v2 = nn.Linear(channels, channels, bias=bias)

    def forward(self, x_l: torch.Tensor, x_h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        
        B,C,H,W = x_l.size()

        x_l = x_l.permute(0, 2, 3, 1)
        x_h = x_h.permute(0, 2, 3, 1)

        x_l_v = self.v1(x_l)
        x_h_v = self.v2(x_h)

        q = self.q(self.ln1(x_l)) # (B,H,W,C)
        k = self.k(self.ln2(x_h)) # (B,H,W,C)

        attn = (q @ k.transpose(2, 3)) * (1.0 / math.sqrt(C))

        attn_l = F.softmax(attn.transpose(2,3),dim = -1) # (B, H, W, W)
        attn_h = F.softmax(attn, dim=-1)  # (B, H, W, W)

        x_l_o = attn_l @ x_l_v    # (B, H, W, C)
        x_h_o = attn_h @ x_h_v

        x_h = x_h + x_l_o
        x_l = x_l + x_h_o

        x_h = x_h.permute(0, 3, 1, 2)
        x_l = x_l.permute(0, 3, 1, 2)
        return x_l, x_h