import math
import torch
import einops
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from model.ffc import FFC_BN_ACT, SpectralTransform

# def conv3x3(in_planes, out_planes, stride=1) -> nn.Conv2d:
#     """
#     Returns a Conv2d object with 
#     in_channels=in_planes
#     out_channels=out_planes
#     stride=stride
#     kernel_size=(3,3)
#     padding=1
#     bias = False #due to batch norm layers
#     """
#     return nn.Conv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def default_conv(in_channels, out_channels, kernel_size, bias=True, groups = 1):
    wn = lambda x:torch.nn.utils.weight_norm(x)
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, groups = groups)

# class LayerNorm(nn.Module):
#     r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
#     The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
#     shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
#     with shape (batch_size, channels, height, width).
#     """
#     def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
#         super().__init__()
#         self.weight = nn.Parameter(torch.ones(normalized_shape))
#         self.bias = nn.Parameter(torch.zeros(normalized_shape))
#         self.eps = eps
#         self.data_format = data_format
#         if self.data_format not in ["channels_last", "channels_first"]:
#             raise NotImplementedError 
#         self.normalized_shape = (normalized_shape, )
    
#     def forward(self, x):
#         if self.data_format == "channels_last":
#             return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
#         elif self.data_format == "channels_first":
#             u = x.mean(1, keepdim=True)
#             s = (x - u).pow(2).mean(1, keepdim=True)
#             x = (x - u) / torch.sqrt(s + self.eps)
#             x = self.weight[:, None, None] * x + self.bias[:, None, None]
#             return x

class Scale(nn.Module):
    def __init__(self, value=1e-3) -> None:
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor([value]))
    def forward(self, x):
        return x * self.scale

# class MeanShift(nn.Conv2d):
#     def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
#         super(MeanShift, self).__init__(3, 3, kernel_size=1)
#         std = torch.Tensor(rgb_std)
#         self.weight.data = torch.eye(3).view(3, 3, 1, 1)
#         self.weight.data.div_(std.view(3, 1, 1, 1))
#         self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
#         self.bias.data.div_(std)
#         self.requires_grad = False

# class BasicBlock(nn.Sequential):
#     def __init__(
#         self, in_channels, out_channels, kernel_size, stride=1, bias=False,
#         bn=True, act=nn.ReLU(True)):

#         m = [nn.Conv2d(
#             in_channels, out_channels, kernel_size,
#             padding=(kernel_size//2), stride=stride, bias=bias)
#         ]
#         if bn: m.append(nn.BatchNorm2d(out_channels))
#         if act is not None: m.append(act)
#         super(BasicBlock, self).__init__(*m)

# class ResBlock(nn.Module):
#     def __init__(
#         self, conv, n_feats, kernel_size,
#         bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

#         super(ResBlock, self).__init__()
#         m = []
#         for i in range(2):
#             m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
#             if bn: m.append(nn.BatchNorm2d(n_feats))
#             if i == 0: m.append(act)

#         self.body = nn.Sequential(*m)
#         self.res_scale = res_scale

#     def forward(self, x):
#         res = self.body(x).mul(self.res_scale)
#         res += x

#         return res

# class LuConv(nn.Module):
#     def __init__(
#         self, conv, n_feats, kernel_size,
#         bias=True, bn=False, act=nn.LeakyReLU(0.05), res_scale=1):
#         super(LuConv, self).__init__()
#         #self.scale1 = Scale(1)
#         #self.scale2 = Scale(1)
#         m = []
#         for i in range(2):
#             m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
#             if bn: m.append(nn.BatchNorm2d(n_feats))
#             if i == 0: m.append(act)

#         self.body = nn.Sequential(*m)
#         self.res_scale = res_scale

#     def forward(self, x):
#         res = self.body(x)
#         return res
        
# class Upsampler(nn.Sequential):
#     def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

#         m = []
#         if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
#             for _ in range(int(math.log(scale, 2))):
#                 m.append(conv(n_feats, 4 * n_feats, 3, bias))
#                 m.append(nn.PixelShuffle(2))
#                 if bn: m.append(nn.BatchNorm2d(n_feats))

#                 if act == 'relu':
#                     m.append(nn.ReLU(True))
#                 elif act == 'prelu':
#                     m.append(nn.PReLU(n_feats))

#         elif scale == 3:
#             m.append(conv(n_feats, 9 * n_feats, 3, bias))
#             m.append(nn.PixelShuffle(3))
#             if bn: m.append(nn.BatchNorm2d(n_feats))

#             if act == 'relu':
#                 m.append(nn.ReLU(True))
#             elif act == 'prelu':
#                 m.append(nn.PReLU(n_feats))
#         else:
#             raise NotImplementedError

#         super(Upsampler, self).__init__(*m)

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

# class SEBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, use_spectral_norm=True) -> None:
#         super().__init__()

#         if use_spectral_norm:
#             self.se = nn.Sequential(
#                 nn.AdaptiveAvgPool2d(4),
#                 spectral_norm(
#                     nn.Conv2d(in_channels,out_channels,4,1,0,bias=False)
#                 ),
#                 Swish(),
#                 spectral_norm(
#                     nn.Conv2d(in_channels,out_channels,1,1,0,bias=False)
#                 ),
#                 nn.Sigmoid()
#             )
#         else:
#             self.se = nn.Sequential(
#                 nn.AdaptiveAvgPool2d(4),
#                 nn.Conv2d(in_channels,out_channels,4,1,0,bias=False),
#                 Swish(),
#                 nn.Conv2d(in_channels,out_channels,1,1,0,bias=False),
#                 nn.Sigmoid()
#             )

#     def forward(self, high_res_features, low_res_features):
#         return high_res_features * self.se(low_res_features)

# class NoiseInjection(nn.Module):
#     def __init__(self):
#         super().__init__()

#         self.weight = nn.Parameter(torch.zeros(1), requires_grad=True)

#     def forward(self, feat, noise=None):
#         if noise is None:
#             batch, _, height, width = feat.shape
#             noise = torch.randn(batch, 1, height, width).to(feat.device)

#         return feat + self.weight * noise

# class upSample(nn.Module):
#     def __init__(self, in_planes, out_planes, scale_factor=2):
#         super(upSample,self).__init__()

#         # self.conv1 = nn.ConvTranspose2d(in_channels=in_planes,out_channels=out_planes,kernel_size=4,stride=2,padding=1,bias=False)
#         self.upsample = nn.Upsample(scale_factor=scale_factor,mode="nearest")
#         self.conv3x3_1 = conv3x3(in_planes,out_planes)
#         self.conv3x3_2 = conv3x3(in_planes=out_planes, out_planes=out_planes)
#         self.conv3x3_3 = conv3x3(in_planes=out_planes, out_planes=out_planes)
#         self.shortcut = nn.Conv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=3, padding=1, stride=1, bias=False)
#         self.batch_norm = nn.BatchNorm2d(out_planes,0.8)
#         self.gamma  = nn.Parameter(torch.randn(1))

#     def forward(self,x):

#         upsample = self.upsample(x)
#         out = self.conv3x3_1(upsample)
#         out = self.conv3x3_2(out)
#         out = self.conv3x3_3(out)
#         out = F.leaky_relu(out,0.2,inplace=True)
#         out = out + self.shortcut(upsample) * self.gamma
#         out = self.batch_norm(out)
#         out = F.leaky_relu(out,0.2,True)
#         return out

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

    def forward(self, x):
        return self.net(x)

# class GLU(nn.Module):
#     def forward(self, x):
#         nc = x.size(1)
#         assert nc % 2 == 0, 'channels dont divide 2!'
#         nc = int(nc/2)
#         return x[:, :nc] * torch.sigmoid(x[:, nc:])

# def UpBlockComp(in_planes, out_planes):
#     block = nn.Sequential(
#         nn.Upsample(scale_factor=2, mode='nearest'),
#         nn.Conv2d(in_planes, out_planes*2, 3, 1, 1, bias=False),
#         #convTranspose2d(in_planes, out_planes*2, 4, 2, 1, bias=False),
#         NoiseInjection(),
#         nn.BatchNorm2d(out_planes*2), GLU(),
#         nn.Conv2d(out_planes, out_planes*2, 3, 1, 1, bias=False),
#         NoiseInjection(),
#         nn.BatchNorm2d(out_planes*2), GLU()
#         )
#     return block

# def UpBlock(in_planes, out_planes):
#     block = nn.Sequential(
#         nn.Upsample(scale_factor=2, mode='nearest'),
#         nn.Conv2d(in_planes, out_planes*2, 3, 1, 1, bias=False),
#         #convTranspose2d(in_planes, out_planes*2, 4, 2, 1, bias=False),
#         nn.BatchNorm2d(out_planes*2), GLU())
#     return block

# class ChannelAttention(nn.Module):
#     def __init__(self, channel, reduction=16) -> None:
#         super().__init__()

#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.conv = nn.Sequential(
#             nn.Conv2d(channel, channel//reduction, 1, padding=0, bias=True),
#             nn.ReLU(True),
#             nn.Conv2d(channel//reduction, channel, 1, padding=0, bias=True),
#             nn.Sigmoid()
#         )

#     def forward(self, x) -> torch.Tensor:
#         out = self.avg_pool(x)
#         out = self.conv(out)
#         return x * out

# class CAB(nn.Module):

#     def __init__(self, num_feat, compress_ratio=4, squeeze_factor=8):
#         super(CAB, self).__init__()

#         self.cab = nn.Sequential(
#             nn.Conv2d(num_feat, num_feat // compress_ratio, 3, 1, 1),
#             nn.GELU(),
#             nn.Conv2d(num_feat // compress_ratio, num_feat, 3, 1, 1),
#             ChannelAttention(num_feat, squeeze_factor)
#             )

#     def forward(self, x):
#         return self.cab(x)

# class FFCResBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0) -> None:
#         super().__init__()

#         self.conv1 = FFC_BN_ACT(
#             in_channels, out_channels, 1, stride=1, padding=0, activation_layer=nn.ReLU
#         )
#         self.conv2 = FFC_BN_ACT(
#             out_channels, out_channels, 1, stride=1, padding=0, activation_layer=nn.ReLU
#         )
    
#     def forward(self, x):
#         B,C,H,W = x.size()
#         in_xl, in_xg = x[:,:C//2], x[:,C//2:]

#         out = self.conv2(self.conv1(x))
#         out_xl, out_xg = out[:,:C//2], out[:,C//2:]
#         out_xl = out_xl + in_xl
#         out_xg = out_xg + in_xg
#         out = torch.cat([out_xl, out_xg], 1)
#         return out

# class Block(nn.Module):
#     def __init__(self, in_channels, out_channels) -> None:
#         super().__init__()

#         self.block = FFCResBlock(in_channels, out_channels, 3, 1, 1)

#         self.combiner = nn.Conv2d(2 * out_channels, out_channels, 1, 1, 0, bias=False)
#         self.tail_conv = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=True)
#         self.bn = nn.BatchNorm2d(out_channels)
#         self.attn = ChannelAttention(in_channels)
#         self.weight1 = Scale()
#         self.weight2 = Scale()

#     def forward(self, x):
#         ffc_out = self.block(x)

#         low_pass_filter = torch.ones_like(ffc_out) / (ffc_out.size(-2) * ffc_out.size(-1))
#         high_pass_features = ffc_out - low_pass_filter
        
#         out = torch.cat([self.weight1(ffc_out), self.weight2(high_pass_features)], dim=1)
#         out = self.tail_conv(self.attn(self.combiner(out)))
#         out = out + x
#         out = F.relu(self.bn(out))
#         return out

# class Block2(nn.Module):
#     def __init__(self, in_channels, out_channels) -> None:
#         super().__init__()

#         self.block = FFCResBlock(in_channels, out_channels, 3, 1, 1)

#         self.conv_head = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(True)
#         )

#         self.conv_tail = nn.Sequential(
#             nn.Conv2d(out_channels, out_channels, 1, 1, 0, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(True)
#         )

#         self.combiner = nn.Conv2d(2 * out_channels, out_channels, 1, 1, 0, bias=True)

#         # self.tail_conv = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=True)
#         # self.bn = nn.BatchNorm2d(out_channels)
#         self.attn = ChannelAttention(in_channels)
#         self.weight1 = Scale()
#         self.weight2 = Scale()

#     def forward(self, x):
#         ffc_out = self.block(x)

#         out = self.conv_head(x)
#         low_pass_filter = torch.ones_like(out) / (out.size(-2) * out.size(-1))
#         out = out - low_pass_filter
#         out = self.conv_tail(out)

#         out = torch.cat([out, ffc_out], 1)
#         out = self.attn(self.combiner(out))
#         out = out + x
#         return out

# class HiFB(nn.Module):
#     def __init__(self, in_channels, out_channels) -> None:
#         super().__init__()

#         self.block = FFCResBlock(in_channels, out_channels, 1, 1, 0)

#         self.conv_head = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(True)
#         )

#         self.conv_tail = nn.Sequential(
#             nn.Conv2d(out_channels, out_channels, 1, 1, 0, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(True)
#         )

#         self.combiner = nn.Conv2d(2 * out_channels, out_channels, 1, 1, 0, bias=True)
#         self.attn = ChannelAttention(in_channels)
#         self.alise = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=True)

    
#     def forward(self, x):
#         ffc_out = x + self.block(x)

#         out = self.conv_head(x)
#         low_pass_filter = torch.ones_like(out) / (out.size(-2) * out.size(-1))
#         out = out - low_pass_filter
#         out = self.conv_tail(out)
#         out = out + x

#         out = torch.cat([out, ffc_out], 1)
#         out = self.alise(self.attn(self.combiner(out)))
#         return out

# class HiFB2(nn.Module):
#     def __init__(self, in_channels, out_channels) -> None:
#         super().__init__()

#         self.block = FFCResBlock(in_channels, out_channels, 3, 1, 1)

#         self.conv_head = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(True)
#         )

#         self.conv_tail = nn.Sequential(
#             nn.Conv2d(out_channels, out_channels, 1, 1, 0, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(True)
#         )

#         self.lamb = nn.Parameter(torch.zeros(out_channels), requires_grad=True)
#         self.combiner = nn.Conv2d(2 * out_channels, out_channels, 1, 1, 0, bias=True)
#         self.attn = ChannelAttention(in_channels)
#         self.alise = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=True)

    
#     def forward(self, x):
#         ffc_out = self.block(x) + x

#         out = self.conv_head(x)
#         low_pass_filter = torch.ones_like(out) / (out.size(-2) * out.size(-1))
#         out = out - low_pass_filter
#         out = out * (1. + self.lamb[None,:, None, None])
#         out = out + low_pass_filter
#         out = self.conv_tail(out)
#         out = out + x

#         out = torch.cat([out, ffc_out], 1)
#         out = self.alise(self.attn(self.combiner(out)))
#         return out

# class HiFB3(nn.Module):
#     def __init__(self, in_channels, out_channels) -> None:
#         super().__init__()

#         # self.block = FFCResBlock(in_channels, out_channels, 3, 1, 1)
#         self.ffc = FFC_BN_ACT(in_channels, out_channels, 3, 0.5, 0.5, 1, 1, activation_layer=nn.ReLU)

#         self.conv_head = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels // 2, 1, 1, 0, bias=False),
#             nn.BatchNorm2d(out_channels // 2),
#             nn.ReLU(True)
#         )

#         self.conv_tail = nn.Sequential(
#             nn.Conv2d(out_channels // 2, out_channels, 1, 1, 0, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(True)
#         )

#         # self.lamb = nn.Parameter(torch.zeros(out_channels // 2), requires_grad=True)
#         self.combiner = nn.Conv2d(2 * out_channels, out_channels, 1, 1, 0, bias=True)
#         self.attn = ChannelAttention(out_channels)

#     def forward(self, x):
#         B,C,H,W = x.size()

#         in_xl, in_xg = x[:,:C//2], x[:,C//2:]

#         ffc_out = self.ffc(x)
        
#         out_xl, out_xg = ffc_out[:,:C//2], ffc_out[:,C//2:]
#         out_xl = out_xl + in_xl
#         out_xg = out_xg + in_xg
        
#         ffc_out = torch.cat([out_xl, out_xg], dim=1)

#         out = self.conv_head(x)
#         low_pass_filter = torch.ones_like(out) / (out.size(-2) * out.size(-1))
#         out = out - low_pass_filter
#         # out = out * (1. + self.lamb[None,:, None, None])
#         # out = out + low_pass_filter
#         out = self.conv_tail(out)
#         out = out + x

#         out = torch.cat([out, ffc_out], 1)
#         out = self.combiner(out)
#         out = self.attn(out)
#         return out

# class HiFB4(nn.Module):
#     def __init__(self, in_channels, out_channels) -> None:
#         super().__init__()

#         # self.block = FFCResBlock(in_channels, out_channels, 3, 1, 1)
#         self.ffc = FFC_BN_ACT(in_channels // 2, out_channels // 2, 1, 0.5, 0.5, 1, 0, activation_layer=nn.ReLU, enable_lfu=True)

#         self.conv_head = nn.Sequential(
#             nn.Conv2d(in_channels // 2, out_channels // 2, 1, 1, 0, bias=False),
#             nn.BatchNorm2d(out_channels // 2),
#             nn.ReLU(True)
#         )

#         self.conv_tail = nn.Sequential(
#             nn.Conv2d(out_channels // 2, out_channels // 2, 1, 1, 0, bias=False),
#             nn.BatchNorm2d(out_channels // 2),
#             nn.ReLU(True)
#         )

#         self.scam = SCAM(out_channels // 2, False)
#         # self.lamb = nn.Parameter(torch.zeros(out_channels // 2), requires_grad=True)
#         # self.combiner = nn.Conv2d(out_channels, out_channels, 1, 1, 0, bias=True)
#         self.attn = ChannelAttention(out_channels)

#     def forward(self, x):
#         B,C,H,W = x.size()

#         in_xl, in_xg = x[:,:C//2], x[:,C//2:]
        
#         fourier_x = in_xl
#         spatial_x = in_xg

#         fourier_xl, fourier_xg = fourier_x[:,:C//4], fourier_x[:,C//4:]

#         ffc_out = self.ffc(fourier_x)

#         ffc_out[:,:C//4] = ffc_out[:,:C//4] + fourier_xl
#         ffc_out[:,C//4:] = ffc_out[:,C//4:] + fourier_xg

#         out = self.conv_head(spatial_x)
#         low_pass_filter = torch.ones_like(out) / (out.size(-2) * out.size(-1))
#         out = out - low_pass_filter
#         out = self.conv_tail(out)
#         out = out + in_xg

#         ffc_out, out = self.scam(ffc_out, out)

#         out = torch.cat([ffc_out, out], 1)
#         # out = self.combiner(out)
#         out = self.attn(out)
#         return out

# class HiFB5(nn.Module):
#     def __init__(self, in_channels, out_channels) -> None:
#         super().__init__()

#         # self.block = FFCResBlock(in_channels, out_channels, 3, 1, 1)
#         self.ffc = FFC_BN_ACT(in_channels // 2, out_channels // 2, 3, 0.5, 0.5, 1, 1, activation_layer=nn.ReLU, enable_lfu=True)
#         self.scam = SCAM(out_channels // 2, False)
        
#         self.attn = ChannelAttention(out_channels)


#     def forward(self, x):
#         B,C,H,W = x.size()

#         in_xl, in_xg = x[:,:C//2], x[:,C//2:]


#         fourier_x = in_xl
#         spatial_x = in_xg

#         fourier_xl, fourier_xg = fourier_x[:,:C//4], fourier_x[:,C//4:]
#         ffc_out = self.ffc(fourier_x)
        
#         ffc_out[:,:C//4] = ffc_out[:,:C//4] + fourier_xl
#         ffc_out[:,C//4:] = ffc_out[:,C//4:] + fourier_xg

#         ffc_out, out = self.scam(ffc_out, spatial_x)
#         # print(ffc_out.shape)
#         out = torch.cat([ffc_out, out], 1)
#         out = self.attn(out)
#         return out

class one_conv(nn.Module):
    def __init__(self, in_channels, growth_rate, kernel_size = 3, relu = True):
        super(one_conv,self).__init__()
        self.conv = nn.Conv2d(in_channels, growth_rate, kernel_size=kernel_size, padding = kernel_size>>1, stride= 1)
        self.flag = relu
        self.conv1 = nn.Conv2d(growth_rate, in_channels, kernel_size=kernel_size, padding = kernel_size>>1, stride= 1)
        if relu:
            self.relu = nn.PReLU(growth_rate)
        self.w = Scale(1.)
    def forward(self,x):
        if self.flag == False:
            output = x + self.w(self.conv1(self.conv(x)))
        else:
            output = x + self.w(self.conv1(self.relu(self.conv(x))))
        return output

class HiFB6(nn.Module):
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

    def forward(self, x):
        B,C,H,W = x.size()

        in_xl, in_xg = x[:,:C//2], x[:,C//2:]
        
        fourier_x = in_xl
        spatial_x = in_xg

        # fourier_xl, fourier_xg = fourier_x[:,:C//4], fourier_x[:,C//4:]

        ffc_out = self.ffc(fourier_x) + fourier_x

        # ffc_out[:,:C//4] = ffc_out[:,:C//4] + fourier_xl
        # ffc_out[:,C//4:] = ffc_out[:,C//4:] + fourier_xg

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

# class HiFB7(nn.Module):
#     def __init__(self, in_channels, out_channels, mlp_ratio=0.5) -> None:
#         super().__init__()

#         self.reduce = nn.Conv2d(in_channels, out_channels//2, 1, 1, 0)
        
#         self.ffc1 = FFC_Wrap(out_channels // 2, out_channels // 2, 3, 0.5, 0.5, 1, 1, activation_layer=nn.ReLU, enable_lfu=True)
        
#         self.body1 = nn.Sequential(
#             *[self.ffc1 for _ in range(3)]
#         )

#         self.alise = nn.Conv2d(out_channels // 2, out_channels, 1, 1, 0)
#         self.attn = ChannelAttention(out_channels//2)
#         # self.alise = nn.Conv2d(out_channels, out_channels,3,1,1)

#         # self.ffc2 = FFC_Wrap(in_channels, out_channels, 3, 0.5, 0.5, 1, 1, activation_layer=nn.ReLU, enable_lfu=True)
 
#     def forward(self, x):
#         B,C,H,W = x.size()
#         out1 = self.reduce(x)
#         out = self.body1(out1)
#         out = self.attn(out)
#         out = self.alise(out)
#         return out + x

# class HiFB8(nn.Module):
#     def __init__(self, in_channels, out_channels, mlp_ratio=0.5) -> None:
#         super().__init__()

#         self.conv = nn.Conv2d(in_channels, out_channels // 2, 1, 1, 0)
#         self.spectral_transform1 = SpectralTransform(out_channels // 2, out_channels // 2, 1, enable_lfu=True)
#         self.spectral_transform2 = SpectralTransform(out_channels // 2, out_channels // 2, 1, enable_lfu=True)
#         self.spectral_transform3 = SpectralTransform(out_channels // 2, out_channels // 2, 1, enable_lfu=True)
#         self.expand = nn.Conv2d(out_channels // 2, out_channels, 1, 1, 0)
 
#     def forward(self, x):
#         B,C,H,W = x.size()
#         out = self.conv(x)
#         out = self.spectral_transform1(out)
#         out = self.spectral_transform2(out)
#         out = self.spectral_transform3(out)
#         out = self.expand(out)
#         return out + x

# class HiFB9(nn.Module):
#     def __init__(self, in_channels, out_channels) -> None:
#         super().__init__()

#         self.ffc = FFC_BN_ACT(in_channels // 2, out_channels // 2, 3, 0.5, 0.5, 1, 1, activation_layer=nn.ReLU, enable_lfu=True)

#         self.conv3_1 = nn.Conv2d(in_channels//2, out_channels // 4, 3, 1, 1, bias=True)
#         self.conv3_2 = nn.Conv2d(out_channels//4, out_channels//2, 3, 1, 1, bias=True)

#         self.scam = SCAM(out_channels // 2, True)

#     def forward(self, x):
#         B,C,H,W = x.size()

#         in_xl, in_xg = x[:,:C//2], x[:,C//2:]
        
#         fourier_x = in_xl
#         spatial_x = in_xg

#         fourier_xl, fourier_xg = fourier_x[:,:C//4], fourier_x[:,C//4:]

#         ffc_out = self.ffc(fourier_x)

#         out = spatial_x
#         out = self.conv3_1(out)
#         out = F.relu(out)
#         out = self.conv3_2(out)
#         # out = out + spatial_x

#         ffc_out, out = self.scam(ffc_out, out)

#         out = torch.cat([ffc_out, out], 1)
#         return out + x

# class HiFB10(nn.Module):
#     def __init__(self, in_channels, out_channels) -> None:
#         super().__init__()

#         self.ffc = FFC_BN_ACT(in_channels, out_channels, 3, 0.5, 0.5, 1, 1, activation_layer=nn.ReLU, enable_lfu=True)
#         self.conv3_1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=True)
#         self.attn = ChannelAttention(out_channels)

#     def forward(self, x):
#         B,C,H,W = x.size()

#         out = self.ffc(x)
#         out = self.conv3_1(out)
#         out = self.attn(out)
#         return out + x

# class ResBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True) -> None:
#         super().__init__()

#         self.block1 = HiFB(in_channels, out_channels)
#         self.block2 = HiFB(out_channels, out_channels)

#     def forward(self, x):
#         out = self.block1(x)        
#         out = self.block2(out)
#         return out + x

# class MS_CAM(nn.Module):

#     def __init__(self, channels=64, reduction=2):
#         super(MS_CAM, self).__init__()
#         inter_channels = int(channels // reduction)

#         self.local_att = nn.Sequential(
#             nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
#             nn.BatchNorm2d(inter_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
#             nn.BatchNorm2d(channels),
#         )

#         self.global_att = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
#             # nn.InstanceNorm2d(inter_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
#             # nn.InstanceNorm2d(channels),
#         )

#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         xl = self.local_att(x)
#         xg = self.global_att(x)
#         xlg = xl + xg
#         wei = self.sigmoid(xlg)
#         return x * wei

# class AFF(nn.Module):
#     '''
#     多特征融合 AFF
#     '''

#     def __init__(self, channels=64, r=4):
#         super(AFF, self).__init__()
#         inter_channels = int(channels // r)

#         self.local_att = nn.Sequential(
#             nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
#             nn.BatchNorm2d(inter_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
#             nn.BatchNorm2d(channels),
#         )

#         self.global_att = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
#             # nn.BatchNorm2d(inter_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
#             # nn.BatchNorm2d(channels),
#         )

#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x, residual):
#         xa = x + residual
#         xl = self.local_att(xa)
#         xg = self.global_att(xa)
#         xlg = xl + xg
#         wei = self.sigmoid(xlg)

#         xo = 2 * x * wei + 2 * residual * (1 - wei)
#         return xo

class SCAM(nn.Module):
    def __init__(self, channels, bias=False) -> None:
        super().__init__()

        self.ln1 = nn.LayerNorm(channels)
        self.ln2 = nn.LayerNorm(channels)

        self.q = nn.Linear(channels, channels, bias=bias)
        self.k = nn.Linear(channels, channels, bias=bias)

        self.v1 = nn.Linear(channels, channels, bias=bias)
        self.v2 = nn.Linear(channels, channels, bias=bias)

    def forward(self, x_l: torch.Tensor, x_h: torch.Tensor):
        
        B,C,H,W = x_l.size()

        x_l = x_l.permute(0, 2, 3, 1)
        x_h = x_h.permute(0, 2, 3, 1)

        x_l_v = self.v1(x_l)
        x_h_v = self.v2(x_h)

        q = self.q(self.ln1(x_l)) # (B,H,W,C)
        k = self.k(self.ln2(x_h)) # (B,H,W,C)

        attn = (q @ k.transpose(2, 3)) / math.sqrt(C)

        attn_l = F.softmax(attn.transpose(2,3),dim = -1) # (B, H, W, W)
        attn_h = F.softmax(attn, dim=-1)  # (B, H, W, W)

        x_l_o = attn_l @ x_l_v    # (B, H, W, C)
        x_h_o = attn_h @ x_h_v

        x_h = x_h + x_l_o
        x_l = x_l + x_h_o

        x_h = x_h.permute(0, 3, 1, 2)
        x_l = x_l.permute(0, 3, 1, 2)
        return x_l, x_h

# class RFB2(nn.Module):
#     def __init__(self, in_channels, out_channels):
        
#         super(RFB2, self).__init__()
        
#         self.conv = nn.Sequential(
#             nn.Conv2d(out_channels, out_channels, 1, 1, 0, bias=True),
#         )

#         self.hfb1 = HiFB3(out_channels, out_channels)
#         self.hfb2 = HiFB3(out_channels, out_channels)

#         self.conv_final = nn.Sequential(
#             nn.Conv2d(out_channels, out_channels, 1, 1, 0, bias=True),
#         )

#         self.attn = CAB(out_channels)
#         self.alise = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)

#     def forward(self, x):
#         out = self.conv(x)
        
#         hfb = self.hfb1(out)
#         hfb = self.hfb2(hfb)

#         hfb = hfb + out
#         out = self.conv_final(hfb)
#         out = self.alise(self.attn(out))
#         return out + x

# class RFB(nn.Module):
#     def __init__(self, in_channels, out_channels):
        
#         super(RFB, self).__init__()
#         # wn = lambda x:torch.nn.utils.weight_norm(x)

#         self.fb1 = HiFB2(in_channels, out_channels)     
#         self.fb2 = HiFB2(out_channels, out_channels)
#         self.fb3 = HiFB2(out_channels, out_channels)
#         self.conv = BasicConv(2 * out_channels, out_channels, 1, 1, 0)
        
#         self.downsample = nn.AvgPool2d(2)
#         self.attention = ChannelAttention(out_channels)

#     def forward(self, x):
#         out = self.fb1(x)
#         downsample = self.downsample(out)
#         high = out - F.interpolate(downsample, size = out.size()[-2:], mode='bilinear', align_corners=True)
#         high = self.fb2(high)

#         out = torch.cat([out, high], dim = 1)
#         out = self.conv(out)
#         out = self.attention(out)
#         out = self.fb3(out)
#         out = out + x
#         return out

# class PixelAttention(nn.Module):
#     def __init__(self, channels) -> None:
#         super().__init__()
#         self.attn = nn.Sequential(
#             nn.Conv2d(channels, channels, 1, 1, 0, bias=True),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         y = self.attn(x)
#         out = torch.mul(x,y)
#         return out