# Fast Fourier Convolution NeurIPS 2020
# original implementation https://github.com/pkumivision/FFC/blob/main/model_zoo/ffc.py
# paper https://proceedings.neurips.cc/paper/2020/file/2fd5d41ec6cfab47e32164d5624269b1-Paper.pdf

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.jit as jit

class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        res = x * y.expand_as(x)
        return res


class FFCSE_block(nn.Module):

    def __init__(self, channels, ratio_g):
        super(FFCSE_block, self).__init__()
        in_cg = int(channels * ratio_g)
        in_cl = channels - in_cg
        r = 16

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv1 = nn.Conv2d(channels, channels // r,
                               kernel_size=1, bias=True)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv_a2l = None if in_cl == 0 else nn.Conv2d(
            channels // r, in_cl, kernel_size=1, bias=True)
        self.conv_a2g = None if in_cg == 0 else nn.Conv2d(
            channels // r, in_cg, kernel_size=1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x if type(x) is tuple else (x, 0)
        id_l, id_g = x

        x = id_l if type(id_g) is int else torch.cat([id_l, id_g], dim=1)
        x = self.avgpool(x)
        x = self.relu1(self.conv1(x))

        x_l = 0 if self.conv_a2l is None else id_l * \
            self.sigmoid(self.conv_a2l(x))
        x_g = 0 if self.conv_a2g is None else id_g * \
            self.sigmoid(self.conv_a2g(x))
        return x_l, x_g


class FourierUnit(nn.Module):

    def __init__(self, in_channels, out_channels, groups=1, spatial_scale_factor=None, spatial_scale_mode='bilinear',
                 spectral_pos_encoding=False, use_se=False, se_kwargs=None, ffc3d=False, fft_norm='ortho'):
        # bn_layer not used
        super(FourierUnit, self).__init__()
        self.groups = groups

        self.conv_layer = torch.nn.Conv2d(in_channels=in_channels * 2 + (2 if spectral_pos_encoding else 0),
                                          out_channels=out_channels * 2,
                                          kernel_size=1, stride=1, padding=0, groups=self.groups, bias=False)
        self.bn = torch.nn.BatchNorm2d(out_channels * 2)
        self.relu = torch.nn.ReLU()

        # squeeze and excitation block
        self.use_se = use_se
        if use_se:
            if se_kwargs is None:
                se_kwargs = {}
            self.se = SELayer(self.conv_layer.in_channels, **se_kwargs)

        self.spatial_scale_factor = spatial_scale_factor
        self.spatial_scale_mode = spatial_scale_mode
        self.spectral_pos_encoding = spectral_pos_encoding
        self.ffc3d = ffc3d
        self.fft_norm = fft_norm

    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]

        if self.spatial_scale_factor is not None:
            orig_size = x.shape[-2:]
            x = F.interpolate(x, scale_factor=self.spatial_scale_factor, mode=self.spatial_scale_mode, align_corners=False)

        r_size = x.size()
        # (batch, c, h, w/2+1, 2)
        # fft_dim = (-3, -2, -1) if self.ffc3d else (-2, -1)
        fft_dim = (-2,-1)
        ffted = torch.fft.rfftn(x, dim=fft_dim, norm=self.fft_norm)
        ffted = torch.stack((ffted.real, ffted.imag), dim=-1)
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()  # (batch, c, 2, h, w/2+1)
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])

        # if self.spectral_pos_encoding:
        #     height, width = ffted.shape[-2:]
        #     coords_vert = torch.linspace(0, 1, height)[None, None, :, None].expand(batch, 1, height, width).to(ffted)
        #     coords_hor = torch.linspace(0, 1, width)[None, None, None, :].expand(batch, 1, height, width).to(ffted)
        #     ffted = torch.cat((coords_vert, coords_hor, ffted), dim=1)

        # if self.use_se:
        #     ffted = self.se(ffted)

        ffted = self.conv_layer(ffted)  # (batch, c*2, h, w/2+1)
        ffted = self.relu(self.bn(ffted))

        ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:]).permute(
            0, 1, 3, 4, 2).contiguous()  # (batch,c, t, h, w/2+1, 2)
        ffted = torch.complex(ffted[..., 0], ffted[..., 1])

        ifft_shape_slice = x.shape[-3:] if self.ffc3d else x.shape[-2:]
        output = torch.fft.irfftn(ffted, s=ifft_shape_slice, dim=fft_dim, norm=self.fft_norm)

        if self.spatial_scale_factor is not None:
            output = F.interpolate(output, size=orig_size, mode=self.spatial_scale_mode, align_corners=False)

        return output


class SpectralTransform(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, groups=1, enable_lfu=False, **fu_kwargs):
        # bn_layer not used
        super(SpectralTransform, self).__init__()

        self.enable_lfu = enable_lfu
        if stride == 2:
            self.downsample = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        else:
            self.downsample = nn.Identity()

        self.stride = stride
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels //
                      2, kernel_size=1, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True)
        )
        self.fu = FourierUnit(
            out_channels // 2, out_channels // 2, groups, **fu_kwargs)
        if self.enable_lfu:
            self.lfu = FourierUnit(
                out_channels // 2, out_channels // 2, groups)
        self.conv2 = torch.nn.Conv2d(
            out_channels // 2, out_channels, kernel_size=1, groups=groups, bias=False)

    
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.downsample(x)
        x = self.conv1(x)
        output = self.fu(x)

        if self.enable_lfu:
            # n, c, h, w = x.shape
            # split_no = 2
            # split_s = h // split_no
            
            # xs = torch.cat(torch.split(
            #     x[:, :c // 4], split_s, dim=-2), dim=1).contiguous()
            
            # xs = torch.cat(torch.split(xs, split_s, dim=-1),
            #                dim=1).contiguous()
            # xs = self.lfu(xs)
            # xs = xs.repeat(1, 1, split_no, split_no).contiguous()

            n, c, h, w = x.shape
            split_no = 2
            split_s_h = h // split_no
            split_s_w = w // split_no

            xs = x
            if h % 2 == 1:
                split_s_h += 1
                # x = F.pad(x, (0,0,1,0))
                xs = F.interpolate(xs, size=(h+1,w), mode='bilinear', align_corners=True)
            
            xs = torch.cat(torch.split(
                xs[:, :c // 4], split_s_h, dim=-2), dim=1).contiguous()

            if w % 2 == 1:
                split_s_w += 1
                # xs = F.pad(xs, (0, 0, 0, 1))
                xs = F.interpolate(xs, size=(xs.size(-2),xs.size(-1)+1), mode='bilinear', align_corners=True)
            
            xs = torch.cat(torch.split(xs, split_s_w, dim=-1),
                           dim=1).contiguous()
            xs = self.lfu(xs)
            xs = xs.repeat(1, 1, split_no, split_no).contiguous()
            xs = xs[:,:,:h,:w]
        else:
            xs = torch.zeros(0, device=x.device)

        output = self.conv2(x + output + xs)

        return output


class FFC(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 ratio_gin=0.5, ratio_gout=0.5, stride=1, padding=0,
                 dilation=1, groups=1, bias=False, enable_lfu=False, efficient_conv=False,
                 padding_type='reflect', gated=False, **spectral_kwargs):

        super(FFC, self).__init__()

        assert stride == 1 or stride == 2, "Stride should be 1 or 2."
        self.stride = stride

        in_cg = int(in_channels * ratio_gin)
        in_cl = in_channels - in_cg
        out_cg = int(out_channels * ratio_gout)
        out_cl = out_channels - out_cg
        #groups_g = 1 if groups == 1 else int(groups * ratio_gout)
        #groups_l = 1 if groups == 1 else groups - groups_g

        self.ratio_gin = ratio_gin
        self.ratio_gout = ratio_gout
        self.global_in_num = in_cg

        module = nn.Identity if in_cl == 0 or out_cl == 0 else nn.Conv2d
        if efficient_conv:
            self.convl2l = nn.Sequential(
                nn.Conv2d(in_cl, in_cl, 3, 1, 1, groups=in_cl, bias=bias),
                nn.Conv2d(in_cl, out_cl, 1, 1, 0, bias=bias)
            )
        else:
            self.convl2l = module(in_cl, out_cl, kernel_size,
                                stride, padding, dilation, groups, bias, padding_mode=padding_type)

        module = nn.Identity if in_cl == 0 or out_cg == 0 else nn.Conv2d
        if efficient_conv:
            self.convl2g = nn.Sequential(
                nn.Conv2d(in_cl, in_cl, 3, 1, 1, groups=in_cl, bias=bias),
                nn.Conv2d(in_cl, out_cg, 1, 1, 0, bias=bias)
            )
        else:
            self.convl2g = module(in_cl, out_cg, kernel_size,
                                stride, padding, dilation, groups, bias, padding_mode=padding_type)

        module = nn.Identity if in_cg == 0 or out_cl == 0 else nn.Conv2d
        if efficient_conv:
            self.convg2l = nn.Sequential(
                nn.Conv2d(in_cg, in_cg, 3, 1, 1, groups=in_cg, bias=bias),
                nn.Conv2d(in_cg, out_cl, 1, 1, 0, bias=bias)
            )
        else:
            self.convg2l = module(in_cg, out_cl, kernel_size,
                            stride, padding, dilation, groups, bias, padding_mode=padding_type)

        module = nn.Identity if in_cg == 0 or out_cg == 0 else SpectralTransform
        self.convg2g = module(
            in_cg, out_cg, stride, 1 if groups == 1 else groups // 2, enable_lfu, **spectral_kwargs)

        self.gated = gated
        module = nn.Identity if in_cg == 0 or out_cl == 0 or not self.gated else nn.Conv2d
        self.gate = module(in_channels, 2, 1)

    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B,C,H,W = x.size()
        # print(x.size())
        # x_l, x_g = x if type(x) is tuple else (x, 0)
        x_l, x_g = x[:,:C//2], x[:,C//2:]
        out_xl: torch.Tensor = torch.zeros(1, device=x.device)
        out_xg: torch.Tensor = torch.zeros(1, device=x.device)

        # if self.gated:
        #     total_input_parts = [x_l]
        #     if torch.as_tensor(x_g):
        #         total_input_parts.append(x_g)
        #     total_input = torch.cat(total_input_parts, dim=1)

        #     gates = torch.sigmoid(self.gate(total_input))
        #     g2l_gate, l2g_gate = gates.chunk(2, dim=1)
        # else:
        #     g2l_gate, l2g_gate = torch.ones(1, device=x.device), torch.ones(1, device=x.device)

        g2l_gate, l2g_gate = 1, 1

        if self.ratio_gout != 1:
            out_xl = self.convl2l(x_l) + self.convg2l(x_g) * g2l_gate
        if self.ratio_gout != 0:
            out_xg = self.convl2g(x_l) * l2g_gate + self.convg2g(x_g)

        out = torch.cat([out_xl, out_xg], dim=1)
        return out


class FFC_BN_ACT(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size, ratio_gin=0.5, ratio_gout=0.5,
                 stride=1, padding=0, dilation=1, groups=1, bias=False,
                 norm_layer=nn.BatchNorm2d, activation_layer=nn.Identity,
                 efficient_conv=False,
                 padding_type='reflect',
                 enable_lfu=False, **kwargs):

        super(FFC_BN_ACT, self).__init__()

        self.ffc = FFC(in_channels, out_channels, kernel_size,
                       ratio_gin, ratio_gout, stride, padding, dilation,
                       groups, bias, enable_lfu, efficient_conv=efficient_conv, padding_type=padding_type, **kwargs)
        lnorm = nn.Identity if ratio_gout == 1 else norm_layer
        gnorm = nn.Identity if ratio_gout == 0 else norm_layer
        global_channels = int(out_channels * ratio_gout)
        self.bn_l = lnorm(out_channels - global_channels)
        self.bn_g = gnorm(global_channels)

        lact = nn.Identity if ratio_gout == 1 else activation_layer
        gact = nn.Identity if ratio_gout == 0 else activation_layer
        self.act_l = lact()
        self.act_g = gact()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_ffc = self.ffc(x)
        B,C,H,W = x_ffc.size()
        x_l, x_g = x_ffc[:,:C//2], x_ffc[:,C//2:]
        x_l = self.act_l(self.bn_l(x_l))
        x_g = self.act_g(self.bn_g(x_g))
        return torch.cat([x_l, x_g], dim=1)