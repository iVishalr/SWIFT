import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
from model.attention import PatchEmbed, PatchUnEmbed, SwinTransformerBlockV2
from model.modules import RFB, PixelShuffleUpsample, default_conv
from typing import List, Tuple, Optional

__all__ = ['swift_x2', 'swift_x3', 'swift_x4']

class FSTB(nn.Module):
    '''
    Fourier-Swin Transformer Block (FSTB)
    '''
    def __init__(self, 
        out_channels: int,
        embd_dim: int, 
        num_heads: int,
        depth: int,
        rfbs: int,
        window_size: int,
        img_size: int, 
        mlp_ratio: float = 0.5,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        drop: Optional[float] = 0.,
        attn_drop: Optional[float] = 0.,
        drop_path: Optional[float] = 0.,
        patch_size: int = 1, 
        norm_layer=nn.LayerNorm, 
        act_layer=nn.ReLU,
        feat_scale: Optional[bool] = False,
        attn_scale: Optional[bool] = True,
        residual_conv="1conv"
    ) -> None:
        
        super(FSTB, self).__init__()
        
        self.depth = depth
        self.embd_dim = embd_dim

        if self.depth != 0:
            self.patch_embed = PatchEmbed(
                img_size=img_size,
                patch_size=patch_size,
                in_chans=out_channels,
                embed_dim=embd_dim,
                norm_layer=norm_layer
            )

            self.patch_unembd = PatchUnEmbed(
                img_size=img_size,
                patch_size=patch_size,
                in_chans=out_channels,
                embed_dim=embd_dim,
                norm_layer=norm_layer
            )

            num_patches = self.patch_embed.num_patches
            patches_resolution = self.patch_embed.patches_resolution

        self.rfbs = nn.ModuleList([
            RFB(out_channels, out_channels)
            for _ in range(rfbs)
        ])

        self.blocks = nn.ModuleList([
            SwinTransformerBlockV2(
                dim=embd_dim,
                input_resolution=(patches_resolution[0], patches_resolution[1]),
                num_heads=num_heads,
                window_size=window_size,
                shift_size= 0 if (i%2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                drop_path=drop_path[i] if isinstance(drop_path,list) else drop_path,
                attn_drop=attn_drop,
                norm_layer=norm_layer,
                act_layer=act_layer,
                feat_scale=feat_scale,
                attn_scale=attn_scale
            )
            for i in range(depth)
        ])

        if residual_conv == '1conv':
            self.conv = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        elif residual_conv == '3conv':
            # to save parameters and memory
            self.conv = nn.Sequential(
                nn.Conv2d(embd_dim, embd_dim // 4, 3, 1, 1),
                nn.PReLU(embd_dim // 4),
                nn.Conv2d(embd_dim // 4, embd_dim // 4, 1, 1, 0),
                nn.PReLU(embd_dim // 4),
                nn.Conv2d(embd_dim // 4, embd_dim, 3, 1, 1)
            )

    def forward(self, x: torch.Tensor, x_size: Tuple[int, int]) -> torch.Tensor:
        out = x
        if self.depth != 0:
            # extract patches
            out = self.patch_embed(out)
            # pass through S2TL
            out = out.contiguous()
            for block in self.blocks:
                out = block(out, x_size)

            # reform the image
            out = self.patch_unembd(out, x_size)
            out = out.contiguous()
            
        for rfb in self.rfbs:
            out = rfb(out)

        out = self.conv(out) + x
        return out
        
class SWIFT(nn.Module):
    def __init__(self,
        img_size: int = 64,
        patch_size: int = 1, 
        in_channels: int = 3,
        embd_dim: int = 60,
        depths: List[int] = [6,6,6,6],
        rfbs: List[int] = [2,2,2,2],
        num_heads: List[int] = [6,6,6,6],
        window_size: int = 8,
        mlp_ratio: float = 1,
        scale: int = 2,
        drop: Optional[float] = 0.,
        drop_path: Optional[float] = 0.1,
        attn_drop: Optional[float] = 0.,
        qk_scale: Optional[float] = None,
        qkv_bias: Optional[bool] = True,
        img_range: Optional[float] = 1.,
        ape: Optional[bool] = False,
        patch_norm: Optional[bool] = True,
        residual_conv: Optional[str] = "3conv",
        act_layer = nn.ReLU,
        norm_layer=nn.LayerNorm,
        feat_scale: Optional[bool] = True,
        attn_scale: Optional[bool] = False,
        ) -> None:
        
        super(SWIFT, self).__init__()
        
        in_channels = in_channels
        out_channels = in_channels
        num_feat = embd_dim
        self.scale = scale
        self.img_range = img_range
        
        if in_channels == 3:
            # RGB mean for DIV2K
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1,1,1,1)

        self.window_size = window_size
        
        # ------------ Shallow Feature Extraction ---------- #
        modules_head = [
            default_conv(
                in_channels = in_channels,
                out_channels = embd_dim,
                kernel_size = 3,
                bias = True,
                groups = 1
            ),
        ]
        
        # ------------- Deep Feature Extraction ------------- #
        self.num_layers = len(depths)
        self.embd_dim = embd_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.mlp_ratio = mlp_ratio

        dpr = [x.item() for x in torch.linspace(0, drop_path, sum(depths))]

        self.layers = nn.ModuleList()
        
        for i_layer in range(self.num_layers):
            layer = FSTB(
                out_channels=embd_dim,
                embd_dim=embd_dim,
                depth=depths[i_layer],
                rfbs=rfbs[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, 
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer+1])],
                norm_layer=norm_layer,
                img_size=img_size,
                patch_size=patch_size,
                act_layer=act_layer,
                residual_conv=residual_conv,
                feat_scale=feat_scale,
                attn_scale=attn_scale
            )

            self.layers.append(layer)

        # build the last conv layer in deep feature extraction
        if residual_conv == '1conv':
            self.conv_after_body = nn.Conv2d(embd_dim, embd_dim, 3, 1, 1)
        elif residual_conv == '3conv':
            # to save parameters and memory
            self.conv_after_body = nn.Sequential(
                nn.Conv2d(embd_dim, embd_dim // 4, 3, 1, 1),
                nn.PReLU(embd_dim // 4),
                nn.Conv2d(embd_dim // 4, embd_dim // 4, 1, 1, 0),
                nn.PReLU(embd_dim // 4),
                nn.Conv2d(embd_dim // 4, embd_dim, 3, 1, 1)
            )

        # ------------- High Quality Image Reconstruction --------- #
        self.conv_before_upsample = nn.Sequential(
            nn.Conv2d(num_feat, num_feat, 3, 1, 1),
        )

        modules_tail = []

        if scale == 2 or scale == 4:
            modules_tail.append(PixelShuffleUpsample(num_feat, num_feat))
        
        if scale == 3:
            modules_tail.append(PixelShuffleUpsample(num_feat, num_feat, scale=3))
        
        if scale == 4:
            modules_tail.append(PixelShuffleUpsample(num_feat, num_feat))

        modules_tail.append(
            default_conv(
                in_channels = num_feat,
                out_channels = out_channels,
                kernel_size = 3,
            )
        )

        self.head = nn.Sequential(*modules_head)
        self.tail = nn.Sequential(*modules_tail)
        self.apply(self._init_weights)

    def check_image_size(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.size()
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x_size = (x.shape[2], x.shape[3])
        for layer in self.layers:
            x = layer(x, x_size)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H,W = x.shape[2:]
        x = self.check_image_size(x)
        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range

        # ----------------- Shallow Features --------------- #
        shallow_features = self.head(x)

        # ----------------- Deep Features ------------------ #
        out = self.forward_features(shallow_features)
        out = self.conv_after_body(out)
        out = out + shallow_features
         
        # ------------------- Upsampler --------------------- #
        out = self.conv_before_upsample(out)

        out = self.tail(out)

        out = out / self.img_range + self.mean
        return out[:, :, :H*self.scale, :W*self.scale]

    def _init_weights(self, module):
        if isinstance(module, nn.BatchNorm2d):
            nn.init.constant_(module.weight,1.0)
            nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=.02)
            if isinstance(module, (nn.Linear)) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, (nn.LayerNorm)):
            module.weight.data.fill_(1.0)
            module.bias.data.zero_()

    def load_state_dict(self, state_dict, strict=False):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') >= 0:
                        print('Replace pre-trained upsampler to new one...')
                    else:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))

def swift_x2(**kwargs):
    model = SWIFT(
        img_size=64,
        patch_size=1,
        in_channels=3,
        embd_dim=64,
        rfbs=[2, 2, 2, 2],
        depths=[2, 2, 2, 2],
        num_heads=[8, 8, 8, 8],
        mlp_ratio=1,
        window_size=8,
        residual_conv="3conv",
        scale=2,
        act_layer=nn.GELU,
        feat_scale=False,
        attn_scale=True,
        **kwargs
    )
    return model

def swift_x3(**kwargs):
    model = SWIFT(
        img_size=64,
        patch_size=1,
        in_channels=3,
        embd_dim=64,
        rfbs=[2, 2, 2, 2],
        depths=[2, 2, 2, 2],
        num_heads=[8, 8, 8, 8],
        mlp_ratio=1,
        window_size=8,
        residual_conv="3conv",
        scale=3,
        act_layer=nn.GELU,
        feat_scale=False,
        attn_scale=True,
        **kwargs
    )
    return model

def swift_x4(**kwargs):
    model = SWIFT(
        img_size=64,
        patch_size=1,
        in_channels=3,
        embd_dim=64,
        rfbs=[2, 2, 2, 2],
        depths=[2, 2, 2, 2],
        num_heads=[8, 8, 8, 8],
        mlp_ratio=1,
        window_size=8,
        residual_conv="3conv",
        scale=4,
        act_layer=nn.GELU,
        feat_scale=False,
        attn_scale=True,
        **kwargs
    )
    return model