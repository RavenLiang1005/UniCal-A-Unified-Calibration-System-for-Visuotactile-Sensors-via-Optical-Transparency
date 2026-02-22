"""
UniCal Decoders: Unified Reconstruction Modules

This module implements lightweight decoders for:
- Contact mask segmentation
- Depth map reconstruction
- Force map estimation

All decoders share a common U-Net style architecture with depthwise
separable convolutions for efficiency.

Author: Chenxin Liang, Shoujie Li
"""

from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from typing import Dict, List, Tuple, Optional


# =====================================================================
# Helper Functions (keep name compatible with old DW decoder)
# =====================================================================


def upsample(x: torch.Tensor):
    """Upsample feature map (old DW decoder uses this name/signature)."""
    return F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)


# =====================================================================
# Basic Building Blocks (old-compatible names)
# =====================================================================

class DWConvBlock(nn.Module):
    """Depthwise Separable Conv + SE with GroupNorm, name kept as in old version."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super(DWConvBlock, self).__init__()

        def NormLayer(c: int):
            # old version used GroupNorm(4, C)
            return nn.GroupNorm(4, c)

        self.depthwise = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size, stride, padding=1, groups=in_ch, bias=False),
            NormLayer(in_ch),
            nn.ReLU(inplace=True),
        )

        self.mix = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 1, bias=False),
            NormLayer(in_ch),
            nn.ReLU(inplace=True),
        )

        self.pointwise = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            NormLayer(out_ch),
            nn.ReLU(inplace=True),
        )

        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_ch, out_ch // 8, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch // 8, out_ch, 1),
            nn.Sigmoid(),
        )

        self.gamma = nn.Parameter(torch.ones(1) * 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.depthwise(x)
        out = self.mix(out)
        out = self.pointwise(out)
        out = out * self.se(out)
        if out.shape == identity.shape:
            out = out + self.gamma * identity
        return out


class Conv3x3(nn.Module):
    """3x3 conv for output layers (name as in old depth_decoder)."""

    def __init__(self, in_ch: int, out_ch: int):
        super(Conv3x3, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x.contiguous())


class DoubleConv(nn.Module):
    """U-Net style double conv for mask decoder (GN instead of BN)."""

    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.GroupNorm(4, cout),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.GroupNorm(4, cout),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# =====================================================================
# Depth Decoder (DWDepthDecoder) – key/structure compatible with old one
# =====================================================================

class DWDepthDecoder(nn.Module):
    """
    Depthwise version decoder
    (Structure + weight keys unchanged)
    """

    def __init__(self, num_ch_enc,
                 scales=range(4),
                 num_output_channels=1,
                 use_skips=True):

        super().__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.scales = scales

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])

        self.convs = OrderedDict()

        for i in range(4, -1, -1):

            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            self.convs[("upconv", i, 0)] = DWConvBlock(num_ch_in, self.num_ch_dec[i])

            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]

            self.convs[("upconv", i, 1)] = DWConvBlock(num_ch_in, self.num_ch_dec[i])

        for s in self.scales:
            self.convs[("dispconv", s)] = Conv3x3(
                self.num_ch_dec[s],
                self.num_output_channels
            )

        self.decoder = nn.ModuleList(list(self.convs.values()))

    def forward(self, input_features):

        outputs = {}
        x = input_features[-1]

        for i in range(4, -1, -1):

            x = self.convs[("upconv", i, 0)](x)
            x = upsample(x)

            if self.use_skips and i > 0:
                x = torch.cat([x, input_features[i - 1]], dim=1)

            x = self.convs[("upconv", i, 1)](x)

            if i in self.scales:
                outputs[("depth", i)] = self.convs[("dispconv", i)](x)

        return outputs


# =====================================================================
# Force Decoder (DWForceDecoder) – mirror old DWForceDecoder naming
# =====================================================================

class DWForceDecoder(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True):
        super(DWForceDecoder, self).__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = "bilinear"
        self.scales = list(scales)

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])

        self.convs = OrderedDict()
        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = DWConvBlock(num_ch_in, num_ch_out)

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = DWConvBlock(num_ch_in, num_ch_out)

        # 输出层：forceconv，与旧版一致
        for s in self.scales:
            self.convs[("forceconv", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)

        self.decoder = nn.ModuleList(list(self.convs.values()))

    def forward(self, input_features: List[torch.Tensor], mask: Optional[torch.Tensor] = None) -> Dict[Tuple[str, int], torch.Tensor]:
        self.outputs: Dict[Tuple[str, int], torch.Tensor] = {}
        x = input_features[-1]

        for i in range(4, -1, -1):
            x = self.convs[("upconv", i, 0)](x)
            x = [upsample(x)]

            if self.use_skips and i > 0:
                x += [input_features[i - 1]]

            x = torch.cat(x, dim=1).clone()
            x = self.convs[("upconv", i, 1)](x)

            if i in self.scales:
                force_map = self.convs[("forceconv", i)](x)
                if mask is not None:
                    B, C, H, W = force_map.shape
                    mask_resized = F.interpolate(mask, size=(H, W), mode="bilinear", align_corners=False)
                    force_map = force_map * mask_resized
                self.outputs[("force_map", i)] = force_map

        return self.outputs

