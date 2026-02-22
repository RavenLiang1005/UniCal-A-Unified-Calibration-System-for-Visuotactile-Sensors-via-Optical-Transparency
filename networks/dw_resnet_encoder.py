from __future__ import absolute_import, division, print_function
import numpy as np
import torch
import torch.nn as nn


# ============================================================
#  GroupNorm Helper
# ============================================================
def GN(num_channels, num_groups=8):
    """
    Safe GroupNorm:
    Automatically adjust group number so that
    num_channels % num_groups == 0
    """
    g = num_groups
    while num_channels % g != 0 and g > 1:
        g -= 1
    return nn.GroupNorm(g, num_channels)


# ============================================================
#  Depthwise + Pointwise Convolution Block (GN Version)
# ============================================================
class DWConvBlock(nn.Module):
    """
    Depthwise → Mix → Pointwise → SE → Residual

    ⚠ Important:
    Structure kept unchanged to ensure compatibility
    with existing state_dict.
    """

    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super(DWConvBlock, self).__init__()

        # ----- Depthwise -----
        self.depthwise = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size,
                      stride=stride, padding=padding,
                      groups=in_ch, bias=False),
            GN(in_ch),
            nn.ReLU(inplace=True)
        )

        # ----- Channel Mixing -----
        self.mix = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 1, bias=False),
            GN(in_ch),
            nn.ReLU(inplace=True)
        )

        # ----- Pointwise -----
        self.pointwise = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            GN(out_ch),
            nn.ReLU(inplace=True)
        )

        # ----- SE Attention -----
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_ch, out_ch // 8, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch // 8, out_ch, 1),
            nn.Sigmoid()
        )

        # Residual scaling (learnable)
        self.gamma = nn.Parameter(torch.ones(1) * 0.5)

    def forward(self, x):
        identity = x

        out = self.depthwise(x)
        out = self.mix(out)
        out = self.pointwise(out)

        # SE re-weighting
        out = out * self.se(out)

        # Residual connection (only if shape matches)
        if out.shape == identity.shape:
            out = out + self.gamma * identity

        return out


# ============================================================
#  Residual Block
# ============================================================
class DWBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(DWBasicBlock, self).__init__()

        self.conv1 = DWConvBlock(inplanes, planes, stride=stride)
        self.conv2 = DWConvBlock(planes, planes)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        return nn.functional.relu(out, inplace=True)


# ============================================================
#  DWResNet Encoder (GN Version)
# ============================================================
class DWResNetEncoder(nn.Module):
    """
    Lightweight ResNet-style encoder using
    depthwise separable convolution + GroupNorm.

    """

    def __init__(self, num_layers=18,
                 num_input_images=1):

        super(DWResNetEncoder, self).__init__()

        assert num_layers in [18, 34]
        blocks = {18: [2, 2, 2, 2],
                  34: [3, 4, 6, 3]}[num_layers]

        self.inplanes = 64
        self.num_ch_enc = np.array([64, 64, 128, 256, 512])

        # --------------------------------------------------
        # Stem
        # --------------------------------------------------
        self.conv1 = nn.Sequential(
            nn.Conv2d(num_input_images * 3, 64,
                      kernel_size=7, stride=2,
                      padding=3, bias=False),
            GN(64),
            nn.ReLU(inplace=True)
        )

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # --------------------------------------------------
        # Residual Stages
        # --------------------------------------------------
        self.layer1 = self._make_layer(64, blocks[0])
        self.layer2 = self._make_layer(128, blocks[1], stride=2)
        self.layer3 = self._make_layer(256, blocks[2], stride=2)
        self.layer4 = self._make_layer(512, blocks[3], stride=2)

        # --------------------------------------------------
        # Weight Initialization (Conv only)
        # --------------------------------------------------
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight,
                    mode='fan_out',
                    nonlinearity='relu'
                )

    # =======================================================
    #  Layer Builder
    # =======================================================
    def _make_layer(self, planes, blocks, stride=1):
        downsample = None

        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes,
                          kernel_size=1,
                          stride=stride,
                          bias=False),
                GN(planes),
            )

        layers = []
        layers.append(DWBasicBlock(self.inplanes, planes, stride, downsample))

        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(DWBasicBlock(self.inplanes, planes))

        return nn.Sequential(*layers)

    # =======================================================
    #  Forward
    # =======================================================
    def forward(self, input_image):
        """
        Returns:
            List of multi-scale feature maps
        """

        features = []

        # normalize to [0,1]
        x = input_image / 255.0

        # Stem
        x = self.conv1(x)
        features.append(x)

        # Stage 1
        x = self.maxpool(x)
        x = self.layer1(x)
        features.append(x)

        # Stage 2
        x = self.layer2(x)
        features.append(x)

        # Stage 3
        x = self.layer3(x)
        features.append(x)

        # Stage 4
        x = self.layer4(x)
        features.append(x)

        return features