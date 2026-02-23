
import torch
import torch.nn as nn
import torch.nn.functional as F


def GN(C, groups=8):
    """Automatically select a divisible number of groups for GroupNorm."""
    g = groups
    while C % g != 0 and g > 1:
        g -= 1
    return nn.GroupNorm(g, C)


class DWConvBlock(nn.Module):
    """Depthwise + Pointwise convolution block with SE attention and soft residual."""
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super(DWConvBlock, self).__init__()

        self.depthwise = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size, stride, padding, groups=in_ch, bias=False),
            GN(in_ch),
            nn.ReLU(inplace=True)
        )

        self.mix = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 1, bias=False),
            GN(in_ch),
            nn.ReLU(inplace=True)
        )

        self.pointwise = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            GN(out_ch),
            nn.ReLU(inplace=True)
        )

        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_ch, out_ch // 8, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch // 8, out_ch, 1),
            nn.Sigmoid()
        )

        self.gamma = nn.Parameter(torch.ones(1) * 0.5)

    def forward(self, x):
        identity = x
        out = self.depthwise(x)
        out = self.mix(out)
        out = self.pointwise(out)
        out = out * self.se(out)

        if out.shape == identity.shape:
            out = out + self.gamma * identity

        return out


class DWDoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            DWConvBlock(in_ch, out_ch),
            DWConvBlock(out_ch, out_ch)
        )

    def forward(self, x):
        return self.conv(x)


class DWMaskDecoder(nn.Module):
    def __init__(self, num_ch_enc, base_ch=64, out_ch=1, scales=range(4)):
        super().__init__()
        self.scales = scales
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        self.bottleneck = DWDoubleConv(num_ch_enc[-1], base_ch * 8)

        self.up3 = DWDoubleConv(base_ch * 8 + num_ch_enc[3], base_ch * 4)
        self.up2 = DWDoubleConv(base_ch * 4 + num_ch_enc[2], base_ch * 2)
        self.up1 = DWDoubleConv(base_ch * 2 + num_ch_enc[1], base_ch)
        self.up0 = DWDoubleConv(base_ch + num_ch_enc[0], base_ch)

        self.out_convs = nn.ModuleDict({
            str(s): nn.Conv2d(
                base_ch if s == 0 else base_ch * (2 ** (s - 1)),
                out_ch, kernel_size=1
            )
            for s in self.scales
        })

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def load_pretrained(self, ckpt_path, strict=True):
        state_dict = torch.load(ckpt_path, map_location="cpu")
        if "decoder" in state_dict:
            state_dict = state_dict["decoder"]
        self.load_state_dict(state_dict, strict=strict)
        print(f"✅ DWMaskDecoder loaded weights from {ckpt_path}")

    def forward(self, features):
        f1, f2, f3, f4, f5 = features
        outputs = {}

        x = self.bottleneck(f5)

        x = self.up(x)
        x = torch.cat([x, f4], dim=1)
        x = self.up3(x)
        if 3 in self.scales:
            outputs[("mask", 3)] = self.out_convs["3"](x)

        x = self.up(x)
        x = torch.cat([x, f3], dim=1)
        x = self.up2(x)
        if 2 in self.scales:
            outputs[("mask", 2)] = self.out_convs["2"](x)

        x = self.up(x)
        x = torch.cat([x, f2], dim=1)
        x = self.up1(x)
        if 1 in self.scales:
            outputs[("mask", 1)] = self.out_convs["1"](x)

        x = self.up(x)
        x = torch.cat([x, f1], dim=1)
        x = self.up0(x)
        if 0 in self.scales:
            outputs[("mask", 0)] = self.out_convs["0"](x)

        return outputs
