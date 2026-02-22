import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================
#                       GroupNorm Helper
# =========================================================
def GN(C, groups=8):
    """自动选择可整除的 group 数"""
    g = groups
    while C % g != 0 and g > 1:
        g -= 1
    return nn.GroupNorm(g, C)


# =========================================================
#                GN 版本 Depthwise + Pointwise 模块
# =========================================================
class DWConvBlock(nn.Module):
    """
    DWConvBlock + 1x1 mix + SE + soft residual
    - 所有 BatchNorm2d 已替换为 GroupNorm
    """
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super(DWConvBlock, self).__init__()
        mid_ch = max(in_ch // 2, 8)

        # Depthwise
        self.depthwise = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size, stride, padding, groups=in_ch, bias=False),
            GN(in_ch),
            nn.ReLU(inplace=True)
        )

        # 1x1 通道混合
        self.mix = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 1, bias=False),
            GN(in_ch),
            nn.ReLU(inplace=True)
        )

        # Pointwise
        self.pointwise = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            GN(out_ch),
            nn.ReLU(inplace=True)
        )

        # SE 注意力
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


# =========================================================
#                GN 版本 DWDoubleConv
# =========================================================
class DWDoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            DWConvBlock(in_ch, out_ch),
            DWConvBlock(out_ch, out_ch)
        )

    def forward(self, x):
        return self.conv(x.contiguous())


# =========================================================
#                   GN 版本 DWMaskDecoder
# =========================================================
class DWMaskDecoder(nn.Module):
    def __init__(self, num_ch_enc, base_ch=64, out_ch=1, scales=range(4)):
        super().__init__()
        self.scales = scales
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        # bottleneck
        self.bottleneck = DWDoubleConv(num_ch_enc[-1], base_ch * 8)

        # Skip connections
        self.up3 = DWDoubleConv(base_ch * 8 + num_ch_enc[3], base_ch * 4)
        self.up2 = DWDoubleConv(base_ch * 4 + num_ch_enc[2], base_ch * 2)
        self.up1 = DWDoubleConv(base_ch * 2 + num_ch_enc[1], base_ch)
        self.up0 = DWDoubleConv(base_ch + num_ch_enc[0], base_ch)

        # 输出层
        self.out_convs = nn.ModuleDict({
            str(s): nn.Conv2d(
                base_ch if s == 0 else base_ch * (2 ** (s - 1)),
                out_ch, kernel_size=1
            )
            for s in self.scales
        })

        # 初始化 conv
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def load_pretrained(self, ckpt_path, strict=True):
        state_dict = torch.load(ckpt_path, map_location="cpu")
        if "decoder" in state_dict:
            state_dict = state_dict["decoder"]
        self.load_state_dict(state_dict, strict=strict)
        print(f"✅ DWMaskDecoder (GroupNorm版) loaded weights from {ckpt_path}")

    def forward(self, features):
        f1, f2, f3, f4, f5 = [f.contiguous() for f in features]
        outputs = {}

        # bottleneck
        x = self.bottleneck(f5)

        # stage 4
        x = self.up(x)
        x = torch.cat([x, f4], dim=1).clone()
        x = self.up3(x)
        if 3 in self.scales:
            outputs[("mask", 3)] = self.out_convs["3"](x)

        # stage 3
        x = self.up(x)
        x = torch.cat([x, f3], dim=1).clone()
        x = self.up2(x)
        if 2 in self.scales:
            outputs[("mask", 2)] = self.out_convs["2"](x)

        # stage 2
        x = self.up(x)
        x = torch.cat([x, f2], dim=1).clone()
        x = self.up1(x)
        if 1 in self.scales:
            outputs[("mask", 1)] = self.out_convs["1"](x)

        # stage 1
        x = self.up(x)
        x = torch.cat([x, f1], dim=1).clone()
        x = self.up0(x)
        if 0 in self.scales:
            outputs[("mask", 0)] = self.out_convs["0"](x)

        return outputs
