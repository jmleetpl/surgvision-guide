"""Model zoo for the toolkit. Every model takes a normalised RGB batch and returns
logits [B, K, H, W] (segmentation) or [B, K] (classification).

Only ImageNet-pretrained *backbones* are loaded; heads start untrained. Pick an
architecture by name in config.training.arch, or compare several with
model_search.py. Sizes are rough parameter counts to help CPU/GPU planning.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models as tvm
from torchvision.models import segmentation as tvs

SEG_ARCHS = {
    "unet_resnet18": "U-Net, ResNet18 encoder (~14M) — default, fast on CPU",
    "unet_resnet34": "U-Net, ResNet34 encoder (~24M)",
    "lraspp_mobilenet": "LR-ASPP, MobileNetV3-Large (~3M) — lightest, real-time candidate",
    "deeplabv3_mobilenet": "DeepLabV3, MobileNetV3-Large (~11M)",
    "deeplabv3_resnet50": "DeepLabV3, ResNet50 (~40M) — needs GPU",
}
CLS_ARCHS = {
    "resnet18": "ResNet18 (~11M) — default",
    "resnet50": "ResNet50 (~24M) — needs GPU",
    "efficientnet_b0": "EfficientNet-B0 (~5M)",
    "mobilenet_v3_large": "MobileNetV3-Large (~5M) — lightest",
    "convnext_tiny": "ConvNeXt-Tiny (~28M) — needs GPU",
}
DEFAULT_ARCH = {"segmentation": "unet_resnet18", "classification": "resnet18"}


def _weights(name: str, pretrained: bool):
    if not pretrained:
        return None
    return {"resnet18": tvm.ResNet18_Weights.IMAGENET1K_V1, "resnet34": tvm.ResNet34_Weights.IMAGENET1K_V1,
            "resnet50": tvm.ResNet50_Weights.IMAGENET1K_V1, "efficientnet_b0": tvm.EfficientNet_B0_Weights.IMAGENET1K_V1,
            "mobilenet_v3_large": tvm.MobileNet_V3_Large_Weights.IMAGENET1K_V1,
            "convnext_tiny": tvm.ConvNeXt_Tiny_Weights.IMAGENET1K_V1}[name]


def _load(fn, **kw):
    try:
        return fn(**kw)
    except Exception as e:  # offline Colab / hospital PC without internet
        raise SystemExit(f"[STOP] ImageNet weights unavailable ({e}). Restore internet or rerun with "
                         "--no-pretrained and REPORT that the backbone was randomly initialised.")


class Up(nn.Module):
    def __init__(self, cin, cskip, cout):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(cin + cskip, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout),
                                  nn.ReLU(inplace=True), nn.Conv2d(cout, cout, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(cout), nn.ReLU(inplace=True))

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], 1))


class ResUNet(nn.Module):
    """U-Net decoder on a ResNet18/34 encoder (both have 64-64-128-256-512 channels)."""

    def __init__(self, depth: int, n_classes: int, pretrained: bool):
        super().__init__()
        name = f"resnet{depth}"
        e = _load(getattr(tvm, name), weights=_weights(name, pretrained))
        self.stem = nn.Sequential(e.conv1, e.bn1, e.relu)
        self.pool, self.l1, self.l2, self.l3, self.l4 = e.maxpool, e.layer1, e.layer2, e.layer3, e.layer4
        self.u3, self.u2, self.u1, self.u0 = Up(512, 256, 256), Up(256, 128, 128), Up(128, 64, 64), Up(64, 64, 32)
        self.head = nn.Conv2d(32, n_classes, 1)

    def forward(self, x):
        s0 = self.stem(x)
        s1 = self.l1(self.pool(s0))
        s2 = self.l2(s1)
        s3 = self.l3(s2)
        y = self.u3(self.l4(s3), s3)
        y = self.u0(self.u1(self.u2(y, s2), s1), s0)
        return F.interpolate(self.head(y), size=x.shape[-2:], mode="bilinear", align_corners=False)


class TVSeg(nn.Module):
    """torchvision segmentation model -> plain logits tensor."""

    def __init__(self, fn, backbone_w, n_classes, pretrained):
        super().__init__()
        self.m = _load(fn, weights=None, weights_backbone=backbone_w if pretrained else None, num_classes=n_classes)

    def forward(self, x):
        return self.m(x)["out"]


def build_model(task: str, k: int, pretrained: bool, arch: str | None = None) -> nn.Module:
    arch = arch or DEFAULT_ARCH[task]
    if task == "segmentation":
        if arch not in SEG_ARCHS:
            raise SystemExit(f"[STOP] unknown segmentation arch {arch!r}; choose from {list(SEG_ARCHS)}")
        if arch.startswith("unet_resnet"):
            return ResUNet(int(arch[-2:]), k, pretrained)
        if arch == "lraspp_mobilenet":
            return TVSeg(tvs.lraspp_mobilenet_v3_large, tvm.MobileNet_V3_Large_Weights.IMAGENET1K_V1, k, pretrained)
        if arch == "deeplabv3_mobilenet":
            m = TVSeg(tvs.deeplabv3_mobilenet_v3_large, tvm.MobileNet_V3_Large_Weights.IMAGENET1K_V1, k, pretrained)
            return m
        return TVSeg(tvs.deeplabv3_resnet50, tvm.ResNet50_Weights.IMAGENET1K_V1, k, pretrained)
    if arch not in CLS_ARCHS:
        raise SystemExit(f"[STOP] unknown classification arch {arch!r}; choose from {list(CLS_ARCHS)}")
    m = _load(getattr(tvm, arch), weights=_weights(arch, pretrained))
    if arch.startswith("resnet"):
        m.fc = nn.Linear(m.fc.in_features, k)
    elif arch == "convnext_tiny":
        m.classifier[2] = nn.Linear(m.classifier[2].in_features, k)
    else:  # efficientnet / mobilenet: last Linear of the classifier
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, k)
    return m


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
