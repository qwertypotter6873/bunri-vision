"""Neural network architectures for Bunri-Vision AI.

Includes:
1. ScratchResSECNN: 5-stage Residual Depthwise-Separable + Squeeze-and-Excitation CNN
   trained completely from scratch (random initialization baseline).
2. TransferClassifier: Unified wrapper around pretrained MobileNetV3-Small and ResNet-50
   for transfer learning / fine-tuning comparisons.
3. BunriConvNeXtGeM: Proposed domain-specialized classifier combining ConvNeXt-Tiny
   (7x7 depthwise receptive fields) + Generalized Mean (GeM) texture pooling +
   LayerNorm-Dropout head + learnable/calibrated Temperature parameter, plus
   dual-output ONNX exportability (logits + spatial feature map for fast Grad-CAM
   without needing PyTorch on the production inference server!).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class SqueezeExcite(nn.Module):
    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        mid = max(8, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, mid, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(mid, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(self.pool(x))


class ResSepBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=stride, padding=1, groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, groups=out_ch, bias=False),
            nn.BatchNorm2d(out_ch),
            SqueezeExcite(out_ch),
        )
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(x) + self.shortcut(x))


class ScratchResSECNN(nn.Module):
    """Custom 5-stage Residual SE-CNN trained completely from scratch."""

    def __init__(self, num_classes: int = 9, dropout: float = 0.3) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.SiLU(inplace=True),
        )
        self.stage1 = nn.Sequential(ResSepBlock(48, 64, stride=1), ResSepBlock(64, 64, stride=1))
        self.stage2 = nn.Sequential(ResSepBlock(64, 128, stride=2), ResSepBlock(128, 128, stride=1))
        self.stage3 = nn.Sequential(ResSepBlock(128, 256, stride=2), ResSepBlock(256, 256, stride=1))
        self.stage4 = nn.Sequential(ResSepBlock(256, 384, stride=2), ResSepBlock(384, 384, stride=1))
        self.stage5 = ResSepBlock(384, 512, stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(512),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        feat = self.stage5(x)
        return self.head(self.pool(feat))


class GeMPooling(nn.Module):
    """Generalized Mean Pooling for local surface/texture cue aggregation."""

    def __init__(self, p: float = 3.0, eps: float = 1e-6) -> None:
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p = torch.clamp(self.p, min=1.0, max=6.0)
        pos = F.gelu(x) + self.eps
        pos = torch.clamp(pos, min=self.eps)
        pooled = pos.pow(p).mean(dim=(-2, -1), keepdim=True).pow(1.0 / p)
        avg = x.mean(dim=(-2, -1), keepdim=True)
        return (0.5 * avg + 0.5 * pooled).flatten(1)


class TransferClassifier(nn.Module):
    """Standard transfer-learning / fine-tuning wrapper for MobileNetV3 & ResNet-50."""

    def __init__(self, backbone_name: str, num_classes: int = 9, pretrained: bool = True, dropout: float = 0.25) -> None:
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )
        with torch.no_grad():
            in_features = int(self.backbone(torch.zeros(1, 3, 224, 224)).shape[-1])
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        return self.head(feat)


class BunriConvNeXtGeM(nn.Module):
    """Proposed model: ConvNeXt-Tiny + Dual Avg/GeM Texture Pooling + Calibrated Head."""

    def __init__(
        self,
        backbone_name: str = "convnext_tiny.fb_in22k_ft_in1k",
        num_classes: int = 9,
        pretrained: bool = True,
        dropout: float = 0.30,
    ) -> None:
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="",
        )
        in_features = self.backbone.num_features  # 768 for convnext_tiny
        self.gem = GeMPooling(p=3.0)
        self.norm = nn.LayerNorm(in_features)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(in_features, num_classes)
        self.register_buffer("temperature", torch.ones(1, dtype=torch.float32))

    def forward_features_and_logits(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        spatial_feat = self.backbone(x)  # (B, C, 7, 7)
        pooled = self.gem(spatial_feat)  # (B, C)
        normed = self.norm(pooled)
        logits = self.fc(self.drop(normed))
        calibrated_logits = logits / torch.clamp(self.temperature, min=0.2, max=5.0)
        # Class-specific CAM map via 1x1 conv with cloned classifier weights (safe under ONNX quantization)
        cam_w = self.fc.weight.clone().unsqueeze(-1).unsqueeze(-1)
        cam_maps = F.conv2d(spatial_feat, cam_w)
        return logits, calibrated_logits, cam_maps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits, _, _ = self.forward_features_and_logits(x)
        return logits


class ONNXExportWrapper(nn.Module):
    """Exports both calibrated logits and 9-class spatial CAM activation maps (7x7) in a single ONNX pass!"""

    def __init__(self, model: BunriConvNeXtGeM) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _, calibrated_logits, cam_maps = self.model.forward_features_and_logits(x)
        return calibrated_logits, cam_maps


def build_model(arch: str, num_classes: int = 9, pretrained: bool = True) -> nn.Module:
    if arch == "scratch_cnn":
        return ScratchResSECNN(num_classes=num_classes)
    if arch == "mobilenetv3_small":
        return TransferClassifier("mobilenetv3_small_100.lamb_in1k", num_classes=num_classes, pretrained=pretrained)
    if arch == "resnet50":
        return TransferClassifier("resnet50.a1_in1k", num_classes=num_classes, pretrained=pretrained)
    if arch == "convnext_tiny_gem":
        return BunriConvNeXtGeM("convnext_tiny.fb_in22k_ft_in1k", num_classes=num_classes, pretrained=pretrained)
    raise ValueError(f"Unknown architecture: {arch}")
