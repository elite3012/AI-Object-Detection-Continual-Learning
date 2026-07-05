from __future__ import annotations

import torch
from torch import nn


class ConvNormAct(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, *, stride: int = 1) -> None:
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )


class SqueezeExcitation(nn.Module):
    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        hidden = max(8, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.gate = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs * self.gate(self.pool(inputs))


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        stride: int = 1,
        attention: bool = False,
        residual: bool = True,
    ) -> None:
        super().__init__()
        self.residual = residual
        self.body = nn.Sequential(
            ConvNormAct(in_channels, out_channels, stride=stride),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.attention = (
            SqueezeExcitation(out_channels) if attention and residual else nn.Identity()
        )
        if residual and (stride != 1 or in_channels != out_channels):
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()
        self.activation = nn.SiLU(inplace=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.attention(self.body(inputs))
        if not self.residual:
            return self.activation(features)
        return self.activation(features + self.shortcut(inputs))


class PestNetS(nn.Module):
    """Compact residual CNN trained from scratch for the approved IP102 subset."""

    def __init__(
        self,
        num_classes: int,
        *,
        width: int = 32,
        dropout: float = 0.25,
        use_attention: bool = True,
        use_residual: bool = True,
    ) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2")
        if width < 8:
            raise ValueError("width must be at least 8")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")

        channels = (width, width * 2, width * 4, width * 8)
        self.features = nn.Sequential(
            ConvNormAct(3, channels[0], stride=2),
            ResidualBlock(channels[0], channels[0], residual=use_residual),
            ResidualBlock(channels[0], channels[1], stride=2, residual=use_residual),
            ResidualBlock(channels[1], channels[1], residual=use_residual),
            ResidualBlock(channels[1], channels[2], stride=2, residual=use_residual),
            ResidualBlock(channels[2], channels[2], residual=use_residual),
            ResidualBlock(
                channels[2],
                channels[3],
                stride=2,
                attention=use_attention,
                residual=use_residual,
            ),
            ResidualBlock(
                channels[3],
                channels[3],
                attention=use_attention,
                residual=use_residual,
            ),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(channels[-1], num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(inputs))


class SimpleCNN(nn.Module):
    """Small baseline CNN used to keep PestNet-S comparisons honest."""

    def __init__(self, num_classes: int, *, width: int = 32, dropout: float = 0.2) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2")
        channels = (width, width * 2, width * 4)
        self.net = nn.Sequential(
            ConvNormAct(3, channels[0]),
            nn.MaxPool2d(2),
            ConvNormAct(channels[0], channels[1]),
            nn.MaxPool2d(2),
            ConvNormAct(channels[1], channels[2]),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(channels[2], num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


def build_model(
    name: str,
    *,
    num_classes: int,
    width: int = 32,
    dropout: float = 0.25,
) -> nn.Module:
    normalized = name.lower().replace("-", "_")
    if normalized == "pestnet_s":
        return PestNetS(num_classes=num_classes, width=width, dropout=dropout)
    if normalized == "pestnet_s_no_attention":
        return PestNetS(
            num_classes=num_classes,
            width=width,
            dropout=dropout,
            use_attention=False,
        )
    if normalized == "pestnet_s_no_residual":
        return PestNetS(
            num_classes=num_classes,
            width=width,
            dropout=dropout,
            use_attention=False,
            use_residual=False,
        )
    if normalized == "simple_cnn":
        return SimpleCNN(num_classes=num_classes, width=width, dropout=dropout)
    raise ValueError(f"Unsupported model architecture: {name}")


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
