"""
CBAM: Convolutional Block Attention Module (Woo et al., 2018)
Implements sequential Channel Attention + Spatial Attention.

Channel Attention:
  Mc(F) = σ(W1·δ(W0·Favg) + W1·δ(W0·Fmax))

Spatial Attention:
  Ms(F') = σ(f^{7×7}([AvgPool(F'); MaxPool(F')]))

Refined: F' = Mc(F) ⊗ F, then F'' = Ms(F') ⊗ F'
"""

import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """Channel attention: learns which feature channels are important."""

    def __init__(self, channels, reduction_ratio=16):
        super().__init__()
        mid = max(channels // reduction_ratio, 1)
        self.shared_mlp = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=False),
            nn.Linear(mid, channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()

        # Global average pooling -> (B, C)
        avg_pool = x.mean(dim=(2, 3))
        # Global max pooling -> (B, C)
        max_pool = x.amax(dim=(2, 3))

        # Shared MLP on both
        avg_out = self.shared_mlp(avg_pool)
        max_out = self.shared_mlp(max_pool)

        # Combine and apply sigmoid
        attention = self.sigmoid(avg_out + max_out)  # (B, C)
        return attention.unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)


class SpatialAttention(nn.Module):
    """Spatial attention: learns which spatial locations are important."""

    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Channel-wise pooling -> (B, 1, H, W)
        avg_pool = x.mean(dim=1, keepdim=True)
        max_pool = x.amax(dim=1, keepdim=True)

        # Concatenate and convolve
        combined = torch.cat([avg_pool, max_pool], dim=1)  # (B, 2, H, W)
        attention = self.sigmoid(self.conv(combined))  # (B, 1, H, W)
        return attention


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module.
    Applies channel attention then spatial attention sequentially.
    """

    def __init__(self, channels, reduction_ratio=16, spatial_kernel=7):
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(spatial_kernel)

    def forward(self, x):
        # Channel attention: F' = Mc(F) ⊗ F
        x = x * self.channel_attention(x)
        # Spatial attention: F'' = Ms(F') ⊗ F'
        x = x * self.spatial_attention(x)
        return x


class C2fCBAM(nn.Module):
    """
    C2f block from YOLOv8 followed by CBAM attention.
    This wraps an existing C2f module and adds CBAM after it.
    Used for injecting attention at P3, P4, P5 backbone levels.
    """

    def __init__(self, c2f_module, channels, reduction_ratio=16):
        super().__init__()
        self.c2f = c2f_module
        self.cbam = CBAM(channels, reduction_ratio)

    def forward(self, x):
        x = self.c2f(x)
        x = self.cbam(x)
        return x


if __name__ == "__main__":
    # Quick test
    cbam = CBAM(channels=256, reduction_ratio=16)
    x = torch.randn(2, 256, 20, 20)
    out = cbam(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {out.shape}")
    print(f"CBAM params: {sum(p.numel() for p in cbam.parameters()):,}")
