"""Memory-aware 3-D adaptation of FusionRegister's MRB.

The original Modality Retainment Block is implemented for 2-D infrared/visible
fusion.  This version keeps its three main ideas for volumetric registration:
local correlation with each input, multi-window gated MLP mixing, and
source/target-specific attention followed by residual detail restoration.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class LocalCorrelation3D(nn.Module):
    """Local cost volume between two equally shaped 3-D feature maps."""

    def __init__(self, max_disp=1):
        super().__init__()
        self.max_disp = int(max_disp)

    @property
    def out_channels(self):
        return (2 * self.max_disp + 1) ** 3

    def forward(self, x1, x2):
        if x1.shape != x2.shape:
            raise ValueError(
                'LocalCorrelation3D expects equal shapes, got '
                f'{tuple(x1.shape)} and {tuple(x2.shape)}'
            )

        p = self.max_disp
        if p == 0:
            return torch.mean(x1 * x2, dim=1, keepdim=True)

        _, _, depth, height, width = x1.shape
        x2_padded = F.pad(x2, (p, p, p, p, p, p))
        correlations = []
        for dz in range(-p, p + 1):
            for dy in range(-p, p + 1):
                for dx in range(-p, p + 1):
                    z0, y0, x0 = p + dz, p + dy, p + dx
                    shifted = x2_padded[
                        :, :, z0:z0 + depth, y0:y0 + height, x0:x0 + width
                    ]
                    correlations.append(torch.mean(x1 * shifted, dim=1, keepdim=True))
        return torch.cat(correlations, dim=1)

class WindowGMLP3D(nn.Module):
    """Apply the spatial-gating MLP independently in non-overlapping 3-D windows."""

    def __init__(self, channels, window_size):
        super().__init__()
        self.channels = int(channels)
        self.window_size = int(window_size)
        window_volume = self.window_size ** 3

        self.in_project = nn.Linear(self.channels, self.channels * 2)
        self.activation = nn.GELU()
        self.spatial_gate = nn.Linear(window_volume, window_volume)
        self.out_project = nn.Linear(self.channels, self.channels)

    def _split_windows(self, x):
        batch, channels, depth, height, width = x.shape
        win = self.window_size
        pad_d = (win - depth % win) % win
        pad_h = (win - height % win) % win
        pad_w = (win - width % win) % win
        x = F.pad(x, (0, pad_w, 0, pad_h, 0, pad_d))

        padded_shape = x.shape[2:]
        grid_d, grid_h, grid_w = (size // win for size in padded_shape)
        x = x.reshape(
            batch, channels,
            grid_d, win, grid_h, win, grid_w, win
        )
        x = x.permute(0, 2, 4, 6, 3, 5, 7, 1).contiguous()
        x = x.reshape(batch, grid_d * grid_h * grid_w, win ** 3, channels)
        metadata = (depth, height, width, grid_d, grid_h, grid_w)
        return x, metadata

    def _merge_windows(self, x, metadata):
        depth, height, width, grid_d, grid_h, grid_w = metadata
        batch = x.shape[0]
        win = self.window_size
        x = x.reshape(batch, grid_d, grid_h, grid_w, win, win, win, self.channels)
        x = x.permute(0, 7, 1, 4, 2, 5, 3, 6).contiguous()
        x = x.reshape(batch, self.channels, grid_d * win, grid_h * win, grid_w * win)
        return x[:, :, :depth, :height, :width].contiguous()

    def forward(self, x):
        windows, metadata = self._split_windows(x)
        shortcut = windows
        gates = self.activation(self.in_project(windows))
        content, gate = torch.chunk(gates, 2, dim=-1)
        gate = self.spatial_gate(gate.transpose(-1, -2)).transpose(-1, -2)
        windows = self.out_project(content * (gate + 1.0)) + shortcut
        return self._merge_windows(windows, metadata)

class MultiWindowGMLP3D(nn.Module):
    """Adaptively combine 1x1x1, 3x3x3 and 5x5x5 gMLP windows."""

    def __init__(self, channels, window_sizes=(1, 3, 5), dropout=0.2):
        super().__init__()
        self.channels = int(channels)
        self.branches = nn.ModuleList([
            WindowGMLP3D(self.channels, window_size) for window_size in window_sizes
        ])
        hidden = max(self.channels // 4, 1)
        self.reweight = nn.Sequential(
            nn.Linear(self.channels, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, self.channels * len(window_sizes)),
            nn.Dropout(dropout),
        )
        self.out_project = nn.Conv3d(self.channels, self.channels, kernel_size=1)

    def forward(self, x):
        branch_outputs = [branch(x) for branch in self.branches]
        pooled = torch.stack(branch_outputs, dim=0).sum(dim=0).mean(dim=(2, 3, 4))
        weights = self.reweight(pooled)
        weights = weights.view(x.shape[0], len(self.branches), self.channels)
        weights = torch.softmax(weights, dim=1)

        mixed = torch.zeros_like(x)
        for index, branch_output in enumerate(branch_outputs):
            mixed = mixed + branch_output * weights[:, index, :, None, None, None]
        return self.out_project(mixed) + x

class ChannelAttention3D(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.body = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(channels, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.body(x)

class SpatialAttention3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv3d(2, 1, kernel_size=7, padding=3)

    def forward(self, x):
        maximum = torch.max(x, dim=1, keepdim=True).values
        average = torch.mean(x, dim=1, keepdim=True)
        weight = torch.sigmoid(self.conv(torch.cat([maximum, average], dim=1)))
        return x * weight

class ResidualAttention3D(nn.Module):
    def __init__(self, channels, attention):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(channels, channels, kernel_size=3, padding=1),
        )
        if attention == 'channel':
            self.attention = ChannelAttention3D(channels)
        elif attention == 'spatial':
            self.attention = SpatialAttention3D()
        else:
            raise ValueError(f'Unsupported attention type: {attention}')

    def forward(self, x):
        return x + self.attention(self.body(x))

class MRBBranch3D(nn.Module):
    def __init__(self, channels, attention, corr_max_disp=1):
        super().__init__()
        self.correlation = LocalCorrelation3D(corr_max_disp)
        self.fuse = nn.Conv3d(
            channels * 2 + self.correlation.out_channels,
            channels,
            kernel_size=3,
            padding=1,
        )
        self.mixer = MultiWindowGMLP3D(channels)
        self.attention = ResidualAttention3D(channels, attention)

    def forward(self, fused_feature, source_feature):
        correlation = self.correlation(fused_feature, source_feature)
        x = self.fuse(torch.cat([fused_feature, correlation, source_feature], dim=1))
        shortcut = x
        x = self.mixer(x)
        x = self.attention(x)
        return x + shortcut

class ModalityRetainmentBlock3D(nn.Module):
    """Restore source/target details in a joint volumetric registration feature."""

    def __init__(self, channels, corr_max_disp=1):
        super().__init__()
        # Mirrors FusionRegister: one channel-attention branch and one
        # spatial-attention branch, followed by residual detail aggregation.
        self.source_branch = MRBBranch3D(
            channels, attention='channel', corr_max_disp=corr_max_disp
        )
        self.target_branch = MRBBranch3D(
            channels, attention='spatial', corr_max_disp=corr_max_disp
        )

    def forward(self, fused_feature, source_feature, target_feature):
        if not (fused_feature.shape == source_feature.shape == target_feature.shape):
            raise ValueError(
                'MRB inputs must have equal [B,C,D,H,W] shapes, got '
                f'{tuple(fused_feature.shape)}, {tuple(source_feature.shape)}, '
                f'{tuple(target_feature.shape)}'
            )
        source_detail = self.source_branch(fused_feature, source_feature)
        target_detail = self.target_branch(fused_feature, target_feature)
        return fused_feature + source_detail + target_detail
