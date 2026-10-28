"""
LitePulse.py

LitePulse v2: a lightweight multi-ROI rPPG student model.

    RGB(+ΔRGB) x 3 ROIs
            │
     Shared PulseStem   (per-ROI, weight-shared)
            │
     PulseBlock x N      (per-ROI, weight-shared)
            │
  Adaptive Physiological ROI Fusion (APRF)
            │
     Pulse Spectral Attention
            │
        Temporal CNN
            │
      Projection Head
            │
       Predicted BVP

Expected input tensor shape: (B, T, C, H, W)
    C = NUM_DATA_TYPES * NUM_ROIS * CHANNELS_PER_ROI
    e.g. DATA_TYPE=["Raw","DiffNormalized"], 3 ROIs, 3 channels/ROI -> C=18

This assumes the channel layout produced by BaseLoader.preprocess(), which
concatenates data types along the last axis in order, i.e. for C=18 the
layout is:
    [Raw_forehead(3), Raw_left(3), Raw_right(3),
     Diff_forehead(3), Diff_left(3), Diff_right(3)]
NOT interleaved per-ROI. regroup_by_roi() below undoes this so each ROI's
branch gets its own contiguous (Raw+Diff) channel block, i.e. "RGB + ΔRGB"
per ROI as in the PulseStem design discussion.

>>> If your DATA_TYPE config list or channel counts differ, update
>>> NUM_DATA_TYPES / CHANNELS_PER_ROI in LitePulseConfig accordingly, or
>>> this regrouping will silently mix the wrong channels together.
"""

import math
from dataclasses import dataclass, field
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


ROI_ORDER = ["forehead", "left_cheek", "right_cheek"]


@dataclass
class LitePulseConfig:
    num_rois: int = 3
    channels_per_roi: int = 3          # RGB
    num_data_types: int = 2            # e.g. Raw + DiffNormalized -> "RGB + ΔRGB"
    stem_channels: Tuple[int, int, int] = (16, 24, 32)
    num_pulse_blocks: int = 3
    fps: float = 30.0
    pulse_band_hz: Tuple[float, float] = (0.7, 2.5)
    dropout: float = 0.1

    @property
    def stem_in_channels(self) -> int:
        return self.num_data_types * self.channels_per_roi


def regroup_by_roi(x: torch.Tensor, cfg: LitePulseConfig) -> List[torch.Tensor]:
    """
    x: (B, T, C, H, W) with C = num_data_types * num_rois * channels_per_roi,
       laid out as num_data_types contiguous blocks, each containing
       num_rois contiguous ROI blocks (see module docstring).
    returns: list of length num_rois, each (B, T, num_data_types*channels_per_roi, H, W)
             with that ROI's channels from every data type concatenated together.
    """
    r, c, t = cfg.num_rois, cfg.channels_per_roi, cfg.num_data_types
    per_type_width = r * c
    expected_C = t * per_type_width
    if x.shape[2] != expected_C:
        raise ValueError(
            f"LitePulse expected C={expected_C} (num_data_types={t} x "
            f"num_rois={r} x channels_per_roi={c}) but got C={x.shape[2]}. "
            "Check LitePulseConfig against your DATA_TYPE preprocessing config."
        )
    roi_tensors = []
    for roi_idx in range(r):
        parts = [
            x[:, :, dt * per_type_width + roi_idx * c: dt * per_type_width + (roi_idx + 1) * c]
            for dt in range(t)
        ]
        roi_tensors.append(torch.cat(parts, dim=2))  # (B, T, t*c, H, W)
    return roi_tensors


class PulseStem(nn.Module):
    """
    Gradual channel expansion (in -> 16 -> 24 -> 32) instead of an
    aggressive single-shot compression, so subtle chrominance changes
    (the pulse signal is ~1-2% of pixel intensity) survive the stem.
    Applied per-frame (folds T into the batch dim), shared across ROIs
    by using the same module instance for every ROI branch.
    """

    def __init__(self, cfg: LitePulseConfig):
        super().__init__()
        c1, c2, c3 = cfg.stem_channels
        self.net = nn.Sequential(
            nn.Conv2d(cfg.stem_in_channels, c1, kernel_size=3, padding=1),
            nn.BatchNorm2d(c1), nn.ReLU(inplace=True),
            nn.Conv2d(c1, c2, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(c2), nn.ReLU(inplace=True),
            nn.Conv2d(c2, c3, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(c3), nn.ReLU(inplace=True),
        )
        self.out_channels = c3

    def forward(self, x):
        # x: (B, T, Cin, H, W) -> (B, T, Cout, H/4, W/4)
        b, t, c, h, w = x.shape
        x = x.reshape(b * t, c, h, w)
        x = self.net(x)
        _, co, ho, wo = x.shape
        return x.reshape(b, t, co, ho, wo)


class PulseBlock(nn.Module):
    """
    One temporal-spatial residual block: depthwise temporal conv (mixes
    information across frames per-channel) followed by a spatial conv,
    residual-connected. Shared across ROI branches (same module instance
    called on each ROI's feature stream) so this is the "Shared LitePulse
    Encoder" from the architecture sketch.
    """

    def __init__(self, channels: int, dropout: float = 0.1):
        super().__init__()
        self.temporal = nn.Conv1d(channels, channels, kernel_size=3, padding=1,
                                   groups=channels)  # depthwise over time
        self.spatial = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn_t = nn.BatchNorm1d(channels)
        self.bn_s = nn.BatchNorm2d(channels)
        self.drop = nn.Dropout2d(dropout)

    def forward(self, x):
        # x: (B, T, C, H, W)
        b, t, c, h, w = x.shape

        # temporal mixing: (B, T, C, H, W) -> (B*H*W, C, T)
        xt = x.permute(0, 3, 4, 2, 1).reshape(b * h * w, c, t)
        xt = F.relu(self.bn_t(self.temporal(xt)))
        xt = xt.reshape(b, h, w, c, t).permute(0, 4, 3, 1, 2)  # (B, T, C, H, W)

        # spatial mixing: (B, T, C, H, W) -> (B*T, C, H, W)
        xs = xt.reshape(b * t, c, h, w)
        xs = F.relu(self.bn_s(self.spatial(xs)))
        xs = self.drop(xs)
        xs = xs.reshape(b, t, c, h, w)

        return x + xs  # residual


class ROIEncoder(nn.Module):
    """Stem + N PulseBlocks + spatial global-average-pool -> (B, T, C) per ROI."""

    def __init__(self, cfg: LitePulseConfig):
        super().__init__()
        self.stem = PulseStem(cfg)
        self.blocks = nn.ModuleList([
            PulseBlock(self.stem.out_channels, cfg.dropout)
            for _ in range(cfg.num_pulse_blocks)
        ])
        self.out_channels = self.stem.out_channels

    def forward(self, x):
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        return x.mean(dim=[3, 4])  # (B, T, C) global-average-pool over H, W


class AdaptivePhysiologicalROIFusion(nn.Module):
    """
    APRF: instead of a static concat + 1x1 conv, a lightweight quality
    estimator scores each ROI's reliability per frame, softmax-normalizes
    those scores across ROIs, and fuses with the resulting weighted sum.
    An occluded/noisy ROI's weight is pushed toward 0 automatically rather
    than being hand-tuned or fixed at train time.
    """

    def __init__(self, channels: int, num_rois: int):
        super().__init__()
        self.quality_head = nn.Sequential(
            nn.Linear(channels, channels // 2), nn.ReLU(inplace=True),
            nn.Linear(channels // 2, 1),
        )
        self.num_rois = num_rois

    def forward(self, roi_feats: List[torch.Tensor]):
        """
        roi_feats: list of num_rois tensors, each (B, T, C)
        returns: fused (B, T, C), weights (B, T, num_rois)
        """
        stacked = torch.stack(roi_feats, dim=2)  # (B, T, R, C)
        logits = self.quality_head(stacked).squeeze(-1)  # (B, T, R)
        weights = torch.softmax(logits, dim=-1)  # sum to 1 across ROIs
        fused = (stacked * weights.unsqueeze(-1)).sum(dim=2)  # (B, T, C)
        return fused, weights


class PulseSpectralAttention(nn.Module):
    """
    Physiology-guided channel attention: for each feature channel, compute
    the FFT energy fraction that falls in the plausible heart-rate band
    (default 0.7-2.5 Hz), and use that fraction (not raw variance) as the
    attention weight. Channels whose temporal pattern doesn't look like a
    pulse (motion, lighting flicker, blinks -- all far higher variance
    than the ~1-3% pulse signal) get suppressed.
    """

    def __init__(self, fps: float, band_hz: Tuple[float, float]):
        super().__init__()
        self.fps = fps
        self.band_hz = band_hz

    def forward(self, x):
        # x: (B, T, C)
        b, t, c = x.shape
        xf = x - x.mean(dim=1, keepdim=True)  # remove DC before FFT
        spec = torch.fft.rfft(xf, dim=1)  # (B, T//2+1, C)
        power = spec.real.pow(2) + spec.imag.pow(2)

        freqs = torch.fft.rfftfreq(t, d=1.0 / self.fps).to(x.device)  # (T//2+1,)
        lo, hi = self.band_hz
        band_mask = (freqs >= lo) & (freqs <= hi)  # (T//2+1,)

        band_energy = power[:, band_mask, :].sum(dim=1)  # (B, C)
        total_energy = power.sum(dim=1) + 1e-8            # (B, C)
        energy_ratio = band_energy / total_energy          # (B, C) in [0, 1]

        attn = torch.sigmoid((energy_ratio - energy_ratio.mean(dim=-1, keepdim=True)) * 4.0)
        return x * attn.unsqueeze(1), energy_ratio  # (B, T, C), (B, C) for logging


class TemporalCNN(nn.Module):
    """Refines the fused, attention-weighted feature sequence over time."""

    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(channels), nn.ReLU(inplace=True),
            nn.Conv1d(channels, channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(channels), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # x: (B, T, C) -> (B, C, T) -> conv -> (B, T, C)
        x = x.transpose(1, 2)
        x = self.net(x)
        return x.transpose(1, 2)


class LitePulse(nn.Module):
    def __init__(self, cfg: LitePulseConfig = None):
        super().__init__()
        self.cfg = cfg or LitePulseConfig()

        self.roi_encoder = ROIEncoder(self.cfg)  # shared across ROI branches
        C = self.roi_encoder.out_channels

        self.fusion = AdaptivePhysiologicalROIFusion(C, self.cfg.num_rois)
        self.spectral_attn = PulseSpectralAttention(self.cfg.fps, self.cfg.pulse_band_hz)
        self.temporal_cnn = TemporalCNN(C)
        self.proj_head = nn.Linear(C, 1)

        # Per-ROI auxiliary heads used only for the ROI-consistency loss
        # (see LitePulseTrainer). Cheap: one Linear per ROI on top of the
        # already-shared ROI features.
        self.aux_roi_heads = nn.ModuleList([nn.Linear(C, 1) for _ in range(self.cfg.num_rois)])

    def forward(self, x, return_aux=False):
        """
        x: (B, T, C_in, H, W) channel-stacked multi-ROI input (see module
           docstring for the expected channel layout).
        returns: bvp (B, T) predicted signal.
                 if return_aux: also (roi_weights (B,T,R), roi_preds list[(B,T)],
                                       spectral_energy_ratio (B, C))
        """
        roi_inputs = regroup_by_roi(x, self.cfg)  # list of (B, T, Cin_roi, H, W)
        roi_feats = [self.roi_encoder(roi_x) for roi_x in roi_inputs]  # list of (B, T, C)

        fused, roi_weights = self.fusion(roi_feats)          # (B, T, C), (B, T, R)
        attended, energy_ratio = self.spectral_attn(fused)     # (B, T, C), (B, C)
        temporal = self.temporal_cnn(attended)                 # (B, T, C)
        bvp = self.proj_head(temporal).squeeze(-1)              # (B, T)

        if not return_aux:
            return bvp

        roi_preds = [head(feat).squeeze(-1) for head, feat in zip(self.aux_roi_heads, roi_feats)]
        return bvp, roi_weights, roi_preds, energy_ratio