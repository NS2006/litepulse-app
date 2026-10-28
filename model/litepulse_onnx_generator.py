"""
Exports the LitePulse checkpoint to ONNX for the browser version
(web/, run with ONNX Runtime Web).

    pip install onnx onnxruntime
    python litepulse_onnx_generator.py
    copy litepulse.onnx ..\\web\\models\\litepulse.onnx

PulseSpectralAttention is swapped for an export-friendly copy: the window is
always 180 frames, so rfft + the boolean band mask become two fixed DFT
matrix multiplies. Same math, but only plain ops (MatMul, Mul, ReduceSum)
that every ONNX Runtime backend supports. The script refuses to finish if
the saved .onnx does not match the original PyTorch model.
"""

import copy

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn

from LitePulse import LitePulse, LitePulseConfig, PulseSpectralAttention


CKPT = "./pth_model/UBFC_UBFC_PURE_litepulseFDCMAEv2_epoch29.pth.zip"
OUTPUT = "litepulse.onnx"
TEMPORAL_FRAMES = 180

cfg = LitePulseConfig(
    num_rois=3,
    channels_per_roi=3,
    num_data_types=2,
    num_pulse_blocks=3,
    fps=30.0,
)


class PulseSpectralAttentionDFT(nn.Module):
    """PulseSpectralAttention with rfft written as fixed cos/sin matmuls."""

    def __init__(self, original: PulseSpectralAttention, t: int):
        super().__init__()
        n = torch.arange(t, dtype=torch.float64)
        k = torch.arange(t // 2 + 1, dtype=torch.float64)
        angle = 2 * torch.pi * k[:, None] * n[None, :] / t

        freqs = torch.fft.rfftfreq(t, d=1.0 / original.fps).double()
        lo, hi = original.band_hz
        band = ((freqs >= lo) & (freqs <= hi)).double()

        self.register_buffer("cos", torch.cos(angle).float())  # (F, T)
        self.register_buffer("sin", torch.sin(angle).float())  # (F, T)
        self.register_buffer("band", band.float())              # (F,)

    def forward(self, x):
        # x: (B, T, C)
        xf = x - x.mean(dim=1, keepdim=True)
        real = torch.matmul(self.cos, xf)  # (B, F, C)
        imag = torch.matmul(self.sin, xf)
        power = real * real + imag * imag

        band_energy = (power * self.band[None, :, None]).sum(dim=1)  # (B, C)
        total_energy = power.sum(dim=1) + 1e-8
        energy_ratio = band_energy / total_energy

        attn = torch.sigmoid((energy_ratio - energy_ratio.mean(dim=-1, keepdim=True)) * 4.0)
        return x * attn.unsqueeze(1), energy_ratio


class LitePulseExport(nn.Module):
    """
    Same outputs as the .ptl export:
        bvp           (1, 180)      predicted pulse signal
        roi_weights   (1, 180, 3)   APRF weights [forehead, left cheek, right cheek]
        energy_ratio  (1, 32)       PulseSpectralAttention band-energy ratio per channel
    """

    def __init__(self, model: LitePulse):
        super().__init__()
        self.model = model

    def forward(self, x):
        bvp, roi_weights, _roi_preds, energy_ratio = self.model(x, return_aux=True)
        return bvp, roi_weights, energy_ratio


model = LitePulse(cfg)
model.load_state_dict(torch.load(CKPT, map_location="cpu", weights_only=True))
model.eval()

export_model = copy.deepcopy(model)
export_model.spectral_attn = PulseSpectralAttentionDFT(model.spectral_attn, TEMPORAL_FRAMES)
export_model = LitePulseExport(export_model).eval()
reference_model = LitePulseExport(model).eval()

example = torch.randn(1, TEMPORAL_FRAMES, 18, 36, 36)
names = ["bvp", "roi_weights", "energy_ratio"]

torch.onnx.export(
    export_model,
    (example,),
    OUTPUT,
    input_names=["input"],
    output_names=names,
    opset_version=17,
    dynamo=False,
)
onnx.checker.check_model(onnx.load(OUTPUT))

# Check the file the browser will actually load against the original model.
session = ort.InferenceSession(OUTPUT, providers=["CPUExecutionProvider"])

print("Original PyTorch model vs saved .onnx:")
for seed in range(3):
    torch.manual_seed(seed)
    check_input = torch.randn(1, TEMPORAL_FRAMES, 18, 36, 36)

    with torch.no_grad():
        reference_output = reference_model(check_input)
    onnx_output = session.run(None, {"input": check_input.numpy()})

    for name, ref, out in zip(names, reference_output, onnx_output):
        diff = np.abs(ref.numpy() - out).max()
        print(f"  seed {seed} {name:13s} max|diff|={diff:.2e}")
        assert np.allclose(ref.numpy(), out, atol=1e-4), \
            f"Saved .onnx does not match the PyTorch model in {name}"

print("\nSaved:")
print(OUTPUT)
