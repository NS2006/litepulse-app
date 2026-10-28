import torch
from torch.jit.mobile import _load_for_lite_interpreter
from torch.utils.mobile_optimizer import MobileOptimizerType, optimize_for_mobile

from LitePulse import LitePulse, LitePulseConfig


CKPT = "./pth_model/UBFC_UBFC_PURE_litepulseFDCMAEv2_epoch29.pth.zip"

cfg = LitePulseConfig(
    num_rois=3,
    channels_per_roi=3,
    num_data_types=2,
    num_pulse_blocks=3,
    fps=30.0,
)


class LitePulseExport(torch.nn.Module):
    """
    Output of the .ptl (what the Android app reads):
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

checkpoint = torch.load(
    CKPT,
    map_location="cpu"
)

model.load_state_dict(checkpoint)
model.eval()

export_model = LitePulseExport(model).eval()


# LitePulse input:
# [batch, temporal, channels, height, width]
example = torch.randn(
    1,
    180,
    18,
    36,
    36
)


with torch.no_grad():

    # The original single-output model, to make sure the wrapper
    # did not change the pulse prediction.
    original_bvp = model(example)

    reference_output = export_model(example)

    traced_model = torch.jit.trace(
        export_model,
        example,
        check_trace=False
    )

    traced_output = traced_model(example)

    names = ["bvp", "roi_weights", "energy_ratio"]

    print("Reference vs traced:")
    for name, ref, traced in zip(names, reference_output, traced_output):
        print(
            f"  {name:13s} shape={tuple(ref.shape)} "
            f"max|diff|={(ref - traced).abs().max().item():.2e}"
        )
        assert torch.allclose(ref, traced, atol=1e-5), \
            f"TorchScript trace mismatch in {name}"

    assert torch.allclose(original_bvp, reference_output[0], atol=1e-6), \
        "Wrapper changed the BVP output"

    print("\nroi_weights sum over ROIs (should be 1):",
          traced_output[1].sum(dim=-1).mean().item())


# CONV_BN_FUSION must stay off: the ROI encoder is one module called three
# times (once per ROI), and the pass folds BatchNorm into the shared conv
# weights once per call site. The result looked fine to load but gave
# APRF weights of [1, 0, 0] and a corrupted BVP.
optimized_model = optimize_for_mobile(
    traced_model,
    optimization_blocklist={MobileOptimizerType.CONV_BN_FUSION}
)

optimized_model._save_for_lite_interpreter(
    "litepulse.ptl"
)

# Check the file the app will actually load, not just the traced model.
saved_model = _load_for_lite_interpreter("litepulse.ptl")

print("\nReference vs saved .ptl:")
for seed in range(3):
    torch.manual_seed(seed)
    check_input = torch.randn(1, 180, 18, 36, 36)

    with torch.no_grad():
        reference_output = export_model(check_input)
        saved_output = saved_model(check_input)

    for name, ref, saved in zip(names, reference_output, saved_output):
        diff = (ref - saved).abs().max().item()
        print(f"  seed {seed} {name:13s} max|diff|={diff:.2e}")
        assert torch.allclose(ref, saved, atol=1e-4), \
            f"Saved .ptl does not match the PyTorch model in {name}"

print("\nSaved:")
print("litepulse.ptl")