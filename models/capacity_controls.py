"""Active capacity controls: parameter matching is measured, never padded."""
from torch import nn
import torch


def parameters(module):
    return sum(p.numel() for p in module.parameters())


class MatchedResidualConv(nn.Module):
    """Local residual bottleneck with width nearest a declared parameter budget.

    All weights are used. This controls capacity approximately, not FLOPs or
    receptive field. Width and actual mismatch are exported to protocol.json.
    """
    def __init__(self, channels, budget):
        super().__init__()
        # C->h 1x1, h->h 3x3, h->C 1x1; biases in all three.
        count = lambda h: 2 * channels * h + 9 * h * h + 2 * h + channels
        hi = 1
        while count(hi) < budget:
            hi *= 2
        width = min(range(1, hi + 1), key=lambda h: abs(count(h) - budget))
        self.net = nn.Sequential(nn.Conv2d(channels, width, 1), nn.GELU(),
                                 nn.Conv2d(width, width, 3, padding=1), nn.GELU(),
                                 nn.Conv2d(width, channels, 1))
        self.width = width
        self.reference_parameters = int(budget)
        self.control_parameters = parameters(self)
        self.relative_parameter_error = (self.control_parameters - budget) / max(budget, 1)

    def forward(self, x):
        return x + .1 * self.net(x)


class IndependentScaleConvControl(nn.Module):
    """Replace a multi-scale module by independent local convolutions per scale."""
    def __init__(self, channels, reference):
        super().__init__()
        budget = parameters(reference)
        total = sum(c * c for c in channels)
        self.blocks = nn.ModuleList([MatchedResidualConv(c, round(budget * c * c / total)) for c in channels])
        self.reference_parameters = budget
        self.control_parameters = parameters(self)
        self.relative_parameter_error = (self.control_parameters - budget) / max(budget, 1)

    def forward(self, features):
        return [block(feature) for block, feature in zip(self.blocks, features)]


class IntensityPrior(nn.Module):
    """Parameter-free intensity histograms, with the SAME learned HOG modulators.

    This isolates orientation-histogram input from modulation capacity. It is
    an intensity-prior control, not a claim to have removed every handcrafted
    preprocessing operation or matched the cost of HOG extraction.
    """
    def __init__(self, bins, cell_size):
        super().__init__()
        self.bins, self.cell_size = bins, cell_size

    def forward(self, x):
        gray = x.mean(1, keepdim=True)
        low, high = gray.amin((2, 3), keepdim=True), gray.amax((2, 3), keepdim=True)
        gray = (gray - low) / (high - low).clamp_min(1e-6)
        centers = torch.linspace(0., 1., self.bins, device=x.device, dtype=x.dtype)[None, :, None, None]
        # Distinct intensity bins, rather than replicated channels with redundant
        # modulation weights. No directional gradients or orientation histograms.
        hist = (1 - (gray - centers).abs() * max(self.bins - 1, 1)).clamp_min(0.)
        if self.cell_size > 1:
            hist = torch.nn.functional.avg_pool2d(hist, self.cell_size, stride=1, padding=self.cell_size // 2)
            hist = hist[..., :x.shape[-2], :x.shape[-1]]
        return (hist / (hist.mean((2, 3), keepdim=True) + 1e-6)).clamp(max=10.)


class FeedForwardRefinementControl(nn.Module):
    """Exactly the IRB parameters, one conventional residual update.

    The original step scalars gate distinct channel groups of one residual,
    instead of becoming redundant copies in one average. The denoiser is
    evaluated once, with the same tanh residual nonlinearity. Same parameter
    count, different computation, receptive field and FLOPs.
    """
    def __init__(self, reference):
        super().__init__()
        self.reference_parameters = parameters(reference)
        self.denoiser = reference.denoiser
        self.step_scale = reference.step_scale
        if not 0 < self.step_scale.numel() <= reference.denoiser[0].in_channels:
            raise ValueError("Feedforward control needs one nonempty channel group per retained scale")
        self.control_parameters = parameters(self)
        self.relative_parameter_error = 0.

    def forward(self, x):
        residual = torch.tanh(self.denoiser(x))
        groups = torch.tensor_split(residual, self.step_scale.numel(), dim=1)
        return x + torch.cat([scale * group for scale, group in zip(self.step_scale, groups)], dim=1)
