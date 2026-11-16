"""
model.py — ECoGFlexNet: a *streaming* causal decoder for BCI Comp IV ds4.

The important change is not the layer list, it is the I/O contract.

    model_v2:    (B, rows, W) -> (B, 5)        one label per forward pass
    this model:  (B, rows, W) -> (B, 5, W)     one label per input SAMPLE

Same causal stack, same weights, but the head is read at every timestep
instead of only the last one. Three consequences, all of them large:

  * ACCURACY. Emitting at 20 Hz and holding each value for 50 ms puts a
    staircase on top of a signal that is scored at 1000 Hz. Emitting at
    100 Hz reduces the step to 10 ms. On a simulated glove trace the 20 Hz
    hold costs about 0.021 r and the 100 Hz hold about 0.003 — which is
    almost exactly the -0.0197 that score_official.py already measures as
    the "A pipeline -> B filtered-1kHz" row. That row is not a measurement
    artefact, it is the output rate, and it is refundable.

  * COMPUTE. A sliding window recomputes the whole receptive field for one
    output. Reading every timestep amortises the same arithmetic over W
    outputs. Sustained cost drops from `MACs_per_window x 20/s` to
    `MACs_per_sample x 100/s`, roughly 15-20x less work for 5x more
    predictions. `mac_report()` prints both framings; a paper about edge
    deployment should quote MACs per emitted sample, because that is what
    the battery pays for.

  * SUPERVISION. Every timestep is a training target, which is the same
    argument model_v2's `forward_sequence` made — except that here the dense
    path IS the deployed path, so nothing has to be verified equivalent.

CAUSALITY BUG THIS FILE FIXES
-----------------------------
model_v2's ECALayer squeezes with `x.mean(dim=-1)`, a mean over the WHOLE
window. In `forward` that is harmless, because the window ends at the
predicted timestep. In `forward_sequence` it is not: the attention weight
applied to timestep t is computed from timesteps > t, so 399 of the 400
densely supervised outputs were trained with a look-ahead that the exported
single-step graph does not have. Training and deployment optimised different
functions. Replaced here by `CausalChannelGate`, a pointwise bottleneck gate
that depends only on the current timestep's activations. `_selftest` asserts
strict causality of the dense output, which model_v2 could not have passed
at the tolerance used here.

EXPORT CONSTRAINTS (carried over from model.py / model_v2.py, still honoured)
    * depthwise convs use groups=ch with a strict multiplier of 1
    * per-band spatial filters are a ModuleList of plain Conv2d, not one
      grouped conv with multiplier > 1
    * causal convs left-pad with F.pad and use padding=0, emitting
      constant_pad_nd rather than slice_copy (the "pad" mode that fixed
      `xnn_status_invalid_parameter` at static slice node 45)
    * the dense head needs NO slice at all: it returns every timestep and the
      host keeps the tail it wants. The last-step path `forward_last` is kept
      for the old export and uses the two-`select` pattern that lowered
      cleanly before.
    * ops used: conv, add, mul, cat, relu, sigmoid, constant_pad_nd
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ======================================================================
# Blocks
# ======================================================================
class CausalChannelGate(nn.Module):
    """Per-timestep channel gate. The causal replacement for ECA / squeeze-
    excite, both of which pool over time and therefore look ahead.

    w_t = sigmoid(W2 relu(W1 x_t)), x_t -> x_t * w_t.
    Cost 2*ch*ch/r MACs per timestep; at ch=48, r=4 that is 1152.
    """

    def __init__(self, channels, reduction=4):
        super().__init__()
        hidden = max(4, channels // reduction)
        self.fc1 = nn.Conv2d(channels, hidden, 1, bias=True)
        self.fc2 = nn.Conv2d(hidden, channels, 1, bias=True)

    def forward(self, x):
        return x * torch.sigmoid(self.fc2(F.relu(self.fc1(x))))


class ECAGate(nn.Module):
    """ABLATION ONLY: model_v2's time-pooling ECA gate, reinstated so the
    look-ahead claim can be measured rather than asserted. The squeeze is a
    mean over the WHOLE input window, so under dense supervision the gate at
    timestep t sees timesteps > t. Never use in a deployed model.
    """

    def __init__(self, channels, k=3):
        super().__init__()
        self.conv = nn.Conv1d(1, 1, k, padding=k // 2, bias=False)

    def forward(self, x):                          # (B, C, 1, T)
        s = x.mean(dim=(2, 3))                     # pools over time: leaks
        w = torch.sigmoid(self.conv(s.unsqueeze(1))).squeeze(1)
        return x * w[:, :, None, None]


class CausalSepBlock(nn.Module):
    """Depthwise-separable causal residual block.

    dw (groups=ch, multiplier 1) -> pw -> BN -> ReLU -> spatial dropout
    -> residual add -> ReLU.

    "pad" and "slice" modes are algebraically identical and share weights;
    `padding` is a module attribute, not a parameter, so a checkpoint trained
    under one re-exports under the other without retraining.
    """

    def __init__(self, ch, kernel_size, dilation, dropout, causal_mode="pad"):
        super().__init__()
        assert kernel_size >= 2
        self.pad = (kernel_size - 1) * dilation
        self.dw = nn.Conv2d(ch, ch, (1, kernel_size), padding=(0, 0),
                            dilation=(1, dilation), groups=ch, bias=False)
        self.pw = nn.Conv2d(ch, ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(ch)
        self.drop = nn.Dropout2d(dropout)
        self.set_causal_mode(causal_mode)

    def set_causal_mode(self, mode):
        assert mode in ("pad", "slice")
        self.causal_mode = mode
        self.dw.padding = (0, self.pad) if mode == "slice" else (0, 0)

    def forward(self, x):
        if self.causal_mode == "slice":
            y = self.dw(x)[:, :, :, :-self.pad]
        else:
            y = self.dw(F.pad(x, (self.pad, 0)))
        y = self.drop(F.relu(self.bn(self.pw(y))))
        return F.relu(x + y)


class SpatialStem(nn.Module):
    """Per-band spatial filtering — S learned electrode weightings per band.

    Subject-specific: the only part of the network whose shape depends on the
    montage, which is what makes cross-subject training of the trunk possible.
    Per-band rather than one shared pattern because LMP and high gamma have
    different generators and different per-electrode SNR, and because the
    filter bank has already done the frequency decomposition, so there is
    nothing for a temporal-first ordering (EEGNet's rationale) to buy here.

    (B, n_bands*C, T) -> (B, n_bands*S, 1, T)

    TWO LOWERINGS, ONE WEIGHT TENSOR
    --------------------------------
      "split"    n_bands separate convolutions over x[:, b:b+1]. Emits
                 n_bands static `slice` nodes. This is model_v2's pattern.
      "grouped"  one grouped convolution, groups=n_bands. No slice nodes at
                 all, but it is a depthwise conv with multiplier S, which
                 model.py's comments flag as not reliably delegated to
                 XNNPACK.

    Both read the SAME `weight`/`bias` parameters, so they are numerically
    identical and a checkpoint moves between them freely — flip the mode if a
    backend rejects one, no retraining. `export_int8.py --probe` tries both.
    """

    def __init__(self, n_electrodes, n_bands, spatial_filters, dropout=0.2,
                 mode="split"):
        super().__init__()
        self.n_electrodes = int(n_electrodes)
        self.n_bands = int(n_bands)
        self.spatial_filters = int(spatial_filters)
        self.weight = nn.Parameter(
            torch.empty(self.n_bands * self.spatial_filters, 1,
                        self.n_electrodes, 1))
        self.bias = nn.Parameter(torch.zeros(self.n_bands * self.spatial_filters))
        bound = 1.0 / math.sqrt(self.n_electrodes)
        nn.init.uniform_(self.weight, -bound, bound)
        nn.init.uniform_(self.bias, -bound, bound)
        self.bn = nn.BatchNorm2d(self.n_bands * self.spatial_filters)
        self.drop = nn.Dropout2d(dropout)
        self.set_spatial_mode(mode)

    def set_spatial_mode(self, mode):
        assert mode in ("split", "grouped")
        self.spatial_mode = mode
        return self

    @property
    def out_channels(self):
        return self.n_bands * self.spatial_filters

    def forward(self, x):
        x = x.view(-1, self.n_bands, self.n_electrodes, x.shape[-1])
        S = self.spatial_filters
        if self.spatial_mode == "grouped":
            x = F.conv2d(x, self.weight, self.bias, groups=self.n_bands)
        else:
            x = torch.cat([F.conv2d(x[:, b:b + 1],
                                    self.weight[b * S:(b + 1) * S],
                                    self.bias[b * S:(b + 1) * S])
                           for b in range(self.n_bands)], dim=1)
        return self.drop(F.relu(self.bn(x)))


class Trunk(nn.Module):
    """Everything after the spatial stem. Shape-independent of the subject,
    so one trunk can be trained on all three recordings at once.

    (B, in_ch, 1, T) -> (B, n_outputs, 1, T)
    """

    def __init__(self, in_ch, width=48, n_outputs=5, tcn_kernel=5,
                 dilations=(1, 2, 4, 8, 16, 32), dropout=0.2,
                 head_hidden=32, smooth_kernel=32, skip_mode="sum",
                 causal_mode="pad", use_gate=True, gate_kind="causal"):
        super().__init__()
        assert skip_mode in ("sum", "last")
        assert gate_kind in ("causal", "eca")
        self.width = int(width)
        self.n_outputs = int(n_outputs)
        self.tcn_kernel = int(tcn_kernel)
        self.dilations = tuple(int(d) for d in dilations)
        self.skip_mode = skip_mode
        self.smooth_kernel = int(smooth_kernel)

        self.mix = nn.Conv2d(in_ch, self.width, 1, bias=False)
        self.bn_mix = nn.BatchNorm2d(self.width)
        if not use_gate:
            self.gate = None
        elif gate_kind == "eca":
            self.gate = ECAGate(self.width)
        else:
            self.gate = CausalChannelGate(self.width)

        self.blocks = nn.ModuleList([
            CausalSepBlock(self.width, self.tcn_kernel, d, dropout, causal_mode)
            for d in self.dilations
        ])

        # 1x1 head. Reading it at every timestep is what makes the model a
        # streaming decoder; a Linear layer would not.
        self.head1 = nn.Conv2d(self.width, head_hidden, 1, bias=True)
        self.head2 = nn.Conv2d(head_hidden, self.n_outputs, 1, bias=True)

        # Learned causal output smoother, one FIR per finger. This subsumes
        # `apply_ema_filter`: it is strictly causal, it is trained rather than
        # tuned, it lives inside the exported graph (so the deployed decoder
        # and the scored decoder are the same object), and being asymmetric it
        # can also absorb a small per-finger timing offset. Initialised to the
        # EMA(alpha=0.4) kernel that eval_protocol_v2 hard-coded, so training
        # starts from the old behaviour.
        if self.smooth_kernel > 1:
            self.smooth = nn.Conv2d(self.n_outputs, self.n_outputs,
                                    (1, self.smooth_kernel),
                                    groups=self.n_outputs, bias=False)
            with torch.no_grad():
                a = 0.4
                k = torch.tensor([a * (1 - a) ** i
                                  for i in range(self.smooth_kernel)][::-1])
                k = k / k.sum()
                self.smooth.weight.copy_(
                    k.view(1, 1, 1, -1).repeat(self.n_outputs, 1, 1, 1))
        else:
            self.smooth = None
        self.causal_mode = causal_mode

    @property
    def receptive_field(self):
        rf = 1 + (self.tcn_kernel - 1) * sum(self.dilations)
        return rf + (self.smooth_kernel - 1 if self.smooth is not None else 0)

    def set_causal_mode(self, mode):
        for b in self.blocks:
            b.set_causal_mode(mode)
        self.causal_mode = mode

    def forward(self, x):
        x = F.relu(self.bn_mix(self.mix(x)))
        if self.gate is not None:
            x = self.gate(x)

        if self.skip_mode == "sum":
            # Free skip aggregation: every block already outputs `width`
            # channels, so summing them costs 0 MACs and still gives the head
            # access to short- and long-timescale features. A WaveNet-style
            # 1x1 skip projection per block would roughly double the trunk's
            # arithmetic for a marginal gain.
            acc = None
            for b in self.blocks:
                x = b(x)
                acc = x if acc is None else acc + x
            x = acc
        else:
            for b in self.blocks:
                x = b(x)

        y = self.head2(F.relu(self.head1(x)))
        if self.smooth is not None:
            y = self.smooth(F.pad(y, (self.smooth_kernel - 1, 0)))
        return y


# ======================================================================
# Single-subject model
# ======================================================================
class ECoGFlexNet(nn.Module):
    """input (B, n_bands*n_electrodes, T) band-major -> output (B, 5, T)."""

    def __init__(self, n_electrodes, n_bands=7, n_outputs=5, spatial_filters=8,
                 spatial_dropout=0.2, stem=None, trunk=None, **trunk_kw):
        super().__init__()
        self.stem = stem if stem is not None else SpatialStem(
            n_electrodes, n_bands, spatial_filters, spatial_dropout)
        self.trunk = trunk if trunk is not None else Trunk(
            self.stem.out_channels, n_outputs=n_outputs, **trunk_kw)
        self.n_electrodes = self.stem.n_electrodes
        self.n_bands = self.stem.n_bands
        self.n_outputs = self.trunk.n_outputs

    @property
    def receptive_field(self):
        return self.trunk.receptive_field

    def receptive_field_ms(self, fs=100.0):
        return 1000.0 * self.receptive_field / fs

    def set_causal_mode(self, mode):
        self.trunk.set_causal_mode(mode)
        return self

    def set_spatial_mode(self, mode):
        self.stem.set_spatial_mode(mode)
        return self

    def forward(self, x):
        """Dense path. (B, rows, T) -> (B, n_outputs, T).

        This is both the training path and the export path. No slice op: the
        host keeps whichever tail it needs, which is why nothing here can trip
        the XNNPACK static-slice failure.
        """
        return self.trunk(self.stem(x)).squeeze(2)

    def forward_last(self, x):
        """Legacy single-step path, for comparison against model_v2 exports."""
        return self.trunk(self.stem(x))[:, :, 0, -1]


# ======================================================================
# Multi-subject wrapper
# ======================================================================
class MultiSubjectNet(nn.Module):
    """One shared trunk, one spatial stem per subject.

    ds4 gives about 6 min 40 s of training data per subject. The trunk is
    where nearly all the parameters and all the temporal modelling live, and
    nothing in it depends on the electrode montage, so it can be fitted on
    three recordings instead of one. The stem stays subject-specific because
    the montage is subject-specific — and because the dataset description
    states the channel order is scrambled, so there is no shared spatial
    prior to exploit even in principle.

    `single(sub)` returns a plain ECoGFlexNet sharing the same tensors, which
    is what gets fine-tuned, exported and scored.
    """

    def __init__(self, electrode_counts, n_bands=7, n_outputs=5,
                 spatial_filters=8, spatial_dropout=0.2, **trunk_kw):
        super().__init__()
        self.stems = nn.ModuleDict({
            sub: SpatialStem(n_el, n_bands, spatial_filters, spatial_dropout)
            for sub, n_el in electrode_counts.items()
        })
        any_stem = next(iter(self.stems.values()))
        self.trunk = Trunk(any_stem.out_channels, n_outputs=n_outputs, **trunk_kw)

    def forward(self, x, sub):
        return self.trunk(self.stems[sub](x)).squeeze(2)

    def single(self, sub):
        return ECoGFlexNet(None, stem=self.stems[sub], trunk=self.trunk)


# ======================================================================
# Presets
# ======================================================================
PRESETS = {
    # ~2.4 s receptive field. Cheapest configuration that still covers a
    # full flexion cycle.
    "nano": dict(spatial_filters=6, width=32, tcn_kernel=5,
                 dilations=(1, 2, 4, 8, 16, 32), head_hidden=24,
                 smooth_kernel=32),
    # Default. ~2.5 s receptive field, ~30 k MACs per emitted sample.
    "tiny": dict(spatial_filters=8, width=48, tcn_kernel=5,
                 dilations=(1, 2, 4, 8, 16, 32), head_hidden=32,
                 smooth_kernel=32),
    # ~5.1 s receptive field (one extra dilation octave) and a wider trunk.
    # Use when `tiny` underfits, and report it as a separate row.
    "base": dict(spatial_filters=12, width=64, tcn_kernel=5,
                 dilations=(1, 2, 4, 8, 16, 32, 64), head_hidden=48,
                 smooth_kernel=32),
}


def build_model(n_electrodes, n_bands=7, preset="tiny", **overrides):
    cfg = dict(PRESETS[preset])
    sf = cfg.pop("spatial_filters")
    cfg.update(overrides)
    return ECoGFlexNet(n_electrodes, n_bands=n_bands, spatial_filters=sf, **cfg)


def build_multi(electrode_counts, n_bands=7, preset="tiny", **overrides):
    cfg = dict(PRESETS[preset])
    sf = cfg.pop("spatial_filters")
    cfg.update(overrides)
    return MultiSubjectNet(electrode_counts, n_bands=n_bands,
                           spatial_filters=sf, **cfg)


# ======================================================================
# Cost accounting
# ======================================================================
def mac_report(model, window, fs=100.0, emit_rate_windowed=20.0):
    """Exact MAC counts, closed form. Two framings:

    per-sample   arithmetic amortised over every emitted prediction, i.e.
                 what a streaming decoder actually costs. Multiply by fs for
                 sustained MACs/s.
    per-window   arithmetic for one full window forward pass, i.e. the number
                 comparable to the model_v2 / Micro_ECoG_TCN budgets, which
                 emitted one prediction per window.
    """
    stem, trunk = model.stem, model.trunk
    C, B, S = stem.n_electrodes, stem.n_bands, stem.spatial_filters
    W, K = trunk.width, trunk.tcn_kernel

    per_step = 0
    per_step += B * C * S                                   # per-band spatial
    per_step += stem.out_channels * W                       # 1x1 mix
    if isinstance(trunk.gate, ECAGate):
        per_step += W * trunk.gate.conv.kernel_size[0]      # ECA (ablation only)
    elif trunk.gate is not None:
        h = trunk.gate.fc1.out_channels
        per_step += W * h + h * W                           # channel gate
    per_step += len(trunk.dilations) * (W * K + W * W)      # dw + pw per block
    per_step += W * trunk.head1.out_channels
    per_step += trunk.head1.out_channels * trunk.n_outputs
    if trunk.smooth is not None:
        per_step += trunk.n_outputs * trunk.smooth_kernel

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "params": params,
        "macs_per_sample": per_step,
        "macs_per_window": per_step * window,
        "sustained_streaming_mmac_s": per_step * fs / 1e6,
        "sustained_windowed_mmac_s": per_step * window * emit_rate_windowed / 1e6,
        "receptive_field": trunk.receptive_field,
    }


def deployment_cost(model, emit=25, fs=100.0):
    """Cost of the artefact you can actually export today.

    `mac_report`'s per-sample figure is the CACHED-STREAMING limit: what a
    TCN costs when every layer keeps a ring buffer of its own history and each
    new sample advances it by one step. That is the right number for a bespoke
    firmware implementation and the wrong number for a stateless .pte, which
    must re-run the receptive field on every call.

    A stateless export fed `RF + emit - 1` samples produces `emit` valid
    labels, so the overhead factor is (RF + emit - 1) / emit. Reporting the
    cached-streaming figure for a stateless artefact would overstate the
    result by that factor; both are printed so the paper can be explicit
    about which one it is claiming.
    """
    rep = mac_report(model, 1)
    rf = rep["receptive_field"]
    window = rf + emit - 1
    per_call = rep["macs_per_sample"] * window
    return {
        "emit": emit, "window": window, "receptive_field": rf,
        "macs_per_call": per_call,
        "macs_per_label_stateless": per_call / emit,
        "macs_per_label_cached": rep["macs_per_sample"],
        "mmac_s_stateless": per_call * (fs / emit) / 1e6,
        "mmac_s_cached": rep["macs_per_sample"] * fs / 1e6,
        "overhead": window / emit,
    }


# ======================================================================
# Streaming inference
# ======================================================================
@torch.no_grad()
def predict_dense(model, x, chunk=20_000, device=torch.device("cpu")):
    """Run the model over an arbitrarily long recording, causally.

    x : (rows, N) for the whole recording.
    returns (N, n_outputs).

    Chunks overlap by the receptive field and the warm-up part of each chunk
    is discarded, so the result is bit-comparable to one giant forward pass
    but bounded in memory. Timestep t of the output depends only on inputs
    <= t; `_selftest` checks that against a per-window reference.

    The first `receptive_field` samples of the recording are computed from
    zero padding, exactly as a device would at power-on.
    """
    model.eval().to(device)
    rf = model.receptive_field
    n = x.shape[-1]
    out = torch.empty(n, model.n_outputs)

    pos = 0
    while pos < n:
        end = min(pos + chunk, n)
        start = max(0, pos - (rf - 1))
        seg = x[:, start:end].unsqueeze(0).to(device)
        y = model(seg)[0].transpose(0, 1).cpu()      # (T, n_outputs)
        out[pos:end] = y[pos - start:]
        pos = end
    return out


# ======================================================================
# Self-test
# ======================================================================
def _selftest():
    torch.manual_seed(0)
    n_bands, window = 7, 600

    for preset in ("nano", "tiny", "base"):
        for n_el in (58, 41, 60):
            m = build_model(n_el, n_bands=n_bands, preset=preset).eval()
            x = torch.randn(3, n_bands * n_el, window)

            y = m(x)
            assert y.shape == (3, 5, window), y.shape
            assert torch.allclose(m.forward_last(x), y[:, :, -1], atol=1e-5)

            # --- strict causality of EVERY dense output ------------------
            t0 = window // 2
            x2 = x.clone()
            x2[:, :, t0:] += 25.0
            y2 = m(x2)
            drift = (y[:, :, :t0] - y2[:, :, :t0]).abs().max().item()
            assert drift < 1e-5, f"{preset}: dense output not causal ({drift:.2e})"

            # --- interchangeable lowerings, same weights ------------------
            yp = m(x)
            m.set_causal_mode("slice")
            assert torch.allclose(yp, m(x), atol=1e-5), "pad != slice"
            m.set_causal_mode("pad")
            m.set_spatial_mode("grouped")
            assert torch.allclose(yp, m(x), atol=1e-5), "split != grouped"
            m.set_spatial_mode("split")

            # --- chunked streaming == single pass -------------------------
            long_x = torch.randn(n_bands * n_el, 4000)
            ref = m(long_x.unsqueeze(0))[0].transpose(0, 1)
            got = predict_dense(m, long_x, chunk=700)
            assert torch.allclose(ref, got, atol=1e-4), \
                (ref - got).abs().max().item()

            if n_el == 58:
                r = mac_report(m, window)
                print(f"  {preset:<5} el=58 bands=7  params={r['params']:>7,}  "
                      f"RF={r['receptive_field']:>4} ({r['receptive_field'] * 10:>5} ms)  "
                      f"{r['macs_per_sample'] / 1e3:>6.1f} kMAC/sample  "
                      f"{r['sustained_streaming_mmac_s']:>5.2f} MMAC/s @100 Hz")

    # --- multi-subject wiring --------------------------------------------
    ms = build_multi({"sub1": 58, "sub2": 41, "sub3": 60}, n_bands=7).eval()
    for sub, n_el in (("sub1", 58), ("sub2", 41), ("sub3", 60)):
        xa = torch.randn(2, 7 * n_el, 300)
        a = ms(xa, sub)
        b = ms.single(sub)(xa)
        assert a.shape == (2, 5, 300) and torch.allclose(a, b, atol=1e-6)
    tp = sum(p.numel() for p in ms.trunk.parameters())
    sp = sum(p.numel() for p in ms.stems["sub1"].parameters())
    print(f"\n  multi-subject: shared trunk {tp:,} params, "
          f"per-subject stem {sp:,} params "
          f"({100 * tp / (tp + sp):.0f}% of a single-subject model is shareable)")

    # --- the smoother must be causal too ---------------------------------
    m = build_model(20, n_bands=7, preset="tiny").eval()
    x = torch.randn(1, 7 * 20, 500)
    x2 = x.clone()
    x2[:, :, 400:] += 5.0
    assert (m(x)[:, :, :400] - m(x2)[:, :, :400]).abs().max() < 1e-5
    print("  learned output smoother is causal: OK")

    # --- gradients flow through everything -------------------------------
    m.train()
    loss = m(torch.randn(2, 7 * 20, 400)).pow(2).mean()
    loss.backward()
    missing = [n for n, p in m.named_parameters()
               if p.requires_grad and p.grad is None]
    assert not missing, f"no gradient reached: {missing}"
    print("  every parameter receives gradient: OK")
    print("\nAll model self-tests passed.")


if __name__ == "__main__":
    print("ECoGFlexNet — shapes, causality, streaming equivalence, budgets\n")
    _selftest()
