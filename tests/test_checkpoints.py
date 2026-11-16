"""The 15 shipped FP32 models (3 subjects x 5 seeds) load, have the size the
paper reports, and are causal. No data needed."""
import glob
import os

import pytest
import torch

import protocol as P
from score import load_model

CKPTS = sorted(glob.glob(os.path.join(P.CKPT_DIR, "paper_sub*_s*.pt")))


def test_fifteen_checkpoints():
    assert len(CKPTS) == 15


@pytest.mark.parametrize("path", CKPTS, ids=os.path.basename)
def test_checkpoint(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model, _ = load_model(path, int(ck["n_electrodes"]), int(ck["n_bands"]))
    assert model.receptive_field == 284
    assert sum(p.numel() for p in model.parameters()) == ck["params"]

    rows = int(ck["n_electrodes"]) * int(ck["n_bands"])
    torch.manual_seed(0)
    x = torch.randn(1, rows, 600)
    with torch.no_grad():
        y = model(x)
        x2 = x.clone()
        x2[:, :, 400:] += 5.0          # change only the future
        y2 = model(x2)
    assert y.shape == (1, 5, 600)
    assert torch.isfinite(y).all()
    assert torch.allclose(y[:, :, :400], y2[:, :, :400], atol=1e-5), "output saw the future"


def test_sub1_size_matches_paper():
    ck = torch.load(os.path.join(P.CKPT_DIR, "paper_sub1_s0.pt"),
                    map_location="cpu", weights_only=False)
    assert ck["params"] == 25369
