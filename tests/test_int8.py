"""The shipped INT8 ExecuTorch models run and keep the right shapes. Needs
`pip install -r requirements-export.txt`; skipped otherwise."""
import glob
import os

import numpy as np
import pytest
import torch

import protocol as P

rt = pytest.importorskip("executorch.runtime")

PTES = sorted(glob.glob(os.path.join(P.CKPT_DIR, "paper_ptq_sub*_s*_int8.pte")))
ROWS = {"sub1": 62 * 7, "sub2": 48 * 7, "sub3": 64 * 7}   # electrodes x bands
RF, EMIT = 284, 10


def test_fifteen_int8_models():
    assert len(PTES) == 15


@pytest.mark.parametrize("path", PTES, ids=os.path.basename)
def test_pte_runs(path):
    sub = os.path.basename(path).split("_")[2]
    method = rt.Runtime.get().load_program(path).load_method("forward")
    x = torch.from_numpy(np.random.default_rng(0).standard_normal(
        (1, ROWS[sub], RF + EMIT - 1)).astype(np.float32))
    y = method.execute((x,))[0]
    assert tuple(y.shape) == (1, 5, RF + EMIT - 1)
    assert torch.isfinite(y).all()
