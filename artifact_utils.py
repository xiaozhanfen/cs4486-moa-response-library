"""Shared artifact utilities; independent of previous competition solutions."""
import io
import json
import shutil

import numpy as np
import torch
from scipy.special import logit

from config import ROOT


def safe_logit(p):
    return logit(np.clip(np.asarray(p, dtype=np.float64), 1e-15, 1 - 1e-15))


def quantize_checkpoint(path, max_code=127):
    """Store weight matrices or convolution kernels with one scale per output."""
    if not 1 <= max_code <= 127:
        raise ValueError("Signed weight codes must fit in INT8")
    arrays = {}
    state = torch.load(path, map_location="cpu", weights_only=True)
    for key, tensor in state.items():
        values = tensor.numpy()
        if np.issubdtype(values.dtype, np.floating) and values.ndim >= 2 and values[0].size > 1:
            axes = tuple(range(1, values.ndim))
            scale = np.max(np.abs(values), axis=axes, keepdims=True) / np.float32(max_code)
            scale[scale == 0] = 1
            arrays[key] = np.rint(values / scale).clip(-max_code, max_code).astype(np.int8)
            arrays[key + ".scale"] = scale
        else:
            arrays[key] = values
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    return buffer.getvalue()



def write_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")



def copy_runtime(dataset, notebook):
    for filename in ("models.py", "infer_engine.py"):
        shutil.copyfile(ROOT / filename, dataset / filename)
    template = ROOT / "notebooks/kaggle_inference.ipynb"
    nb = json.loads(template.read_text(encoding="utf-8"))
    nb["cells"][0]["source"] = (ROOT / "kernel.py").read_text(encoding="utf-8").splitlines(True)
    write_json(notebook / "kernel.ipynb", nb)
