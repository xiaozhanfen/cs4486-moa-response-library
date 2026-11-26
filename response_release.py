"""Build and verify the standalone Kaggle input for the full-data response ensemble."""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd
import torch

from artifact_utils import copy_runtime, quantize_checkpoint, write_json
from config import DATA_DIR, MODELS_DIR, ROOT, RUNS_DIR
from infer_engine import predict_member
from refit_response import FULL
from response_study import STUDY, apply_blend, loss, read_study, safe_logit


RELEASE = ROOT / "release_rebuilt"
DATASET = "xiaozhanfen/moa-response-library-weights"
NOTEBOOK = "xiaozhanfen/moa-response-library"


def build():
    fit = json.loads((STUDY / "refit.json").read_bytes())
    report = json.loads((STUDY / "selected_cv.json").read_bytes())
    if not report["passed"] or fit["tags"] != report["neural_members"]:
        raise ValueError("Refit and accepted five-fold study differ")
    if RELEASE.exists():
        raise FileExistsError(f"Preserving existing release at {RELEASE}")
    stage = RELEASE.with_name(RELEASE.name + ".building")
    stage.mkdir(parents=True)
    dataset, notebook = stage / "dataset", stage / "notebook"
    dataset.mkdir()
    notebook.mkdir()
    copy_runtime(dataset, notebook)
    for name in ["response_features.py", "drug_library.py"]:
        shutil.copyfile(ROOT / name, dataset / name)
    params = report["deployment"]
    slopes = np.array(params["coefficients"])
    if (slopes < 0).any() or slopes.sum() <= 0:
        raise ValueError("Neural mixture coefficients are invalid")
    members = []
    for tag, weight in zip(fit["tags"], slopes/slopes.sum(), strict=True):
        cfg = json.loads((RUNS_DIR / tag / "config.json").read_bytes())
        chosen = {k:cfg[k] for k in ["kind", "hidden", "dropout", "in_dim", "out_dim", "target_cols"]}
        chosen.update(folds=1, seeds=fit["seeds"], storage="row-int8", feature_run=fit["recipes"][tag]["feature_run"])
        members.append({"tag": tag, "weight": float(weight), "config": chosen})
    manifest = {"release": "response-library", "training_scope": "all_treated", "blend_space": "fold-logit",
                "folds": 1, "drug_library": True, "members": members, "calibration": {
                    "slope": float(slopes.sum()), "intercept": params["intercept"],
                    "library_low": params["library_low"], "library_high": params["library_high"]}}
    write_json(dataset / "moa_manifest.json", manifest)
    with zipfile.ZipFile(dataset / "weights_features.moa", "w") as z:
        for p in sorted([*FULL.glob("*/feat/global.npz"), FULL / "drug_library/fold0.npz"]):
            z.write(p, "models/" + p.relative_to(FULL).as_posix())
    for seed in fit["seeds"]:
        with zipfile.ZipFile(dataset / f"weights_seed{seed}.moa", "w") as z:
            for tag in fit["tags"]:
                rel = Path(tag) / f"seed{seed}/fold0"
                z.writestr("models/" + (rel / "model.npz").as_posix(), quantize_checkpoint(FULL / rel / "model.pt"))
    write_json(dataset / "dataset-metadata.json", {"title": "MoA Response Library Weights", "id": DATASET,
               "licenses": [{"name": "other"}], "description": "Private course weights trained from the official lish-moa data for academic use."})
    write_json(notebook / "kernel-metadata.json", {"id": NOTEBOOK, "title": "MoA Response Library",
               "code_file": "kernel.ipynb", "language": "python", "kernel_type": "notebook", "is_private": True,
               "enable_gpu": False, "enable_tpu": False, "enable_internet": False,
               "dataset_sources": [DATASET], "competition_sources": ["lish-moa"], "kernel_sources": []})
    stage.rename(RELEASE)


def holdout_precision(root, report):
    """Check how much the deployment storage format changes labeled fold-zero predictions."""
    tags = report["neural_members"]
    ids, folds, y, _, _, original, library, confidence = read_study(tags)
    valid = folds == 0
    frame = pd.read_csv(DATA_DIR / "train_features.csv").set_index("sig_id").loc[ids[valid]].reset_index()
    packed = []
    for tag in tags:
        cfg = json.loads((RUNS_DIR / tag / "config.json").read_bytes())
        feature_run = cfg.get("feature_run") or tag
        features = root / feature_run / "feat"
        features.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(MODELS_DIR / feature_run / "feat/fold0.npz", features / "fold0.npz")
        for seed in cfg["seeds"]:
            folder = Path(tag) / f"seed{seed}/fold0"
            (root / folder).mkdir(parents=True, exist_ok=True)
            (root / folder / "model.npz").write_bytes(quantize_checkpoint(MODELS_DIR / folder / "model.pt"))
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pred = predict_member(tag, cfg["kind"], cfg["hidden"], frame, root, device,
                              config=dict(cfg, storage="row-int8"), only_fold=0)
        packed.append(safe_logit(pred))
    combined = np.stack(packed, axis=-1)
    args = (library[valid], confidence[valid], report["deployment"])
    delta = loss(y[valid], apply_blend(combined, *args)) - loss(y[valid], apply_blend(original[valid], *args))
    if abs(delta) >= 1e-5:
        raise ValueError(f"INT8 log-loss drift on fold zero exceeds tolerance: {delta}")
    return delta




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["build", "precision"])
    parser.add_argument("--release-dir", type=Path, default=RELEASE)
    args = parser.parse_args()
    RELEASE = args.release_dir.resolve()
    if RELEASE == (ROOT / "response_release").resolve():
        parser.error("The scored V1 bundle is immutable; use a fresh release directory")
    if args.action == "build":
        build()
    else:
        from predict import workspace
        report = json.loads((STUDY / "selected_cv.json").read_bytes())
        with workspace() as scratch:
            print("INT8 fold-zero log-loss delta:", holdout_precision(scratch, report))
