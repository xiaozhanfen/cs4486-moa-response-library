"""Run the final ensemble on an actual input CSV; no stored predictions are used."""
import argparse
import json
import os
import shutil
import uuid
import zipfile
from contextlib import contextmanager
from pathlib import Path

# Set before importing NumPy/PyTorch, so small matrix products do not oversubscribe.
for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(variable, "4")

import pandas as pd
import torch

from config import DATA_DIR, ROOT
from drug_library import library_for_test
from infer_engine import blend_and_write, run_inference


@contextmanager
def workspace():
    parent = ROOT / ".work"
    parent.mkdir(exist_ok=True)
    path = parent / uuid.uuid4().hex
    path.mkdir()
    try:
        yield path
    finally:
        if path.resolve().parent != parent.resolve():
            raise ValueError("Unexpected temporary cleanup path")
        shutil.rmtree(path)


def unpack(dataset, destination):
    if (dataset / "models").is_dir():
        return dataset / "models"
    archives = sorted(dataset.glob("weights_*.moa"))
    if not archives:
        raise FileNotFoundError(f"No weight archives in {dataset}")
    for path in archives:
        with zipfile.ZipFile(path) as archive:
            for member in archive.namelist():
                if not (destination / member).resolve().is_relative_to(destination.resolve()):
                    raise ValueError("Archive member escapes extraction directory")
            archive.extractall(destination)
    return destination / "models"


def predict(data_dir, output, dataset, device="cpu"):
    test = pd.read_csv(data_dir / "test_features.csv")
    template = pd.read_csv(data_dir / "sample_submission.csv")
    manifest = json.loads((dataset / "moa_manifest.json").read_bytes())
    targets = list(template.drop(columns="sig_id"))
    if targets != manifest["members"][0]["config"]["target_cols"]:
        raise ValueError("Submission label order differs from the trained model")
    if set(template.sig_id) != set(test.sig_id):
        template = pd.DataFrame(0.0, index=range(len(test)), columns=targets)
        template.insert(0, "sig_id", test.sig_id.to_numpy())
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    output.parent.mkdir(parents=True, exist_ok=True)
    with workspace() as scratch:
        models = unpack(dataset, scratch)
        predictions, treated = run_inference(test, models, device, manifest)
        library = library_for_test(test, treated, models, len(targets), manifest["folds"])
        result = blend_and_write(test, predictions, treated, template,
            {member["tag"]: member["weight"] for member in manifest["members"]}, output,
            calibration=manifest["calibration"], library=library)
    print(f"Written {output.resolve()} — {len(result)} rows, {len(targets)} probabilities per row", flush=True)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=DATA_DIR)
    p.add_argument("--output", type=Path, default=ROOT / "outputs/submission.csv")
    p.add_argument("--dataset", type=Path, default=ROOT / "response_release/dataset")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = p.parse_args()
    predict(args.data_dir, args.output, args.dataset, args.device)
