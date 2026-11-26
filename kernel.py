"""Kaggle entrypoint: infer on the actual competition test data."""
import json
import os
import sys
import tempfile
import zipfile
from pathlib import Path

import pandas as pd
import torch


def unique_file(root, name):
    hits = sorted(Path(root).rglob(name))
    if len(hits) != 1:
        raise FileNotFoundError(f"Expected one {name} under {root}, found {len(hits)}")
    return hits[0]


def main():
    with tempfile.TemporaryDirectory() as scratch:
        predict(Path(scratch))


def predict(scratch):
    input_root = Path(os.environ.get("KAGGLE_INPUT_DIR", "/kaggle/input"))
    output_root = Path(os.environ.get("KAGGLE_OUTPUT_DIR", "/kaggle/working"))
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = unique_file(input_root, "moa_manifest.json")
    bundle = manifest_path.parent
    sys.path.insert(0, str(bundle))
    from infer_engine import run_inference, blend_and_write

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    models_dir = bundle / "models"
    if not models_dir.exists():
        model_root = scratch
        archives = sorted(bundle.glob("weights_*.zip")) + sorted(bundle.glob("weights_*.moa"))
        for archive_path in archives or [bundle / "models.zip"]:
            with zipfile.ZipFile(archive_path) as archive:
                archive.extractall(model_root)
        models_dir = model_root / "models"
    test_path = unique_file(input_root, "test_features.csv")
    test = pd.read_csv(test_path)
    sub = pd.read_csv(test_path.parent / "sample_submission.csv")
    target_cols = [c for c in sub if c != "sig_id"]
    expected_cols = manifest["members"][0]["config"]["target_cols"]
    if target_cols != expected_cols:
        raise ValueError("Submission target columns differ from model output order")
    # Some reruns replace test_features but leave the public sample's row IDs.
    # Use the sample as a column schema, and always predict the actual test rows.
    if set(sub["sig_id"]) != set(test["sig_id"]):
        sub = pd.DataFrame(0.0, index=range(len(test)), columns=target_cols)
        sub.insert(0, "sig_id", test["sig_id"].to_numpy())
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    device = os.environ.get("KAGGLE_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Inference: {len(test)} rows, device={device}", flush=True)
    preds, treat = run_inference(test, models_dir, device, manifest)
    options = {}
    if manifest.get("blend_space") == "fold-logit":
        options["calibration"] = manifest["calibration"]
    if manifest.get("drug_library"):
        from drug_library import library_for_test
        options["library"] = library_for_test(test, treat, models_dir, len(target_cols), manifest["folds"])
    result = blend_and_write(test, preds, treat, sub,
                             {m["tag"]: m["weight"] for m in manifest["members"]},
                             output_root / "submission.csv", **options)
    print("submission.csv written:", result.shape, flush=True)


if __name__ == "__main__":
    main()
