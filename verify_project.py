"""Verify portable source, immutable Kaggle bundle and real/changing input rows."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(variable, "4")

import nbformat
import numpy as np
import pandas as pd

from artifact_utils import safe_logit, write_json
from config import DATA_DIR, ROOT, RUNS_DIR
from predict import predict, workspace
from response_study import apply_blend


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate(result, test, template):
    targets = list(template.drop(columns="sig_id"))
    order = template.sig_id if set(template.sig_id) == set(test.sig_id) else test.sig_id
    np.testing.assert_array_equal(result.sig_id, order)
    assert list(result) == ["sig_id", *targets] and result.sig_id.is_unique
    values = result[targets].to_numpy(dtype=float)
    assert np.isfinite(values).all() and ((values >= 0) & (values <= 1)).all()
    controls = test.loc[test.cp_type.eq("ctl_vehicle"), "sig_id"]
    assert not result.set_index("sig_id").loc[controls].to_numpy().any()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--release-dir", type=Path, default=ROOT / "response_release")
    p.add_argument("--reference", choices=["official", "fp32"], default="official")
    p.add_argument("--skip-notebook", action="store_true")
    args = p.parse_args()
    release = args.release_dir.resolve()
    dataset = release / "dataset"
    original = json.loads((ROOT / "evidence/scored_release_check.json").read_bytes())
    immutable = release == (ROOT / "response_release").resolve()
    if immutable:
        for name, expected in original["sha256"].items():
            if sha256(release/name) != expected:
                raise ValueError(f"Scored V1 artifact changed: {name}")
    if args.reference == "official":
        # Reference is a saved Kaggle output, not a label file or an inference input.
        expected = pd.read_csv(ROOT / "evidence/kaggle_v1/submission.csv").set_index("sig_id")
    else:
        report = json.loads((RUNS_DIR/"response_study/selected_cv.json").read_bytes())
        with np.load(RUNS_DIR/"response_study/refit_visible.npz") as saved:
            z = np.stack([safe_logit(saved[t]) for t in report["neural_members"]], axis=-1)
            values = apply_blend(z, saved["vote"], saved["confidence"], report["deployment"])
            expected = pd.DataFrame(values, index=saved["sig_id"], columns=report_targets(dataset))
    test = pd.read_csv(DATA_DIR / "test_features.csv")
    template = pd.read_csv(DATA_DIR / "sample_submission.csv")
    if args.reference == "fp32":
        expected = expected.reindex(test.sig_id).fillna(0.)
    receipt = {"reference": args.reference, "immutable_v1_hashes_passed": immutable, "passed": []}
    started = time.perf_counter()
    with workspace() as root:
        data = root / "data"
        data.mkdir()

        def run(frame, sample):
            frame.to_csv(data/"test_features.csv", index=False)
            sample.to_csv(data/"sample_submission.csv", index=False)
            result = predict(data, root/"source.csv", dataset, "cpu")
            validate(result, frame, sample)
            return result.set_index("sig_id")

        actual = run(test, template)
        difference = np.abs(actual.to_numpy() - expected.loc[actual.index].to_numpy())
        if args.reference == "official":
            np.testing.assert_allclose(actual, expected.loc[actual.index], atol=2e-6, rtol=2e-5)
        elif difference.mean() > 5e-5 or difference.max() > .02:
            raise ValueError("Quantization drift from FP32 reference exceeds accepted tolerance")
        receipt.update(rows=len(actual), targets=len(actual.columns), max_abs_difference=float(difference.max()),
                       mean_abs_difference=float(difference.mean()))
        receipt["passed"].append("full_source_inference")

        if not args.skip_notebook:
            # Execute the preserved notebook's predict function in a fresh process.
            # Using an inherited-ACL workspace avoids Windows tempfile ACL issues.
            notebook = nbformat.read(release/"notebook/kernel.ipynb", as_version=4)
            nbformat.validate(notebook)
            runner = root / "notebook_runner.py"
            runner.write_text("import json,sys\nfrom pathlib import Path\n"
                "nb=json.loads(Path(sys.argv[1]).read_bytes())\n"
                "scope={'__name__':'verified_notebook'}\n"
                "exec(compile(''.join(nb['cells'][0]['source']),sys.argv[1],'exec'),scope)\n"
                "scope['predict'](Path(sys.argv[2]))\n", encoding="utf-8")
            inputs, out = root/"input", root/"working"
            (inputs/"weights").mkdir(parents=True)
            (inputs/"competition").mkdir()
            for path in dataset.iterdir():
                if path.is_file():
                    try:
                        os.link(path, inputs/"weights"/path.name)
                    except OSError:
                        import shutil
                        shutil.copyfile(path, inputs/"weights"/path.name)
            test.to_csv(inputs/"competition/test_features.csv", index=False)
            template.to_csv(inputs/"competition/sample_submission.csv", index=False)
            scratch = root/"notebook_scratch"
            scratch.mkdir()
            env = dict(os.environ, KAGGLE_INPUT_DIR=str(inputs), KAGGLE_OUTPUT_DIR=str(out),
                       KAGGLE_DEVICE="cpu", PYTHONDONTWRITEBYTECODE="1")
            env.pop("PYTHONPATH", None)
            subprocess.run([sys.executable, str(runner), str(release/"notebook/kernel.ipynb"), str(scratch)],
                           cwd=root, env=env, check=True)
            notebook_result = pd.read_csv(out/"submission.csv")
            validate(notebook_result, test, template)
            np.testing.assert_allclose(notebook_result.set_index("sig_id"), actual, atol=1e-7, rtol=1e-6)
            receipt["passed"].append("frozen_notebook_matches_source")

        tiny = pd.concat([test.loc[test.cp_type.eq("trt_cp") & test.cp_time.eq(24) & test.cp_dose.eq("D1")].head(3),
                          test.loc[test.cp_type.eq("ctl_vehicle")].head(2)]).copy()
        reference = actual.loc[tiny.sig_id].copy()
        tiny["sig_id"] = [f"new_group_test_{i}" for i in range(len(tiny))]
        reference.index = tiny.sig_id
        reversed_sample = template.head(len(tiny)).copy()
        reversed_sample["sig_id"] = tiny.sig_id.to_numpy()[::-1]
        for sample in [reversed_sample, template]:
            changed = run(tiny, sample)
            np.testing.assert_allclose(changed, reference.loc[changed.index], atol=3e-6, rtol=2e-5)
        receipt["passed"] += ["new_ids", "missing_condition_categories", "reversed_template", "stale_template"]
        run(tiny.tail(2), template)
        run(tiny.head(0), template)
        receipt["passed"] += ["all_controls", "empty_input"]

    receipt.update(seconds=time.perf_counter()-started, python=sys.version.split()[0],
        source_sha256={p.name:sha256(p) for p in ROOT.glob("*.py")},
        release_sha256={p.relative_to(release).as_posix():sha256(p)
                        for folder in [dataset,release/"notebook"] for p in folder.iterdir() if p.is_file()})
    destination = ROOT / "evidence/final_verification.json" if immutable else release / "verification.json"
    write_json(destination, receipt)
    print(json.dumps({k:v for k,v in receipt.items() if not k.endswith("sha256")}, indent=2))


def report_targets(dataset):
    return json.loads((dataset/"moa_manifest.json").read_bytes())["members"][0]["config"]["target_cols"]


if __name__ == "__main__":
    main()
