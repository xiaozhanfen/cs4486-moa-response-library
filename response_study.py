"""Same-fold comparison with the scored Course5 model. No leaderboard scores enter fitting."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

from artifact_utils import safe_logit, write_json
from config import BIO_COLS, CLIP, DATA_DIR, FOLD_FILE, MODELS_DIR, ROOT, RUNS_DIR
from drug_library import fit_library, measurements, predict_library
from folds import read_treated_folds
from metrics import competition_log_loss as loss


STUDY = RUNS_DIR / "response_study"
LIBRARY = MODELS_DIR / "drug_library"
BASELINE = ROOT / "evidence/baseline_course5"
COURSE5_SCORES = {"submission": 57937726, "public": 0.01866, "private": 0.01629}


def study_rows():
    frame = pd.read_csv(DATA_DIR / "train_features.csv", usecols=["sig_id", "cp_type"])
    ids = frame.loc[frame.cp_type.eq("trt_cp"), "sig_id"].to_numpy(dtype=str)
    labels = pd.read_csv(DATA_DIR / "train_targets_scored.csv").set_index("sig_id")
    drugs = pd.read_csv(DATA_DIR / "train_drug.csv").set_index("sig_id").drug_id
    folds = read_treated_folds(FOLD_FILE, frame, drugs.to_dict(), 5)
    return ids, folds, labels.loc[ids].to_numpy(dtype=np.float64), list(labels), drugs.loc[ids].to_numpy()


def build_library():
    ids, folds, y, targets, drugs = study_rows()
    train = pd.read_csv(DATA_DIR / "train_features.csv").set_index("sig_id").loc[ids]
    x = measurements(train[BIO_COLS].to_numpy())
    out, confidence = np.empty_like(y, dtype=np.float32), np.empty(len(y), dtype=np.float32)
    LIBRARY.mkdir(parents=True, exist_ok=True)
    for f in range(5):
        tr, va = folds != f, folds == f
        state = fit_library(x[tr], drugs[tr], y[tr])
        out[va], confidence[va] = predict_library(x[va], state)
        np.savez_compressed(LIBRARY / f"fold{f}.npz", **state)
    STUDY.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(STUDY / "library_oof.npz", sig_id=ids, folds=folds, targets=np.asarray(targets),
                        probabilities=out, confidence=confidence)


def read_study(tags, compact_predictions=None):
    ids, folds, y, targets, drugs = study_rows()
    with np.load(BASELINE / "crossfit.npz") as saved:
        np.testing.assert_array_equal(saved["sig_id"], ids)
        np.testing.assert_array_equal(saved["folds"], folds)
        np.testing.assert_array_equal(saved["targets"], targets)
        baseline = saved["oof"].astype(float)
    def read_member(tag):
        run = RUNS_DIR / tag
        split = pd.read_csv(run / "folds.csv")
        np.testing.assert_array_equal(split.sig_id, ids)
        np.testing.assert_array_equal(split.fold, folds)
        if json.loads((run / "config.json").read_bytes())["target_cols"] != targets:
            raise ValueError(f"{tag}: target order mismatch")
        if compact_predictions is not None:
            return np.load(Path(compact_predictions) / f"{tag}.npy")
        return np.load(run / "oof.npy")
    logits = []
    logits.extend(safe_logit(read_member(t)) for t in tags)
    with np.load(STUDY / "library_oof.npz") as saved:
        np.testing.assert_array_equal(saved["sig_id"], ids)
        np.testing.assert_array_equal(saved["folds"], folds)
        np.testing.assert_array_equal(saved["targets"], targets)
        library, confidence = saved["probabilities"].astype(float), saved["confidence"].astype(float)
    return ids, folds, y, drugs, baseline, np.stack(logits, axis=-1), library, confidence


def fit_blend(logits, library, confidence, y, *, use_library=True):
    """Nonnegative neural logit coefficients, one intercept and two soft library weights."""
    n_heads = logits.shape[-1]
    x = logits.reshape(-1, n_heads)
    # The features are highly correlated. Center them to condition the intercept.
    center = x.mean(0)
    x = np.ascontiguousarray(x - center)
    c = confidence[:, None]
    def objective(theta):
        neural = expit(x @ theta[:n_heads] + theta[n_heads]).reshape(y.shape)
        gate = theta[-2] * (1 - c) + theta[-1] * c
        p = np.clip(neural + gate * (library - neural), CLIP, 1 - CLIP)
        ll = -np.where(y, np.log(p), np.log1p(-p)).mean()
        derivative = (p - y) / (p * (1 - p))
        dz = derivative * (1 - gate) * neural * (1 - neural)
        dg = derivative * (library - neural)
        return ll, np.r_[x.T @ dz.ravel() / y.size, dz.mean(), (dg * (1-c)).mean(), (dg * c).mean()]

    start = np.r_[np.full(n_heads, 1/n_heads), center.mean(), [0.1, 0.25] if use_library else [0., 0.]]
    library_limit = 0.6 if use_library else 0.
    fit = minimize(objective, start, method="L-BFGS-B", jac=True,
                   bounds=[(0, 3)] * n_heads + [(None, None), (0, library_limit), (0, library_limit)],
                   options={"ftol": 2e-12, "gtol": 2e-9, "maxiter": 160})
    if not fit.success or not np.isfinite(fit.x).all():
        raise RuntimeError(f"Blend fit failed: {fit.message}")
    w = fit.x[:n_heads]
    return {"coefficients": w.tolist(), "intercept": float(fit.x[n_heads] - center @ w),
            "library_low": float(fit.x[-2]), "library_high": float(fit.x[-1])}


def apply_blend(logits, library, confidence, fit):
    neural = expit(logits @ np.asarray(fit["coefficients"]) + fit["intercept"])
    gate = ((1 - confidence) * fit["library_low"] + confidence * fit["library_high"])[:, None]
    return np.clip(neural + gate * (library - neural), CLIP, 1 - CLIP)


def bootstrap_drugs(y, pred, baseline, drugs):
    def row_losses(p):
        p = np.clip(p, CLIP, 1 - CLIP)
        return -np.where(y, np.log(p), np.log1p(-p)).mean(1)
    _, ix, n = np.unique(drugs, return_inverse=True, return_counts=True)
    totals = np.bincount(ix, weights=row_losses(pred) - row_losses(baseline))
    rng = np.random.default_rng(4486)
    draws = rng.integers(len(n), size=(1200, len(n)))
    return np.quantile(totals[draws].sum(1) / n[draws].sum(1), [0.025, 0.975]).tolist()


def ablate_library(tags):
    _, folds, y, _, _, logits, lib, conf = read_study(tags)
    pred = np.empty_like(y)
    for fold in range(5):
        tr, va = folds != fold, folds == fold
        fit = fit_blend(logits[tr], lib[tr], conf[tr], y[tr], use_library=False)
        pred[va] = apply_blend(logits[va], lib[va], conf[va], fit)
    report = {"five_fold_neural_only": loss(y, pred),
              "uncalibrated_single_branches": {tag: loss(y, expit(logits[..., i])) for i,tag in enumerate(tags)}}
    write_json(STUDY/"ablation.json", report)
    print(json.dumps(report, indent=2), flush=True)


def assess(tags, name, compact_predictions=None):
    ids, folds, y, drugs, baseline, logits, lib, conf = read_study(tags, compact_predictions)
    outer, within = np.empty_like(y), np.empty_like(y)
    halves = np.array([hashlib.sha256(d.encode()).digest()[0] % 2 for d in drugs])
    course5 = json.loads((BASELINE / "course5_validation.json").read_bytes())
    prior_disjoint = {r["fold"]:r["within_fold_disjoint_drugs"] for r in course5["per_fold"]}
    per_fold = []
    def predict_split(tr, va):
        fit = fit_blend(logits[tr], lib[tr], conf[tr], y[tr])
        return apply_blend(logits[va], lib[va], conf[va], fit)
    for f in range(5):
        va = folds == f
        outer[va] = predict_split(~va, va)
        for side in (0, 1):
            hold = va & (halves == side)
            within[hold] = predict_split(va & ~hold, hold)
        result = {"fold": f, "course5": loss(y[va], baseline[va]), "outer": loss(y[va], outer[va]),
                  "course5_disjoint_drugs": prior_disjoint[f],
                  "disjoint_drugs": loss(y[va], within[va])}
        per_fold.append(result)
        print(result, flush=True)
    full = fit_blend(logits, lib, conf, y)
    small = pd.Series(drugs).map(pd.Series(drugs).value_counts()).to_numpy() <= 18
    n_total = len(pd.read_csv(DATA_DIR / "train_features.csv", usecols=["sig_id"]))
    treated = loss(y, outer)
    ci = bootstrap_drugs(y, outer, baseline, drugs)
    report = {"course5_official": COURSE5_SCORES, "neural_members": list(tags),
              "deployment": full, "course5_treated": loss(y, baseline), "treated": treated,
              "all_rows": treated * len(y) / n_total - np.log1p(-CLIP) * (1 - len(y)/n_total),
              "disjoint_drugs": loss(y, within),
              "course5_disjoint_drugs": course5["scores"]["calibration_within_fold_disjoint_drugs"]["treated"],
              "course5_small_drugs": loss(y[small], baseline[small]),
              "small_drugs": loss(y[small], outer[small]), "paired_drug_bootstrap_95pct": ci, "folds": per_fold}
    paired_folds_improved = all(r["outer"] < r["course5"] and r["disjoint_drugs"] < r["course5_disjoint_drugs"]
                                for r in per_fold)
    report["passed"] = paired_folds_improved and ci[1] < 0 and report["small_drugs"] < report["course5_small_drugs"]
    write_json(STUDY / f"{name}.json", report)
    np.savez_compressed(STUDY / f"{name}_oof.npz", sig_id=ids, folds=folds, probabilities=outer.astype(np.float32))
    print(json.dumps({k:v for k,v in report.items() if k != "folds"}, indent=2), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-library", action="store_true")
    parser.add_argument("--tags", nargs="+")
    parser.add_argument("--compact-predictions", type=Path)
    parser.add_argument("--name", default="validation")
    parser.add_argument("--ablate-library", action="store_true")
    args = parser.parse_args()
    if args.prepare_library:
        build_library()
    if args.tags and args.ablate_library:
        ablate_library(args.tags)
    elif args.tags:
        assess(args.tags, args.name, args.compact_predictions)
