"""Refit the validated response models at a CV-derived epoch with no test-label selection."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from artifact_utils import write_json
from config import BIO_COLS, DATA_DIR, MODELS_DIR, RUNS_DIR
from drug_library import fit_library, measurements, predict_library
from features import FoldFeatureBuilder
from models import build_model
from response_features import center_signatures, fit_response, transform_response
from response_study import STUDY, apply_blend, fit_blend, loss, read_study, safe_logit, study_rows
from trainer import auxiliary_warmup, profile_labels, set_seed, train_one_fold


FULL = MODELS_DIR / "response_full"
TAGS = ["response_cnn5", "response_profile5", "response_aux5"]


def checkpoint_epoch(tag, cfg, folds, omit=None):
    # BatchNorm increments once per training minibatch. The auxiliary epoch uses
    # the same batches, so subtract it before taking the median scored epoch.
    epochs = []
    for seed in cfg["seeds"]:
        for fold in range(5):
            if fold == omit:
                continue
            state = torch.load(MODELS_DIR / tag / f"seed{seed}/fold{fold}/model.pt",
                               map_location="cpu", weights_only=True)
            steps = next(v.item() for k, v in state.items() if k.endswith("num_batches_tracked"))
            batches = math.ceil(np.count_nonzero(folds != fold) / cfg["batch_size"])
            if steps % batches:
                raise ValueError("Checkpoint's BatchNorm counter does not match its split")
            epochs.append(steps // batches - int(cfg.get("aux_warmup", False)))
    chosen = int(np.rint(np.median(epochs)))
    if not 1 <= chosen <= cfg["epochs"]:
        raise ValueError("Invalid selected epoch")
    return chosen, epochs


def refit_features(kind, train, evaluate, output=None):
    a = train[BIO_COLS].to_numpy(dtype=np.float64)
    b = evaluate[BIO_COLS].to_numpy(dtype=np.float64)
    if kind == "relative":
        a, b = center_signatures(a), center_signatures(b)
        state = fit_response(a)
        if output:
            np.savez_compressed(output, **state)
        return transform_response(a, state), transform_response(b, state)
    builder = FoldFeatureBuilder()
    conditions = ["cp_time", "cp_dose"]
    builder.fit(a, train[conditions], train[conditions], evaluate[conditions], conditions)
    if output:
        builder.save(output)
    return builder.transform(a, train[conditions]), builder.transform(b, evaluate[conditions])


def run(mode, tags, seeds):
    ids, folds, y, targets, drugs = study_rows()
    y = y.astype(np.float32)
    train = pd.read_csv(DATA_DIR / "train_features.csv").set_index("sig_id").loc[ids].reset_index()
    aux_df = pd.read_csv(DATA_DIR / "train_targets_nonscored.csv").set_index("sig_id").loc[ids]
    aux = aux_df.to_numpy(dtype=np.float32)
    cfgs = {tag: json.loads((RUNS_DIR / tag / "config.json").read_bytes()) for tag in tags}
    full = mode == "full"
    configs = {}
    for tag, cfg in cfgs.items():
        epoch, observed = checkpoint_epoch(tag, cfg, folds, omit=None if full else 0)
        configs[tag] = {"stop_epoch": epoch, "cv_checkpoint_epochs": observed}
    if full:
        study = json.loads((STUDY / "selected_cv.json").read_bytes())
        audit = json.loads((STUDY / "refit_audit.json").read_bytes())
        if not study["passed"] or not audit["larger_training_set_improved"] or tags != study["neural_members"] or tags != audit["tags"]:
            raise ValueError("Full-data release requires matching successful validation and refit audit")
        if FULL.exists():
            raise FileExistsError(f"Existing full-data fit at {FULL}")
        FULL.mkdir(parents=True)
        evaluate = pd.read_csv(DATA_DIR / "test_features.csv")
        evaluate = evaluate.loc[evaluate.cp_type.eq("trt_cp")].reset_index(drop=True)
        splits = [("full", np.ones(len(y), dtype=bool), seeds)]
    else:
        evaluate = train.loc[folds == 0].reset_index(drop=True)
        audit_seeds = [13, 23, 37]
        splits = [("larger", folds != 0, audit_seeds)]
        splits += [("smaller", (folds != 0) & (folds != i), [seed])
                   for i, seed in enumerate(audit_seeds, 1)]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictions = {}
    for scope, tr, run_seeds in splits:
        features, owners = {}, {}
        xt = train.loc[tr]
        drug_state = fit_library(measurements(xt[BIO_COLS].to_numpy()), drugs[tr], y[tr])
        vote, confidence = predict_library(measurements(evaluate[BIO_COLS].to_numpy()), drug_state)
        bucket = predictions.setdefault(scope, {name: [] for name in [*tags, "vote", "confidence"]})
        bucket["vote"].append(vote)
        bucket["confidence"].append(confidence)
        if full:
            (FULL / "drug_library").mkdir()
            np.savez_compressed(FULL / "drug_library/fold0.npz", **drug_state)
        for tag in tags:
            cfg, recipe = cfgs[tag], configs[tag]
            kind = cfg["features"]
            if kind not in features:
                owners[kind] = tag
                dest = None
                if full:
                    dest = FULL / tag / "feat/global.npz"
                    dest.parent.mkdir(parents=True)
                features[kind] = refit_features(kind, xt, evaluate, dest)
            X, X_eval = features[kind]
            recipe["feature_run"] = owners[kind] if owners[kind] != tag else None
            profile, target = profile_labels(y[tr], aux[tr]) if cfg["kind"] == "profile_mlp" else (None, y[tr])
            pos = None
            if cfg.get("frequency_weight"):
                counts = y[tr].sum(0)
                pos = torch.from_numpy(np.log(100 + counts.min()) / np.log(100 + counts)).to(device)
            for seed in run_seeds:
                set_seed(seed)
                model = build_model(cfg["kind"], X.shape[1], len(targets), cfg["hidden"], cfg["dropout"], profile).to(device)
                if cfg.get("aux_warmup"):
                    choice = auxiliary_warmup(model, X, y[tr], aux[tr], cfg["batch_size"], seed, device, cfg["bf16"])
                    if full:
                        recipe["warmup_targets"] = aux_df.columns[choice].tolist()
                _, pred, _, state = train_one_fold(model, X, target, None, None, X_eval, cfg["epochs"],
                    cfg["batch_size"], cfg["lr"], cfg["weight_decay"], cfg["max_lr"], cfg["pct_start"],
                    cfg["smoothing"], seed, device, False, cfg["bf16"], pos, recipe["stop_epoch"])
                bucket[tag].append(pred)
                if full:
                    dest = FULL / tag / f"seed{seed}/fold0/model.pt"
                    dest.parent.mkdir(parents=True)
                    torch.save(state, dest)
                print(f"{scope} {tag} seed={seed} train={len(X)} epoch={recipe['stop_epoch']}/{cfg['epochs']}", flush=True)
        del features
    mean = {scope: {k: np.mean(v, axis=0, dtype=np.float64) for k,v in pred.items()}
            for scope,pred in predictions.items()}
    if full:
        write_json(STUDY / "refit.json", {"scope": "all_treated", "rows": len(y), "seeds": seeds,
                                          "tags": tags, "recipes": configs, "targets": targets})
        np.savez_compressed(STUDY / "refit_visible.npz", sig_id=evaluate.sig_id.to_numpy(dtype=str), **mean["full"])
    else:
        _, f, yy, _, _, logits, lib, c = read_study(tags)
        fit = fit_blend(logits[f != 0], lib[f != 0], c[f != 0], yy[f != 0])
        scores = {}
        for scope, pred in mean.items():
            z = np.stack([safe_logit(pred[t]) for t in tags], axis=-1)
            result = apply_blend(z, pred["vote"], pred["confidence"], fit)
            scores[scope] = {"combined": loss(yy[f == 0], result)}
            scores[scope].update({t: loss(yy[f == 0], pred[t]) for t in tags})
        audit = {"validation_fold": 0, "validation_rows": int((f == 0).sum()), "tags": tags,
                 "seeds": audit_seeds, "omitted_extra_fold_per_smaller_seed": [1, 2, 3], "scores": scores,
                 "larger_training_set_improved": scores["larger"]["combined"] < scores["smaller"]["combined"],
                 "epochs": {tag:r["stop_epoch"] for tag,r in configs.items()}}
        write_json(STUDY / "refit_audit.json", audit)
        print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["audit", "full"])
    p.add_argument("--tags", nargs="+", default=TAGS)
    p.add_argument("--seeds", nargs="+", type=int, default=list(range(8)))
    args = p.parse_args()
    run(args.mode, args.tags, args.seeds)
