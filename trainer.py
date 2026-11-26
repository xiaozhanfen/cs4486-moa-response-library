"""Final five-fold trainer. Run through reproduce.py for the accepted recipes."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from config import CLIP, CV_SEED, DATA_DIR, FOLD_FILE, MODELS_DIR, N_FOLDS, RUNS_DIR
from features import FoldFeatureBuilder, extract_bio_matrix
from folds import read_treated_folds
from metrics import competition_log_loss
from models import SmoothBCEwLogits, build_model


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



def load_raw(data_dir: Path):
    train = pd.read_csv(data_dir / "train_features.csv")
    test = pd.read_csv(data_dir / "test_features.csv")
    scored = pd.read_csv(data_dir / "train_targets_scored.csv")
    nonscored = pd.read_csv(data_dir / "train_targets_nonscored.csv")
    submission = pd.read_csv(data_dir / "sample_submission.csv")
    drug = None
    if (data_dir / "train_drug.csv").exists():
        drug = pd.read_csv(data_dir / "train_drug.csv")
    return train, test, scored, nonscored, submission, drug



def align_by_sig_id(feat: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """按 sig_id 对齐标签，防止文件重排导致错位。"""
    ids, indexed = pd.Index(feat["sig_id"]), labels.set_index("sig_id")
    if ids.hasnans or indexed.index.hasnans or not ids.is_unique or not indexed.index.is_unique:
        raise ValueError("Training feature and label IDs must be unique and present")
    if len(ids) != len(indexed) or not ids.isin(indexed.index).all():
        raise ValueError("Training feature and label IDs differ")
    return indexed.reindex(ids).reset_index()



def predict_proba(model, X: np.ndarray, device, batch_size: int = 2048) -> np.ndarray:
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.from_numpy(X[i:i + batch_size]).to(device)
            outs.append(torch.sigmoid(model(xb)).cpu().numpy())
    return np.concatenate(outs)



def auxiliary_warmup(model, x, scored, auxiliary, batch_size, seed, device, bf16):
    """Select auxiliary labels using only this fold's train rows and warm up its encoder."""
    def normalized_columns(matrix):
        centered = matrix.astype(np.float64) - matrix.mean(axis=0, dtype=np.float64)
        return centered / np.maximum(np.linalg.norm(centered, axis=0), 1e-12)

    correlations = np.abs(normalized_columns(auxiliary).T @ normalized_columns(scored)).max(axis=1)
    selected = np.argsort(-correlations, kind="stable")[:min(33, np.count_nonzero(correlations))]
    if not len(selected):
        raise ValueError("No variable auxiliary targets in this fold")
    scored_head = model.head[-1]
    model.head[-1] = torch.nn.Linear(scored_head.in_features, len(selected)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    loader = DataLoader(TensorDataset(torch.from_numpy(x), torch.from_numpy(auxiliary[:, selected].copy())),
                        batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed))
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, total_steps=len(loader),
                                                    pct_start=0.1, div_factor=1000)
    model.train()
    for xb, yb in loader:
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=bf16):
            loss = torch.nn.functional.binary_cross_entropy_with_logits(model(xb.to(device)).float(), yb.to(device))
        loss.backward()
        optimizer.step()
        schedule.step()
    model.head[-1] = scored_head
    return selected.tolist()



def profile_labels(scored, aux):
    annotation = np.column_stack([scored, aux])
    if not np.isin(annotation, [0, 1]).all():
        raise ValueError("Profile supervision requires binary labels")
    codes = np.packbits(annotation.astype(np.uint8), axis=1)
    _, first, indices = np.unique(codes, axis=0, return_index=True, return_inverse=True)
    return scored[first], np.column_stack([scored, indices]).astype(np.float32)



def train_one_fold(model, X_tr, y_tr, X_val, y_val, X_test, epochs, batch_size, lr, wd,
                   max_lr, pct_start, smoothing, seed, device, verbose, bf16=False, pos_weight=None,
                   stop_epoch=None) -> tuple:
    if stop_epoch is not None and not 1 <= stop_epoch <= epochs:
        raise ValueError("Stop epoch must be inside the scheduled training cycle")
    if (X_val is None) != (y_val is None):
        raise ValueError("Supply both validation features and labels, or neither for a refit")
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr.astype(np.float32))),
        batch_size=batch_size, shuffle=True, num_workers=0,
        generator=torch.Generator().manual_seed(seed),
    )
    total_steps = epochs * len(loader)
    sched = None
    if max_lr > 0:
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=max_lr, pct_start=pct_start, div_factor=1e3,
            total_steps=total_steps,
        )
    criterion = SmoothBCEwLogits(smoothing, pos_weight) if smoothing > 0 else torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val, best_state, best_preds = np.inf, None, None
    for _ in range(epochs if stop_epoch is None else stop_epoch):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=bf16):
                if hasattr(model, "training_loss"):
                    loss = model.training_loss(xb, yb, criterion)
                else:
                    loss = criterion(model(xb).float(), yb)
            loss.backward()
            opt.step()
            if sched is not None:
                sched.step()
        if X_val is not None:
            val_pred = predict_proba(model, X_val, device)
            val_ll = competition_log_loss(y_val, val_pred)
            if val_ll < best_val:
                best_val, best_preds = val_ll, val_pred
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if X_val is None:
        best_val = None
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    test_pred = predict_proba(model, X_test, device)
    if verbose and best_val is not None:
        print(f"      fold done, val log loss = {best_val:.6f}", flush=True)
    return best_preds, test_pred, best_val, best_state


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tag", required=True)
    p.add_argument("--kind", choices=["response_cnn", "profile_mlp"], required=True)
    p.add_argument("--features", choices=["original", "relative"], required=True)
    p.add_argument("--data-dir", type=Path, default=DATA_DIR)
    p.add_argument("--fold-file", type=Path, default=FOLD_FILE)
    p.add_argument("--feature-run", default=None)
    p.add_argument("--seeds", default="0,1,2,3,4")
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--hidden", type=int, default=4096)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--max-lr", type=float, default=0.01)
    p.add_argument("--pct-start", type=float, default=0.1)
    p.add_argument("--smoothing", type=float, default=0.001)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--aux-warmup", action="store_true")
    p.add_argument("--frequency-weight", action="store_true")
    args = p.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.bf16 and (device.type != "cuda" or not torch.cuda.is_bf16_supported()):
        raise ValueError("BF16 training needs a compatible CUDA device")
    if args.aux_warmup and args.kind != "response_cnn":
        raise ValueError("Auxiliary warmup requires ResponseCNN")
    out_dir, model_dir = RUNS_DIR / args.tag, MODELS_DIR / args.tag
    if out_dir.exists() or model_dir.exists():
        raise FileExistsError("Preserving existing artifacts; choose fresh MOA_RUNS_DIR/MOA_MODELS_DIR")
    out_dir.mkdir(parents=True)
    model_dir.mkdir(parents=True)

    train, test, scored, nonscored, _, drug = load_raw(args.data_dir)
    target_df = align_by_sig_id(train, scored)
    target_cols = list(target_df.drop(columns="sig_id"))
    treat = train.cp_type.eq("trt_cp").to_numpy()
    test_treat = test.cp_type.eq("trt_cp").to_numpy()
    train_t, test_t = train.loc[treat].reset_index(drop=True), test.reset_index(drop=True)
    y = target_df[target_cols].to_numpy(np.float32)[treat]
    sig_ids = train_t.sig_id.to_numpy()
    drug_map = dict(zip(drug.sig_id, drug.drug_id)) if drug is not None else None
    folds = read_treated_folds(args.fold_file, train, drug_map, N_FOLDS)
    pd.DataFrame({"sig_id": sig_ids, "fold": folds}).to_csv(out_dir / "folds.csv", index=False)
    aux_frame = align_by_sig_id(train, nonscored).drop(columns="sig_id")
    aux = aux_frame.to_numpy(dtype=np.float32)[treat]
    profiles = {}
    if args.kind == "profile_mlp":
        for fold in range(N_FOLDS):
            tr = folds != fold
            profiles[fold] = profile_labels(y[tr], aux[tr])

    x = extract_bio_matrix(train_t)
    xt = extract_bio_matrix(test_t)
    cond = ["cp_time", "cp_dose"]
    feat_cache = {}
    if args.feature_run:
        from config import ROOT
        from infer_engine import build_features
        source = RUNS_DIR / args.feature_run
        previous = json.loads((source / "config.json").read_bytes())
        np.testing.assert_array_equal(read_treated_folds(source / "folds.csv", train, drug_map, N_FOLDS), folds)
        previous_data = Path(previous["data_dir"])
        if not previous_data.is_absolute():
            previous_data = ROOT / previous_data
        if (previous.get("feature_run") or previous.get("global_fit") or previous.get("fa_pool")
                or previous["features"] != args.features or previous_data.resolve() != args.data_dir.resolve()):
            raise ValueError("Shared transforms must use identical training data, folds and feature recipe")
        for fold in range(N_FOLDS):
            with np.load(MODELS_DIR / args.feature_run / "feat" / f"fold{fold}.npz") as packed:
                state = dict(packed)
            full = build_features(train_t, state)
            feat_cache[fold] = full[folds != fold], full, build_features(test_t, state)
    elif args.features == "relative":
        from response_features import center_signatures, fit_response, transform_response
        centered, centered_test = center_signatures(x), center_signatures(xt)
        feat_dir = model_dir / "feat"
        feat_dir.mkdir()
        for fold in range(N_FOLDS):
            tr = folds != fold
            state = fit_response(centered[tr])
            full = transform_response(centered, state)
            feat_cache[fold] = full[tr], full, transform_response(centered_test, state)
            np.savez_compressed(feat_dir / f"fold{fold}.npz", **state)
    else:
        feat_dir = model_dir / "feat"
        feat_dir.mkdir()
        for fold in range(N_FOLDS):
            tr = np.flatnonzero(folds != fold)
            builder = FoldFeatureBuilder()
            builder.fit(x[tr], train_t[cond].iloc[tr], train_t[cond], test_t[cond], cond)
            feat_cache[fold] = (builder.transform(x[tr], train_t[cond].iloc[tr]),
                                builder.transform(x, train_t[cond]), builder.transform(xt, test_t[cond]))
            builder.save(feat_dir / f"fold{fold}.npz")

    in_dim, out_dim = feat_cache[0][0].shape[1], len(target_cols)
    oof = np.zeros((len(train_t), out_dim), dtype=np.float64)
    test_sum = np.zeros((len(test_t), out_dim), dtype=np.float64)
    scores, auxiliary_targets = {}, {}
    print(f"[{args.tag}] device={device} seeds={seeds} input={in_dim} output={out_dim}", flush=True)
    for seed in seeds:
        # Deliberately seed once before the fold loop, matching the accepted run.
        set_seed(seed)
        oof_s, test_s, fold_scores = np.zeros_like(oof), np.zeros((test_treat.sum(), out_dim)), []
        for fold in range(N_FOLDS):
            tr, va = folds != fold, folds == fold
            xtr, full, xte = feat_cache[fold]
            profile, labels = profiles.get(fold, (None, y[tr]))
            model = build_model(args.kind, in_dim, out_dim, args.hidden, args.dropout, profile).to(device)
            if args.aux_warmup:
                chosen = auxiliary_warmup(model, xtr, y[tr], aux[tr], args.batch_size, seed, device, args.bf16)
                auxiliary_targets[fold] = [aux_frame.columns[i] for i in chosen]
            pos = None
            if args.frequency_weight:
                counts = y[tr].sum(axis=0)
                pos = torch.from_numpy(np.log(100 + counts.min()) / np.log(100 + counts)).to(device)
            val_pred, test_pred, best_val, state = train_one_fold(
                model, xtr, labels, full[va], y[va], xte[test_treat], args.epochs, args.batch_size,
                args.lr, args.weight_decay, args.max_lr, args.pct_start, args.smoothing,
                seed, device, True, args.bf16, pos)
            oof_s[va] = np.clip(val_pred, CLIP, 1 - CLIP)
            test_s += np.clip(test_pred, CLIP, 1 - CLIP) / N_FOLDS
            fold_scores.append(best_val)
            sd = model_dir / f"seed{seed}/fold{fold}"
            sd.mkdir(parents=True)
            torch.save(state, sd / "model.pt")
        oof_s = np.clip(oof_s, CLIP, 1 - CLIP)
        oof += oof_s / len(seeds)
        test_full = np.zeros_like(test_sum)
        test_full[test_treat] = np.clip(test_s, CLIP, 1 - CLIP)
        test_sum += test_full / len(seeds)
        scores[seed] = competition_log_loss(y, oof_s)
        np.save(out_dir / f"oof_seed{seed}.npy", oof_s.astype(np.float32))
        np.save(out_dir / f"test_seed{seed}.npy", test_full.astype(np.float32))
        print(f"seed={seed} CV={scores[seed]:.9f} folds={fold_scores}", flush=True)
    oof, test_sum = np.clip(oof, CLIP, 1-CLIP), np.clip(test_sum, CLIP, 1-CLIP)
    test_sum[~test_treat] = 0
    np.save(out_dir / "oof.npy", oof.astype(np.float32))
    np.save(out_dir / "test.npy", test_sum.astype(np.float32))
    cfg = vars(args).copy()
    cfg.pop("tag")
    cfg.update(seeds=seeds, folds=N_FOLDS, cv_seed=CV_SEED, targets="scored", fa=1, stats=1,
               global_fit=0, fa_pool=0, in_dim=in_dim, out_dim=out_dim, target_cols=target_cols,
               n_treat=len(y), n_test=len(test_t), test_treat_mask_sum=int(test_treat.sum()),
               cv_ensemble=competition_log_loss(y, oof), cv_per_seed=scores)
    if auxiliary_targets:
        cfg["auxiliary_targets_by_fold"] = auxiliary_targets
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False, default=str)+"\n", encoding="utf-8")
    print(f"[{args.tag}] ENSEMBLE CV={cfg['cv_ensemble']:.9f}", flush=True)


if __name__ == "__main__":
    main()

