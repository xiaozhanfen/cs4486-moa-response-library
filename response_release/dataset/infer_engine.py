"""真推理引擎（本地验证 + Kaggle 内核共用）。

从保存的模型权重与预处理参数重建每个成员的测试集预测，
流程与 trainer.py 训练时完全一致，输出与 runs/<tag>/test.npy 对齐（本地验证 max diff < 1e-6）。

成员（全部为非零融合权重的简化集合）:
  simple      (5 种子 x 5 折, 折内 fit, 每折一个预处理参数包)
  simple_g    (6 种子 x 5 折, 全局 fit, 1 个预处理参数包)
  simple10    (6 种子 x 10 折, 全局 fit)
  simple10_x  (4 种子 x 10 折, 全局 fit)
  simple_gx   (5 种子 x 5 折, 全局 fit)
  mlp3        (2 种子 x 5 折, 全局 fit, 3 隐藏层)
"""
import hashlib
import json
import os
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import ndtri
from scipy.stats import kurtosis, skew

from models import SimpleNN, SimpleNN3

BOUNDS_THRESHOLD = 1e-7
CLIP = 1e-15
GENE_COLS = [f"g-{i}" for i in range(772)]
CELL_COLS = [f"c-{i}" for i in range(100)]
N_G, N_C = 772, 100
FA_G, FA_C = 90, 50


def load_fold_state(folder, storage="fp32"):
    import torch

    if storage == "fp32":
        return torch.load(folder / "model.pt", map_location="cpu", weights_only=True)
    if storage != "row-int8":
        raise ValueError(f"Unsupported checkpoint storage: {storage}")
    with np.load(folder / "model.npz", allow_pickle=False) as archive:
        state = {}
        for name in archive.files:
            if name.endswith(".scale"):
                continue
            values = archive[name]
            if values.dtype == np.int8:
                values = values.astype(np.float32) * archive[name + ".scale"]
            state[name] = torch.from_numpy(values)
    return state


def predict_attached_cnn(test, member, device, *, by_fold=False, input_root=None):
    """Reuse the team's previously uploaded, hash-identified CNN and its own predictor."""
    input_root = Path(input_root or os.environ.get("KAGGLE_INPUT_DIR", "/kaggle/input"))
    relative = "models/member_0/manifest.json"
    expected = member["manifest_sha256"]
    with tempfile.TemporaryDirectory() as scratch:
        roots = [p.parents[2] for p in input_root.rglob("member_0/manifest.json")
                 if hashlib.sha256(p.read_bytes()).hexdigest() == expected
                 and p.parent.parent.name == "models"]
        if not roots:
            for path in input_root.rglob("moa_inference_bundle.zip"):
                with zipfile.ZipFile(path) as archive:
                    if relative not in archive.namelist():
                        continue
                    if hashlib.sha256(archive.read(relative)).hexdigest() != expected:
                        continue
                    selected = [n for n in archive.namelist()
                                if n.startswith(("src/", "models/member_0/"))]
                    archive.extractall(scratch, selected)
                    roots.append(Path(scratch))
        if len(roots) != 1:
            raise ValueError("Attach exactly one matching cs4486-moa-candidate-v6 dataset")
        sys.path.insert(0, str(roots[0]))
        try:
            from src.predict import predict_artifact
            origin = roots[0] / "models/member_0"
            if by_fold:
                # The original verified predictor accepts a one-fold manifest.
                # Point it at each original fold without copying or editing weights.
                saved = json.loads((origin / "manifest.json").read_bytes())
                single = Path(scratch) / "one_fold"
                single.mkdir()
                predictions = []
                folds = sorted(saved["folds"], key=lambda f: f["fold"])
                if [f["fold"] for f in folds] != list(range(member["config"]["folds"])):
                    raise ValueError("CNN and MLP fold numbers differ")
                for fold in folds:
                    source = dict(fold, path=str((origin / fold["path"]).resolve()))
                    one = dict(saved, n_folds=1, folds=[source])
                    (single / "manifest.json").write_text(json.dumps(one), encoding="utf-8")
                    pred, targets = predict_artifact(single, test, device)
                    predictions.append(pred)
                pred = np.stack(predictions)
            else:
                pred, targets = predict_artifact(origin, test, device)
        finally:
            sys.path.pop(0)
        if targets != member["config"]["target_cols"]:
            raise ValueError("Attached CNN target order differs from the MLP")
        return pred

def manual_qt(X: np.ndarray, quantiles: np.ndarray, refs: np.ndarray) -> np.ndarray:
    """逐列复现 sklearn QuantileTransformer(normal) 的 transform（与训练一致）。"""
    clip_min = ndtri(BOUNDS_THRESHOLD - np.spacing(1))
    clip_max = ndtri(1 - (BOUNDS_THRESHOLD - np.spacing(1)))
    out = np.empty_like(X, dtype=np.float64)
    for j in range(X.shape[1]):
        q = quantiles[:, j]
        x = X[:, j]
        lower = x - BOUNDS_THRESHOLD < q[0]
        upper = x + BOUNDS_THRESHOLD > q[-1]
        v = 0.5 * (np.interp(x, q, refs) - np.interp(-x, -q[::-1], -refs[::-1]))
        v[upper] = 1.0
        v[lower] = 0.0
        out[:, j] = np.clip(ndtri(v), clip_min, clip_max)
    return out


def manual_fa_transform(X: np.ndarray, mean: np.ndarray, comp: np.ndarray,
                        noise: np.ndarray) -> np.ndarray:
    """复现 sklearn FactorAnalysis.transform 的 EM 后验投影（含噪声方差）。"""
    # Same posterior as sklearn, without inverting a 772 x 772 diagonal matrix.
    scaled = comp / noise
    cov_z = np.linalg.inv(np.eye(comp.shape[0]) + scaled @ comp.T)
    return (X - mean) @ scaled.T @ cov_z


def build_features(test: pd.DataFrame, feat: dict) -> np.ndarray:
    """按保存的参数重建 1032 维特征。feat 来自 models/<tag>/feat/<...>.npz。"""
    if str(feat.get("feature_kind", "")) == "relative_pca":
        from response_features import center_signatures, transform_response
        return transform_response(center_signatures(test[GENE_COLS + CELL_COLS].to_numpy()), feat)
    use_fa = bool(feat["use_fa"])
    use_stats = bool(feat["use_stats"])
    cond_cols = [str(c) for c in feat["cond_cols"]]
    X = test[GENE_COLS + CELL_COLS].to_numpy(dtype=np.float64)
    X1 = manual_qt(X, feat["qt1_quantiles"].astype(np.float64),
                   feat["qt1_references"].astype(np.float64))
    parts = [X1]
    if use_fa:
        parts += [
            manual_fa_transform(X1[:, :N_G], feat["fa_g_mean"], feat["fa_g_comp"],
                                feat["fa_g_noise"].astype(np.float64)),
            manual_fa_transform(X1[:, N_G:N_G + N_C], feat["fa_c_mean"], feat["fa_c_comp"],
                                feat["fa_c_noise"].astype(np.float64)),
        ]
    if use_stats:
        def _stats(x):
            return np.nan_to_num(np.vstack(
                [x.sum(1), x.mean(1), x.std(1), kurtosis(x, axis=1), skew(x, axis=1)]).T,
                nan=0.0, posinf=0.0, neginf=0.0)
        g, c = X1[:, :N_G], X1[:, N_G:N_G + N_C]
        gc = np.hstack([g, c])
        parts += [_stats(g), _stats(c), _stats(gc)]
    # Hidden batches may omit categories; retain the training feature layout.
    categories = {"cp_time": [24, 48, 72], "cp_dose": ["D1", "D2"]}
    blocks = []
    for col in cond_cols:
        values = list(feat.get(f"categories_{col}", categories[col]))
        if not test[col].isin(values).all():
            raise ValueError(f"Unknown {col} category")
        blocks.append(np.column_stack([(test[col] == v).to_numpy(dtype=float) for v in values]))
    onehot = np.hstack(blocks)
    parts.append(onehot)
    full = np.hstack(parts)
    return manual_qt(full, feat["qt2_quantiles"].astype(np.float64),
                     feat["qt2_references"].astype(np.float64)).astype(np.float32)


def predict_member(tag: str, arch: str, hidden: int, test: pd.DataFrame,
                   models_dir: Path, device: str = "cpu", treat_mask=None,
                   config=None, batch_size=2048, only_fold=None) -> np.ndarray:
    """返回该成员在 test 全部行上的预测（对照组 0）。

    逐折加载模型并立即释放，避免大权重文件反复加载时的内存压力；
    输出与训练时保存的 test.npy 对齐（本地验证 max diff < 1e-6）。
    """
    import gc

    import torch
    from models import build_model

    if config is None:
        import json
        config = json.loads((models_dir.parent / "runs" / tag / "config.json").read_text(encoding="utf-8"))
    feat_dir = models_dir / (config.get("feature_run") or tag) / "feat"
    global_npz = feat_dir / "global.npz"
    has_global = global_npz.exists()
    seed_dirs = [models_dir / tag / f"seed{s}" for s in config["seeds"]]
    n_seeds = len(seed_dirs)
    n_folds = config.get("folds", 5)
    fold_numbers = range(n_folds) if only_fold is None else (only_fold,)
    if only_fold is not None and only_fold not in range(n_folds):
        raise ValueError(f"Unknown fold {only_fold}")
    n_targets = config["out_dim"]
    mask = np.ones(len(test), dtype=bool) if treat_mask is None else np.asarray(treat_mask, dtype=bool)
    selected = test.loc[mask]
    pred_sum = np.zeros((len(test), n_targets), dtype=np.float64)
    if not len(selected):
        return pred_sum
    features = {}
    for sd in seed_dirs:
        seed_pred = np.zeros((len(selected), n_targets), dtype=np.float64)
        for fold in fold_numbers:
            fd = sd / f"fold{fold}"
            fp = global_npz if has_global else feat_dir / f"fold{fold}.npz"
            if fp not in features:
                with np.load(fp, allow_pickle=False) as feat:
                    features[fp] = build_features(selected, dict(feat))
            Xf = features[fp]
            if Xf.shape[1] != config["in_dim"]:
                raise ValueError(f"{tag}: feature width mismatch")
            state = load_fold_state(fd, config.get("storage", "fp32"))
            model = build_model(arch, Xf.shape[1], n_targets, hidden, config.get("dropout", 0.25),
                                state.get("profile_targets"))
            model.load_state_dict(state)
            model.to(device).eval()
            with torch.inference_mode():
                for start in range(0, len(Xf), batch_size):
                    xb = torch.from_numpy(Xf[start:start + batch_size]).to(device)
                    out = torch.sigmoid(model(xb)).cpu().numpy().astype(np.float64)
                    seed_pred[start:start + len(out)] += np.clip(out, CLIP, 1 - CLIP) / len(fold_numbers)
            del model
            gc.collect()
        pred_sum[mask] += np.clip(seed_pred, CLIP, 1 - CLIP) / n_seeds
    pred_sum[mask] = np.clip(pred_sum[mask], CLIP, 1 - CLIP)
    return pred_sum


def run_inference(test: pd.DataFrame, models_dir: Path, device: str = "cpu", manifest=None):
    if not test["cp_type"].isin(["trt_cp", "ctl_vehicle"]).all():
        raise ValueError("Unknown or missing cp_type")
    treat = (test["cp_type"] == "trt_cp").to_numpy()
    if manifest is None:
        import json
        manifest = json.loads((models_dir.parent / "moa_manifest.json").read_text(encoding="utf-8"))
    preds = {}
    by_fold = manifest.get("blend_space") == "fold-logit"
    for member in manifest["members"]:
        tag, cfg = member["tag"], member["config"]
        print(f"Predicting {tag}", flush=True)
        if cfg["kind"] == "projected_cnn":
            preds[tag] = predict_attached_cnn(test, member, device, by_fold=by_fold)
        elif by_fold:
            preds[tag] = np.stack([
                predict_member(tag, cfg["kind"], cfg["hidden"], test, models_dir,
                               device, treat, config=cfg, only_fold=fold)
                for fold in range(cfg["folds"])
            ])
        else:
            preds[tag] = predict_member(tag, cfg["kind"], cfg["hidden"], test,
                                        models_dir, device, treat, config=cfg)
    return preds, treat


def blend_and_write(test: pd.DataFrame, preds: dict, treat, sub_template: pd.DataFrame,
                    weights: dict, out_path: str, calibration=None, library=None) -> pd.DataFrame:
    from scipy.special import expit, logit
    out = np.zeros_like(next(iter(preds.values())), dtype=np.float64)
    if not np.isclose(sum(weights.values()), 1):
        raise ValueError("Blend weights must sum to one")
    for tag, w in weights.items():
        if w > 0:
            out += w * logit(np.clip(preds[tag], CLIP, 1 - CLIP))
    if calibration is not None:
        if out.ndim != 3:
            raise ValueError("Fold calibration needs fold-by-sample-by-label logits")
        out = expit(calibration["slope"] * out + calibration["intercept"])
        if library is not None:
            vote, match = library
            if vote.shape != out.shape or match.shape != out.shape[:2]:
                raise ValueError("Drug-library fold or row order differs")
            gate = ((1 - match) * calibration["library_low"] + match * calibration["library_high"])[..., None]
            out += gate * (vote - out)
        out = out.mean(axis=0)
    else:
        if out.ndim != 2:
            raise ValueError("Missing fitted calibration for a fold ensemble")
        out = expit(out)
    out = np.clip(out, CLIP, 1 - CLIP)
    out[~treat] = 0.0
    target_cols = [c for c in sub_template.columns if c != "sig_id"]
    sub = sub_template.copy()
    if test["sig_id"].duplicated().any() or sub["sig_id"].duplicated().any():
        raise ValueError("Duplicate sig_id")
    if set(test["sig_id"]) != set(sub["sig_id"]):
        raise ValueError("Test/template sig_id mismatch")
    aligned = pd.DataFrame(out, index=test["sig_id"], columns=target_cols).loc[sub["sig_id"]]
    if not np.isfinite(aligned.to_numpy()).all():
        raise ValueError("Non-finite predictions")
    sub[target_cols] = aligned.to_numpy()
    sub.to_csv(out_path, index=False)
    return sub
