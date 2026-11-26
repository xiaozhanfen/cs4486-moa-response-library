"""Sample-relative measurements and fold-local PCA for the response branch."""
import numpy as np


def center_signatures(raw):
    """Remove a sample's own measurement offset separately for genes and cells."""
    raw = np.asarray(raw, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] != 872 or not np.isfinite(raw).all():
        raise ValueError("Expected 772 finite gene and 100 finite cell measurements per row")
    result = raw.copy()
    for block in (result[:, :772], result[:, 772:]):
        quartiles = np.quantile(block, [0.25, 0.75], axis=1)
        block -= quartiles.mean(axis=0)[:, None]
    cells = result[:, 772:]
    cells /= 4.0 + np.quantile(np.abs(cells), 0.75, axis=1)[:, None]
    return result


def fit_response(centered):
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import QuantileTransformer

    qt = QuantileTransformer(n_quantiles=min(100, len(centered)), output_distribution="normal",
                             subsample=None, random_state=42)
    ranks = qt.fit_transform(centered)
    state = {"feature_kind": "relative_pca", "quantiles": qt.quantiles_,
             "references": qt.references_}
    for name, block, count in (("gene", ranks[:, :772], 50), ("cell", ranks[:, 772:], 15)):
        pca = PCA(min(count, *block.shape), svd_solver="randomized", random_state=42).fit(block)
        state[name + "_mean"] = pca.mean_
        state[name + "_components"] = pca.components_
        state[name + "_scale"] = np.sqrt(pca.explained_variance_).clip(1e-8)
    return state


def transform_response(centered, state):
    from infer_engine import manual_qt

    ranks = manual_qt(centered, state["quantiles"], state["references"])
    parts = [ranks]
    for name, block in (("gene", ranks[:, :772]), ("cell", ranks[:, 772:])):
        projection = (block - state[name + "_mean"]) @ state[name + "_components"].T
        parts.append(projection / state[name + "_scale"])
    return np.hstack(parts).astype(np.float32)
