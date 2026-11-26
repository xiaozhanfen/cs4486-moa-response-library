"""Read and validate the frozen five-fold split used by all final experiments.

Drugs with <=18 treated examples stay together. Nine larger groups may cross folds.
This is a mixed protocol, not a claim of generalization to entirely unseen drugs.
"""
import numpy as np
import pandas as pd

RARE_DRUG_THRESHOLD = 18


def read_treated_folds(path, train, drug_map, n_splits):
    """Load a frozen split by ID. Full-train manifests mark controls with -1."""
    saved = pd.read_csv(path).set_index("sig_id", verify_integrity=True)["fold"]
    treated = train.loc[train.cp_type.eq("trt_cp"), "sig_id"]
    if not train.sig_id.is_unique or not treated.isin(saved.index).all():
        raise ValueError("Fold manifest does not cover the unique treated training IDs")
    controls = train.loc[train.cp_type.eq("ctl_vehicle"), "sig_id"]
    extra = saved.drop(index=treated)
    if not extra.index.isin(controls).all() or not extra.eq(-1).all():
        raise ValueError("Unexpected IDs or non-control assignments in fold manifest")
    ids = saved.loc[treated]
    if not ids.isin(range(n_splits)).all() or ids.nunique() != n_splits:
        raise ValueError("Fold manifest must cover every requested fold")
    if drug_map is None:
        raise ValueError("A drug-aware split requires train_drug.csv")
    drug = treated.map(drug_map)
    if drug.isna().any():
        raise ValueError("Missing treated drug IDs")
    groups = pd.DataFrame({"drug": drug.to_numpy(), "fold": ids.to_numpy()}).groupby("drug").fold
    if ((groups.size() <= RARE_DRUG_THRESHOLD) & (groups.nunique() > 1)).any():
        raise ValueError("A small drug group is split across folds")
    return ids.to_numpy(dtype=np.int64)

