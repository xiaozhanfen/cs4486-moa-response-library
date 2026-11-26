"""Recompute a minimal label-frequency baseline on the exact frozen split."""
import json

import numpy as np

from config import DATA_DIR, ROOT
from metrics import competition_log_loss
from response_study import study_rows


def main():
    ids, folds, y, targets, _ = study_rows()
    oof = np.empty_like(y)
    for fold in range(5):
        oof[folds == fold] = y[folds != fold].mean(axis=0)
    treated = competition_log_loss(y, oof)
    import pandas as pd
    total = len(pd.read_csv(DATA_DIR/"train_features.csv", usecols=["sig_id"]))
    result = {"method": "training-fold label frequency; controls zero", "folds": 5,
              "treated": treated, "all_rows": treated*len(y)/total,
              "treated_rows": len(ids), "total_rows": total}
    output = ROOT / "outputs/baseline_frequency"
    output.mkdir(parents=True, exist_ok=True)
    (output/"metrics.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    np.savez_compressed(output/"oof.npz", sig_id=ids, folds=folds, targets=targets, pred=oof.astype(np.float32))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
