"""Verify the report's 15 source combinations using saved OOF predictions."""

import itertools
import json
from pathlib import Path
import sys

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from response_study import apply_blend, fit_blend, loss, read_study


def main():
    tags = ["response_cnn5", "response_profile5", "response_aux5"]
    _, folds, truth, _, _, logits, library, confidence = read_study(tags)
    scores = {"library": loss(truth, library)}
    with threadpool_limits(limits=4):
        for size in range(1, len(tags) + 1):
            for columns in itertools.combinations(range(len(tags)), size):
                selected = logits[..., list(columns)]
                for use_library in (False, True):
                    prediction = np.empty_like(truth)
                    for fold in range(5):
                        train, valid = folds != fold, folds == fold
                        fit = fit_blend(selected[train], library[train], confidence[train],
                                        truth[train], use_library=use_library)
                        prediction[valid] = apply_blend(selected[valid], library[valid],
                                                       confidence[valid], fit)
                    members = [tags[column] for column in columns]
                    if use_library:
                        members.append("library")
                    name = "+".join(members)
                    scores[name] = loss(truth, prediction)
                    print(f"{name}: {scores[name]:.10f}", flush=True)
    output = ROOT / "evidence/report_combinations.json"
    report = {
        "protocol": "Fixed OOF sources, fusion refitted on other four folds for each scored fold",
        "rows": len(truth),
        "library_only": "Original uncalibrated OOF library votes",
        "scores": scores,
    }
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
