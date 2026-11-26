# Data

`folds.csv` is the frozen validation split used in the recorded experiments.

The local working copy contains the six official competition CSVs under `raw/`:

```text
train_features.csv
train_targets_scored.csv
train_targets_nonscored.csv
train_drug.csv
test_features.csv
sample_submission.csv
```

The shareable project ZIP excludes these raw data. Obtain them from the official
[Mechanisms of Action competition](https://www.kaggle.com/competitions/lish-moa/data)
after accepting the competition terms, then place them in `data/raw/`.
For inference alone only `test_features.csv` and `sample_submission.csv` are needed.
Alternatively use `python predict.py --data-dir "your-data-folder"`.
For retraining use `python reproduce.py all --data-dir "your-data-folder"`.
No API keys or account credentials belong in this directory.
