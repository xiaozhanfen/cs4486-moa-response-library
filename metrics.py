"""Kaggle lish-moa 官方指标：206 个计分标签的列平均二元 log loss。

y_true / y_pred 形状均为 (n, 206)。概率裁剪到 [1e-15, 1-1e-15]，与官方一致。
"""
import numpy as np

from config import CLIP


def competition_log_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    pred = np.asarray(y_pred, dtype=np.float64)
    truth = np.asarray(y_true)
    if pred.ndim != 2 or not pred.size or pred.shape != truth.shape:
        raise ValueError("Expected aligned nonempty sample-by-label matrices")
    if not np.isfinite(pred).all() or not np.isfinite(truth).all():
        raise ValueError("Non-finite validation inputs")
    pred = np.clip(pred, CLIP, 1.0 - CLIP)
    # Each label has the same row count. A matrix mean equals the columnwise mean.
    return float(-np.where(truth > 0.5, np.log(pred), np.log1p(-pred)).mean())
