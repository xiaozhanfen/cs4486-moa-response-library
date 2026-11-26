"""特征构建：全部变换只在本折训练部分 fit，测试/验证只 transform —— 防泄漏。

输入（处理样本）：
  * 872 个生物特征（g-772 + c-100）逐列 RankGauss（QuantileTransformer(n_quantiles=100, normal)）
  * FactorAnalysis：g- -> 90 维、c- -> 50 维（无监督降维，不含标签信息）
  * 统计特征：g / c / g+c 各自的 sum、mean、std、kurtosis、skew，共 15 维
  * cp_time / cp_dose one-hot（3+2=5 维）
合计：872 + 140 + 15 + 5 = 1032 维。
"""
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew
from sklearn.decomposition import FactorAnalysis
from sklearn.preprocessing import QuantileTransformer

from config import BIO_COLS, CELL_COLS, COND_COLS, GENE_COLS

FA_G = 90   # g- 特征的因子数
FA_C = 50   # c- 特征的因子数
QT_N = 100  # RankGauss 分位数


class FoldFeatureBuilder:
    """在给定训练折上 fit，为 train/val/test 生成同一套特征。

    两级 RankGauss（冠军方案的做法）：
      第 1 级：872 个原始生物特征逐列 QuantileTransformer(normal)；
      第 2 级：拼接 FA 与统计特征后，对全部列再做一次 QuantileTransformer(normal)，
               统一不同来源特征的量纲，改善 NN 优化条件数。
    两级变换都只在本折训练部分 fit。
    """

    def __init__(self, use_fa: bool = True, use_stats: bool = True, seed: int = 123):
        self.use_fa = use_fa
        self.use_stats = use_stats
        self.seed = seed
        self.qt = None
        self.qt2 = None
        self.fa_g = None
        self.fa_c = None
        self.out_dim = None
        self.cond_cols = None
        self.categories = None

    def _assemble(self, X: np.ndarray, df: pd.DataFrame) -> np.ndarray:
        n_g = len(GENE_COLS)
        n_c = len(CELL_COLS)
        parts = [self.qt.transform(X)]
        if self.use_fa:
            parts += [self.fa_g.transform(parts[0][:, :n_g]),
                      self.fa_c.transform(parts[0][:, n_g:n_g + n_c])]
        if self.use_stats:
            g, c = parts[0][:, :n_g], parts[0][:, n_g:n_g + n_c]
            gc = np.hstack([g, c])

            def _stats(x):
                return np.nan_to_num(np.vstack(
                    [x.sum(1), x.mean(1), x.std(1), kurtosis(x, axis=1), skew(x, axis=1)]).T,
                    nan=0.0, posinf=0.0, neginf=0.0)

            parts += [_stats(g), _stats(c), _stats(gc)]
        onehot = np.hstack([np.column_stack([(df[c] == v).to_numpy(dtype=np.float32)
                                            for v in self.categories[c]])
                            for c in self.cond_cols])
        parts.append(onehot)
        return np.hstack(parts).astype(np.float64)

    def fit(self, X_tr: np.ndarray, df_tr: pd.DataFrame, df_val: pd.DataFrame,
            df_test: pd.DataFrame, cond_cols: list[str],
            X_pool: np.ndarray | None = None, df_pool: pd.DataFrame | None = None):
        """X_pool/df_pool：可选的额外无监督样本（如测试集特征），与训练数据合并
        用于 FA 与第二级 RankGauss 的拟合。全程不接触任何标签。"""
        n_g = len(GENE_COLS)
        n_c = len(CELL_COLS)
        self.cond_cols = cond_cols
        self.categories = {c: sorted(df_tr[c].unique().tolist()) for c in cond_cols}

        self.qt = QuantileTransformer(n_quantiles=QT_N, output_distribution="normal",
                                      random_state=self.seed, subsample=int(2e5))
        self.qt.fit(X_tr)

        X_tr_t = self.qt.transform(X_tr)
        if self.use_fa:
            if X_pool is not None:
                src = np.vstack([X_tr_t, self.qt.transform(X_pool)])
            else:
                src = X_tr_t
            self.fa_g = FactorAnalysis(n_components=FA_G, random_state=1903).fit(src[:, :n_g])
            self.fa_c = FactorAnalysis(n_components=FA_C, random_state=1903).fit(src[:, n_g:n_g + n_c])

        full_tr = self._assemble(X_tr, df_tr)
        if X_pool is not None and df_pool is not None:
            full_tr = np.vstack([full_tr, self._assemble(X_pool, df_pool)])
        self.qt2 = QuantileTransformer(n_quantiles=QT_N, output_distribution="normal",
                                       random_state=self.seed, subsample=int(2e5))
        self.qt2.fit(full_tr)
        self.out_dim = full_tr.shape[1]
        return self

    def transform(self, X: np.ndarray, df: pd.DataFrame) -> np.ndarray:
        return self.qt2.transform(self._assemble(X, df)).astype(np.float32)

    def save(self, path) -> None:
        """保存全部 fit 参数（本地复现用）。"""
        payload = {
            "qt1_quantiles": self.qt.quantiles_,
            "qt1_references": self.qt.references_,
            "qt2_quantiles": self.qt2.quantiles_,
            "qt2_references": self.qt2.references_,
            "qt_n_quantiles": int(self.qt.n_quantiles_),
            "use_fa": bool(self.use_fa), "use_stats": bool(self.use_stats),
            "out_dim": int(self.out_dim), "cond_cols": list(self.cond_cols),
        }
        for c in self.cond_cols:
            payload[f"categories_{c}"] = self.categories[c]
        if self.use_fa:
            payload.update({
                "fa_g_mean": self.fa_g.mean_, "fa_g_comp": self.fa_g.components_,
                "fa_c_mean": self.fa_c.mean_, "fa_c_comp": self.fa_c.components_,
                "fa_g_noise": self.fa_g.noise_variance_,
                "fa_c_noise": self.fa_c.noise_variance_,
            })
        np.savez_compressed(path, **payload)


def extract_bio_matrix(df: pd.DataFrame) -> np.ndarray:
    """取出 872 维生物特征矩阵（未变换）。"""
    return df[BIO_COLS].to_numpy(dtype=np.float64)
