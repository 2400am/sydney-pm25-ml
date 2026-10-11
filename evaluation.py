"""Shared evaluation: metrics, paired block bootstrap CIs, stratified reporting.

Why block bootstrap: days at one station are autocorrelated and stations on the same day are
correlated, so resampling individual rows gives CIs that are far too narrow. We resample whole
blocks (calendar weeks for temporal tests, stations for spatial tests)."""
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, brier_score_loss, f1_score,
                             mean_absolute_error, precision_score, recall_score, roc_auc_score)
import config as C


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def r2(y, p):
    y, p = np.asarray(y), np.asarray(p)
    return float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))


REG_METRICS = {"MAE": mean_absolute_error, "RMSE": rmse, "R2": r2}


def clf_metrics(y, prob, thr):
    yhat = prob >= thr
    out = {"n": len(y), "positives": int(y.sum()), "PR_AUC": np.nan, "ROC_AUC": np.nan,
           "Brier": brier_score_loss(y, prob) if (len(np.unique(y)) > 1 and prob.max() <= 1) else np.nan,
           "recall": recall_score(y, yhat, zero_division=0),
           "precision": precision_score(y, yhat, zero_division=0),
           "F1": f1_score(y, yhat, zero_division=0)}
    if 0 < y.sum() < len(y):
        out["PR_AUC"] = average_precision_score(y, prob)
        out["ROC_AUC"] = roc_auc_score(y, prob)
    return out


def paired_block_bootstrap(y, pred_model, pred_base, blocks, metric, n_boot=C.N_BOOT, seed=C.SEED):
    """Returns (point estimate of metric(base) - metric(model) or model - base, 95% CI).
    Positive = model better for error metrics; we pass sign via `metric` choice below."""
    rng = np.random.default_rng(seed)
    y, a, b, blocks = map(np.asarray, (y, pred_model, pred_base, blocks))
    uniq = np.unique(blocks)
    idx_by_block = {k: np.flatnonzero(blocks == k) for k in uniq}
    point = metric(y, a, b)
    stats = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        ii = np.concatenate([idx_by_block[k] for k in pick])
        try:
            stats.append(metric(y[ii], a[ii], b[ii]))
        except ValueError:      # e.g. a resample with no positives
            continue
    lo, hi = np.nanpercentile(stats, [2.5, 97.5])
    return point, lo, hi


def rmse_reduction(y, model, base):        # relative: 0.10 = 10% lower RMSE than baseline
    return 1 - rmse(y, model) / rmse(y, base)


def prauc_gain(y, model, base):
    return average_precision_score(y, model) - average_precision_score(y, base)


def strata(df):
    """Masks for the strata the feedback asked for: smoke vs typical conditions."""
    fire = df.fire_period.values.astype(bool)
    return {"all": np.ones(len(df), bool), "fire_2019_20": fire, "non_fire": ~fire}
