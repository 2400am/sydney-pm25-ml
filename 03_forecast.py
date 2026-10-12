"""Objective 1: next-day WHO exceedance classification.

Experiments (all thresholds/hyper-parameters chosen on VAL_YEAR, test year touched once):
  E1 monitored stations, unseen time         -> train <=2022, test 2024
  E2 unmonitored stations, unseen time       -> combined holdout: a spatial cluster removed from training
                                                AND its own PM history replaced by neighbour-IDW inputs
  E3 extreme-event stress test               -> train without Jul-2019..Jun-2020, test on the fire period
  E4 does fire data distort typical days?    -> 2024 test, trained with vs without the fire season
  Oracle                                     -> E1 + observed next-day weather (upper bound only)
"""
import json
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import config as C
import evaluation as E
import features as F

pm = pd.read_pickle(C.PROC / "pm25_hourly_qc.pkl")
daily = pd.read_pickle(C.PROC / "pm25_daily.pkl")
sites = pd.read_csv(C.PROC / "sites.csv")
wx = pd.read_csv(C.RAW / "weather_hourly.csv", parse_dates=["time"])
X = F.build_forecast_table(pm, wx, daily, sites)
X.to_pickle(C.PROC / "forecast_table.pkl")
print(f"forecast table: {len(X):,} station-days, positive rate {X.y.mean():.3f}")

FEATS = F.OWN_PM + F.NET + F.WX + F.CAL
yr = X.target_date.dt.year.astype(str)
train_m, val_m, test_m = X.target_date <= C.TRAIN_END, yr == C.VAL_YEAR, yr == C.TEST_YEAR
fire_season = X.target_date.between("2019-07-01", "2020-06-30")


def fit_models(tr, feats):
    hgb = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                         l2_regularization=1.0, random_state=C.SEED)
    hgb.fit(tr[feats], tr.y)
    lr = make_pipeline(SimpleImputer(), StandardScaler(), LogisticRegression(max_iter=2000))
    lr.fit(tr[feats], tr.y)
    clim = tr.assign(m=tr.target_date.dt.month).groupby(["site_id", "m"]).y.mean()
    return {"gbm": hgb, "logistic": lr, "clim": clim}


def predict_all(models, df, feats):
    p = {"gbm": models["gbm"].predict_proba(df[feats])[:, 1],
         "logistic": models["logistic"].predict_proba(df[feats])[:, 1]}
    key = pd.MultiIndex.from_arrays([df.site_id, df.target_date.dt.month])
    p["climatology"] = models["clim"].reindex(key).fillna(models["clim"].mean()).values
    # (no class re-weighting: keeps probabilities calibrated; imbalance is handled by the threshold rule)
    # persistence: "tomorrow like the last 24 h". Score = last-24h mean, so PR-AUC is defined.
    p["persistence"] = df.pm_mean_24h.fillna(df.net_mean_24h).values
    return p


def pick_threshold(y, prob, min_recall=0.80):
    """Pre-registered rule: highest precision subject to recall >= 0.80 on the validation year."""
    best = (0.5, -1)
    for t in np.unique(np.round(prob, 4)):
        yhat = prob >= t
        rec = (yhat & (y == 1)).sum() / max(y.sum(), 1)
        prec = (yhat & (y == 1)).sum() / max(yhat.sum(), 1)
        if rec >= min_recall and prec > best[1]:
            best = (t, prec)
    return best[0]


def report(name, df, preds, thresholds):
    rows = []
    blocks = df.target_date.dt.to_period("W").astype(str).values   # weekly blocks
    for s_name, mask in E.strata(df).items():
        if mask.sum() == 0 or (s_name != "all" and mask.all()):
            continue
        for m, p in preds.items():
            r = E.clf_metrics(df.y.values[mask], p[mask], thresholds.get(m, C.WHO_24H if m == "persistence" else 0.5))
            rows.append({"experiment": name, "stratum": s_name, "model": m, **r})
        # headline test: gbm vs best baseline PR-AUC, paired weekly block bootstrap
        if df.y.values[mask].sum() > 5:
            base = max(["persistence", "climatology", "logistic"],
                       key=lambda b: E.clf_metrics(df.y.values[mask], preds[b][mask], 0.5)["PR_AUC"])
            pt, lo, hi = E.paired_block_bootstrap(df.y.values[mask], preds["gbm"][mask], preds[base][mask],
                                                  blocks[mask], E.prauc_gain)
            rows.append({"experiment": name, "stratum": s_name, "model": f"gbm minus {base} (PR-AUC)",
                         "PR_AUC": pt, "CI_lo": lo, "CI_hi": hi,
                         "meets_criterion": bool(pt >= C.MIN_PRAUC_GAIN and lo > 0)})
    return rows


results = []

# ---- E1 + oracle ------------------------------------------------------------
models = fit_models(X[train_m], FEATS)
val_p = predict_all(models, X[val_m], FEATS)
thr = {m: pick_threshold(X[val_m].y.values, p) for m, p in val_p.items() if m != "persistence"}
results += report("E1_monitored_unseen_time", X[test_m], predict_all(models, X[test_m], FEATS), thr)

orc = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=C.SEED)
orc.fit(X[train_m][FEATS + F.ORACLE], X[train_m].y)
results += [{"experiment": "Oracle_weather_upper_bound", "stratum": "all", "model": "gbm+observed_next_day_weather",
             **E.clf_metrics(X[test_m].y.values, orc.predict_proba(X[test_m][FEATS + F.ORACLE])[:, 1], thr["gbm"])}]

# ---- E2: combined spatial + temporal holdout --------------------------------
km = KMeans(C.N_SPATIAL_CLUSTERS, n_init=10, random_state=C.SEED).fit(sites[["latitude", "longitude"]])
sites["cluster"] = km.labels_
e2_rows = []
for k in range(C.N_SPATIAL_CLUSTERS):
    held = set(sites.site_id[sites.cluster == k])
    # Every row's own-PM features come from neighbours only; held-out sites are never donors.
    Xn = F.neighbour_substitute(X, sites, F.OWN_PM, exclude_sites=held)
    tr = Xn[train_m & ~Xn.site_id.isin(held)]
    te = Xn[test_m & Xn.site_id.isin(held)]
    if te.empty:
        continue
    mk = fit_models(tr, FEATS)
    p = predict_all(mk, te, FEATS)
    e2_rows.append(te.assign(**{f"p_{m}": v for m, v in p.items()}))
e2 = pd.concat(e2_rows)
results += report("E2_unmonitored_unseen_time(leave-cluster-out)", e2,
                  {m: e2[f"p_{m}"].values for m in ["gbm", "logistic", "climatology", "persistence"]}, thr)

# ---- E3: extreme-event stress test -----------------------------------------
m3 = fit_models(X[train_m & ~fire_season], FEATS)
fire_te = X[X.fire_period]
results += report("E3_fire_period_never_seen_in_training", fire_te, predict_all(m3, fire_te, FEATS), thr)

# ---- E4: does including fire data help/hurt typical (2024) days? -----------
p_with = models["gbm"].predict_proba(X[test_m][FEATS])[:, 1]
p_without = m3["gbm"].predict_proba(X[test_m][FEATS])[:, 1]
pt, lo, hi = E.paired_block_bootstrap(X[test_m].y.values, p_with, p_without,
                                      X[test_m].target_date.dt.to_period("W").astype(str).values, E.prauc_gain)
results.append({"experiment": "E4_effect_of_fire_season_in_training", "stratum": "test_2024",
                "model": "with minus without (PR-AUC)", "PR_AUC": pt, "CI_lo": lo, "CI_hi": hi})

res = pd.DataFrame(results)
res.to_csv(C.OUT / "forecast_results.csv", index=False)
json.dump({k: float(v) for k, v in thr.items()}, open(C.OUT / "forecast_thresholds.json", "w"), indent=1)
pd.set_option("display.width", 200, "display.max_columns", 20)
print(res[["experiment", "stratum", "model", "positives", "PR_AUC", "CI_lo", "CI_hi", "recall", "precision", "F1"]]
      .round(3).to_string(index=False))
