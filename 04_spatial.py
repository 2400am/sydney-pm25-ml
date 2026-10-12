"""Objective 2: estimate daily PM2.5 at a location using only OTHER stations + weather + location.

Schemes (feedback asked whether spatial and temporal holdouts are combined - we report both):
  S1 spatial only        leave-one-station-out (LOSO), all years, stratified fire / non-fire
  S2 spatial + temporal  LOSO, train <=2022 on the other stations, test the held-out station in 2024
  S3 harder spatial      leave-CLUSTER-out + temporal: removes a whole group of nearby stations, so the
                         target is further from any donor - closer to a genuinely unmonitored suburb
Plus: error vs distance-to-nearest-station, 80% prediction-interval coverage, and the share of the
1 km grid that lies further from a station than anything we validated (area of applicability)."""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
import config as C
import evaluation as E
from features import haversine_km

daily = pd.read_pickle(C.PROC / "pm25_daily.pkl")
sites = pd.read_csv(C.PROC / "sites.csv").set_index("site_id")
wx = pd.read_csv(C.RAW / "weather_hourly.csv", parse_dates=["time"])
wx["date"] = wx.time.dt.normalize()
wxd = wx.groupby(["site_id", "date"])[["temperature_2m", "relative_humidity_2m", "wind_speed_10m",
                                       "boundary_layer_height", "precipitation"]].mean().add_prefix("wx_").reset_index()
P = daily.pivot_table(index="date", columns="site_id", values="pm25_mean")
ALL = list(P.columns)
# Raw lat/lon deliberately excluded: with ~15-20 stations a tree uses them to memorise station
# identity and extrapolates badly to new places. Add generalisable static features instead
# (distance to major road, land-use share, elevation, distance to coast) when you have them.
FEATS = ["idw", "nearest_val", "nearest_km", "mean3", "net_mean", "net_std", "n_donors",
         "doy_sin", "doy_cos",
         "wx_temperature_2m", "wx_relative_humidity_2m", "wx_wind_speed_10m", "wx_boundary_layer_height", "wx_precipitation"]


def site_features(s, donors):
    """Features for target site s on every date, using ONLY `donors` (s itself never included)."""
    donors = [d for d in donors if d != s]
    dist = pd.Series(haversine_km(sites.loc[s, "latitude"], sites.loc[s, "longitude"],
                                  sites.loc[donors, "latitude"].values, sites.loc[donors, "longitude"].values), index=donors)
    V = P[donors]
    avail = V.notna()
    w = avail.mul(1 / np.maximum(dist, 0.5) ** 2)
    idw = (V.fillna(0) * w).sum(axis=1) / w.sum(axis=1).replace(0, np.nan)
    order = dist.sort_values().index
    Vo = V[order]
    first = Vo.notna().idxmax(axis=1)   # nearest donor that reported that day
    nearest_val = Vo.bfill(axis=1).iloc[:, 0]
    take = Vo.notna() & (Vo.notna().cumsum(axis=1) <= 3)          # 3 nearest donors that reported
    mean3 = Vo.where(take).sum(axis=1) / take.sum(axis=1).replace(0, np.nan)
    f = pd.DataFrame({"date": P.index, "site_id": s, "idw": idw.values, "nearest_val": nearest_val.values,
                      "nearest_km": dist.reindex(first.values).values, "mean3": mean3.values,
                      "net_mean": V.mean(axis=1).values, "net_std": V.std(axis=1).values,
                      "n_donors": avail.sum(axis=1).values,
                      "latitude": sites.loc[s, "latitude"], "longitude": sites.loc[s, "longitude"]})
    doy = f.date.dt.dayofyear
    f["doy_sin"], f["doy_cos"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
    f = f.merge(daily[["site_id", "date", "pm25_mean", "fire_period"]], on=["site_id", "date"])  # target must exist
    return f.merge(wxd, on=["site_id", "date"], how="left")


def run_fold(test_sites, train_filter, test_filter):
    """Held-out sites are removed from the network entirely: never donors, never training rows."""
    network = [s for s in ALL if s not in test_sites]
    tr = pd.concat([site_features(s, network) for s in network])
    te = pd.concat([site_features(s, network) for s in test_sites])
    tr, te = tr[train_filter(tr)], te[test_filter(te)]
    tr = tr.dropna(subset=["idw"])
    te = te.dropna(subset=["idw"])
    # Residual learning: the model predicts a log-ratio CORRECTION to the IDW estimate. Trees cannot
    # extrapolate, so predicting levels directly fails on smoke days; a correction to IDW does not.
    y = np.log1p(tr.pm25_mean) - np.log1p(tr.idw.clip(lower=0))
    base = np.log1p(te.idw.clip(lower=0))
    kw = dict(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=200,
              l2_regularization=1.0, random_state=C.SEED)
    out = te.copy()
    model = HistGradientBoostingRegressor(**kw).fit(tr[FEATS], y)
    out["gbm"] = np.expm1(base + model.predict(te[FEATS]))
    # 80% interval by station-grouped split-conformal: residuals come from training stations that
    # were left out of their own fit, so the width reflects error at a NEW station. (A quantile GBM
    # was far too narrow: per-station constant features let it memorise station offsets.)
    oof = np.zeros(len(tr))
    for a, b in GroupKFold(n_splits=min(5, tr.site_id.nunique())).split(tr, y, tr.site_id):
        oof[b] = HistGradientBoostingRegressor(**kw).fit(tr[FEATS].iloc[a], y.iloc[a]).predict(tr[FEATS].iloc[b])
    lo_r, hi_r = np.quantile(y.values - oof, [0.10, 0.90])
    pred_log = base + model.predict(te[FEATS])
    out["q10"], out["q90"] = np.expm1(pred_log + lo_r), np.expm1(pred_log + hi_r)
    out["idw_baseline"], out["nearest_baseline"], out["netmean_baseline"] = te.idw, te.nearest_val, te.net_mean
    return out


is_train = lambda d: d.date <= C.TRAIN_END
is_test = lambda d: d.date.dt.year.astype(str) == C.TEST_YEAR
anytime = lambda d: np.ones(len(d), bool)

km = KMeans(C.N_SPATIAL_CLUSTERS, n_init=10, random_state=C.SEED).fit(sites[["latitude", "longitude"]])
clusters = pd.Series(km.labels_, index=sites.index)

schemes = {
    "S1_LOSO_all_years": [run_fold([s], anytime, anytime) for s in ALL],
    "S2_LOSO_plus_test_year": [run_fold([s], is_train, is_test) for s in ALL],
    "S3_leave_cluster_out_plus_test_year": [run_fold(list(clusters.index[clusters == k]), is_train, is_test)
                                            for k in range(C.N_SPATIAL_CLUSTERS)],
}

MODELS = ["gbm", "idw_baseline", "nearest_baseline", "netmean_baseline"]
rows, by_dist = [], []
for name, folds in schemes.items():
    R = pd.concat(folds)
    R.to_pickle(C.PROC / f"spatial_preds_{name}.pkl")
    for s_name, mask in E.strata(R).items():
        if mask.sum() == 0 or (s_name != "all" and mask.all()):
            continue
        r = R[mask]
        for m in MODELS:
            rows.append({"scheme": name, "stratum": s_name, "model": m, "n": len(r),
                         **{k: f(r.pm25_mean, r[m]) for k, f in E.REG_METRICS.items()}})
        rows[-4]["PI80_coverage"] = float(((r.pm25_mean >= r.q10) & (r.pm25_mean <= r.q90)).mean())
        best = min(MODELS[1:], key=lambda m: E.rmse(r.pm25_mean, r[m]))
        pt, lo, hi = E.paired_block_bootstrap(r.pm25_mean, r.gbm, r[best], r.site_id, E.rmse_reduction)  # resample STATIONS
        rows.append({"scheme": name, "stratum": s_name, "model": f"RMSE reduction vs {best}",
                     "RMSE": pt, "CI_lo": lo, "CI_hi": hi,
                     "meets_criterion": bool(pt >= C.MIN_RMSE_REDUCTION and lo > 0)})
    # error vs distance to nearest available station
    R["dist_bin"] = pd.cut(R.nearest_km, [0, 5, 10, 20, 40, 1e9])
    d = R.groupby("dist_bin", observed=True).apply(lambda g: pd.Series({
        "n": len(g), "gbm_RMSE": E.rmse(g.pm25_mean, g.gbm), "idw_RMSE": E.rmse(g.pm25_mean, g.idw_baseline)}),
        include_groups=False).reset_index()
    by_dist.append(d.assign(scheme=name))

res = pd.DataFrame(rows)
res.to_csv(C.OUT / "spatial_results.csv", index=False)
dist = pd.concat(by_dist)
dist.to_csv(C.OUT / "spatial_error_by_distance.csv", index=False)

# ---- Area of applicability on the 1 km grid --------------------------------
b = C.BBOX
lat = np.arange(b["lat_min"], b["lat_max"], C.GRID_KM / 111)
lon = np.arange(b["lon_min"], b["lon_max"], C.GRID_KM / (111 * np.cos(np.radians(-33.9))))
G = pd.DataFrame([(a, o) for a in lat for o in lon], columns=["latitude", "longitude"])
G["nearest_station_km"] = np.min([haversine_km(G.latitude, G.longitude, r.latitude, r.longitude)
                                  for _, r in sites.iterrows()], axis=0)
max_validated = pd.concat(schemes["S3_leave_cluster_out_plus_test_year"]).nearest_km.quantile(0.95)
G["inside_validated_range"] = G.nearest_station_km <= max_validated
G.to_csv(C.OUT / "grid_area_of_applicability.csv", index=False)

pd.set_option("display.width", 200)
print(res.round(3).to_string(index=False))
print("\nError by distance to nearest station\n", dist.round(2).to_string(index=False))
print(f"\n{G.inside_validated_range.mean():.0%} of 1 km grid cells are within the validated distance "
      f"({max_validated:.1f} km); the rest should be shown as 'outside validated range' on any map.")
