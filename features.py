"""Feature construction with an explicit, auditable forecast cutoff.

Forecast for target day D+1 is issued at 16:00 AEST on day D. Allowed inputs are hourly values
whose hour STARTS at or before 15:00 on day D (that hour ends at 16:00). Everything is computed by
reading the hourly series at timestamp D 15:00, so no later value can leak in."""
import numpy as np
import pandas as pd
import config as C

LAST_HOUR = pd.Timedelta(hours=C.FORECAST_CUTOFF_HOUR - 1)


def _per_site_hourly(pm, wx):
    pm = pm.pivot_table(index="time", columns="site_id", values="pm25")
    full = pd.date_range(pm.index.min().normalize(), pm.index.max().normalize() + pd.Timedelta(hours=23), freq="h")
    pm = pm.reindex(full)
    wx = wx.set_index(["time", "site_id"]).sort_index()
    return pm, wx


def build_forecast_table(pm_hourly, wx_hourly, daily, sites):
    pm, wx = _per_site_hourly(pm_hourly, wx_hourly)
    roll = lambda df, h, fn: getattr(df.rolling(h, min_periods=int(h * 0.6)), fn)()
    feats = {
        "pm_mean_24h": roll(pm, 24, "mean"),
        "pm_max_24h": roll(pm, 24, "max"),
        "pm_mean_6h": roll(pm, 6, "mean"),
        "pm_mean_48h": roll(pm, 48, "mean"),
    }
    cut_times = pd.DatetimeIndex(sorted(daily.date.unique())) + LAST_HOUR
    cut_times = cut_times[cut_times.isin(pm.index)]
    rows = []
    for name, f in feats.items():
        s = f.loc[cut_times].stack(future_stack=True).rename(name)
        rows.append(s)
    X = pd.concat(rows, axis=1).reset_index()
    X.columns = ["cutoff", "site_id", *feats.keys()]
    X["date"] = X.cutoff.dt.normalize()              # issue day D
    X["pm_trend_6h"] = X.pm_mean_6h - X.pm_mean_24h

    # Network-wide state at the cutoff (uses only other stations' past values: fine for forecasting)
    net = X.groupby("date").agg(net_mean_24h=("pm_mean_24h", "mean"), net_max_24h=("pm_max_24h", "max"),
                                net_frac_over=("pm_mean_24h", lambda x: (x > C.WHO_24H).mean()))
    X = X.merge(net, on="date")

    # Observed weather, last 24 h up to the cutoff
    w = wx.reset_index()
    rad = np.deg2rad(w.wind_direction_10m)
    w["u"], w["v"] = -w.wind_speed_10m * np.sin(rad), -w.wind_speed_10m * np.cos(rad)
    wcols = ["temperature_2m", "relative_humidity_2m", "wind_speed_10m", "boundary_layer_height",
             "surface_pressure", "u", "v"]
    wp = w.pivot_table(index="time", columns="site_id", values=wcols + ["precipitation"])
    wf = []
    for c in wcols:
        wf.append(wp[c].rolling(24, min_periods=12).mean().loc[cut_times].stack(future_stack=True).rename(f"wx_{c}_24h"))
    wf.append(wp["precipitation"].rolling(24, min_periods=12).sum().loc[cut_times].stack(future_stack=True).rename("wx_precip_24h"))
    W = pd.concat(wf, axis=1).reset_index().rename(columns={"time": "cutoff", "level_0": "cutoff"})
    X = X.merge(W, on=["cutoff", "site_id"], how="left")

    # ORACLE: observed weather on the target day. Not available in real time; reported only as an
    # upper bound on what a perfect weather forecast could add. Never used in the headline model.
    w["date"] = w.time.dt.normalize() - pd.Timedelta(days=1)   # value for D+1 attached to issue day D
    orc = w.groupby(["site_id", "date"])[["wind_speed_10m", "boundary_layer_height", "relative_humidity_2m",
                                          "precipitation"]].mean().add_prefix("oracle_next_").reset_index()
    X = X.merge(orc, on=["site_id", "date"], how="left")

    # Calendar
    X["doy_sin"], X["doy_cos"] = np.sin(2 * np.pi * X.date.dt.dayofyear / 365.25), np.cos(2 * np.pi * X.date.dt.dayofyear / 365.25)
    X["dow"] = X.date.dt.dayofweek

    # Target: valid daily mean on D+1
    tgt = daily[["site_id", "date", "pm25_mean"]].copy()
    tgt["date"] -= pd.Timedelta(days=1)
    X = X.merge(tgt.rename(columns={"pm25_mean": "target_pm25"}), on=["site_id", "date"])
    X["y"] = (X.target_pm25 > C.WHO_24H).astype(int)
    X["target_date"] = X.date + pd.Timedelta(days=1)
    X["fire_period"] = X.target_date.between(C.FIRE_START, C.FIRE_END)
    X = X.merge(sites[["site_id", "latitude", "longitude"]], on="site_id")
    return X


OWN_PM = ["pm_mean_24h", "pm_max_24h", "pm_mean_6h", "pm_mean_48h", "pm_trend_6h"]
NET = ["net_mean_24h", "net_max_24h", "net_frac_over"]
WX = ["wx_temperature_2m_24h", "wx_relative_humidity_2m_24h", "wx_wind_speed_10m_24h",
      "wx_boundary_layer_height_24h", "wx_surface_pressure_24h", "wx_u_24h", "wx_v_24h", "wx_precip_24h"]
CAL = ["doy_sin", "doy_cos", "dow"]
ORACLE = ["oracle_next_wind_speed_10m", "oracle_next_boundary_layer_height",
          "oracle_next_relative_humidity_2m", "oracle_next_precipitation"]


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


def neighbour_substitute(X, sites, cols, exclude_sites=(), power=2):
    """For the 'unmonitored location' forecast: replace a station's own PM history with an
    inverse-distance-weighted estimate built ONLY from other stations (never itself, never any
    station in exclude_sites). This is exactly what an unmonitored suburb would have at 16:00."""
    S = sites.set_index("site_id")
    out = X.copy()
    piv = {c: X.pivot_table(index="date", columns="site_id", values=c) for c in cols}
    for s in X.site_id.unique():
        donors = [d for d in S.index if d != s and d not in exclude_sites and d in piv[cols[0]].columns]
        dist = haversine_km(S.loc[s, "latitude"], S.loc[s, "longitude"], S.loc[donors, "latitude"].values,
                            S.loc[donors, "longitude"].values)
        wts = pd.Series(1 / np.maximum(dist, 0.5) ** power, index=donors)
        m = X.site_id == s
        for c in cols:
            v = piv[c][donors]
            est = (v * wts).sum(axis=1, min_count=1) / v.notna().mul(wts).sum(axis=1).replace(0, np.nan)
            out.loc[m, c] = X.loc[m, "date"].map(est).values
        out.loc[m, "dist_nearest_station_km"] = dist.min()
    return out
