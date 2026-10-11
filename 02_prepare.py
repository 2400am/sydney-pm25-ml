"""QC hourly data, build daily station table, and print the data-volume numbers the feedback asked for
(stations, valid station-days, exceedance rates per split and per stratum)."""
import numpy as np
import pandas as pd
import config as C

pm = pd.read_csv(C.RAW / "pm25_hourly.csv", parse_dates=["time"])
sites = pd.read_csv(C.RAW / "sites.csv")

# ---- QC --------------------------------------------------------------------
n0 = pm.pm25.notna().sum()
pm.loc[(pm.pm25 < -5) | (pm.pm25 > 1000), "pm25"] = np.nan      # physically implausible
pm.loc[pm.pm25.between(-5, 0), "pm25"] = 0                      # small negatives = instrument noise
pm = pm.sort_values(["site_id", "time"])
# flag stuck sensors: identical value for >= 6 consecutive hours
same = pm.groupby("site_id").pm25.transform(lambda s: s.groupby((s != s.shift()).cumsum()).transform("size"))
pm.loc[(same >= 6) & (pm.pm25 > 0), "pm25"] = np.nan
print(f"QC removed {n0 - pm.pm25.notna().sum():,} of {n0:,} hourly values")
pm.to_pickle(C.PROC / "pm25_hourly_qc.pkl")

# ---- daily means with completeness rule ------------------------------------
pm["date"] = pm.time.dt.normalize()
daily = pm.groupby(["site_id", "date"]).pm25.agg(pm25_mean="mean", n_hours="count").reset_index()
daily = daily[daily.n_hours >= C.MIN_HOURS_PER_DAY].drop(columns="n_hours")
daily = daily.merge(sites[["site_id", "latitude", "longitude"]], on="site_id")
daily["split"] = np.select([daily.date <= C.TRAIN_END, daily.date.dt.year.astype(str) == C.VAL_YEAR,
                            daily.date.dt.year.astype(str) == C.TEST_YEAR], ["train", "val", "test"], "other")
daily["fire_period"] = daily.date.between(C.FIRE_START, C.FIRE_END)
daily.to_pickle(C.PROC / "pm25_daily.pkl")

# ---- data-volume report (paste into the proposal/report) -------------------
rep = daily.groupby("split").agg(stations=("site_id", "nunique"), station_days=("pm25_mean", "size"),
                                 who_exceed_rate=("pm25_mean", lambda x: (x > C.WHO_24H).mean()),
                                 nepm_exceed_rate=("pm25_mean", lambda x: (x > C.NEPM_24H).mean()))
strat = daily.groupby("fire_period").agg(station_days=("pm25_mean", "size"),
                                         who_exceed_rate=("pm25_mean", lambda x: (x > C.WHO_24H).mean()))
print("\nBy split\n", rep.round(3), "\n\nBy fire period\n", strat.round(3))
rep.to_csv(C.OUT / "data_volume_by_split.csv")
strat.to_csv(C.OUT / "data_volume_by_fire_period.csv")
