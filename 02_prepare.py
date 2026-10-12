"""Step 2: clean the hourly PM2.5, drop stations without enough data, and build the daily table.

Writes  data/processed/sites.csv            stations actually used from here on
        data/processed/pm25_hourly_qc.pkl   cleaned hourly readings
        data/processed/pm25_daily.pkl       one row per station per day (the modelling table)
Prints  station coverage, the biggest hourly readings, and counts per split (paste into the report)."""
import numpy as np
import pandas as pd
import config as C

pm = pd.read_csv(C.RAW / "pm25_hourly.csv", parse_dates=["time"])
sites = pd.read_csv(C.RAW / "sites.csv")

# ---- 1. QC on hourly values ------------------------------------------------
n0 = pm.pm25.notna().sum()
pm.loc[(pm.pm25 < -5) | (pm.pm25 > C.MAX_PLAUSIBLE_HOURLY), "pm25"] = np.nan   # physically implausible
pm.loc[pm.pm25.between(-5, 0), "pm25"] = 0                       # small negatives = instrument noise around zero
pm = pm.sort_values(["site_id", "time"]).reset_index(drop=True)
# stuck sensor: exactly the same non-zero value for 6+ hours in a row
run_len = pm.groupby("site_id").pm25.transform(lambda s: s.groupby((s != s.shift()).cumsum()).transform("size"))
pm.loc[(run_len >= 6) & (pm.pm25 > 0), "pm25"] = np.nan
print(f"QC removed {n0 - pm.pm25.notna().sum():,} of {n0:,} valid hourly values")

print("\nTen highest hourly readings kept (should mostly be Nov 2019 - Jan 2020 bushfire smoke):")
top = pm.nlargest(10, "pm25").merge(sites[["site_id", "site_name"]], on="site_id")
print(top[["site_name", "time", "pm25"]].to_string(index=False))

# ---- 2. daily means with the 75% completeness rule -------------------------
pm["date"] = pm.time.dt.normalize()
daily = pm.groupby(["site_id", "date"]).pm25.agg(pm25_mean="mean", n_hours="count").reset_index()
daily = daily[daily.n_hours >= C.MIN_HOURS_PER_DAY].drop(columns="n_hours")

# ---- 3. keep only stations with enough valid days --------------------------
cov = daily.groupby("site_id").agg(valid_days=("date", "size"), first_day=("date", "min"), last_day=("date", "max"))
cov = sites.set_index("site_id")[["site_name", "region"]].join(cov).fillna({"valid_days": 0})
cov["kept"] = cov.valid_days >= C.MIN_VALID_DAYS
print(f"\nStation coverage (kept = at least {C.MIN_VALID_DAYS} valid days):")
print(cov.sort_values("valid_days", ascending=False).to_string())
keep = cov.index[cov.kept]
sites = sites[sites.site_id.isin(keep)]
sites.to_csv(C.PROC / "sites.csv", index=False)
pm = pm[pm.site_id.isin(keep)]
daily = daily[daily.site_id.isin(keep)]
pm.drop(columns="date").to_pickle(C.PROC / "pm25_hourly_qc.pkl")

# ---- 4. labels used later ----------------------------------------------------
daily = daily.merge(sites[["site_id", "latitude", "longitude"]], on="site_id")
daily["split"] = np.select([daily.date <= C.TRAIN_END, daily.date.dt.year.astype(str) == C.VAL_YEAR,
                            daily.date.dt.year.astype(str) == C.TEST_YEAR], ["train", "val", "test"], "other")
daily["fire_period"] = daily.date.between(C.FIRE_START, C.FIRE_END)
daily.to_pickle(C.PROC / "pm25_daily.pkl")

# ---- 5. data-volume report (paste into the report) --------------------------
rep = daily.groupby("split").agg(stations=("site_id", "nunique"), station_days=("pm25_mean", "size"),
                                 who_exceed_rate=("pm25_mean", lambda x: (x > C.WHO_24H).mean()),
                                 nepm_exceed_rate=("pm25_mean", lambda x: (x > C.NEPM_24H).mean()))
strat = daily.groupby("fire_period").agg(station_days=("pm25_mean", "size"),
                                         who_exceed_rate=("pm25_mean", lambda x: (x > C.WHO_24H).mean()))
print(f"\n{len(sites)} stations kept, {len(daily):,} valid station-days")
print("\nBy split\n", rep.round(3), "\n\nBy fire period (Oct 2019 - Feb 2020)\n", strat.round(3))
rep.to_csv(C.OUT / "data_volume_by_split.csv")
strat.to_csv(C.OUT / "data_volume_by_fire_period.csv")
cov.to_csv(C.OUT / "station_coverage.csv")
