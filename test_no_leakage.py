"""Proves the forecast cutoff is respected: scrambling every PM2.5 and weather value AFTER 16:00 on
day D must not change any non-oracle feature for issue day D.   Run: python test_no_leakage.py"""
import numpy as np
import pandas as pd
import config as C
import features as F

pm = pd.read_pickle(C.PROC / "pm25_hourly_qc.pkl")
daily = pd.read_pickle(C.PROC / "pm25_daily.pkl")
sites = pd.read_csv(C.RAW / "sites.csv")
wx = pd.read_csv(C.RAW / "weather_hourly.csv", parse_dates=["time"])

D = pd.Timestamp("2021-06-15")
win = lambda df: df[(df.time >= D - pd.Timedelta(days=4)) & (df.time < D + pd.Timedelta(days=3))].copy()
pm_s, wx_s = win(pm), win(wx)
d_s = daily[(daily.date >= D - pd.Timedelta(days=4)) & (daily.date < D + pd.Timedelta(days=3))]

base = F.build_forecast_table(pm_s, wx_s, d_s, sites)
cut = D + pd.Timedelta(hours=C.FORECAST_CUTOFF_HOUR)
rng = np.random.default_rng(1)
pm_x, wx_x = pm_s.copy(), wx_s.copy()
pm_x.loc[pm_x.time >= cut, "pm25"] = rng.uniform(0, 500, (pm_x.time >= cut).sum())
for c in ["temperature_2m", "relative_humidity_2m", "wind_speed_10m", "boundary_layer_height", "precipitation"]:
    wx_x.loc[wx_x.time >= cut, c] = rng.uniform(0, 1000, (wx_x.time >= cut).sum())
scr = F.build_forecast_table(pm_x, wx_x, d_s, sites)

cols = F.OWN_PM + F.NET + F.WX + F.CAL
a = base[base.date == D].set_index("site_id")[cols].sort_index()
b = scr[scr.date == D].set_index("site_id")[cols].sort_index()
assert len(a) > 0, "no rows for test day"
pd.testing.assert_frame_equal(a, b)
o1 = base[base.date == D].set_index("site_id")[F.ORACLE].sort_index()
o2 = scr[scr.date == D].set_index("site_id")[F.ORACLE].sort_index()
assert not o1.equals(o2), "oracle features SHOULD change (sanity check that scrambling worked)"
print(f"PASS: {len(a)} stations x {len(cols)} features unchanged after scrambling post-cutoff data; oracle features changed as expected.")
