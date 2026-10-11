"""Synthetic stand-in data in the SAME format as 01_download.py, so the whole pipeline can be
tested before the real download works. Delete data/raw/* and run 01_download.py for real data.
Results on synthetic data mean nothing scientifically."""
import numpy as np
import pandas as pd
import config as C

rng = np.random.default_rng(0)
n_sites = 15
sites = pd.DataFrame({
    "site_id": np.arange(100, 100 + n_sites),
    "site_name": [f"SYN_{i}" for i in range(n_sites)],
    "latitude": rng.uniform(-34.1, -33.6, n_sites),
    "longitude": rng.uniform(150.7, 151.3, n_sites),
})
sites.to_csv(C.RAW / "sites.csv", index=False)

hours = pd.date_range(C.STUDY_START, f"{C.STUDY_END} 23:00", freq="h")
days = hours.normalize().unique()
nd = len(days)
doy = days.dayofyear.values

# Regional daily driver: AR(1) + winter peak + 2019-20 smoke episode
reg = np.zeros(nd)
for i in range(1, nd):
    reg[i] = 0.7 * reg[i - 1] + rng.normal(0, 0.35)
reg += 0.25 * np.cos(2 * np.pi * (doy - 180) / 365)
fire = (days >= C.FIRE_START) & (days <= C.FIRE_END)
reg[fire] += rng.gamma(2, 0.6, fire.sum())
day_idx = np.searchsorted(days, hours.normalize())

# weather (shared, slightly site-perturbed)
wind = np.clip(4 + 2 * np.sin(2 * np.pi * hours.hour.values / 24) + rng.normal(0, 1.5, len(hours)), 0.1, None)
blh = np.clip(600 + 500 * np.sin(np.pi * (hours.hour.values - 6) / 12).clip(0) + rng.normal(0, 100, len(hours)), 50, None)
temp = 18 + 6 * np.cos(2 * np.pi * (hours.dayofyear.values - 15) / 365) + 4 * np.sin(2 * np.pi * (hours.hour.values - 9) / 24)
rh = np.clip(70 - 1.5 * (temp - 18) + rng.normal(0, 8, len(hours)), 10, 100)

pm_rows, wx_rows = [], []
for k, s in sites.iterrows():
    offset = 0.3 * (s.longitude - 151.0) * -3 + rng.normal(0, 0.15)  # western sites dirtier
    log_pm = 1.8 + offset + reg[day_idx] - 0.04 * (wind - 4) - 0.0004 * (blh - 600) \
        + 0.15 * np.sin(2 * np.pi * (hours.hour.values - 20) / 24) + rng.normal(0, 0.25, len(hours))
    pm = np.exp(log_pm)
    pm[rng.random(len(hours)) < 0.03] = np.nan             # random dropouts
    start = rng.integers(0, len(hours) - 24 * 60)
    pm[start:start + 24 * rng.integers(5, 60)] = np.nan     # an instrument outage
    pm_rows.append(pd.DataFrame({"site_id": s.site_id, "time": hours, "pm25": np.round(pm, 1)}))
    wx_rows.append(pd.DataFrame({"site_id": s.site_id, "time": hours,
                                 "temperature_2m": temp + rng.normal(0, 0.5, len(hours)),
                                 "relative_humidity_2m": rh, "wind_speed_10m": wind,
                                 "wind_direction_10m": rng.uniform(0, 360, len(hours)),
                                 "precipitation": rng.exponential(0.05, len(hours)),
                                 "surface_pressure": 1013 + rng.normal(0, 5, len(hours)),
                                 "boundary_layer_height": blh}))
pd.concat(pm_rows).to_csv(C.RAW / "pm25_hourly.csv", index=False)
pd.concat(wx_rows).to_csv(C.RAW / "weather_hourly.csv", index=False)
print("synthetic data written to", C.RAW)
