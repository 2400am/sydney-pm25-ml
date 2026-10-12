import pandas as pd
sites = pd.read_csv("data/raw/sites.csv")
pm = pd.read_csv("data/raw/pm25_hourly.csv", parse_dates=["time"])
wx = pd.read_csv("data/raw/weather_hourly.csv", parse_dates=["time"])

print(f"{len(sites)} stations, {len(pm):,} PM2.5 rows, {len(wx):,} weather rows\n")
summary = pm.groupby("site_id").agg(first=("time", "min"), last=("time", "max"),
                                    hours=("pm25", "count"), mean=("pm25", "mean"), max=("pm25", "max"))
summary = sites.set_index("site_id")[["site_name"]].join(summary).sort_values("first")
print(summary.round(1).to_string())