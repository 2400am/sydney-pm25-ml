"""Download the raw data for the project.

  1. sites.csv          - list of NSW government air-quality monitoring stations in the Sydney box
  2. pm25_hourly.csv    - hourly PM2.5 measured at those stations, 2016-2024 (NSW DCCEEW Air Quality API)
  3. weather_hourly.csv - hourly weather at each station's location (ERA5 reanalysis via Open-Meteo)

Downloads go in small chunks that are saved as they arrive (data/raw/chunks/), so if it crashes
you can just run it again and it carries on where it stopped.

Run:  python 01_download.py
API guide: https://www.environment.nsw.gov.au/sites/default/files/air-quality-application-programming-interface-user-guide-210346.pdf
"""
import time
import pandas as pd
import requests
import config as C

NSW = "https://data.airquality.nsw.gov.au/api/Data"
HDR = {"Content-Type": "application/json", "accept": "application/json"}
CHUNKS = C.RAW / "chunks"
CHUNKS.mkdir(exist_ok=True)


def post_once(url, body, tries=2):
    """Returns the parsed JSON, or None if the server keeps failing (so the caller can split the request)."""
    for attempt in range(1, tries + 1):
        try:
            r = requests.post(url, json=body, headers=HDR, timeout=180)
            if r.status_code == 200:
                return r.json()
            print(f"    server said {r.status_code} ({r.text[:80].strip()}...)")
        except requests.RequestException as e:
            print(f"    request failed ({e.__class__.__name__})")
        time.sleep(3 * attempt)
    return None


def fetch_pm25(site_ids, start, end):
    """Ask for PM2.5 for these stations between start and end. If the server chokes (it crashes on big
    requests with a 502), split the request in half - first by stations, then by dates - and try again."""
    body = {"Parameters": ["PM2.5"], "Sites": [int(x) for x in site_ids],
            "StartDate": f"{start:%Y-%m-%d}", "EndDate": f"{end:%Y-%m-%d}",
            "Categories": ["Averages"], "SubCategories": ["Hourly"], "Subcategories": ["Hourly"],
            "Frequency": ["Hourly average"]}
    data = post_once(f"{NSW}/get_Observations", body)
    if data is not None:
        return data if isinstance(data, list) else []
    if len(site_ids) > 1:
        half = len(site_ids) // 2
        print(f"    -> splitting into {half} + {len(site_ids) - half} stations")
        return fetch_pm25(site_ids[:half], start, end) + fetch_pm25(site_ids[half:], start, end)
    if (end - start).days > 7:
        mid = start + (end - start) / 2
        mid = mid.normalize()
        print(f"    -> splitting dates at {mid:%Y-%m-%d}")
        return fetch_pm25(site_ids, start, mid) + fetch_pm25(site_ids, mid, end)
    print(f"    !! giving up on station {site_ids[0]} {start:%Y-%m-%d}..{end:%Y-%m-%d} (logged as a gap)")
    return []


# ---------------------------------------------------------------- 1. stations
def get_sites():
    sites = pd.DataFrame(requests.get(f"{NSW}/get_SiteDetails", headers=HDR, timeout=60).json())
    sites = sites.rename(columns={"Site_Id": "site_id", "SiteName": "site_name", "Latitude": "latitude",
                                  "Longitude": "longitude", "Region": "region"})
    sites = sites.dropna(subset=["latitude", "longitude"])          # drops region summary rows
    sites = sites[~sites.region.astype(str).str.contains("Test", case=False)]   # 'Test Site' is not a real station
    b = C.BBOX
    in_box = sites.latitude.between(b["lat_min"], b["lat_max"]) & sites.longitude.between(b["lon_min"], b["lon_max"])
    return sites[in_box].reset_index(drop=True)


# ---------------------------------------------------------------- 2. PM2.5
def get_pm25(site_ids):
    """One request per month to start with; fetch_pm25 splits it further if the server chokes."""
    starts = pd.date_range(C.STUDY_START, C.STUDY_END, freq="MS")
    for s in starts:
        e = s + pd.offsets.MonthBegin(1)
        f = CHUNKS / f"pm25_{s:%Y-%m}.csv"
        if f.exists():
            continue
        print(f"  {s:%Y-%m}: requesting...")
        data = fetch_pm25(list(site_ids), s, e)
        if len(data) == 0:
            print(f"  {s:%Y-%m}: 0 rows")
            pd.DataFrame(columns=["site_id", "time", "pm25"]).to_csv(f, index=False)
            continue
        raw = pd.json_normalize(data)
        out = pd.DataFrame({
            "site_id": raw["Site_Id"],
            # 'Hour' 1..24 = the hour ENDING at that time -> convert to hour-START timestamp
            "time": pd.to_datetime(raw["Date"]) + pd.to_timedelta(raw["Hour"].astype(int) - 1, unit="h"),
            "pm25": pd.to_numeric(raw["Value"], errors="coerce"),
        })
        out.to_csv(f, index=False)
        print(f"  {s:%Y-%m}: {len(out):,} rows from {out.site_id.nunique()} stations")
        time.sleep(1)

    parts = [pd.read_csv(p) for p in sorted(CHUNKS.glob("pm25_*.csv"))]
    pm = pd.concat(parts, ignore_index=True).drop_duplicates(["site_id", "time"])
    if pm.empty:
        raise RuntimeError("No PM2.5 rows came back at all - paste the messages above to Claude.")
    return pm


# ---------------------------------------------------------------- 3. weather
def get_weather(sites):
    """ERA5 reanalysis = a best estimate of the weather that ACTUALLY happened (not a forecast).
    Open-Meteo's free tier has hourly limits, so on a 429 we wait and retry; finished stations are cached."""
    vars_ = ("temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m,precipitation,"
             "surface_pressure,boundary_layer_height")
    for _, p in sites.iterrows():
        f = CHUNKS / f"weather_{p.site_id}.csv"
        if f.exists():
            continue
        for attempt in range(6):
            r = requests.get("https://archive-api.open-meteo.com/v1/archive", params=dict(
                latitude=p.latitude, longitude=p.longitude, start_date=C.STUDY_START, end_date=C.STUDY_END,
                hourly=vars_, timezone="Australia/Brisbane"),      # Brisbane = AEST all year, matches NSW data
                timeout=300)
            if r.status_code == 200:
                break
            wait = 60 * (attempt + 1)
            print(f"  weather {p.site_name}: {r.status_code} {r.text[:150]} - waiting {wait}s")
            time.sleep(wait)
        r.raise_for_status()
        h = pd.DataFrame(r.json()["hourly"])
        h["site_id"] = p.site_id
        h.to_csv(f, index=False)
        print(f"  weather {p.site_name}: {len(h):,} hours")
        time.sleep(2)
    return pd.concat([pd.read_csv(f) for f in sorted(CHUNKS.glob("weather_*.csv"))], ignore_index=True)


if __name__ == "__main__":
    sites = get_sites()
    print(f"{len(sites)} stations in the {C.REGION} box:", ", ".join(sites.site_name))
    sites.to_csv(C.RAW / "sites.csv", index=False)

    print("\nDownloading hourly PM2.5 in 1-month chunks...")
    pm = get_pm25(sites.site_id)
    pm.to_csv(C.RAW / "pm25_hourly.csv", index=False)
    print(f"Saved pm25_hourly.csv: {len(pm):,} rows, {pm.site_id.nunique()} stations with PM2.5")

    # keep only stations that actually measure PM2.5 (many measure only ozone, NO2, etc.)
    sites = sites[sites.site_id.isin(pm.site_id.unique())]
    sites.to_csv(C.RAW / "sites.csv", index=False)

    print("\nDownloading weather per station...")
    get_weather(sites).to_csv(C.RAW / "weather_hourly.csv", index=False)
    print("Saved weather_hourly.csv. All done - next: python 02_prepare.py")