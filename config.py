"""Single source of truth for every design decision the feedback asked us to pin down.
Change values here, not inside the scripts, so the report and the code never disagree."""
from pathlib import Path

DATA = Path(__file__).parent / "data"
RAW, PROC, OUT = DATA / "raw", DATA / "processed", Path(__file__).parent / "outputs"
for p in (RAW, PROC, OUT):
    p.mkdir(parents=True, exist_ok=True)

# ---- Study region and dates ------------------------------------------------
REGION = "Greater Sydney"          # NSW DCCEEW regions: Sydney East / North-west / South-west
BBOX = dict(lat_min=-34.35, lat_max=-33.45, lon_min=150.55, lon_max=151.35)
STUDY_START, STUDY_END = "2016-01-01", "2024-12-31"
TRAIN_END = "2022-12-31"           # train: 2016-2022
VAL_YEAR = "2023"                  # tuning / threshold choice only
TEST_YEAR = "2024"                 # touched ONCE, at the end

# Exceptional smoke period, reported as its own stratum (never mixed silently).
FIRE_START, FIRE_END = "2019-10-01", "2020-02-29"

# ---- Times ------------------------------------------------------------------
# All timestamps are AEST (UTC+10, no daylight saving), hour-START convention.
# NSW API 'Hour' = 1..24 for the hour ENDING at that time -> start = Hour-1. Verify in week 1.
TZ = "Etc/GMT-10"
FORECAST_CUTOFF_HOUR = 16          # forecast issued 16:00 AEST on day D using hours 00-15 of D and earlier
# Target: calendar-day (00:00-23:59 AEST) mean PM2.5 on D+1.

# ---- Targets ----------------------------------------------------------------
WHO_24H = 15.0                     # WHO 2021 24-h PM2.5 guideline (ug/m3)
NEPM_24H = 25.0                    # Australian NEPM standard, reported as a secondary threshold
MIN_HOURS_PER_DAY = 18             # 75% completeness rule for a valid daily mean
MIN_VALID_DAYS = 365               # drop stations with less than a year of valid days
MAX_PLAUSIBLE_HOURLY = 1500        # ug/m3; bushfire smoke can exceed 1000 for an hour, instruments rarely beyond this

# ---- Spatial unit ----------------------------------------------------------
GRID_KM = 1.0                      # prediction surface: 1 km grid cells (point prediction at cell centre)
N_SPATIAL_CLUSTERS = 5             # leave-cluster-out: harder than leave-one-station-out

# ---- Pre-registered success criteria ---------------------------------------
MIN_RMSE_REDUCTION = 0.10          # spatial: >=10% RMSE reduction vs best baseline, 95% CI excludes 0
MIN_PRAUC_GAIN = 0.05              # forecast: PR-AUC gain >=0.05 vs best baseline, 95% CI excludes 0
N_BOOT = 1000
SEED = 42
