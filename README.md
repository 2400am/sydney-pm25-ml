# Sydney PM2.5: spatial estimation + next-day WHO exceedance

Starter pipeline matching the revised method. Every design choice the marker asked for
(dates, cutoff, spatial unit, thresholds, success criteria) lives in `config.py`.

## Run order
```
pip install -r requirements.txt
python 01_download.py        # real data: NSW DCCEEW Air Quality API + Open-Meteo ERA5 archive
#   (or: python 00_make_synthetic.py   to test the pipeline offline - results meaningless)
python 02_prepare.py         # QC, daily means (>=18 valid hours), prints station/day counts + exceedance rates
python test_no_leakage.py    # must print PASS before you trust any forecast result
python 03_forecast.py        # Objective 1 experiments E1-E4 + oracle -> outputs/forecast_results.csv
python 04_spatial.py         # Objective 2 schemes S1-S3 -> outputs/spatial_results.csv (+ error by distance, grid AoA)
```
Runtime on a laptop: about 5 minutes total for 15 stations x 9 years.

## Week-1 checks with real data
1. `get_ParameterDetails`: confirm the PM2.5 code is `"PM2.5"` and that `Hour` is hour-ENDING (we convert to hour-start).
2. Confirm timestamps are AEST (no daylight saving). Weather is pulled in `Australia/Brisbane` (AEST all year) to match.
3. Paste the counts printed by `02_prepare.py` into the report's data section (replacing our estimates).
4. PurpleAir go/no-go (see the doc): only add low-cost sensors if the criteria are met.

## What each experiment answers
| ID | Question | Holdout |
|---|---|---|
| E1 | Forecast at a monitored station? | time (test 2024) |
| E2 | Forecast at an unmonitored place? Inputs = IDW of other stations' last-24 h PM at 16:00 | cluster of stations AND time |
| E3 | Does it cope with an unseen extreme? | train without Jul 2019-Jun 2020, test on fire period |
| E4 | Does including fire data hurt typical days? | 2024, trained with vs without fire season |
| Oracle | Ceiling if weather forecasts were perfect | E1 + observed next-day weather (not a real-time result) |
| S1 | Spatial estimate at an omitted station | leave-one-station-out, all years, fire/non-fire strata |
| S2 | ... at an omitted station in a future year | LOSO AND time |
| S3 | ... further from any station | leave-cluster-out AND time |

Success = pre-registered minimum gain with a 95% block-bootstrap CI that excludes zero
(`meets_criterion` column). Weekly blocks for forecasts, station blocks for spatial.

## Design notes
- Spatial model predicts a log-ratio correction to IDW (trees can't extrapolate levels on smoke days).
- Raw lat/lon are excluded: with ~20 stations they let the model memorise station identity. Add
  generalisable static features (road density, land use, elevation, distance to coast) in `site_features`.
- Prediction intervals: station-grouped split-conformal (80%). Check `PI80_coverage` is ~0.8.
- `grid_area_of_applicability.csv` flags 1 km cells further from a station than anything validated -
  grey these out on any map rather than showing an estimate.
