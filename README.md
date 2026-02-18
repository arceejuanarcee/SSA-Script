# Reentry Monitor (TIP + TLE) GUI — Latest TIP Only + Map Envelope (Kp + pymsis + KML)

This project is a Windows-friendly Tkinter GUI that:
- Logs in to Space-Track
- Fetches the latest TIP batch (by MSG_EPOCH) for a given NORAD ID
- Fetches the latest TLE (and optionally a short TLE history for B* statistics)
- Builds a decay window from the latest TIP batch
- Plots min/max envelope tracks and intermediate tracks on a world map
- Runs a "proxy" reentry prediction using:
  - NOAA Kp (from SWPC JSON feed) to create an Ap proxy
  - pymsis (MSIS) to estimate atmospheric density near 120 km
  - A simplified descent time and downrange travel model
  - Earth rotation during descent
  - Monte Carlo sampling across the decay window to produce P10/P50/P90 points

Important: This is not an operational-grade reentry solution like EU-SST. It is a best-effort proxy model using public TLE propagation plus simple physics approximations.

---

## What "Latest TIP Only" means

Space-Track TIP messages can contain multiple solutions per message. Your GUI intentionally selects:
- The newest MSG_EPOCH (latest message)
- All rows within that newest MSG_EPOCH batch

The decay window is computed from the DECAY_EPOCH values in that batch.

---

## Fix included: "Invalid TIP window width (start == end)"

Sometimes the latest TIP batch contains:
- Only one DECAY_EPOCH, or
- Multiple rows but all with the same DECAY_EPOCH

That produces a zero-width window (start == end), which breaks Monte Carlo sampling.

This script fixes it by expanding the window when width is 0 seconds, using:
1. A TIP uncertainty field, if present (best-effort scan of common keys), else
2. A GUI fallback uncertainty (minutes), default 48 minutes

The log shows which mode was used:
- tip_spread
- tip_uncertainty
- fallback_uncertainty

---

## Files

- reentry_prediction_v2.py (or your chosen filename): the Tkinter GUI application
- Output folder (default): ./reentry_out
  - tip_history_<NORAD>.csv: run history for each fetch

---

## Requirements

### Python
- Python 3.10+ recommended (3.11 is fine)

### Python packages

Install with:

pip install requests skyfield matplotlib python-dotenv cartopy numpy simplekml pymsis

Notes:
- cartopy can be the hardest dependency on Windows. If you have trouble, install via Conda:
  conda install -c conda-forge cartopy
- simplekml is only needed if you enable KML export features.
- pymsis is required for the density-based portion of the proxy prediction.

---

## Environment variables (.env)

Create a .env file in the same folder as the script:

SPACE_TRACK_USERNAME=your_email_or_username
SPACE_TRACK_PASSWORD=your_password

Optional defaults:
NORAD_CAT_ID=66877
TIP_LIMIT=200
OUT_DIR=./reentry_out

The GUI fields will pre-fill from .env if present.

---

## Running the app

From the script directory:

python reentry_prediction_v2.py

---

## GUI controls (high level)

Top fields:
- SPACE_TRACK_USERNAME, SPACE_TRACK_PASSWORD: Space-Track login
- NORAD_CAT_ID: object ID to query (example: 66877)
- TIP_URL (optional override): only if you want to force a custom TIP query URL

Window sampling and plotting:
- Window Before (min), After (min), Step (sec): how much of the orbit track to draw around the selected epoch
- Intermediate tracks: number of additional dashed tracks between min and max epochs
- Philippines Focus (zoom): zoom map to PH region
- MC samples: Monte Carlo samples for P10/P50/P90 timing selection
- Swath ±km (KML): swath half-width used for KML visualization (if you enable export)
- Use sequential TLE B* bias: fetch a short TLE history and compute robust B* statistics to bias the sampling

Drivers:
- F10.7, F10.7a: used as fallback solar drivers for pymsis
- Fallback uncertainty (min): used to expand TIP window if start == end (default 48)

Buttons:
- Fetch TIP + TLE (latest TIP only): fetch data and compute decay window
- Plot Envelope (latest TIP min–max): plots min/max and intermediate tracks
- Run Kp+MSIS Prediction: runs Monte Carlo selection and proxy impact computation, plots P10/P50/P90
- Export KML ...: optional, if you included full KML functions
- Save Outputs...: optional, if you included full save functions

---

## How the "proxy" prediction works

This model does not do full orbital decay integration. It approximates:

1. Select a time within the TIP window:
   - Monte Carlo samples across the TIP window
   - Optional bias from:
     - B* median/trend (from sequential TLE history)
     - Kp mean around window midpoint

2. Compute the satellite subpoint at that time:
   - SGP4 propagation from TLE
   - Subpoint latitude/longitude from Skyfield

3. Estimate descent time and downrange travel:
   - Query pymsis at 120 km to estimate density
   - Convert density into a descent time proxy (seconds)
   - Use a crude average velocity model to estimate downrange distance
   - Move along the ground-track bearing by that distance

4. Apply Earth rotation during descent:
   - Shift longitude by Earth's sidereal rotation rate over the descent time

Outputs:
- P10 / P50 / P90 times (UTC)
- P10 / P50 / P90 impact proxy lat/lon
- Density, descent time, downrange km, Earth rotation shift

---

## Why this will not match EU-SST perfectly

EU-SST has access to:
- Radar-fused orbit determination (OD)
- Full drag integration (NRLMSISE/JB with history of Ap/F10.7)
- Attitude / breakup modeling assumptions
- Better uncertainty characterization and assimilation

This script uses:
- Public TLE + SGP4
- A simplified impact proxy approach
- Limited drivers (Kp mean, optional B* bias)
- No true breakup modeling

So your result is best treated as a rough proxy, not an operational impact point.

---

## Troubleshooting

### 1) Fetch error: time data ... does not match format "%Y-%m-%d %H:%M:%S"
Cause:
- The code used strict strptime parsing while the string includes a T separator, a Z, or fractional seconds.

Fix:
- Ensure the script uses a tolerant datetime parser (for example, parse_any_datetime_utc()) that accepts:
  - YYYY-MM-DDTHH:MM:SSZ
  - YYYY-MM-DDTHH:MM:SS.sssZ
  - YYYY-MM-DD HH:MM:SS
  - YYYY-MM-DD HH:MM:SS.sss

### 2) Prediction error: Invalid TIP window width (start == end)
Cause:
- Latest TIP batch has only one DECAY_EPOCH (or all identical), producing width 0.

Fix:
- Increase Fallback uncertainty (min) if needed and click Fetch again.
- Check log for the window expansion mode.

### 3) pymsis not installed / import error
Fix:
pip install pymsis

### 4) cartopy installation errors on Windows
Fix options:
- Use Conda:
  conda install -c conda-forge cartopy
- Or ensure a compatible wheel environment.

### 5) Space-Track returns 401/403 or no data
Fix:
- Confirm credentials
- Confirm your Space-Track account access
- Ensure you are not exceeding rate limits

---

## Notes on Space-Track usage

Be mindful of Space-Track terms and rate limits. Avoid polling too frequently. Use reasonable schedules (for example, every 15–30 minutes during active events, otherwise less).

---

## License / Disclaimer

This tool is for internal analysis and educational use. It is not a certified reentry prediction system and should not be used as the sole basis for public safety decisions.
