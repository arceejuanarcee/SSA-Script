#!/usr/bin/env python3
"""
Reentry Monitor GUI (TIP + TLE) — Latest TIP Only + Map Envelope + Kp Prediction + KML Export

Keeps your GUI layout (Fetch / Plot / Run Prediction / Export KML / Save Outputs).

Fixes:
- TIP timestamps with fractional seconds now parse correctly
- TIP window width = 0s (single decay epoch) auto-expands to ±DEFAULT_ZERO_WIDTH_MINUTES
  so MC + prediction can run

Adds:
- NOAA SWPC Planetary Kp fetch (3-hour intervals JSON)
- Kp + sequential TLE B* drag proxy bias the reentry time inside the TIP window
- Prediction outputs: P10/P50/P90 times and points
- KML export: corridor tracks (TIP_min/TIP_max/P50) + swath polygon + P10/P50/P90 points

Notes:
- This is still an operational proxy (like your earlier approach). It is NOT a full aerothermal breakup solver.
- Kp is used as a density/drag proxy to bias timing inside the TIP window.

Requirements:
  pip install requests skyfield matplotlib python-dotenv cartopy numpy simplekml
"""

from __future__ import annotations

import os
import csv
import json
import time
import random
import math
import datetime as dt
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any

import tkinter as tk
from tkinter import ttk, messagebox, filedialog

import requests
import numpy as np
from dotenv import load_dotenv

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from skyfield.api import EarthSatellite, load as sf_load

import cartopy.crs as ccrs
import cartopy.feature as cfeature

try:
    import simplekml
except Exception:
    simplekml = None


# -----------------------------
# Load env
# -----------------------------
load_dotenv()

LOGIN_URL = "https://www.space-track.org/ajaxauth/login"
DEFAULT_TIP_LIMIT = int(os.getenv("TIP_LIMIT", "200"))

# If latest TIP batch collapses to a single decay epoch -> expand to ± this many minutes
DEFAULT_ZERO_WIDTH_MINUTES = int(os.getenv("ZERO_WIDTH_FALLBACK_MIN", "30"))

PH_TZ = dt.timezone(dt.timedelta(hours=8))

# NOAA SWPC Kp products (3-hour planetary K index)
# Linked from the Planetary K-index product page. :contentReference[oaicite:1]{index=1}
NOAA_KP_JSON_PRIMARY = "https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json"
# Backup (sometimes present for 30-day indices; may not include the same schema)
NOAA_KP_JSON_BACKUP = "https://services.swpc.noaa.gov/products/kyoto-dst.json"  # not Kp; used only if primary down


# -----------------------------
# Models
# -----------------------------
@dataclass
class TipSolution:
    msg_epoch: str
    decay_epoch: str
    rev: Optional[int]
    direction: Optional[str]
    lat: Optional[float]
    lon: Optional[float]
    incl: Optional[float]
    high_interest: Optional[str]
    raw: dict


@dataclass
class TLEPoint:
    epoch_utc: dt.datetime
    name: str
    l1: str
    l2: str
    bstar: float


# -----------------------------
# Helpers
# -----------------------------
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def dt_to_iso_z(t: dt.datetime) -> str:
    t = t.astimezone(dt.timezone.utc)
    return t.strftime("%Y-%m-%d %H:%M:%SZ")

def dt_to_iso_ph(t: dt.datetime) -> str:
    t = t.astimezone(PH_TZ)
    return t.strftime("%Y-%m-%d %H:%M:%S (PH)")

def fmt_timedelta(td: dt.timedelta) -> str:
    total = int(abs(td.total_seconds()))
    sign = "-" if td.total_seconds() < 0 else ""
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    if h > 0:
        return f"{sign}{h}h {m}m {s}s"
    if m > 0:
        return f"{sign}{m}m {s}s"
    return f"{sign}{s}s"

def parse_datetime_flexible(s: str) -> dt.datetime:
    """
    Accepts:
      - 'YYYY-MM-DD HH:MM:SS'
      - 'YYYY-MM-DD HH:MM:SS.ssssss'
      - 'YYYY-MM-DDTHH:MM:SS'
      - 'YYYY-MM-DDTHH:MM:SS.ssssss'
      - with optional trailing Z
    Returns timezone-aware UTC datetime.
    """
    if not s:
        raise ValueError("Empty datetime string")
    s = s.strip()
    if s.endswith("Z"):
        s = s[:-1]

    # Common Space-Track TIP format: "YYYY-MM-DD HH:MM:SS"
    # But you encountered ISO with fractional seconds.
    # Try a few.
    fmts = [
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S",
    ]
    for f in fmts:
        try:
            return dt.datetime.strptime(s, f).replace(tzinfo=dt.timezone.utc)
        except Exception:
            pass

    # Last resort: fromisoformat
    try:
        x = dt.datetime.fromisoformat(s)
        if x.tzinfo is None:
            x = x.replace(tzinfo=dt.timezone.utc)
        return x.astimezone(dt.timezone.utc)
    except Exception as e:
        raise ValueError(f"Unparsable datetime: {s}") from e

def retry_get(session: requests.Session, url: str, tries: int = 6, timeout: int = 30) -> requests.Response:
    last_exc = None
    for i in range(tries):
        try:
            r = session.get(url, timeout=timeout)
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep((2 ** i) + random.random())
                continue
            r.raise_for_status()
            return r
        except Exception as e:
            last_exc = e
            time.sleep((2 ** i) + random.random())
    raise RuntimeError(f"GET failed after retries: {url}") from last_exc

def spacetrack_login(username: str, password: str) -> requests.Session:
    if not username or not password:
        raise RuntimeError("Missing SPACE_TRACK_USERNAME / SPACE_TRACK_PASSWORD (or GUI input).")
    s = requests.Session()
    r = s.post(LOGIN_URL, data={"identity": username, "password": password}, timeout=30)
    r.raise_for_status()
    return s

def fetch_tip(session: requests.Session, norad_id: int, tip_url_override: str = "", limit: int = DEFAULT_TIP_LIMIT) -> list:
    if tip_url_override.strip():
        url = tip_url_override.strip()
    else:
        url = (
            f"https://www.space-track.org/basicspacedata/query/class/tip/"
            f"NORAD_CAT_ID/{norad_id}/orderby/MSG_EPOCH%20desc/limit/{limit}/format/json"
        )
    r = retry_get(session, url)
    txt = r.text.strip()
    return r.json() if txt.startswith("[") else json.loads(txt)

def parse_tip_solutions(tip_json: list) -> List[TipSolution]:
    sols: List[TipSolution] = []
    for row in tip_json:
        sols.append(
            TipSolution(
                msg_epoch=(row.get("MSG_EPOCH") or "").strip(),
                decay_epoch=(row.get("DECAY_EPOCH") or "").strip(),
                rev=int(row["REV"]) if str(row.get("REV", "")).isdigit() else None,
                direction=row.get("DIRECTION"),
                lat=float(row["LAT"]) if row.get("LAT") not in (None, "") else None,
                lon=float(row["LON"]) if row.get("LON") not in (None, "") else None,
                incl=float(row["INCL"]) if row.get("INCL") not in (None, "") else None,
                high_interest=row.get("HIGH_INTEREST"),
                raw=row,
            )
        )

    def key(sol: TipSolution):
        try:
            return parse_datetime_flexible(sol.msg_epoch).timestamp()
        except Exception:
            return 0.0

    sols.sort(key=key, reverse=True)
    return sols

def select_latest_tip_batch(solutions: List[TipSolution]) -> List[TipSolution]:
    if not solutions:
        return []
    newest_msg = solutions[0].msg_epoch
    if not newest_msg:
        return solutions[:1]
    return [s for s in solutions if s.msg_epoch == newest_msg]

def compute_tip_window_from_batch(solutions_latest_batch: List[TipSolution]) -> Tuple[Optional[dt.datetime], Optional[dt.datetime], List[dt.datetime]]:
    decays: List[dt.datetime] = []
    for s in solutions_latest_batch:
        if s.decay_epoch:
            try:
                decays.append(parse_datetime_flexible(s.decay_epoch))
            except Exception:
                pass
    if not decays:
        return None, None, []
    return min(decays), max(decays), decays

def fetch_latest_tle(session: requests.Session, norad_id: int) -> Tuple[str, str, str]:
    tle_url = (
        f"https://www.space-track.org/basicspacedata/query/class/gp/"
        f"NORAD_CAT_ID/{norad_id}/orderby/EPOCH%20desc/limit/1/format/tle"
    )
    r = retry_get(session, tle_url)
    lines = [ln.strip() for ln in r.text.splitlines() if ln.strip()]
    if len(lines) < 2:
        raise RuntimeError("TLE fetch returned insufficient lines.")
    if lines[0].startswith("1 ") and lines[1].startswith("2 "):
        name = f"NORAD {norad_id}"
        l1, l2 = lines[0], lines[1]
    else:
        if len(lines) < 3:
            raise RuntimeError("TLE fetch returned insufficient lines (expected name+2 lines).")
        name = lines[0]
        l1, l2 = lines[1], lines[2]
    return name, l1, l2

def fetch_tle_history(session: requests.Session, norad_id: int, limit: int = 25) -> List[TLEPoint]:
    url = (
        f"https://www.space-track.org/basicspacedata/query/class/gp/"
        f"NORAD_CAT_ID/{norad_id}/orderby/EPOCH%20desc/limit/{int(limit)}/format/json"
    )
    r = retry_get(session, url)
    data = r.json()
    if not isinstance(data, list) or not data:
        return []

    out: List[TLEPoint] = []
    for row in data:
        name = row.get("OBJECT_NAME") or f"NORAD {norad_id}"
        l1 = row.get("TLE_LINE1")
        l2 = row.get("TLE_LINE2")
        epoch = row.get("EPOCH")
        if not (l1 and l2 and epoch):
            continue
        try:
            epoch_dt = parse_datetime_flexible(epoch)
        except Exception:
            continue
        bstar = parse_bstar_from_tle_line1(l1)
        out.append(TLEPoint(epoch_utc=epoch_dt, name=name, l1=l1.strip(), l2=l2.strip(), bstar=bstar))

    out.sort(key=lambda x: x.epoch_utc)  # oldest -> newest
    return out

def parse_bstar_from_tle_line1(l1: str) -> float:
    """
    Standard TLE B* format around columns 54-61 (8 chars) like ' 34123-4'
    meaning +0.34123E-4.
    """
    try:
        s = l1[53:61].strip()
        if len(s) < 7:
            return float("nan")
        mant = s[:-2]
        exp = s[-2:]
        sign = 1.0
        if mant[0] == "-":
            sign = -1.0
            mant = mant[1:]
        elif mant[0] == "+":
            mant = mant[1:]
        mantissa = sign * float(f"0.{mant}")
        exponent = int(exp)
        return mantissa * (10 ** exponent)
    except Exception:
        return float("nan")

def robust_bstar_stats(tles: List[TLEPoint]) -> Dict[str, float]:
    vals = np.array([x.bstar for x in tles if np.isfinite(x.bstar)], dtype=float)
    if len(vals) < 5:
        return {"bstar_med": float("nan"), "bstar_mad": float("nan"), "bstar_trend_per_day": float("nan")}

    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med))) + 1e-30

    n = min(15, len(tles))
    x0 = tles[-n].epoch_utc
    xs = np.array([(tles[-n+i].epoch_utc - x0).total_seconds() / 86400 for i in range(n)], dtype=float)
    ys = np.array([tles[-n+i].bstar for i in range(n)], dtype=float)
    try:
        m = float(np.polyfit(xs, ys, 1)[0])
    except Exception:
        m = float("nan")

    return {"bstar_med": med, "bstar_mad": mad, "bstar_trend_per_day": m}

def split_dateline_segments(lats: List[float], lons: List[float], jump_deg: float = 180.0):
    if not lats or not lons or len(lats) != len(lons):
        return []
    segs = []
    cur_lat = [lats[0]]
    cur_lon = [lons[0]]
    for i in range(1, len(lats)):
        if abs(lons[i] - lons[i-1]) > jump_deg:
            segs.append((cur_lat, cur_lon))
            cur_lat = [lats[i]]
            cur_lon = [lons[i]]
        else:
            cur_lat.append(lats[i])
            cur_lon.append(lons[i])
    segs.append((cur_lat, cur_lon))
    return segs

def groundtrack_corridor(
    sat: EarthSatellite,
    t_center: dt.datetime,
    minutes_before: int,
    minutes_after: int,
    step_seconds: int
) -> Tuple[List[float], List[float], List[dt.datetime]]:
    ts = sf_load.timescale()
    start = t_center - dt.timedelta(minutes=minutes_before)
    end = t_center + dt.timedelta(minutes=minutes_after)

    times_dt: List[dt.datetime] = []
    cur = start
    while cur <= end:
        times_dt.append(cur)
        cur += dt.timedelta(seconds=step_seconds)

    t_sf = ts.from_datetimes(times_dt)
    geoc = sat.at(t_sf)
    sub = geoc.subpoint()

    lats = list(sub.latitude.degrees)
    lons_raw = list(sub.longitude.degrees)
    lons = [((x + 180) % 360) - 180 for x in lons_raw]
    return lats, lons, times_dt

def subpoint_at_time(sat: EarthSatellite, t_utc: dt.datetime) -> Tuple[float, float]:
    ts = sf_load.timescale()
    t_sf = ts.from_datetime(t_utc)
    sub = sat.at(t_sf).subpoint()
    lat = float(sub.latitude.degrees)
    lon = float(sub.longitude.degrees)
    lon = ((lon + 180) % 360) - 180
    return lat, lon


# -----------------------------
# NOAA Kp
# -----------------------------
def fetch_noaa_kp_json() -> list:
    # No session needed (public)
    r = requests.get(NOAA_KP_JSON_PRIMARY, timeout=20)
    r.raise_for_status()
    return r.json()

def kp_max_last_24h(kp_rows: list, ref_utc: dt.datetime) -> float:
    """
    NOAA Kp JSON is typically a table-like list where first row is header.
    We'll parse rows that look like:
      ["time_tag","kp_index",...]
    and filter last 24 hours from ref_utc.
    """
    if not isinstance(kp_rows, list) or len(kp_rows) < 2:
        return float("nan")

    header = kp_rows[0]
    # Expected columns: time_tag, kp_index
    try:
        time_idx = header.index("time_tag")
        kp_idx = header.index("kp_index")
    except Exception:
        # Some variants use different header naming; try fallbacks
        time_idx = 0
        kp_idx = 1

    lo = ref_utc - dt.timedelta(hours=24)
    best = float("-inf")
    found = False

    for row in kp_rows[1:]:
        if not isinstance(row, list) or len(row) <= max(time_idx, kp_idx):
            continue
        try:
            t = parse_datetime_flexible(str(row[time_idx]))
            kp = float(row[kp_idx])
        except Exception:
            continue
        if lo <= t <= ref_utc:
            found = True
            if kp > best:
                best = kp

    return float(best) if found else float("nan")


# -----------------------------
# Prediction model (MC inside TIP window)
# -----------------------------
def mc_sample_time_inside_window(
    wmin: dt.datetime,
    wmax: dt.datetime,
    use_bstar: bool,
    bstar_stats: Dict[str, float],
    kp_value: float,
    n: int = 2000,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Samples reentry time INSIDE [wmin, wmax] using a beta distribution.

    Bias is derived from:
      - Sequential TLE B* (drag proxy): higher + rising -> earlier
      - Kp (geomagnetic activity proxy): higher -> earlier (more drag)
    """
    if seed is not None:
        np.random.seed(seed)

    width_s = max(1.0, (wmax - wmin).total_seconds())

    # Base mean at 0.5 (mid-window)
    mean = 0.5

    # --- Kp bias ---
    # Kp ranges 0-9. We'll map to a conservative shift up to ~±0.20 of the window.
    if np.isfinite(kp_value):
        # higher kp -> earlier => mean moves toward 0
        mean += float(np.clip(-(kp_value - 3.0) * 0.03, -0.20, 0.10))

    # --- B* bias ---
    if use_bstar:
        med = bstar_stats.get("bstar_med", float("nan"))
        mad = bstar_stats.get("bstar_mad", float("nan"))
        trend = bstar_stats.get("bstar_trend_per_day", float("nan"))

        if np.isfinite(med):
            # log magnitude proxy; higher magnitude -> earlier
            mean += float(np.clip(-0.06 * np.tanh(math.log10(abs(med) + 1e-15) + 7.0), -0.12, 0.12))
        if np.isfinite(trend):
            # increasing trend -> earlier
            mean += float(np.clip(-0.08 * np.tanh(trend * 5e4), -0.12, 0.12))

        # Spread control: more uncertainty (high MAD) -> flatter
        flatness = 0.55
        if np.isfinite(mad) and np.isfinite(med) and abs(med) > 0:
            rel = mad / (abs(med) + 1e-30)
            flatness = float(np.clip(0.35 + 2.0 * rel, 0.35, 0.95))
        k = (1.0 - flatness) * 18.0 + 2.0
    else:
        # Without bstar, keep a moderate concentration
        k = 10.0

    mean = float(np.clip(mean, 0.05, 0.95))
    a = mean * k
    b = (1.0 - mean) * k

    u = np.random.beta(a, b, size=int(n))
    ts = np.array([wmin.timestamp() + float(x) * width_s for x in u], dtype=float)
    return ts


# -----------------------------
# KML helpers
# -----------------------------
def _kml_color(aabbggrr: str) -> str:
    # simplekml expects aabbggrr; caller provides it.
    return aabbggrr

def export_kml_tracks_and_points(
    kml_path: str,
    tracks: List[Dict[str, Any]],
    points: List[Dict[str, Any]],
    swath_polygon: Optional[List[Tuple[float, float]]] = None
) -> None:
    if simplekml is None:
        raise RuntimeError("simplekml not installed. pip install simplekml")

    kml = simplekml.Kml()

    # Tracks
    for tr in tracks:
        ls = kml.newlinestring(name=tr["name"])
        ls.coords = list(zip(tr["lons"], tr["lats"]))
        ls.altitudemode = simplekml.AltitudeMode.clamptoground
        ls.extrude = 0
        if "color" in tr:
            ls.style.linestyle.color = _kml_color(tr["color"])
        if "width" in tr:
            ls.style.linestyle.width = int(tr["width"])

    # Points
    for p in points:
        pt = kml.newpoint(name=p["name"], coords=[(p["lon"], p["lat"])])
        if "description" in p:
            pt.description = p["description"]
        if "icon_href" in p:
            pt.style.iconstyle.icon.href = p["icon_href"]

    # Swath polygon (optional)
    if swath_polygon:
        pol = kml.newpolygon(name="Reentry Swath (± km)")
        pol.outerboundaryis = [(lon, lat) for (lon, lat) in swath_polygon]
        pol.style.polystyle.fill = 0  # outline only
        pol.style.linestyle.width = 2

    kml.save(kml_path)

def build_simple_swath_polygon(lats: List[float], lons: List[float], half_width_km: float) -> List[Tuple[float, float]]:
    """
    Builds a crude buffer polygon around a linestring by offsetting left/right in lat/lon degrees.

    This is an approximation suitable for quick GIS overlays.
    For better accuracy, you'd buffer in a projected CRS (e.g., EPSG:3857/3395) using shapely+pyproj.
    """
    if not lats or not lons or len(lats) < 2:
        return []

    # Convert km to degrees scale
    # 1 deg lat ~ 111 km; 1 deg lon ~ 111*cos(lat) km
    left = []
    right = []

    for i in range(len(lats)):
        lat = lats[i]
        lon = lons[i]

        # local heading using neighbor
        j0 = max(0, i - 1)
        j1 = min(len(lats) - 1, i + 1)
        dlat = lats[j1] - lats[j0]
        dlon = lons[j1] - lons[j0]
        # normal vector (perpendicular)
        nx, ny = -dlat, dlon
        norm = math.hypot(nx, ny) + 1e-12
        nx /= norm
        ny /= norm

        dlat_deg = (half_width_km / 111.0) * nx
        dlon_deg = (half_width_km / (111.0 * max(0.2, math.cos(math.radians(lat))))) * ny

        left.append(( ((lon + dlon_deg + 180) % 360) - 180, lat + dlat_deg ))
        right.append(( ((lon - dlon_deg + 180) % 360) - 180, lat - dlat_deg ))

    # polygon: left forward + right reversed + close
    poly = left + right[::-1] + [left[0]]
    return [(lon, lat) for (lon, lat) in poly]


# -----------------------------
# History CSV
# -----------------------------
def history_path(out_dir: str, norad_id: int) -> str:
    return os.path.join(out_dir, f"tip_history_{norad_id}.csv")

def read_last_history_row(csv_path: str) -> Optional[dict]:
    if not os.path.exists(csv_path):
        return None
    try:
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        return rows[-1] if rows else None
    except Exception:
        return None

def append_history_row(csv_path: str, row: dict) -> None:
    file_exists = os.path.exists(csv_path)
    fieldnames = list(row.keys())
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            w.writeheader()
        w.writerow(row)


# -----------------------------
# GUI
# -----------------------------
class ReentryGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Reentry Monitor (TIP + TLE) — Latest TIP Only + Map Envelope (Kp Prediction + KML)")
        self.geometry("1240x820")

        self.out_dir = os.getenv("OUT_DIR", "./reentry_out")
        ensure_dir(self.out_dir)

        # State
        self.tip_raw: Optional[list] = None
        self.solutions_all: List[TipSolution] = []
        self.solutions_latest: List[TipSolution] = []
        self.tle: Optional[Tuple[str, str, str]] = None
        self.tle_hist: List[TLEPoint] = []
        self.bstar_stats: Dict[str, float] = {}

        self.window_min: Optional[dt.datetime] = None
        self.window_max: Optional[dt.datetime] = None

        self.latest_sat: Optional[EarthSatellite] = None

        # Prediction results
        self.pred: Dict[str, Any] = {}

        # UI vars (match your GUI style)
        self.var_user = tk.StringVar(value=os.getenv("SPACE_TRACK_USERNAME", ""))
        self.var_pass = tk.StringVar(value=os.getenv("SPACE_TRACK_PASSWORD", ""))
        self.var_norad = tk.StringVar(value=os.getenv("NORAD_CAT_ID", "66877"))
        self.var_tip_url = tk.StringVar(value=os.getenv("TIP_URL", ""))

        self.var_before = tk.StringVar(value="90")
        self.var_after = tk.StringVar(value="90")
        self.var_step = tk.StringVar(value="30")
        self.var_mid_tracks = tk.StringVar(value="5")

        self.var_ph_focus = tk.BooleanVar(value=False)

        # Prediction knobs shown in your screenshot
        self.var_mc_samples = tk.StringVar(value="2000")
        self.var_swath_km = tk.StringVar(value="100")
        self.var_use_bstar = tk.BooleanVar(value=True)

        self._build_ui()
        self._build_plot()

    def _build_ui(self):
        top = ttk.Frame(self, padding=10)
        top.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(top, text="SPACE_TRACK_USERNAME").grid(row=0, column=0, sticky="w")
        ttk.Entry(top, textvariable=self.var_user, width=26).grid(row=0, column=1, padx=6)

        ttk.Label(top, text="SPACE_TRACK_PASSWORD").grid(row=0, column=2, sticky="w")
        ttk.Entry(top, textvariable=self.var_pass, width=26, show="*").grid(row=0, column=3, padx=6)

        ttk.Label(top, text="NORAD_CAT_ID").grid(row=0, column=4, sticky="w")
        ttk.Entry(top, textvariable=self.var_norad, width=10).grid(row=0, column=5, padx=6)

        ttk.Label(top, text="TIP_URL (optional override)").grid(row=1, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(top, textvariable=self.var_tip_url, width=120).grid(row=1, column=1, columnspan=5, sticky="we", pady=(8, 0))

        row2 = ttk.Frame(self, padding=(10, 0, 10, 6))
        row2.pack(side=tk.TOP, fill=tk.X)

        def add_field(label: str, var: tk.StringVar, w: int = 8):
            ttk.Label(row2, text=label).pack(side=tk.LEFT)
            ttk.Entry(row2, textvariable=var, width=w).pack(side=tk.LEFT, padx=(6, 14))

        add_field("Window Before (min)", self.var_before, 8)
        add_field("After (min)", self.var_after, 8)
        add_field("Step (sec)", self.var_step, 8)
        add_field("Intermediate tracks", self.var_mid_tracks, 10)

        ttk.Checkbutton(row2, text="Philippines Focus (zoom)", variable=self.var_ph_focus, command=self._apply_extent).pack(side=tk.LEFT, padx=(10, 0))

        row3 = ttk.Frame(self, padding=(10, 0, 10, 10))
        row3.pack(side=tk.TOP, fill=tk.X)

        add_field("MC samples", self.var_mc_samples, 8)
        add_field("Swath ±km (KML)", self.var_swath_km, 10)
        ttk.Checkbutton(row3, text="Use sequential TLE B* bias", variable=self.var_use_bstar).pack(side=tk.LEFT, padx=(6, 0))

        btns = ttk.Frame(self, padding=(10, 0, 10, 10))
        btns.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(btns, text="Fetch TIP + TLE (latest TIP only)", command=self.on_fetch).pack(side=tk.LEFT)
        ttk.Button(btns, text="Plot Envelope (latest TIP min–max)", command=self.on_plot_envelope).pack(side=tk.LEFT, padx=8)
        ttk.Button(btns, text="Run Kp-Based Prediction", command=self.on_run_prediction).pack(side=tk.LEFT, padx=8)
        ttk.Button(btns, text="Export KML (corridor + swath + points)", command=self.on_export_kml).pack(side=tk.LEFT, padx=8)
        ttk.Button(btns, text="Save Outputs…", command=self.on_save_outputs).pack(side=tk.LEFT, padx=8)

        self.status = tk.Text(self, height=10)
        self.status.pack(side=tk.BOTTOM, fill=tk.X)
        self._log("Ready. Fetch → Plot → Run Prediction → Export KML / Save Outputs.")

    def _build_plot(self):
        center = ttk.Frame(self, padding=10)
        center.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.fig = plt.Figure(figsize=(11.5, 5.8), dpi=100)
        self.ax = self.fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        self._setup_map()

        self.canvas = FigureCanvasTkAgg(self.fig, master=center)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.canvas.draw()

    def _setup_map(self):
        self.ax.clear()
        self.ax.set_global()
        self.ax.coastlines(linewidth=0.8)
        self.ax.add_feature(cfeature.BORDERS, linewidth=0.4)
        self.ax.add_feature(cfeature.LAND, alpha=0.2)
        self.ax.add_feature(cfeature.OCEAN, alpha=0.1)
        self.ax.gridlines(draw_labels=False, linewidth=0.4, alpha=0.5)
        self._apply_extent()

    def _apply_extent(self):
        if getattr(self, "ax", None) is None:
            return
        if self.var_ph_focus.get():
            self.ax.set_extent([115, 130, 4, 22], crs=ccrs.PlateCarree())
        else:
            self.ax.set_global()
        if getattr(self, "canvas", None) is not None:
            self.canvas.draw_idle()

    def _log(self, msg: str):
        ts = dt.datetime.now().strftime("%H:%M:%S")
        self.status.insert("end", f"[{ts}] {msg}\n")
        self.status.see("end")

    def _get_int(self, var: tk.StringVar, label: str) -> int:
        try:
            return int(var.get().strip())
        except Exception:
            raise ValueError(f"Invalid integer for {label}: {var.get()}")

    def _get_float(self, var: tk.StringVar, label: str) -> float:
        try:
            return float(var.get().strip())
        except Exception:
            raise ValueError(f"Invalid number for {label}: {var.get()}")

    def _plot_track(self, lats: List[float], lons: List[float], linewidth: float, alpha: float, linestyle: str = "-"):
        for seg_lat, seg_lon in split_dateline_segments(lats, lons):
            self.ax.plot(seg_lon, seg_lat, transform=ccrs.PlateCarree(),
                         linewidth=linewidth, alpha=alpha, linestyle=linestyle)

    def _normalize_tip_window(self, wmin: dt.datetime, wmax: dt.datetime) -> Tuple[dt.datetime, dt.datetime, str]:
        """
        If wmin==wmax (0 width), expand around the single epoch.
        """
        width = (wmax - wmin).total_seconds()
        if width >= 1.0:
            return wmin, wmax, ""
        pad = dt.timedelta(minutes=DEFAULT_ZERO_WIDTH_MINUTES)
        return (wmin - pad), (wmax + pad), f"TIP window width was 0s; expanded to ±{DEFAULT_ZERO_WIDTH_MINUTES} min."

    def on_fetch(self):
        try:
            user = self.var_user.get().strip()
            pw = self.var_pass.get().strip()
            norad = self._get_int(self.var_norad, "NORAD_CAT_ID")
            tip_url_override = self.var_tip_url.get().strip()

            self._log("Logging in to Space-Track…")
            session = spacetrack_login(user, pw)

            self._log(f"Fetching TIP (limit={DEFAULT_TIP_LIMIT})…")
            self.tip_raw = fetch_tip(session, norad, tip_url_override, limit=DEFAULT_TIP_LIMIT)
            self.solutions_all = parse_tip_solutions(self.tip_raw)
            self.solutions_latest = select_latest_tip_batch(self.solutions_all)

            self._log("Fetching latest TLE…")
            self.tle = fetch_latest_tle(session, norad)

            # sequential TLEs for B* (optional)
            self.tle_hist = fetch_tle_history(session, norad, limit=25)
            self.bstar_stats = robust_bstar_stats(self.tle_hist) if self.tle_hist else {"bstar_med": float("nan"), "bstar_mad": float("nan"), "bstar_trend_per_day": float("nan")}

            wmin, wmax, decays = compute_tip_window_from_batch(self.solutions_latest)
            if not (wmin and wmax):
                raise RuntimeError("Latest TIP batch has no usable DECAY_EPOCH values.")

            wmin2, wmax2, note = self._normalize_tip_window(wmin, wmax)
            self.window_min, self.window_max = wmin2, wmax2

            name, l1, l2 = self.tle
            ts = sf_load.timescale()
            self.latest_sat = EarthSatellite(l1, l2, name, ts)

            latest_msg_epoch = self.solutions_latest[0].msg_epoch if self.solutions_latest else ""
            width = (self.window_max - self.window_min)

            self._log(f"Latest TIP MSG_EPOCH used: {latest_msg_epoch} (batch rows: {len(self.solutions_latest)})")
            self._log(f"Decay window: {dt_to_iso_z(self.window_min)} to {dt_to_iso_z(self.window_max)} (width {fmt_timedelta(width)})")
            if note:
                self._log(f"NOTE: {note}")

            if self.tle_hist:
                self._log(
                    f"TLE hist loaded: {len(self.tle_hist)} | "
                    f"B* median={self.bstar_stats.get('bstar_med', float('nan')):.3e} "
                    f"MAD={self.bstar_stats.get('bstar_mad', float('nan')):.3e} "
                    f"trend/day={self.bstar_stats.get('bstar_trend_per_day', float('nan')):.3e}"
                )

            # history log
            hist_csv = history_path(self.out_dir, norad)
            now_utc = dt.datetime.now(dt.timezone.utc)
            row = {
                "run_utc": dt_to_iso_z(now_utc),
                "norad_id": str(norad),
                "tip_msg_epoch_used": latest_msg_epoch,
                "latest_batch_count": str(len(self.solutions_latest)),
                "window_start_utc": dt_to_iso_z(self.window_min),
                "window_end_utc": dt_to_iso_z(self.window_max),
                "window_width_sec": str(int((self.window_max - self.window_min).total_seconds())),
                "tip_total_rows_fetched": str(len(self.solutions_all)),
                "tip_limit": str(DEFAULT_TIP_LIMIT),
            }
            append_history_row(hist_csv, row)
            self._log(f"Saved history: {hist_csv}")

        except Exception as e:
            messagebox.showerror("Fetch error", str(e))
            self._log(f"ERROR: {e}")

    def on_plot_envelope(self):
        try:
            if not (self.latest_sat and self.window_min and self.window_max):
                raise RuntimeError("Fetch TIP + TLE first.")

            before_min = self._get_int(self.var_before, "Window Before (min)")
            after_min = self._get_int(self.var_after, "After (min)")
            step_s = self._get_int(self.var_step, "Step (sec)")
            mid_tracks = max(0, self._get_int(self.var_mid_tracks, "Intermediate tracks"))

            wmin = self.window_min
            wmax = self.window_max
            width = wmax - wmin
            wmid = wmin + dt.timedelta(seconds=width.total_seconds() / 2)

            latest_msg_epoch = self.solutions_latest[0].msg_epoch if self.solutions_latest else ""

            self._setup_map()

            # faint intermediate tracks
            if mid_tracks > 0 and width.total_seconds() > 0:
                for i in range(1, mid_tracks + 1):
                    frac = i / (mid_tracks + 1)
                    t_i = wmin + dt.timedelta(seconds=width.total_seconds() * frac)
                    lats_i, lons_i, _ = groundtrack_corridor(self.latest_sat, t_i, before_min, after_min, step_s)
                    self._plot_track(lats_i, lons_i, linewidth=0.9, alpha=0.22, linestyle="--")

            # min/max boundaries (green like your screenshot)
            lats_min, lons_min, _ = groundtrack_corridor(self.latest_sat, wmin, before_min, after_min, step_s)
            lats_max, lons_max, _ = groundtrack_corridor(self.latest_sat, wmax, before_min, after_min, step_s)
            self._plot_track(lats_min, lons_min, linewidth=2.0, alpha=0.90, linestyle="-")
            self._plot_track(lats_max, lons_max, linewidth=2.0, alpha=0.90, linestyle="-")

            self.ax.set_title(
                f"{self.latest_sat.name} — Latest TIP MSG_EPOCH: {latest_msg_epoch}\n"
                f"Decay window: {dt_to_iso_z(wmin)} to {dt_to_iso_z(wmax)} (width {fmt_timedelta(width)})"
            )
            self.canvas.draw()
            self._log("Envelope plotted (latest TIP only): min/max boundaries + intermediate tracks.")

        except Exception as e:
            messagebox.showerror("Plot error", str(e))
            self._log(f"ERROR: {e}")

    def on_run_prediction(self):
        try:
            if not (self.latest_sat and self.window_min and self.window_max):
                raise RuntimeError("Fetch TIP + TLE first.")

            wmin = self.window_min
            wmax = self.window_max
            width_s = (wmax - wmin).total_seconds()
            if width_s < 1.0:
                raise RuntimeError("Invalid TIP window width (still too small after fallback).")

            mc_n = max(200, min(20000, self._get_int(self.var_mc_samples, "MC samples")))
            use_bstar = bool(self.var_use_bstar.get())

            # NOAA Kp around the latest TLE epoch (best we can do operationally)
            kp_val = float("nan")
            if self.tle_hist:
                ref = self.tle_hist[-1].epoch_utc
            else:
                ref = dt.datetime.now(dt.timezone.utc)

            self._log("Fetching NOAA Kp (last 24h max)…")
            try:
                kp_rows = fetch_noaa_kp_json()
                kp_val = kp_max_last_24h(kp_rows, ref)
            except Exception as e:
                kp_val = float("nan")
                self._log(f"WARNING: NOAA Kp fetch failed ({e}). Continuing without Kp bias.")

            if np.isfinite(kp_val):
                self._log(f"NOAA Kp max last 24h (ref={dt_to_iso_z(ref)}): {kp_val:.1f}")
            else:
                self._log("NOAA Kp: unavailable (no bias applied).")

            # Monte Carlo time sampling inside the TIP window
            ts_samples = mc_sample_time_inside_window(
                wmin, wmax,
                use_bstar=use_bstar,
                bstar_stats=self.bstar_stats,
                kp_value=kp_val,
                n=mc_n
            )

            # percentiles
            p10 = dt.datetime.fromtimestamp(float(np.percentile(ts_samples, 10)), tz=dt.timezone.utc)
            p50 = dt.datetime.fromtimestamp(float(np.percentile(ts_samples, 50)), tz=dt.timezone.utc)
            p90 = dt.datetime.fromtimestamp(float(np.percentile(ts_samples, 90)), tz=dt.timezone.utc)

            lat10, lon10 = subpoint_at_time(self.latest_sat, p10)
            lat50, lon50 = subpoint_at_time(self.latest_sat, p50)
            lat90, lon90 = subpoint_at_time(self.latest_sat, p90)

            # store prediction
            self.pred = {
                "generated_utc": dt_to_iso_z(dt.datetime.now(dt.timezone.utc)),
                "norad_id": int(self.var_norad.get().strip()),
                "tip_window": {"start_utc": dt_to_iso_z(wmin), "end_utc": dt_to_iso_z(wmax), "width_sec": int(width_s)},
                "kp_max_last_24h": kp_val if np.isfinite(kp_val) else None,
                "use_bstar_bias": bool(use_bstar),
                "bstar_stats": self.bstar_stats if use_bstar else None,
                "mc_samples": int(mc_n),
                "p10": {"t_utc": dt_to_iso_z(p10), "lat": lat10, "lon": lon10},
                "p50": {"t_utc": dt_to_iso_z(p50), "lat": lat50, "lon": lon50},
                "p90": {"t_utc": dt_to_iso_z(p90), "lat": lat90, "lon": lon90},
            }

            # plot: corridor + predicted points (P50 as “red dot” equivalent)
            self.on_plot_envelope()
            self.ax.plot([lon50], [lat50], marker="o", markersize=8, transform=ccrs.PlateCarree())
            self.ax.plot([lon10], [lat10], marker="o", markersize=6, transform=ccrs.PlateCarree())
            self.ax.plot([lon90], [lat90], marker="o", markersize=6, transform=ccrs.PlateCarree())
            self.ax.text(lon50 + 2, lat50 + 2, "P50 (best est.)", transform=ccrs.PlateCarree(), fontsize=9)
            self.ax.text(lon10 + 2, lat10 - 2, "P10", transform=ccrs.PlateCarree(), fontsize=9)
            self.ax.text(lon90 + 2, lat90 - 2, "P90", transform=ccrs.PlateCarree(), fontsize=9)
            self.canvas.draw()

            self._log(f"P50 predicted time: {dt_to_iso_z(p50)} | {dt_to_iso_ph(p50)}")
            self._log(f"P50 predicted point: lat={lat50:.2f}, lon={lon50:.2f}")

        except Exception as e:
            messagebox.showerror("Prediction error", str(e))
            self._log(f"ERROR: {e}")

    def on_export_kml(self):
        try:
            if not (self.latest_sat and self.window_min and self.window_max):
                raise RuntimeError("Fetch TIP + TLE first.")
            if not self.pred:
                raise RuntimeError("Run Kp-Based Prediction first.")
            if simplekml is None:
                raise RuntimeError("simplekml not installed. pip install simplekml")

            folder = filedialog.askdirectory(title="Select folder to save KML")
            if not folder:
                return

            before_min = self._get_int(self.var_before, "Window Before (min)")
            after_min = self._get_int(self.var_after, "After (min)")
            step_s = self._get_int(self.var_step, "Step (sec)")
            swath_km = max(0.0, self._get_float(self.var_swath_km, "Swath ±km (KML)"))

            norad = int(self.var_norad.get().strip())
            stamp = dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")

            wmin = self.window_min
            wmax = self.window_max
            p50 = parse_datetime_flexible(self.pred["p50"]["t_utc"])

            # tracks
            tracks = []
            for nm, tt, color, width in [
                ("TIP_min", wmin, "ff00ff00", 3),  # green
                ("TIP_max", wmax, "ff00ff00", 3),  # green
                ("P50_track", p50, "ff00a0ff", 4),  # orange-ish in KML ABGR
            ]:
                lats, lons, _ = groundtrack_corridor(self.latest_sat, tt, before_min, after_min, step_s)
                tracks.append({"name": nm, "lats": lats, "lons": lons, "color": color, "width": width})

            # swath polygon around P50 track (optional)
            swath_poly = None
            if swath_km > 0.0:
                lats50, lons50, _ = groundtrack_corridor(self.latest_sat, p50, before_min, after_min, step_s)
                swath_poly = build_simple_swath_polygon(lats50, lons50, swath_km)

            # points
            pts = []
            for tag in ["p10", "p50", "p90"]:
                pts.append({
                    "name": tag.upper(),
                    "lat": float(self.pred[tag]["lat"]),
                    "lon": float(self.pred[tag]["lon"]),
                    "description": f"time={self.pred[tag]['t_utc']}"
                })

            kml_path = os.path.join(folder, f"reentry_corridor_pred_{norad}_{stamp}.kml")
            export_kml_tracks_and_points(kml_path, tracks, pts, swath_polygon=swath_poly)

            self._log(f"Saved KML: {kml_path}")
            messagebox.showinfo("KML Exported", "KML exported successfully.")

        except Exception as e:
            messagebox.showerror("KML export error", str(e))
            self._log(f"ERROR: {e}")

    def on_save_outputs(self):
        try:
            folder = filedialog.askdirectory(title="Select folder to save outputs")
            if not folder:
                return

            norad = int(self.var_norad.get().strip())
            stamp = dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")

            # Save raw TIP
            if self.tip_raw is not None:
                tip_path = os.path.join(folder, f"tip_raw_{norad}_{stamp}.json")
                with open(tip_path, "w", encoding="utf-8") as f:
                    json.dump(self.tip_raw, f, indent=2)
                self._log(f"Saved: {tip_path}")

            # Save latest batch
            if self.solutions_latest:
                latest_path = os.path.join(folder, f"tip_latest_batch_{norad}_{stamp}.json")
                with open(latest_path, "w", encoding="utf-8") as f:
                    json.dump([s.raw for s in self.solutions_latest], f, indent=2)
                self._log(f"Saved: {latest_path}")

            # Save TLE used
            if self.tle is not None:
                name, l1, l2 = self.tle
                tle_path = os.path.join(folder, f"tle_{norad}_{stamp}.txt")
                with open(tle_path, "w", encoding="utf-8") as f:
                    f.write(name + "\n" + l1 + "\n" + l2 + "\n")
                self._log(f"Saved: {tle_path}")

            # Save prediction JSON
            if self.pred:
                pred_path = os.path.join(folder, f"prediction_{norad}_{stamp}.json")
                with open(pred_path, "w", encoding="utf-8") as f:
                    json.dump(self.pred, f, indent=2)
                self._log(f"Saved: {pred_path}")

            # Save plot PNG
            png_path = os.path.join(folder, f"map_{norad}_{stamp}.png")
            self.fig.savefig(png_path, dpi=220)
            self._log(f"Saved: {png_path}")

            # Copy history CSV
            hist = history_path(self.out_dir, norad)
            if os.path.exists(hist):
                hist_copy = os.path.join(folder, f"tip_history_{norad}.csv")
                with open(hist, "r", encoding="utf-8") as src, open(hist_copy, "w", encoding="utf-8") as dst:
                    dst.write(src.read())
                self._log(f"Saved: {hist_copy}")

            messagebox.showinfo("Saved", "Outputs saved successfully.")

        except Exception as e:
            messagebox.showerror("Save error", str(e))
            self._log(f"ERROR: {e}")


def main():
    app = ReentryGUI()
    app.mainloop()


if __name__ == "__main__":
    main()
