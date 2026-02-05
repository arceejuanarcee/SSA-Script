#!/usr/bin/env python3
"""
Reentry Monitor (TIP + TLE) — Latest TIP Only + Map Envelope + Refined Prediction (TIP LAT/LON + Kp-nearest + MeanMotion trend) + KML

Key refinement to get closer to EU-SST-like "red dot":
- Use Space-Track TIP-provided LAT/LON for the predicted decay point whenever available (DEFAULT ON).
  This is typically far closer to authoritative products than computing a subpoint from a single TLE.

Other refinements:
- Use NOAA Planetary Kp at the time closest to the predicted epoch (not last-24h max).
- Add sequential TLE mean-motion trend as a decay proxy (stronger than B* alone).
- If TIP window width is 0 seconds, auto-expand for MC sampling but keep center anchored.

DISCLAIMER:
- This is still an operational proxy. True debris footprint requires high-fidelity reentry/breakup/winds.
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
# Config
# -----------------------------
load_dotenv()

LOGIN_URL = "https://www.space-track.org/ajaxauth/login"
DEFAULT_TIP_LIMIT = int(os.getenv("TIP_LIMIT", "200"))
PH_TZ = dt.timezone(dt.timedelta(hours=8))

# When latest TIP batch collapses to a single decay epoch
DEFAULT_ZERO_WIDTH_MINUTES = int(os.getenv("ZERO_WIDTH_FALLBACK_MIN", "30"))

# NOAA SWPC Planetary K-index JSON
NOAA_KP_JSON = "https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json"


# -----------------------------
# Models
# -----------------------------
@dataclass
class TipSolution:
    msg_epoch: str
    decay_epoch: str
    rev: Optional[int]
    lat: Optional[float]
    lon: Optional[float]
    raw: dict


@dataclass
class TLEPoint:
    epoch_utc: dt.datetime
    name: str
    l1: str
    l2: str
    bstar: float
    mean_motion_rev_per_day: float


# -----------------------------
# Utils
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
    Returns tz-aware UTC datetime.
    """
    if not s:
        raise ValueError("Empty datetime string")
    s = s.strip()
    if s.endswith("Z"):
        s = s[:-1]

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

    # ISO fallback
    x = dt.datetime.fromisoformat(s)
    if x.tzinfo is None:
        x = x.replace(tzinfo=dt.timezone.utc)
    return x.astimezone(dt.timezone.utc)

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
        raise RuntimeError("Missing Space-Track username/password.")
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
    out: List[TipSolution] = []
    for row in tip_json:
        out.append(
            TipSolution(
                msg_epoch=(row.get("MSG_EPOCH") or "").strip(),
                decay_epoch=(row.get("DECAY_EPOCH") or "").strip(),
                rev=int(row["REV"]) if str(row.get("REV", "")).isdigit() else None,
                lat=float(row["LAT"]) if row.get("LAT") not in (None, "") else None,
                lon=float(row["LON"]) if row.get("LON") not in (None, "") else None,
                raw=row,
            )
        )

    out.sort(key=lambda s: parse_datetime_flexible(s.msg_epoch).timestamp() if s.msg_epoch else 0.0, reverse=True)
    return out

def select_latest_tip_batch(solutions: List[TipSolution]) -> List[TipSolution]:
    if not solutions:
        return []
    newest = solutions[0].msg_epoch
    if not newest:
        return solutions[:1]
    return [s for s in solutions if s.msg_epoch == newest]

def compute_tip_window(latest_batch: List[TipSolution]) -> Tuple[Optional[dt.datetime], Optional[dt.datetime], List[dt.datetime]]:
    decays: List[dt.datetime] = []
    for s in latest_batch:
        if s.decay_epoch:
            try:
                decays.append(parse_datetime_flexible(s.decay_epoch))
            except Exception:
                pass
    if not decays:
        return None, None, []
    return min(decays), max(decays), decays

def normalize_tip_window(wmin: dt.datetime, wmax: dt.datetime) -> Tuple[dt.datetime, dt.datetime, str]:
    width = (wmax - wmin).total_seconds()
    if width >= 1.0:
        return wmin, wmax, ""
    pad = dt.timedelta(minutes=DEFAULT_ZERO_WIDTH_MINUTES)
    return (wmin - pad), (wmax + pad), f"TIP window width was 0s; expanded to ±{DEFAULT_ZERO_WIDTH_MINUTES} min for sampling."

def fetch_latest_tle(session: requests.Session, norad_id: int) -> Tuple[str, str, str]:
    url = (
        f"https://www.space-track.org/basicspacedata/query/class/gp/"
        f"NORAD_CAT_ID/{norad_id}/orderby/EPOCH%20desc/limit/1/format/tle"
    )
    r = retry_get(session, url)
    lines = [ln.strip() for ln in r.text.splitlines() if ln.strip()]
    if len(lines) < 2:
        raise RuntimeError("TLE fetch returned insufficient lines.")
    if lines[0].startswith("1 ") and lines[1].startswith("2 "):
        name = f"NORAD {norad_id}"
        l1, l2 = lines[0], lines[1]
    else:
        if len(lines) < 3:
            raise RuntimeError("TLE fetch returned insufficient lines (expected name+2 lines).")
        name, l1, l2 = lines[0], lines[1], lines[2]
    return name, l1, l2

def parse_bstar_from_tle_line1(l1: str) -> float:
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

def parse_mean_motion_from_tle_line2(l2: str) -> float:
    """
    TLE line 2: mean motion is in columns 53-63 (1-indexed), approx slice [52:63] 0-indexed.
    Example: " 15.12345678"
    """
    try:
        s = l2[52:63].strip()
        return float(s)
    except Exception:
        return float("nan")

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
        mm = parse_mean_motion_from_tle_line2(l2)
        out.append(TLEPoint(epoch_utc=epoch_dt, name=name, l1=l1.strip(), l2=l2.strip(), bstar=bstar, mean_motion_rev_per_day=mm))

    out.sort(key=lambda x: x.epoch_utc)  # oldest -> newest
    return out

def robust_stats(vals: np.ndarray) -> Tuple[float, float]:
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med))) + 1e-30
    return med, mad

def compute_tle_trends(tles: List[TLEPoint]) -> Dict[str, float]:
    """
    Returns robust median/mad for B* and mean-motion, plus linear trend/day.
    """
    out: Dict[str, float] = {}
    if len(tles) < 5:
        return {
            "bstar_med": float("nan"), "bstar_mad": float("nan"), "bstar_trend_per_day": float("nan"),
            "mm_med": float("nan"), "mm_mad": float("nan"), "mm_trend_per_day": float("nan")
        }

    bvals = np.array([t.bstar for t in tles if np.isfinite(t.bstar)], dtype=float)
    mvals = np.array([t.mean_motion_rev_per_day for t in tles if np.isfinite(t.mean_motion_rev_per_day)], dtype=float)

    if len(bvals) >= 5:
        bmed, bmad = robust_stats(bvals)
    else:
        bmed, bmad = float("nan"), float("nan")

    if len(mvals) >= 5:
        mmed, mmad = robust_stats(mvals)
    else:
        mmed, mmad = float("nan"), float("nan")

    # Trend using last N
    n = min(15, len(tles))
    x0 = tles[-n].epoch_utc
    xs = np.array([(tles[-n+i].epoch_utc - x0).total_seconds() / 86400 for i in range(n)], dtype=float)

    # B* trend
    btrend = float("nan")
    try:
        ys_b = np.array([tles[-n+i].bstar for i in range(n)], dtype=float)
        if np.all(np.isfinite(ys_b)):
            btrend = float(np.polyfit(xs, ys_b, 1)[0])
    except Exception:
        pass

    # Mean motion trend (rev/day per day)
    mmtrend = float("nan")
    try:
        ys_m = np.array([tles[-n+i].mean_motion_rev_per_day for i in range(n)], dtype=float)
        if np.all(np.isfinite(ys_m)):
            mmtrend = float(np.polyfit(xs, ys_m, 1)[0])
    except Exception:
        pass

    return {
        "bstar_med": bmed, "bstar_mad": bmad, "bstar_trend_per_day": btrend,
        "mm_med": mmed, "mm_mad": mmad, "mm_trend_per_day": mmtrend
    }

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
# NOAA Kp (nearest-time)
# -----------------------------
def fetch_noaa_kp_rows() -> list:
    r = requests.get(NOAA_KP_JSON, timeout=20)
    r.raise_for_status()
    return r.json()

def kp_nearest(kp_rows: list, target_utc: dt.datetime, max_age_hours: int = 72) -> float:
    """
    Returns kp_index nearest to target_utc (within +-max_age_hours).
    NOAA file is usually header row + rows: [time_tag, kp_index, ...]
    """
    if not isinstance(kp_rows, list) or len(kp_rows) < 2:
        return float("nan")

    header = kp_rows[0]
    try:
        t_idx = header.index("time_tag")
        k_idx = header.index("kp_index")
    except Exception:
        t_idx, k_idx = 0, 1

    best_dt = None
    best_kp = float("nan")
    best_abs = None

    lo = target_utc - dt.timedelta(hours=max_age_hours)
    hi = target_utc + dt.timedelta(hours=max_age_hours)

    for row in kp_rows[1:]:
        if not isinstance(row, list) or len(row) <= max(t_idx, k_idx):
            continue
        try:
            t = parse_datetime_flexible(str(row[t_idx]))
            kp = float(row[k_idx])
        except Exception:
            continue
        if not (lo <= t <= hi):
            continue
        d = abs((t - target_utc).total_seconds())
        if best_abs is None or d < best_abs:
            best_abs = d
            best_dt = t
            best_kp = kp

    return float(best_kp) if best_dt else float("nan")


# -----------------------------
# MC Sampling: combine Kp + mean-motion trend + (optional) B*
# -----------------------------
def mc_sample_time(
    wmin: dt.datetime,
    wmax: dt.datetime,
    mc_n: int,
    use_bstar: bool,
    trends: Dict[str, float],
    kp_at_target: float
) -> np.ndarray:
    """
    Samples time inside window using Beta distribution.

    Bias drivers:
      - Kp: higher => earlier
      - Mean motion trend: more negative/decay signals earlier (proxy)
      - B* trend/level (optional): higher => earlier

    IMPORTANT: This does NOT create new physics. It's a bias/uncertainty model.
    """
    width_s = max(1.0, (wmax - wmin).total_seconds())

    # base mean at midpoint
    mean = 0.5

    # Kp -> bias (conservative but stronger than previous)
    # Typical Kp around 1-4. Strong storms 6-9.
    if np.isfinite(kp_at_target):
        # shift up to about -0.30 (earlier) for very high Kp
        mean += float(np.clip(-(kp_at_target - 2.5) * 0.05, -0.30, 0.15))

    # Mean motion trend bias:
    # If mean motion is increasing (often as orbit decays, mean motion increases),
    # stronger increase => earlier.
    mmtrend = trends.get("mm_trend_per_day", float("nan"))
    if np.isfinite(mmtrend):
        # Scale: typical changes are small; amplify with tanh
        mean += float(np.clip(-0.18 * np.tanh(mmtrend * 200.0), -0.18, 0.18))

    # B* bias (optional)
    if use_bstar:
        bmed = trends.get("bstar_med", float("nan"))
        btrend = trends.get("bstar_trend_per_day", float("nan"))
        if np.isfinite(bmed):
            mean += float(np.clip(-0.10 * np.tanh(math.log10(abs(bmed) + 1e-15) + 7.0), -0.12, 0.12))
        if np.isfinite(btrend):
            mean += float(np.clip(-0.12 * np.tanh(btrend * 5e4), -0.12, 0.12))

    mean = float(np.clip(mean, 0.03, 0.97))

    # Concentration: use combined uncertainty from mm_mad and bstar_mad
    k = 10.0
    mm_med = trends.get("mm_med", float("nan"))
    mm_mad = trends.get("mm_mad", float("nan"))
    if np.isfinite(mm_med) and np.isfinite(mm_mad) and abs(mm_med) > 0:
        rel = mm_mad / (abs(mm_med) + 1e-30)
        # higher rel => flatter => smaller k
        k = float(np.clip(14.0 / (1.0 + 8.0 * rel), 3.0, 18.0))

    a = mean * k
    b = (1.0 - mean) * k
    u = np.random.beta(a, b, size=int(mc_n))
    ts = np.array([wmin.timestamp() + float(x) * width_s for x in u], dtype=float)
    return ts


# -----------------------------
# KML helpers
# -----------------------------
def export_kml(path: str, tracks: List[Dict[str, Any]], points: List[Dict[str, Any]], swath_poly: Optional[List[Tuple[float, float]]] = None) -> None:
    if simplekml is None:
        raise RuntimeError("simplekml not installed. pip install simplekml")
    kml = simplekml.Kml()

    for tr in tracks:
        ls = kml.newlinestring(name=tr["name"])
        ls.coords = list(zip(tr["lons"], tr["lats"]))
        ls.altitudemode = simplekml.AltitudeMode.clamptoground
        ls.extrude = 0
        if "color" in tr:
            ls.style.linestyle.color = tr["color"]  # aabbggrr
        if "width" in tr:
            ls.style.linestyle.width = int(tr["width"])

    for p in points:
        pt = kml.newpoint(name=p["name"], coords=[(p["lon"], p["lat"])])
        if "description" in p:
            pt.description = p["description"]

    if swath_poly:
        pol = kml.newpolygon(name="Swath (±km)")
        pol.outerboundaryis = [(lon, lat) for (lon, lat) in swath_poly]
        pol.style.polystyle.fill = 0
        pol.style.linestyle.width = 2

    kml.save(path)

def build_simple_swath_polygon(lats: List[float], lons: List[float], half_width_km: float) -> List[Tuple[float, float]]:
    if not lats or not lons or len(lats) < 2:
        return []

    left = []
    right = []
    for i in range(len(lats)):
        lat = lats[i]
        lon = lons[i]
        j0 = max(0, i - 1)
        j1 = min(len(lats) - 1, i + 1)
        dlat = lats[j1] - lats[j0]
        dlon = lons[j1] - lons[j0]

        nx, ny = -dlat, dlon
        norm = math.hypot(nx, ny) + 1e-12
        nx /= norm
        ny /= norm

        dlat_deg = (half_width_km / 111.0) * nx
        dlon_deg = (half_width_km / (111.0 * max(0.2, math.cos(math.radians(lat))))) * ny

        left.append(( ((lon + dlon_deg + 180) % 360) - 180, lat + dlat_deg ))
        right.append(( ((lon - dlon_deg + 180) % 360) - 180, lat - dlat_deg ))

    poly = left + right[::-1] + [left[0]]
    return [(lon, lat) for (lon, lat) in poly]


# -----------------------------
# GUI
# -----------------------------
class ReentryGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Reentry Monitor (TIP + TLE) — Latest TIP Only + Map Envelope (Refined Prediction + KML)")
        self.geometry("1240x820")

        self.out_dir = os.getenv("OUT_DIR", "./reentry_out")
        ensure_dir(self.out_dir)

        # State
        self.tip_raw: Optional[list] = None
        self.sol_all: List[TipSolution] = []
        self.sol_latest: List[TipSolution] = []

        self.window_min: Optional[dt.datetime] = None
        self.window_max: Optional[dt.datetime] = None
        self.window_center: Optional[dt.datetime] = None

        self.tle_latest: Optional[Tuple[str, str, str]] = None
        self.tle_hist: List[TLEPoint] = []
        self.trends: Dict[str, float] = {}

        self.sat: Optional[EarthSatellite] = None

        self.pred: Dict[str, Any] = {}
        self.last_kp: Optional[float] = None

        # UI vars (kept in the same style as your GUI)
        self.var_user = tk.StringVar(value=os.getenv("SPACE_TRACK_USERNAME", ""))
        self.var_pass = tk.StringVar(value=os.getenv("SPACE_TRACK_PASSWORD", ""))
        self.var_norad = tk.StringVar(value=os.getenv("NORAD_CAT_ID", "66877"))
        self.var_tip_url = tk.StringVar(value=os.getenv("TIP_URL", ""))

        self.var_before = tk.StringVar(value="90")
        self.var_after = tk.StringVar(value="90")
        self.var_step = tk.StringVar(value="30")
        self.var_mid_tracks = tk.StringVar(value="5")
        self.var_ph_focus = tk.BooleanVar(value=False)

        self.var_mc_samples = tk.StringVar(value="2000")
        self.var_swath_km = tk.StringVar(value="100")
        self.var_use_bstar = tk.BooleanVar(value=True)

        # BIG refinement switch
        self.var_use_tip_latlon = tk.BooleanVar(value=True)

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
        ttk.Checkbutton(row3, text="Use TIP LAT/LON for point (recommended)", variable=self.var_use_tip_latlon).pack(side=tk.LEFT, padx=(12, 0))

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
            self.ax.plot(seg_lon, seg_lat, transform=ccrs.PlateCarree(), linewidth=linewidth, alpha=alpha, linestyle=linestyle)

    def on_fetch(self):
        try:
            user = self.var_user.get().strip()
            pw = self.var_pass.get().strip()
            norad = self._get_int(self.var_norad, "NORAD_CAT_ID")
            tip_url = self.var_tip_url.get().strip()

            self._log("Logging in to Space-Track…")
            session = spacetrack_login(user, pw)

            self._log(f"Fetching TIP (limit={DEFAULT_TIP_LIMIT})…")
            self.tip_raw = fetch_tip(session, norad, tip_url, limit=DEFAULT_TIP_LIMIT)
            self.sol_all = parse_tip_solutions(self.tip_raw)
            self.sol_latest = select_latest_tip_batch(self.sol_all)

            if not self.sol_latest:
                raise RuntimeError("No TIP solutions returned.")

            wmin0, wmax0, _ = compute_tip_window(self.sol_latest)
            if not (wmin0 and wmax0):
                raise RuntimeError("Latest TIP batch has no usable DECAY_EPOCH values.")

            # center is true center of *original* window (even if zero width)
            self.window_center = wmin0 + (wmax0 - wmin0) / 2

            wmin, wmax, note = normalize_tip_window(wmin0, wmax0)
            self.window_min, self.window_max = wmin, wmax

            latest_msg = self.sol_latest[0].msg_epoch
            self._log(f"Latest TIP MSG_EPOCH used: {latest_msg} (batch rows: {len(self.sol_latest)})")
            self._log(f"Decay window: {dt_to_iso_z(self.window_min)} to {dt_to_iso_z(self.window_max)} (width {fmt_timedelta(self.window_max - self.window_min)})")
            if note:
                self._log(f"NOTE: {note}")

            self._log("Fetching latest TLE…")
            self.tle_latest = fetch_latest_tle(session, norad)
            name, l1, l2 = self.tle_latest
            ts = sf_load.timescale()
            self.sat = EarthSatellite(l1, l2, name, ts)

            self._log("Fetching TLE history…")
            self.tle_hist = fetch_tle_history(session, norad, limit=25)
            self.trends = compute_tle_trends(self.tle_hist) if self.tle_hist else {}
            if self.tle_hist:
                self._log(
                    f"TLE hist loaded: {len(self.tle_hist)} | "
                    f"MM med={self.trends.get('mm_med', float('nan')):.6f} rev/day "
                    f"trend/day={self.trends.get('mm_trend_per_day', float('nan')):.6e} | "
                    f"B* med={self.trends.get('bstar_med', float('nan')):.3e}"
                )

        except Exception as e:
            messagebox.showerror("Fetch error", str(e))
            self._log(f"ERROR: {e}")

    def on_plot_envelope(self):
        try:
            if not (self.sat and self.window_min and self.window_max):
                raise RuntimeError("Fetch TIP + TLE first.")

            before_min = self._get_int(self.var_before, "Window Before (min)")
            after_min = self._get_int(self.var_after, "After (min)")
            step_s = self._get_int(self.var_step, "Step (sec)")
            mid_tracks = max(0, self._get_int(self.var_mid_tracks, "Intermediate tracks"))

            wmin = self.window_min
            wmax = self.window_max
            width = wmax - wmin

            latest_msg = self.sol_latest[0].msg_epoch if self.sol_latest else ""

            self._setup_map()

            if mid_tracks > 0 and width.total_seconds() > 0:
                for i in range(1, mid_tracks + 1):
                    frac = i / (mid_tracks + 1)
                    t_i = wmin + dt.timedelta(seconds=width.total_seconds() * frac)
                    lats_i, lons_i, _ = groundtrack_corridor(self.sat, t_i, before_min, after_min, step_s)
                    self._plot_track(lats_i, lons_i, linewidth=0.9, alpha=0.22, linestyle="--")

            lats_min, lons_min, _ = groundtrack_corridor(self.sat, wmin, before_min, after_min, step_s)
            lats_max, lons_max, _ = groundtrack_corridor(self.sat, wmax, before_min, after_min, step_s)
            self._plot_track(lats_min, lons_min, linewidth=2.0, alpha=0.90)
            self._plot_track(lats_max, lons_max, linewidth=2.0, alpha=0.90)

            self.ax.set_title(
                f"{self.sat.name} — Latest TIP MSG_EPOCH: {latest_msg}\n"
                f"Decay window: {dt_to_iso_z(wmin)} to {dt_to_iso_z(wmax)} (width {fmt_timedelta(width)})"
            )
            self.canvas.draw()
            self._log("Envelope plotted (latest TIP only).")

        except Exception as e:
            messagebox.showerror("Plot error", str(e))
            self._log(f"ERROR: {e}")

    def _tip_point_for_time(self, t_utc: dt.datetime) -> Optional[Tuple[float, float, str]]:
        """
        If TIP LAT/LON exists, choose the TIP solution whose DECAY_EPOCH is closest to t_utc.
        Returns (lat, lon, source_note) or None.
        """
        if not self.sol_latest:
            return None

        best = None
        best_abs = None
        for s in self.sol_latest:
            if s.lat is None or s.lon is None or not s.decay_epoch:
                continue
            try:
                td = abs((parse_datetime_flexible(s.decay_epoch) - t_utc).total_seconds())
            except Exception:
                continue
            if best_abs is None or td < best_abs:
                best_abs = td
                best = s

        if best is None:
            return None

        lon = ((best.lon + 180) % 360) - 180
        return float(best.lat), float(lon), "TIP_LATLON_nearest_DECAY_EPOCH"

    def on_run_prediction(self):
        try:
            if not (self.sat and self.window_min and self.window_max):
                raise RuntimeError("Fetch TIP + TLE first.")

            wmin = self.window_min
            wmax = self.window_max
            width_s = (wmax - wmin).total_seconds()
            if width_s < 1.0:
                raise RuntimeError("TIP window still too small after fallback.")

            mc_n = max(200, min(20000, self._get_int(self.var_mc_samples, "MC samples")))
            use_bstar = bool(self.var_use_bstar.get())
            use_tip_latlon = bool(self.var_use_tip_latlon.get())

            # choose a target epoch for Kp-nearest: use the TRUE center of original window if available
            t_target = self.window_center or (wmin + (wmax - wmin) / 2)

            self._log("Fetching NOAA Kp…")
            kp_val = float("nan")
            try:
                kp_rows = fetch_noaa_kp_rows()
                kp_val = kp_nearest(kp_rows, t_target)
            except Exception as e:
                self._log(f"WARNING: NOAA Kp fetch failed ({e}). Continuing without Kp bias.")

            self.last_kp = kp_val if np.isfinite(kp_val) else None
            if np.isfinite(kp_val):
                self._log(f"Kp nearest {dt_to_iso_z(t_target)}: {kp_val:.1f}")
            else:
                self._log("Kp: unavailable (no bias applied).")

            # Monte Carlo sample time inside window
            ts_samples = mc_sample_time(wmin, wmax, mc_n, use_bstar, self.trends, kp_val)

            p10_t = dt.datetime.fromtimestamp(float(np.percentile(ts_samples, 10)), tz=dt.timezone.utc)
            p50_t = dt.datetime.fromtimestamp(float(np.percentile(ts_samples, 50)), tz=dt.timezone.utc)
            p90_t = dt.datetime.fromtimestamp(float(np.percentile(ts_samples, 90)), tz=dt.timezone.utc)

            def get_point(t: dt.datetime) -> Tuple[float, float, str]:
                if use_tip_latlon:
                    tippt = self._tip_point_for_time(t)
                    if tippt:
                        lat, lon, src = tippt
                        return lat, lon, src
                lat, lon = subpoint_at_time(self.sat, t)
                return lat, lon, "TLE_subpoint"

            p10_lat, p10_lon, p10_src = get_point(p10_t)
            p50_lat, p50_lon, p50_src = get_point(p50_t)
            p90_lat, p90_lon, p90_src = get_point(p90_t)

            self.pred = {
                "generated_utc": dt_to_iso_z(dt.datetime.now(dt.timezone.utc)),
                "norad_id": int(self.var_norad.get().strip()),
                "tip_msg_epoch": self.sol_latest[0].msg_epoch if self.sol_latest else None,
                "tip_window": {"start_utc": dt_to_iso_z(wmin), "end_utc": dt_to_iso_z(wmax), "width_sec": int(width_s)},
                "kp_nearest_time": dt_to_iso_z(t_target),
                "kp_value": float(kp_val) if np.isfinite(kp_val) else None,
                "use_bstar_bias": use_bstar,
                "use_tip_latlon": use_tip_latlon,
                "tle_trends": self.trends,
                "mc_samples": int(mc_n),
                "p10": {"t_utc": dt_to_iso_z(p10_t), "lat": p10_lat, "lon": p10_lon, "point_source": p10_src},
                "p50": {"t_utc": dt_to_iso_z(p50_t), "lat": p50_lat, "lon": p50_lon, "point_source": p50_src},
                "p90": {"t_utc": dt_to_iso_z(p90_t), "lat": p90_lat, "lon": p90_lon, "point_source": p90_src},
            }

            # Plot envelope + points
            self.on_plot_envelope()
            self.ax.plot([p50_lon], [p50_lat], marker="o", markersize=9, transform=ccrs.PlateCarree())
            self.ax.plot([p10_lon], [p10_lat], marker="o", markersize=6, transform=ccrs.PlateCarree())
            self.ax.plot([p90_lon], [p90_lat], marker="o", markersize=6, transform=ccrs.PlateCarree())
            self.ax.text(p50_lon + 2, p50_lat + 2, f"P50 ({p50_src})", transform=ccrs.PlateCarree(), fontsize=9)
            self.ax.text(p10_lon + 2, p10_lat - 2, "P10", transform=ccrs.PlateCarree(), fontsize=9)
            self.ax.text(p90_lon + 2, p90_lat - 2, "P90", transform=ccrs.PlateCarree(), fontsize=9)
            self.canvas.draw()

            self._log(f"P50 time: {self.pred['p50']['t_utc']} | {dt_to_iso_ph(p50_t)}")
            self._log(f"P50 point: lat={p50_lat:.2f}, lon={p50_lon:.2f} (source={p50_src})")

        except Exception as e:
            messagebox.showerror("Prediction error", str(e))
            self._log(f"ERROR: {e}")

    def on_export_kml(self):
        try:
            if simplekml is None:
                raise RuntimeError("simplekml not installed. pip install simplekml")
            if not (self.sat and self.window_min and self.window_max):
                raise RuntimeError("Fetch TIP + TLE first.")
            if not self.pred:
                raise RuntimeError("Run prediction first.")

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
            p50_t = parse_datetime_flexible(self.pred["p50"]["t_utc"])

            tracks = []
            for nm, tt, color, width in [
                ("TIP_min", wmin, "ff00ff00", 3),  # green
                ("TIP_max", wmax, "ff00ff00", 3),  # green
                ("P50_track", p50_t, "ff00a0ff", 4),  # orange-ish
            ]:
                lats, lons, _ = groundtrack_corridor(self.sat, tt, before_min, after_min, step_s)
                tracks.append({"name": nm, "lats": lats, "lons": lons, "color": color, "width": width})

            swath_poly = None
            if swath_km > 0.0:
                lats50, lons50, _ = groundtrack_corridor(self.sat, p50_t, before_min, after_min, step_s)
                swath_poly = build_simple_swath_polygon(lats50, lons50, swath_km)

            points = []
            for tag in ["p10", "p50", "p90"]:
                points.append({
                    "name": tag.upper(),
                    "lat": float(self.pred[tag]["lat"]),
                    "lon": float(self.pred[tag]["lon"]),
                    "description": f"time={self.pred[tag]['t_utc']} | source={self.pred[tag]['point_source']}"
                })

            kml_path = os.path.join(folder, f"reentry_refined_{norad}_{stamp}.kml")
            export_kml(kml_path, tracks, points, swath_poly=swath_poly)

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

            if self.tip_raw is not None:
                tip_path = os.path.join(folder, f"tip_raw_{norad}_{stamp}.json")
                with open(tip_path, "w", encoding="utf-8") as f:
                    json.dump(self.tip_raw, f, indent=2)
                self._log(f"Saved: {tip_path}")

            if self.sol_latest:
                latest_path = os.path.join(folder, f"tip_latest_batch_{norad}_{stamp}.json")
                with open(latest_path, "w", encoding="utf-8") as f:
                    json.dump([s.raw for s in self.sol_latest], f, indent=2)
                self._log(f"Saved: {latest_path}")

            if self.tle_latest is not None:
                name, l1, l2 = self.tle_latest
                tle_path = os.path.join(folder, f"tle_{norad}_{stamp}.txt")
                with open(tle_path, "w", encoding="utf-8") as f:
                    f.write(name + "\n" + l1 + "\n" + l2 + "\n")
                self._log(f"Saved: {tle_path}")

            if self.pred:
                pred_path = os.path.join(folder, f"prediction_{norad}_{stamp}.json")
                with open(pred_path, "w", encoding="utf-8") as f:
                    json.dump(self.pred, f, indent=2)
                self._log(f"Saved: {pred_path}")

            png_path = os.path.join(folder, f"map_{norad}_{stamp}.png")
            self.fig.savefig(png_path, dpi=220)
            self._log(f"Saved: {png_path}")

            messagebox.showinfo("Saved", "Outputs saved successfully.")

        except Exception as e:
            messagebox.showerror("Save error", str(e))
            self._log(f"ERROR: {e}")


def main():
    app = ReentryGUI()
    app.mainloop()

if __name__ == "__main__":
    main()
