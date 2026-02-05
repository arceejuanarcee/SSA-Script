#!/usr/bin/env python3
"""
Reentry Monitor GUI (TIP + TLE) — Latest TIP Only + Map Envelope
+ Kp-biased prediction + KML export

FIX INCLUDED:
- Handles TIP cases where decay window collapses (wmin == wmax).
  -> If window width is 0, it tries to use TIP window fields if present,
     otherwise expands around DECAY_EPOCH by ±fallback minutes (default 90).

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

import numpy as np
import requests
from dotenv import load_dotenv

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from skyfield.api import EarthSatellite, load as sf_load

import cartopy.crs as ccrs
import cartopy.feature as cfeature

import simplekml


# -----------------------------
# Load .env early
# -----------------------------
load_dotenv()

# -----------------------------
# Space-Track endpoint
# -----------------------------
LOGIN_URL = "https://www.space-track.org/ajaxauth/login"
DEFAULT_TIP_LIMIT = int(os.getenv("TIP_LIMIT", "200"))
DEFAULT_TLE_HIST = int(os.getenv("TLE_HIST", "25"))

# NOAA Planetary K-index product (JSON)
NOAA_KP_URL = "https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json"

# Swath half-width for KML polygon (km)
DEFAULT_SWATH_KM = 100.0

# If TIP window collapses, expand around DECAY_EPOCH by ± this many minutes
DEFAULT_FALLBACK_WINDOW_MIN = int(os.getenv("TIP_FALLBACK_WINDOW_MIN", "90"))

EARTH_RADIUS_KM = 6371.0088


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

def parse_any_datetime_utc(s: str) -> dt.datetime:
    """
    Robust datetime parser:
    - "YYYY-MM-DD HH:MM:SS"
    - "YYYY-MM-DDTHH:MM:SS.ffffff"
    - with optional trailing 'Z'
    """
    if not s:
        raise ValueError("Empty datetime string")

    st = s.strip()
    st = st.replace("Z", "")

    fmts = [
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
    ]
    for f in fmts:
        try:
            return dt.datetime.strptime(st, f).replace(tzinfo=dt.timezone.utc)
        except Exception:
            pass

    try:
        return dt.datetime.fromisoformat(st).astimezone(dt.timezone.utc)
    except Exception as e:
        raise ValueError(f"Unparseable datetime: {s}") from e

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
            return parse_any_datetime_utc(sol.msg_epoch).timestamp()
        except Exception:
            return 0.0

    sols.sort(key=key, reverse=True)
    return sols

def select_latest_tip_batch(solutions: List[TipSolution]) -> List[TipSolution]:
    """
    Group strictly by newest MSG_EPOCH string.
    (TIP may still deliver only 1 row for the newest MSG_EPOCH.)
    """
    if not solutions:
        return []
    newest_msg = (solutions[0].msg_epoch or "").strip()
    if not newest_msg:
        return solutions[:1]
    return [s for s in solutions if (s.msg_epoch or "").strip() == newest_msg]

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

def _try_extract_tip_window_fields(latest_batch: List[TipSolution]) -> Optional[Tuple[dt.datetime, dt.datetime]]:
    """
    Some TIP records include explicit window fields (varies by provider/version).
    We'll try common possibilities without assuming.
    """
    if not latest_batch:
        return None

    # check first row (usually enough)
    row = latest_batch[0].raw or {}

    candidates = [
        ("WINDOW_START", "WINDOW_END"),
        ("WINDOW_START_EPOCH", "WINDOW_END_EPOCH"),
        ("WINDOW_START_UTC", "WINDOW_END_UTC"),
        ("START_EPOCH", "END_EPOCH"),
    ]
    for a, b in candidates:
        if row.get(a) and row.get(b):
            try:
                wmin = parse_any_datetime_utc(str(row[a]))
                wmax = parse_any_datetime_utc(str(row[b]))
                if wmax > wmin:
                    return (wmin, wmax)
            except Exception:
                continue
    return None

def compute_tip_window_from_batch(
    solutions_latest_batch: List[TipSolution],
    fallback_minutes: int = DEFAULT_FALLBACK_WINDOW_MIN
) -> Tuple[Optional[dt.datetime], Optional[dt.datetime], List[dt.datetime], bool]:
    """
    Returns (wmin, wmax, decays, used_fallback).

    If latest batch yields only a single DECAY_EPOCH (width 0),
    we:
      1) try TIP explicit window fields if present,
      2) otherwise synthesize window around DECAY_EPOCH by ±fallback_minutes.
    """
    decays: List[dt.datetime] = []
    for s in solutions_latest_batch:
        if s.decay_epoch:
            try:
                decays.append(parse_any_datetime_utc(s.decay_epoch))
            except Exception:
                pass

    if not decays:
        return None, None, [], False

    wmin = min(decays)
    wmax = max(decays)

    # Normal case
    if (wmax - wmin).total_seconds() >= 1:
        return wmin, wmax, decays, False

    # Collapsed window -> try explicit window fields
    win = _try_extract_tip_window_fields(solutions_latest_batch)
    if win is not None:
        return win[0], win[1], decays, True

    # Final fallback: synthesize around the single decay time
    center = wmin
    wmin2 = center - dt.timedelta(minutes=int(fallback_minutes))
    wmax2 = center + dt.timedelta(minutes=int(fallback_minutes))
    return wmin2, wmax2, decays, True

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


# -----------------------------
# NOAA Kp helpers
# -----------------------------
def fetch_noaa_planetary_kp() -> List[Tuple[dt.datetime, float]]:
    r = requests.get(NOAA_KP_URL, timeout=30)
    r.raise_for_status()
    data = r.json()
    rows = data[1:]  # first row is header
    out: List[Tuple[dt.datetime, float]] = []
    for row in rows:
        try:
            t = parse_any_datetime_utc(row[0])
            kp = float(row[1])
            out.append((t, kp))
        except Exception:
            continue
    return out

def nearest_kp_value(kp_rows: List[Tuple[dt.datetime, float]], t_utc: dt.datetime) -> Optional[float]:
    if not kp_rows:
        return None
    best = min(kp_rows, key=lambda x: abs((x[0] - t_utc).total_seconds()))
    return float(best[1])


# -----------------------------
# Sequential TLE B* proxy
# -----------------------------
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

def fetch_tle_history(session: requests.Session, norad_id: int, limit: int = DEFAULT_TLE_HIST) -> List[TLEPoint]:
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
            epoch_dt = parse_any_datetime_utc(epoch)
        except Exception:
            continue
        bstar = parse_bstar_from_tle_line1(l1)
        out.append(TLEPoint(epoch_utc=epoch_dt, name=name, l1=l1.strip(), l2=l2.strip(), bstar=bstar))

    out.sort(key=lambda x: x.epoch_utc)
    return out

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
    m = float(np.polyfit(xs, ys, 1)[0])
    return {"bstar_med": med, "bstar_mad": mad, "bstar_trend_per_day": m}


# -----------------------------
# Time sampling inside TIP window (Kp + B*)
# -----------------------------
def sample_reentry_times(
    wmin: dt.datetime,
    wmax: dt.datetime,
    kp_value: Optional[float],
    bstar_stats: Dict[str, float],
    n: int = 2000,
    seed: Optional[int] = None
) -> np.ndarray:
    if seed is not None:
        np.random.seed(seed)

    width_s = max(1.0, (wmax - wmin).total_seconds())

    bias = 0.0
    if kp_value is not None and np.isfinite(kp_value):
        bias += -0.02 * float(kp_value)

    med = bstar_stats.get("bstar_med", float("nan"))
    trend = bstar_stats.get("bstar_trend_per_day", float("nan"))
    if np.isfinite(med):
        bias += -0.03 * np.tanh(math.log10(abs(med) + 1e-20) + 7.0)
    if np.isfinite(trend):
        bias += -0.03 * np.tanh(trend * 5e4)

    bias = float(np.clip(bias, -0.25, +0.10))
    mean = float(np.clip(0.5 + bias, 0.10, 0.90))

    mad = bstar_stats.get("bstar_mad", float("nan"))
    flatness = 0.55
    if np.isfinite(mad) and np.isfinite(med) and abs(med) > 0:
        rel = mad / (abs(med) + 1e-30)
        flatness = float(np.clip(0.35 + 2.0 * rel, 0.35, 0.95))

    k = (1.0 - flatness) * 18.0 + 2.0
    a = mean * k
    b = (1.0 - mean) * k

    u = np.random.beta(a, b, size=int(n))
    ts = np.array([wmin.timestamp() + float(x) * width_s for x in u], dtype=float)
    return ts

def subpoint_at_time(sat: EarthSatellite, t_utc: dt.datetime) -> Tuple[float, float]:
    ts = sf_load.timescale()
    t_sf = ts.from_datetime(t_utc)
    sub = sat.at(t_sf).subpoint()
    lat = float(sub.latitude.degrees)
    lon = float(sub.longitude.degrees)
    lon = ((lon + 180) % 360) - 180
    return lat, lon


# -----------------------------
# Geodesy for swath polygon
# -----------------------------
def bearing_deg(lat1, lon1, lat2, lon2) -> float:
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dlon = math.radians(lon2 - lon1)
    y = math.sin(dlon) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlon)
    brng = math.degrees(math.atan2(y, x))
    return (brng + 360.0) % 360.0

def destination_point(lat, lon, brng_deg, dist_km) -> Tuple[float, float]:
    ang = dist_km / EARTH_RADIUS_KM
    brng = math.radians(brng_deg)
    phi1 = math.radians(lat)
    lam1 = math.radians(lon)

    phi2 = math.asin(math.sin(phi1) * math.cos(ang) + math.cos(phi1) * math.sin(ang) * math.cos(brng))
    lam2 = lam1 + math.atan2(math.sin(brng) * math.sin(ang) * math.cos(phi1),
                             math.cos(ang) - math.sin(phi1) * math.sin(phi2))
    lat2 = math.degrees(phi2)
    lon2 = (math.degrees(lam2) + 540) % 360 - 180
    return lat2, lon2

def build_swath_polygon(track_lats: List[float], track_lons: List[float], half_width_km: float) -> List[Tuple[float, float]]:
    if len(track_lats) < 2:
        return []

    left: List[Tuple[float, float]] = []
    right: List[Tuple[float, float]] = []

    for i in range(len(track_lats)):
        if i == len(track_lats) - 1:
            lat1, lon1 = track_lats[i-1], track_lons[i-1]
            lat2, lon2 = track_lats[i], track_lons[i]
        else:
            lat1, lon1 = track_lats[i], track_lons[i]
            lat2, lon2 = track_lats[i+1], track_lons[i+1]

        brng = bearing_deg(lat1, lon1, lat2, lon2)
        latL, lonL = destination_point(track_lats[i], track_lons[i], (brng - 90) % 360, half_width_km)
        latR, lonR = destination_point(track_lats[i], track_lons[i], (brng + 90) % 360, half_width_km)
        left.append((lonL, latL))
        right.append((lonR, latR))

    poly = left + right[::-1]
    if poly:
        poly.append(poly[0])
    return poly


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
        self.geometry("1240x860")

        self.out_dir = os.getenv("OUT_DIR", "./reentry_out")
        ensure_dir(self.out_dir)

        # State
        self.tip_raw: Optional[list] = None
        self.solutions_all: List[TipSolution] = []
        self.solutions_latest: List[TipSolution] = []
        self.tle: Optional[Tuple[str, str, str]] = None
        self.window_min: Optional[dt.datetime] = None
        self.window_max: Optional[dt.datetime] = None
        self.used_tip_fallback_window: bool = False

        self.session: Optional[requests.Session] = None
        self.tle_hist: List[TLEPoint] = []
        self.bstar_stats: Dict[str, float] = {}

        self.kp_value: Optional[float] = None
        self.pred_times: Dict[str, dt.datetime] = {}
        self.pred_points: Dict[str, Tuple[float, float]] = {}

        # UI vars
        self.var_user = tk.StringVar(value=os.getenv("SPACE_TRACK_USERNAME", ""))
        self.var_pass = tk.StringVar(value=os.getenv("SPACE_TRACK_PASSWORD", ""))
        self.var_norad = tk.StringVar(value=os.getenv("NORAD_CAT_ID", "66877"))
        self.var_tip_url = tk.StringVar(value=os.getenv("TIP_URL", ""))

        self.var_before = tk.StringVar(value="90")
        self.var_after = tk.StringVar(value="90")
        self.var_step = tk.StringVar(value="30")
        self.var_mid_tracks = tk.StringVar(value="5")

        self.var_ph_focus = tk.BooleanVar(value=False)

        # New knobs (minimal)
        self.var_mc_n = tk.StringVar(value="2000")
        self.var_swath_km = tk.StringVar(value=str(int(DEFAULT_SWATH_KM)))
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

        row2 = ttk.Frame(self, padding=(10, 0, 10, 8))
        row2.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(row2, text="Window Before (min)").pack(side=tk.LEFT)
        ttk.Entry(row2, textvariable=self.var_before, width=8).pack(side=tk.LEFT, padx=(6, 14))

        ttk.Label(row2, text="After (min)").pack(side=tk.LEFT)
        ttk.Entry(row2, textvariable=self.var_after, width=8).pack(side=tk.LEFT, padx=(6, 14))

        ttk.Label(row2, text="Step (sec)").pack(side=tk.LEFT)
        ttk.Entry(row2, textvariable=self.var_step, width=8).pack(side=tk.LEFT, padx=(6, 14))

        ttk.Label(row2, text="Intermediate tracks").pack(side=tk.LEFT)
        ttk.Entry(row2, textvariable=self.var_mid_tracks, width=8).pack(side=tk.LEFT, padx=(6, 14))

        ttk.Checkbutton(row2, text="Philippines Focus (zoom)", variable=self.var_ph_focus, command=self._apply_extent).pack(side=tk.LEFT, padx=(6, 0))

        row3 = ttk.Frame(self, padding=(10, 0, 10, 10))
        row3.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(row3, text="MC samples").pack(side=tk.LEFT)
        ttk.Entry(row3, textvariable=self.var_mc_n, width=8).pack(side=tk.LEFT, padx=(6, 14))

        ttk.Label(row3, text="Swath ±km (KML)").pack(side=tk.LEFT)
        ttk.Entry(row3, textvariable=self.var_swath_km, width=8).pack(side=tk.LEFT, padx=(6, 14))

        ttk.Checkbutton(row3, text="Use sequential TLE B* bias", variable=self.var_use_bstar).pack(side=tk.LEFT, padx=(6, 0))

        btns = ttk.Frame(self, padding=(10, 0, 10, 10))
        btns.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(btns, text="Fetch TIP + TLE (latest TIP only)", command=self.on_fetch).pack(side=tk.LEFT)
        ttk.Button(btns, text="Plot Envelope (latest TIP min–max)", command=self.on_plot_envelope).pack(side=tk.LEFT, padx=8)
        ttk.Button(btns, text="Run Kp-Biased Prediction", command=self.on_predict).pack(side=tk.LEFT, padx=8)
        ttk.Button(btns, text="Export KML (corridor + swath + points)", command=self.on_export_kml).pack(side=tk.LEFT, padx=8)
        ttk.Button(btns, text="Save Outputs…", command=self.on_save_outputs).pack(side=tk.LEFT, padx=8)

        self.status = tk.Text(self, height=10)
        self.status.pack(side=tk.BOTTOM, fill=tk.X)
        self._log("Ready. Fetch → Plot → (optional) Predict → Export KML / Save Outputs.")

    def _build_plot(self):
        center = ttk.Frame(self, padding=10)
        center.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.fig = plt.Figure(figsize=(11.8, 5.8), dpi=100)
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
        self.ax.set_title("Ground-track envelope around latest TIP decay window")
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

    def _plot_track(self, lats: List[float], lons: List[float], linewidth: float, alpha: float, linestyle: str, color: Optional[str] = None):
        for seg_lat, seg_lon in split_dateline_segments(lats, lons):
            self.ax.plot(seg_lon, seg_lat, transform=ccrs.PlateCarree(), linewidth=linewidth, alpha=alpha, linestyle=linestyle, color=color)

    # -----------------------------
    # Actions
    # -----------------------------
    def on_fetch(self):
        try:
            user = self.var_user.get().strip()
            pw = self.var_pass.get().strip()
            norad = self._get_int(self.var_norad, "NORAD_CAT_ID")
            tip_url_override = self.var_tip_url.get().strip()

            self._log("Logging in to Space-Track…")
            self.session = spacetrack_login(user, pw)

            self._log(f"Fetching TIP (limit={DEFAULT_TIP_LIMIT})…")
            self.tip_raw = fetch_tip(self.session, norad, tip_url_override, limit=DEFAULT_TIP_LIMIT)
            self.solutions_all = parse_tip_solutions(self.tip_raw)
            self.solutions_latest = select_latest_tip_batch(self.solutions_all)

            self._log("Fetching latest TLE…")
            self.tle = fetch_latest_tle(self.session, norad)

            wmin, wmax, decays, used_fallback = compute_tip_window_from_batch(
                self.solutions_latest,
                fallback_minutes=DEFAULT_FALLBACK_WINDOW_MIN
            )
            self.window_min, self.window_max = wmin, wmax
            self.used_tip_fallback_window = used_fallback

            if not (wmin and wmax):
                self._log("Latest TIP batch has no usable DECAY_EPOCH (cannot form window).")
                return

            latest_msg_epoch = self.solutions_latest[0].msg_epoch if self.solutions_latest else ""
            width = wmax - wmin

            note = ""
            if used_fallback:
                note = f" (fallback window applied ±{DEFAULT_FALLBACK_WINDOW_MIN}min)"

            self._log(f"Latest TIP MSG_EPOCH used: {latest_msg_epoch} (batch rows: {len(self.solutions_latest)})")
            self._log(f"Decay window: {dt_to_iso_z(wmin)} to {dt_to_iso_z(wmax)} (width {fmt_timedelta(width)}){note}")

            # History row
            hist_csv = history_path(self.out_dir, norad)
            prev = read_last_history_row(hist_csv)
            now_utc = dt.datetime.now(dt.timezone.utc)

            row = {
                "run_utc": dt_to_iso_z(now_utc),
                "norad_id": str(norad),
                "tip_msg_epoch_used": latest_msg_epoch,
                "latest_batch_count": str(len(self.solutions_latest)),
                "window_start_utc": dt_to_iso_z(wmin),
                "window_end_utc": dt_to_iso_z(wmax),
                "window_width_sec": str(int(width.total_seconds())),
                "decay_samples": str(len(decays)),
                "tip_total_rows_fetched": str(len(self.solutions_all)),
                "tip_limit": str(DEFAULT_TIP_LIMIT),
                "used_fallback_window": str(bool(used_fallback)),
                "fallback_minutes": str(DEFAULT_FALLBACK_WINDOW_MIN),
            }
            append_history_row(hist_csv, row)
            self._log(f"Saved history: {hist_csv}")

            if prev and prev.get("window_start_utc"):
                try:
                    prev_start = parse_any_datetime_utc(prev["window_start_utc"])
                    shift = wmin - prev_start
                    self._log(f"Shift vs previous window start: {fmt_timedelta(shift)} (positive = later)")
                except Exception:
                    pass

            # Clear prediction outputs on new fetch
            self.pred_times = {}
            self.pred_points = {}
            self.kp_value = None
            self.tle_hist = []
            self.bstar_stats = {}

        except Exception as e:
            messagebox.showerror("Fetch error", str(e))
            self._log(f"ERROR: {e}")

    def on_plot_envelope(self):
        try:
            if not self.tle or not self.window_min or not self.window_max:
                raise RuntimeError("Fetch TIP + TLE first.")

            before_min = self._get_int(self.var_before, "Window Before (min)")
            after_min = self._get_int(self.var_after, "After (min)")
            step_s = self._get_int(self.var_step, "Step (sec)")
            mid_tracks = max(0, min(30, self._get_int(self.var_mid_tracks, "Intermediate tracks")))

            name, l1, l2 = self.tle
            ts = sf_load.timescale()
            sat = EarthSatellite(l1, l2, name, ts)

            wmin = self.window_min
            wmax = self.window_max
            width = wmax - wmin
            wmid = wmin + (width / 2)

            latest_msg_epoch = self.solutions_latest[0].msg_epoch if self.solutions_latest else ""

            self._setup_map()

            # intermediate faint
            if mid_tracks > 0 and width.total_seconds() > 0:
                for i in range(1, mid_tracks + 1):
                    frac = i / (mid_tracks + 1)
                    t_i = wmin + dt.timedelta(seconds=width.total_seconds() * frac)
                    lats_i, lons_i, _ = groundtrack_corridor(sat, t_i, before_min, after_min, step_s)
                    self._plot_track(lats_i, lons_i, linewidth=0.9, alpha=0.25, linestyle="--", color="goldenrod")

            # boundaries
            lats_min, lons_min, _ = groundtrack_corridor(sat, wmin, before_min, after_min, step_s)
            lats_max, lons_max, _ = groundtrack_corridor(sat, wmax, before_min, after_min, step_s)
            self._plot_track(lats_min, lons_min, linewidth=1.8, alpha=0.95, linestyle="-", color="gold")
            self._plot_track(lats_max, lons_max, linewidth=1.8, alpha=0.95, linestyle="-", color="gold")

            # midpoint
            lats_mid, lons_mid, _ = groundtrack_corridor(sat, wmid, before_min, after_min, step_s)
            self._plot_track(lats_mid, lons_mid, linewidth=2.2, alpha=0.95, linestyle="-", color="limegreen")

            note = ""
            if self.used_tip_fallback_window:
                note = f" (fallback window ±{DEFAULT_FALLBACK_WINDOW_MIN}min)"

            self.ax.set_title(
                f"{name} — Latest TIP MSG_EPOCH: {latest_msg_epoch}\n"
                f"Decay window: {dt_to_iso_z(wmin)} to {dt_to_iso_z(wmax)} (width {fmt_timedelta(width)}){note}"
            )

            # overlay predicted points if present
            if self.pred_points:
                for key, (plat, plon) in self.pred_points.items():
                    ms = 8 if key == "p50" else 6
                    self.ax.plot([plon], [plat], marker="o", markersize=ms, transform=ccrs.PlateCarree(),
                                 linestyle="None", color="red" if key == "p50" else "black", alpha=0.95)
                    self.ax.text(plon + 1.0, plat + 1.0, key.upper(), transform=ccrs.PlateCarree(), fontsize=9)

            self.canvas.draw()
            self._log("Envelope plotted (latest TIP only): min/max boundaries + midpoint track.")

        except Exception as e:
            messagebox.showerror("Plot error", str(e))
            self._log(f"ERROR: {e}")

    def on_predict(self):
        try:
            if not self.session or not self.tle or not self.window_min or not self.window_max:
                raise RuntimeError("Fetch TIP + TLE first.")

            mc_n = max(300, min(50000, self._get_int(self.var_mc_n, "MC samples")))

            name, l1, l2 = self.tle
            ts = sf_load.timescale()
            sat = EarthSatellite(l1, l2, name, ts)

            wmin = self.window_min
            wmax = self.window_max
            width = wmax - wmin
            if width.total_seconds() <= 0:
                raise RuntimeError("TIP window still invalid after fallback (unexpected).")

            # Kp at window midpoint
            self._log("Fetching NOAA planetary Kp…")
            kp_rows = fetch_noaa_planetary_kp()
            wmid = wmin + (width / 2)
            self.kp_value = nearest_kp_value(kp_rows, wmid)
            if self.kp_value is None:
                self._log("WARNING: Could not resolve Kp; running without Kp bias.")
            else:
                self._log(f"Kp nearest to window midpoint ({dt_to_iso_z(wmid)}): {self.kp_value:.1f}")

            # Optional B* bias
            self.bstar_stats = {"bstar_med": float("nan"), "bstar_mad": float("nan"), "bstar_trend_per_day": float("nan")}
            if bool(self.var_use_bstar.get()):
                norad = int(self.var_norad.get().strip())
                self._log(f"Fetching sequential TLE history (last {DEFAULT_TLE_HIST}) for B* proxy…")
                self.tle_hist = fetch_tle_history(self.session, norad, limit=DEFAULT_TLE_HIST)
                if self.tle_hist:
                    self.bstar_stats = robust_bstar_stats(self.tle_hist)
                    self._log(
                        f"B* proxy: median={self.bstar_stats['bstar_med']:.3e}, "
                        f"MAD={self.bstar_stats['bstar_mad']:.3e}, "
                        f"trend/day={self.bstar_stats['bstar_trend_per_day']:.3e}"
                    )
                else:
                    self._log("WARNING: No TLE history returned; continuing without B* bias.")

            self._log("Running Kp-biased Monte Carlo sampling (inside TIP window)…")
            ts_samples = sample_reentry_times(wmin, wmax, self.kp_value, self.bstar_stats, n=mc_n)

            t_p10 = dt.datetime.fromtimestamp(float(np.percentile(ts_samples, 10)), tz=dt.timezone.utc)
            t_p50 = dt.datetime.fromtimestamp(float(np.percentile(ts_samples, 50)), tz=dt.timezone.utc)
            t_p90 = dt.datetime.fromtimestamp(float(np.percentile(ts_samples, 90)), tz=dt.timezone.utc)

            self.pred_times = {"p10": t_p10, "p50": t_p50, "p90": t_p90}

            lat10, lon10 = subpoint_at_time(sat, t_p10)
            lat50, lon50 = subpoint_at_time(sat, t_p50)
            lat90, lon90 = subpoint_at_time(sat, t_p90)
            self.pred_points = {"p10": (lat10, lon10), "p50": (lat50, lon50), "p90": (lat90, lon90)}

            self._log(f"P10 time: {dt_to_iso_z(t_p10)} | lat={lat10:.2f}, lon={lon10:.2f}")
            self._log(f"P50 time: {dt_to_iso_z(t_p50)} | lat={lat50:.2f}, lon={lon50:.2f}")
            self._log(f"P90 time: {dt_to_iso_z(t_p90)} | lat={lat90:.2f}, lon={lon90:.2f}")

            self.on_plot_envelope()

        except Exception as e:
            messagebox.showerror("Prediction error", str(e))
            self._log(f"ERROR: {e}")

    def on_export_kml(self):
        try:
            if not self.tle or not self.window_min or not self.window_max:
                raise RuntimeError("Fetch TIP + TLE first.")
            if not self.pred_times:
                raise RuntimeError("Run Kp-Biased Prediction first.")

            before_min = self._get_int(self.var_before, "Window Before (min)")
            after_min = self._get_int(self.var_after, "After (min)")
            step_s = self._get_int(self.var_step, "Step (sec)")
            swath_km = max(10.0, min(1000.0, self._get_float(self.var_swath_km, "Swath ±km (KML)")))

            name, l1, l2 = self.tle
            ts = sf_load.timescale()
            sat = EarthSatellite(l1, l2, name, ts)

            wmin = self.window_min
            wmax = self.window_max
            t_p50 = self.pred_times["p50"]

            lats_min, lons_min, _ = groundtrack_corridor(sat, wmin, before_min, after_min, step_s)
            lats_max, lons_max, _ = groundtrack_corridor(sat, wmax, before_min, after_min, step_s)
            lats_p50, lons_p50, _ = groundtrack_corridor(sat, t_p50, before_min, after_min, step_s)

            poly = build_swath_polygon(lats_p50, lons_p50, swath_km)

            norad = int(self.var_norad.get().strip())
            stamp = dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            out_path = filedialog.asksaveasfilename(
                title="Save KML",
                defaultextension=".kml",
                initialfile=f"reentry_corridor_{norad}_{stamp}.kml",
                filetypes=[("KML files", "*.kml")]
            )
            if not out_path:
                return

            kml = simplekml.Kml()
            f_tracks = kml.newfolder(name="Tracks")

            def add_line(folder, nm, lats, lons, color_hex=None, width=3):
                ls = folder.newlinestring(name=nm)
                ls.coords = list(zip(lons, lats))
                ls.altitudemode = simplekml.AltitudeMode.clamptoground
                ls.extrude = 0
                if color_hex:
                    ls.style.linestyle.color = color_hex
                ls.style.linestyle.width = width

            add_line(f_tracks, "TIP_min_track", lats_min, lons_min, color_hex="ff00ffff", width=3)
            add_line(f_tracks, "TIP_max_track", lats_max, lons_max, color_hex="ff00ffff", width=3)
            add_line(f_tracks, "P50_centerline", lats_p50, lons_p50, color_hex="ff00ff00", width=4)

            f_swath = kml.newfolder(name=f"Swath_±{int(round(swath_km))}km")
            if poly:
                pol = f_swath.newpolygon(name=f"Swath ±{int(round(swath_km))} km")
                pol.outerboundaryis = poly
                pol.style.polystyle.fill = 0
                pol.style.linestyle.width = 2
                pol.style.linestyle.color = "ff00ffff"

            f_pts = kml.newfolder(name="Impact_Proxy_Points")
            for key in ["p10", "p50", "p90"]:
                t = self.pred_times[key]
                lat, lon = self.pred_points[key]
                p = f_pts.newpoint(name=f"{key.upper()} {dt_to_iso_z(t)}")
                p.coords = [(lon, lat)]
                p.altitudemode = simplekml.AltitudeMode.clamptoground
                if key == "p50":
                    p.style.iconstyle.color = "ff0000ff"
                    p.style.iconstyle.scale = 1.2
                else:
                    p.style.iconstyle.color = "ff000000"
                    p.style.iconstyle.scale = 1.0

            kml.save(out_path)
            self._log(f"KML exported: {out_path}")

        except Exception as e:
            messagebox.showerror("KML export error", str(e))
            self._log(f"ERROR: {e}")

    def on_save_outputs(self):
        try:
            if self.tip_raw is None and self.tle is None:
                raise RuntimeError("Nothing to save yet. Fetch TIP + TLE first.")

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

            if self.solutions_latest:
                latest_path = os.path.join(folder, f"tip_latest_batch_{norad}_{stamp}.json")
                with open(latest_path, "w", encoding="utf-8") as f:
                    json.dump([s.raw for s in self.solutions_latest], f, indent=2)
                self._log(f"Saved: {latest_path}")

            if self.tle is not None:
                name, l1, l2 = self.tle
                tle_path = os.path.join(folder, f"tle_{norad}_{stamp}.txt")
                with open(tle_path, "w", encoding="utf-8") as f:
                    f.write(name + "\n" + l1 + "\n" + l2 + "\n")
                self._log(f"Saved: {tle_path}")

            png_path = os.path.join(folder, f"corridor_envelope_{norad}_{stamp}.png")
            self.fig.savefig(png_path, dpi=220)
            self._log(f"Saved: {png_path}")

            hist = history_path(self.out_dir, norad)
            if os.path.exists(hist):
                hist_copy = os.path.join(folder, f"tip_history_{norad}.csv")
                with open(hist, "r", encoding="utf-8") as src, open(hist_copy, "w", encoding="utf-8") as dst:
                    dst.write(src.read())
                self._log(f"Saved: {hist_copy}")

            if self.pred_times:
                pred_path = os.path.join(folder, f"prediction_summary_{norad}_{stamp}.json")
                payload = {
                    "norad_id": norad,
                    "kp_value": self.kp_value,
                    "bstar_stats": self.bstar_stats,
                    "tip_window": {
                        "start_utc": dt_to_iso_z(self.window_min) if self.window_min else None,
                        "end_utc": dt_to_iso_z(self.window_max) if self.window_max else None,
                        "used_fallback_window": bool(self.used_tip_fallback_window),
                        "fallback_minutes": DEFAULT_FALLBACK_WINDOW_MIN,
                    },
                    "prediction": {
                        k: {
                            "t_utc": dt_to_iso_z(self.pred_times[k]),
                            "lat": float(self.pred_points[k][0]),
                            "lon": float(self.pred_points[k][1]),
                        } for k in ["p10", "p50", "p90"]
                    }
                }
                with open(pred_path, "w", encoding="utf-8") as f:
                    json.dump(payload, f, indent=2)
                self._log(f"Saved: {pred_path}")

            messagebox.showinfo("Saved", "Outputs saved successfully.")

        except Exception as e:
            messagebox.showerror("Save error", str(e))
            self._log(f"ERROR: {e}")


def main():
    app = ReentryGUI()
    app.mainloop()

if __name__ == "__main__":
    main()
