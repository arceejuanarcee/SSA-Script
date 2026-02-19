#!/usr/bin/env python3
"""
Reentry Monitor GUI (TIP + TLE) — LATEST TIP ONLY + Map Envelope
WITH: pymsis MSIS2.1 density + storm-time mode + Monte Carlo time sampling
+ ✅ Physics-based descent integration (density-driven) producing:
    - descent time (s)
    - downrange distance (km)
    - density/velocity/altitude profiles
NOTE (SAFETY):
  This version DOES NOT compute or output final impact lat/lon.

Requirements:
  pip install requests skyfield matplotlib python-dotenv cartopy numpy pymsis

Env vars:
  SPACE_TRACK_USERNAME
  SPACE_TRACK_PASSWORD
  TIP_LIMIT (optional)
  OUT_DIR (optional)
"""

from __future__ import annotations

import os
import csv
import json
import time
import math
import random
import re
import datetime as dt
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any

import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np
import requests
from dotenv import load_dotenv

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import cartopy.crs as ccrs
import cartopy.feature as cfeature

from skyfield.api import EarthSatellite, load as sf_load

try:
    import pymsis
except Exception:
    pymsis = None


# -----------------------------
# Load .env early
# -----------------------------
load_dotenv()

LOGIN_URL = "https://www.space-track.org/ajaxauth/login"
DEFAULT_TIP_LIMIT = int(os.getenv("TIP_LIMIT", "200"))

PH_TZ = dt.timezone(dt.timedelta(hours=8))


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


# -----------------------------
# Helpers: time parsing
# -----------------------------
def parse_any_datetime_utc(s: str) -> dt.datetime:
    if not s:
        raise ValueError("Empty datetime string")
    txt = s.strip()

    if txt.endswith("Z"):
        txt2 = txt[:-1] + "+00:00"
    else:
        txt2 = txt

    try:
        d = dt.datetime.fromisoformat(txt2)
        if d.tzinfo is None:
            d = d.replace(tzinfo=dt.timezone.utc)
        return d.astimezone(dt.timezone.utc)
    except Exception:
        pass

    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S.%f"):
        try:
            d = dt.datetime.strptime(txt, fmt).replace(tzinfo=dt.timezone.utc)
            return d
        except Exception:
            continue

    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f"):
        try:
            d = dt.datetime.strptime(txt, fmt).replace(tzinfo=dt.timezone.utc)
            return d
        except Exception:
            continue

    raise ValueError(f"Unrecognized datetime format: {s!r}")


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


# -----------------------------
# TIP uncertainty parsing (best-effort)
# -----------------------------
def parse_uncertainty_seconds(val: Any) -> Optional[float]:
    """
    Tries to parse uncertainty fields like:
    - "0.8" (assume hours if small) or numeric minutes/seconds (unknown)
    - "~±0h 48m", "±0h 12m", "0h 48m", "48m", "2880s"
    Returns seconds or None.
    """
    if val is None:
        return None

    if isinstance(val, (int, float)):
        x = float(val)
        if x <= 10:
            return x * 3600.0
        if x <= 600:
            return x * 60.0
        return x

    s = str(val).strip().lower()
    if not s:
        return None

    s = s.replace("~", "").replace("≈", "").replace("about", "").replace("+/-", "±").replace("±", "")
    s = s.strip()

    total = 0.0
    found = False

    m = re.search(r"(\d+(?:\.\d+)?)\s*h", s)
    if m:
        total += float(m.group(1)) * 3600.0
        found = True

    m = re.search(r"(\d+(?:\.\d+)?)\s*(m|min|mins|minute|minutes)\b", s)
    if m:
        total += float(m.group(1)) * 60.0
        found = True

    m = re.search(r"(\d+(?:\.\d+)?)\s*(s|sec|secs|second|seconds)\b", s)
    if m:
        total += float(m.group(1))
        found = True

    if found and total > 0:
        return total

    m = re.search(r"(\d+(?:\.\d+)?)", s)
    if m:
        x = float(m.group(1))
        if x <= 10:
            return x * 3600.0
        if x <= 600:
            return x * 60.0
        return x

    return None


def tip_batch_uncertainty_seconds(latest_batch: List[TipSolution]) -> Optional[float]:
    candidate_keys = [
        "EPOCH_UNCERTAINTY",
        "EPOCH_UNC",
        "DECAY_UNCERTAINTY",
        "DECAY_UNC",
        "DECAY_EPOCH_UNCERTAINTY",
        "DECAY_EPOCH_UNC",
        "WINDOW",
        "WINDOW_WIDTH",
        "WINDOW_MINUTES",
    ]
    for s in latest_batch:
        raw = s.raw or {}
        for k in candidate_keys:
            if k in raw and raw[k] not in (None, "", "N/A"):
                sec = parse_uncertainty_seconds(raw[k])
                if sec and sec > 0:
                    return sec
    return None


# -----------------------------
# HTTP helpers
# -----------------------------
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


# -----------------------------
# TIP + TLE
# -----------------------------
def parse_tip_solutions(tip_json: list) -> List[TipSolution]:
    sols: List[TipSolution] = []
    for row in tip_json:
        sols.append(
            TipSolution(
                msg_epoch=(row.get("MSG_EPOCH") or row.get("MSG_EPOCH ", "") or "").strip(),
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
    if not solutions:
        return []
    newest_msg = solutions[0].msg_epoch
    if not newest_msg:
        return solutions[:1]
    return [s for s in solutions if s.msg_epoch == newest_msg]


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


def compute_tip_window_from_latest_batch(
    solutions_latest_batch: List[TipSolution],
    fallback_uncert_minutes: float
) -> Tuple[Optional[dt.datetime], Optional[dt.datetime], List[dt.datetime], str]:
    decays: List[dt.datetime] = []
    for s in solutions_latest_batch:
        if s.decay_epoch:
            try:
                decays.append(parse_any_datetime_utc(s.decay_epoch))
            except Exception:
                pass

    if not decays:
        return None, None, [], "none"

    wmin = min(decays)
    wmax = max(decays)

    if (wmax - wmin).total_seconds() > 0:
        return wmin, wmax, decays, "tip_spread"

    tip_unc_sec = tip_batch_uncertainty_seconds(solutions_latest_batch)
    if tip_unc_sec and tip_unc_sec > 0:
        half = dt.timedelta(seconds=float(tip_unc_sec))
        return wmin - half, wmax + half, decays, "tip_uncertainty"

    half = dt.timedelta(minutes=float(fallback_uncert_minutes))
    return wmin - half, wmax + half, decays, "fallback_uncertainty"


# -----------------------------
# Ground track / subpoint
# -----------------------------
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
# ✅ pymsis MSIS2.1 descent integration (SAFE: no impact lat/lon)
# -----------------------------
def integrate_descent_msis21(
    time_utc: dt.datetime,
    lat_deg: float,
    lon_deg: float,
    *,
    initial_alt_km: float = 120.0,
    initial_velocity_ms: float = 7800.0,
    CdA_over_m: float = 0.02,
    dt_step: float = 1.0,
    max_seconds: int = 2000,
    geomagnetic_activity: int = -1,  # storm-time
) -> Dict[str, Any]:
    """
    Physics-based descent integrator using pymsis MSIS2.1 density.

    Returns descent metadata ONLY:
      - descent_time_s
      - downrange_km (distance traveled along the flight path, not mapped to ground)
      - density/velocity/altitude profiles (for plotting/reporting)

    Safety note: does not compute or return impact coordinates.
    """
    if pymsis is None:
        raise RuntimeError("pymsis not installed. Install: pip install pymsis")

    alt = float(initial_alt_km)
    v = float(initial_velocity_ms)
    t = time_utc.astimezone(dt.timezone.utc)

    elapsed = 0.0
    density_profile: List[float] = []
    velocity_profile: List[float] = []
    altitude_profile: List[float] = []

    # simple descent slope model (empirical) to reduce altitude over time
    descent_slope = 0.02  # tune with validation; keeps this as a proxy

    while alt > 20.0 and v > 300.0 and elapsed < float(max_seconds):
        # pymsis takes naive numpy datetime64; provide UTC naive
        t_naive = t.replace(tzinfo=None)

        rho = pymsis.calculate(
            np.array([np.datetime64(t_naive)]),
            float(lon_deg),
            float(lat_deg),
            float(alt),
            version=2,  # MSIS2.1
            geomagnetic_activity=int(geomagnetic_activity),
        )[0, 0, 0, 0, 0]  # total mass density kg/m^3

        # drag acceleration proxy (ballistic coefficient via CdA_over_m)
        drag_accel = 0.5 * float(rho) * v * v * float(CdA_over_m)  # m/s^2
        dv = drag_accel * float(dt_step)
        v = max(0.0, v - dv)

        # distance traveled along trajectory
        ds = v * float(dt_step)  # meters

        # altitude decrement proxy (not a full 3DOF reentry)
        alt = alt - (ds / 1000.0) * descent_slope

        density_profile.append(float(rho))
        velocity_profile.append(float(v))
        altitude_profile.append(float(alt))

        t += dt.timedelta(seconds=float(dt_step))
        elapsed += float(dt_step)

    avg_v = float(np.mean(velocity_profile)) if velocity_profile else float(initial_velocity_ms)
    downrange_km = (avg_v * elapsed) / 1000.0

    return {
        "descent_time_s": float(elapsed),
        "avg_velocity_ms": float(avg_v),
        "downrange_km": float(downrange_km),
        "density_profile": density_profile,
        "velocity_profile": velocity_profile,
        "altitude_profile": altitude_profile,
        "model_notes": {
            "msis_version": "MSIS2.1 (version=2)",
            "geomagnetic_activity": geomagnetic_activity,
            "CdA_over_m": CdA_over_m,
            "descent_slope": descent_slope,
            "initial_alt_km": initial_alt_km,
            "initial_velocity_ms": initial_velocity_ms,
        },
    }


# -----------------------------
# Monte Carlo time sampling (bias heuristic)
# -----------------------------
def mc_sample_times_with_bias(
    wmin: dt.datetime,
    wmax: dt.datetime,
    n: int,
    seed: Optional[int] = None
) -> np.ndarray:
    if seed is not None:
        np.random.seed(seed)

    width_s = max(1.0, (wmax - wmin).total_seconds())

    # mild center bias (safe default). You can re-introduce your B* heuristics here if desired.
    mean = 0.5
    k = 10.0
    a = mean * k
    b = (1.0 - mean) * k

    u = np.random.beta(a, b, size=int(n))
    ts = np.array([wmin.timestamp() + float(x) * width_s for x in u], dtype=float)
    return ts


# -----------------------------
# Outputs: history CSV
# -----------------------------
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def history_path(out_dir: str, norad_id: int) -> str:
    return os.path.join(out_dir, f"tip_history_{norad_id}.csv")


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
        self.title("Reentry Monitor (TIP + TLE) — Latest TIP Only + Envelope + MSIS2.1 Descent (Safe)")
        self.geometry("1260x860")

        self.out_dir = os.getenv("OUT_DIR", "./reentry_out")
        ensure_dir(self.out_dir)

        self.tip_raw: Optional[list] = None
        self.solutions_all: List[TipSolution] = []
        self.solutions_latest: List[TipSolution] = []
        self.tle: Optional[Tuple[str, str, str]] = None
        self.window_min: Optional[dt.datetime] = None
        self.window_max: Optional[dt.datetime] = None
        self.window_mode: str = "none"

        self.sat: Optional[EarthSatellite] = None

        self.pred: Dict[str, Any] = {}

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

        # Descent integrator params
        self.var_geomag = tk.StringVar(value="-1")     # storm-time
        self.var_cdaom = tk.StringVar(value="0.02")    # CdA/m proxy
        self.var_dt_step = tk.StringVar(value="1.0")
        self.var_max_sec = tk.StringVar(value="2000")

        self.var_fallback_uncert_min = tk.StringVar(value="48")

        self._build_ui()
        self._build_plot()
        self._log("Ready. Fetch → Plot → Run MSIS2.1 Descent (Safe) → Save outputs.")

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

        def add_field(label: str, var: tk.StringVar, w: int = 7):
            ttk.Label(row2, text=label).pack(side=tk.LEFT)
            ttk.Entry(row2, textvariable=var, width=w).pack(side=tk.LEFT, padx=(6, 14))

        add_field("Window Before (min)", self.var_before, 7)
        add_field("After (min)", self.var_after, 7)
        add_field("Step (sec)", self.var_step, 7)
        add_field("Intermediate tracks", self.var_mid_tracks, 7)

        ttk.Checkbutton(row2, text="Philippines Focus (zoom)", variable=self.var_ph_focus, command=self._apply_extent).pack(side=tk.LEFT, padx=(6, 14))

        add_field("MC samples", self.var_mc_samples, 8)
        add_field("Fallback uncertainty (min)", self.var_fallback_uncert_min, 6)

        row3 = ttk.Frame(self, padding=(10, 0, 10, 10))
        row3.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(row3, text="Descent: geomagnetic_activity").pack(side=tk.LEFT)
        ttk.Entry(row3, textvariable=self.var_geomag, width=6).pack(side=tk.LEFT, padx=(6, 14))
        ttk.Label(row3, text="CdA/m").pack(side=tk.LEFT)
        ttk.Entry(row3, textvariable=self.var_cdaom, width=8).pack(side=tk.LEFT, padx=(6, 14))
        ttk.Label(row3, text="dt_step(s)").pack(side=tk.LEFT)
        ttk.Entry(row3, textvariable=self.var_dt_step, width=6).pack(side=tk.LEFT, padx=(6, 14))
        ttk.Label(row3, text="max_seconds").pack(side=tk.LEFT)
        ttk.Entry(row3, textvariable=self.var_max_sec, width=8).pack(side=tk.LEFT, padx=(6, 14))

        btns = ttk.Frame(self, padding=(10, 0, 10, 10))
        btns.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(btns, text="Fetch TIP + TLE (latest TIP only)", command=self.on_fetch).pack(side=tk.LEFT)
        ttk.Button(btns, text="Plot Envelope (latest TIP min–max)", command=self.on_plot_envelope).pack(side=tk.LEFT, padx=8)
        ttk.Button(btns, text="Run MSIS2.1 Descent (Safe)", command=self.on_run_prediction_safe).pack(side=tk.LEFT, padx=8)
        ttk.Button(btns, text="Save Outputs (JSON)", command=self.on_save_outputs).pack(side=tk.LEFT, padx=8)

        self.status = tk.Text(self, height=10)
        self.status.pack(side=tk.BOTTOM, fill=tk.X)

    def _build_plot(self):
        center = ttk.Frame(self, padding=10)
        center.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.fig = plt.Figure(figsize=(11.6, 5.6), dpi=100)
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

            name, l1, l2 = self.tle
            ts = sf_load.timescale()
            self.sat = EarthSatellite(l1, l2, name, ts)

            fallback_min = float(self._get_float(self.var_fallback_uncert_min, "Fallback uncertainty (min)"))
            wmin, wmax, _, mode = compute_tip_window_from_latest_batch(self.solutions_latest, fallback_min)
            self.window_min, self.window_max = wmin, wmax
            self.window_mode = mode

            latest_msg_epoch = self.solutions_latest[0].msg_epoch if self.solutions_latest else ""
            if wmin and wmax:
                width = wmax - wmin
                self._log(f"Latest TIP MSG_EPOCH used: {latest_msg_epoch} (batch rows: {len(self.solutions_latest)})")
                self._log(f"Decay window: {dt_to_iso_z(wmin)} to {dt_to_iso_z(wmax)} (width {fmt_timedelta(width)})")
                self._log(f"Window mode: {mode} (tip_spread / tip_uncertainty / fallback_uncertainty)")
            else:
                self._log(f"Latest TIP MSG_EPOCH: {latest_msg_epoch} (batch rows: {len(self.solutions_latest)})")
                self._log("But no valid DECAY_EPOCH values found in the latest batch.")

            if wmin and wmax:
                hist_csv = history_path(self.out_dir, norad)
                row = {
                    "run_utc": dt_to_iso_z(dt.datetime.now(dt.timezone.utc)),
                    "norad_id": str(norad),
                    "tip_msg_epoch_used": latest_msg_epoch,
                    "latest_batch_count": str(len(self.solutions_latest)),
                    "window_start_utc": dt_to_iso_z(wmin),
                    "window_end_utc": dt_to_iso_z(wmax),
                    "window_width_sec": str(int((wmax - wmin).total_seconds())),
                    "window_mode": mode,
                }
                append_history_row(hist_csv, row)
                self._log(f"Saved history: {hist_csv}")

            self.pred = {}

        except Exception as e:
            messagebox.showerror("Fetch error", str(e))
            self._log(f"ERROR: {e}")

    def on_plot_envelope(self):
        try:
            if not (self.sat and self.solutions_latest and self.window_min and self.window_max):
                raise RuntimeError("Fetch TIP + TLE first (and ensure we have a decay window).")

            before_min = self._get_int(self.var_before, "Window Before (min)")
            after_min = self._get_int(self.var_after, "After (min)")
            step_s = self._get_int(self.var_step, "Step (sec)")
            mid_tracks = max(0, self._get_int(self.var_mid_tracks, "Intermediate tracks"))

            wmin = self.window_min
            wmax = self.window_max
            width = wmax - wmin
            wmid = wmin + (width / 2)

            latest_msg_epoch = self.solutions_latest[0].msg_epoch if self.solutions_latest else ""

            self._setup_map()

            if mid_tracks > 0 and width.total_seconds() > 0:
                for i in range(1, mid_tracks + 1):
                    frac = i / (mid_tracks + 1)
                    t_i = wmin + dt.timedelta(seconds=width.total_seconds() * frac)
                    lats_i, lons_i, _ = groundtrack_corridor(self.sat, t_i, before_min, after_min, step_s)
                    self._plot_track(lats_i, lons_i, linewidth=0.9, alpha=0.25, linestyle="--")

            lats_min, lons_min, _ = groundtrack_corridor(self.sat, wmin, before_min, after_min, step_s)
            lats_max, lons_max, _ = groundtrack_corridor(self.sat, wmax, before_min, after_min, step_s)
            self._plot_track(lats_min, lons_min, linewidth=1.7, alpha=0.92, linestyle="-")
            self._plot_track(lats_max, lons_max, linewidth=1.7, alpha=0.92, linestyle="-")

            sel_lats, sel_lons, _ = groundtrack_corridor(self.sat, wmid, before_min, after_min, step_s)
            self._plot_track(sel_lats, sel_lons, linewidth=2.0, alpha=0.95, linestyle=":")

            title = (
                f"{self.sat.name} — Latest TIP MSG_EPOCH: {latest_msg_epoch}\n"
                f"Decay window: {dt_to_iso_z(wmin)} to {dt_to_iso_z(wmax)} (width {fmt_timedelta(width)}) | mode={self.window_mode}"
            )
            self.ax.set_title(title)
            self.canvas.draw()
            self._log("Envelope plotted.")

        except Exception as e:
            messagebox.showerror("Plot error", str(e))
            self._log(f"ERROR: {e}")

    def on_run_prediction_safe(self):
        try:
            if not (self.sat and self.window_min and self.window_max):
                raise RuntimeError("Fetch TIP + TLE first.")
            if pymsis is None:
                raise RuntimeError("pymsis is required. Install: pip install pymsis")

            wmin, wmax = self.window_min, self.window_max
            width_s = (wmax - wmin).total_seconds()
            if width_s <= 0:
                raise RuntimeError("Zero-width window. Increase 'Fallback uncertainty (min)' and Fetch again.")

            mc_n = max(200, min(50000, self._get_int(self.var_mc_samples, "MC samples")))
            geomag = self._get_int(self.var_geomag, "geomagnetic_activity")
            cdaom = self._get_float(self.var_cdaom, "CdA/m")
            dt_step = self._get_float(self.var_dt_step, "dt_step")
            max_sec = self._get_int(self.var_max_sec, "max_seconds")

            # sample reentry times in the TIP window
            ts_samples = mc_sample_times_with_bias(wmin, wmax, n=mc_n)

            p10 = dt.datetime.fromtimestamp(float(np.percentile(ts_samples, 10)), tz=dt.timezone.utc)
            p50 = dt.datetime.fromtimestamp(float(np.percentile(ts_samples, 50)), tz=dt.timezone.utc)
            p90 = dt.datetime.fromtimestamp(float(np.percentile(ts_samples, 90)), tz=dt.timezone.utc)

            # evaluate descent metrics at p10/p50/p90 using MSIS2.1 density + drag integration
            def eval_one(tt: dt.datetime) -> Dict[str, Any]:
                lat0, lon0 = subpoint_at_time(self.sat, tt)
                descent = integrate_descent_msis21(
                    tt, lat0, lon0,
                    initial_alt_km=120.0,
                    initial_velocity_ms=7800.0,
                    CdA_over_m=cdaom,
                    dt_step=dt_step,
                    max_seconds=max_sec,
                    geomagnetic_activity=geomag,
                )
                return {
                    "time_utc": dt_to_iso_z(tt),
                    "time_ph": dt_to_iso_ph(tt),
                    "subpoint_lat": lat0,
                    "subpoint_lon": lon0,
                    "descent": {
                        "descent_time_s": descent["descent_time_s"],
                        "downrange_km": descent["downrange_km"],
                        "avg_velocity_ms": descent["avg_velocity_ms"],
                        "model_notes": descent["model_notes"],
                    },
                }

            out10 = eval_one(p10)
            out50 = eval_one(p50)
            out90 = eval_one(p90)

            self.pred = {
                "created_utc": dt_to_iso_z(dt.datetime.now(dt.timezone.utc)),
                "norad_id": int(self.var_norad.get().strip()),
                "tip_window": {"start_utc": dt_to_iso_z(wmin), "end_utc": dt_to_iso_z(wmax), "width_sec": int(width_s), "mode": self.window_mode},
                "monte_carlo_times": {"n": int(mc_n), "p10_utc": dt_to_iso_z(p10), "p50_utc": dt_to_iso_z(p50), "p90_utc": dt_to_iso_z(p90)},
                "descent_safe": {"p10": out10, "p50": out50, "p90": out90},
                "note": "Safety: this output does not compute impact lat/lon; it provides descent-time and downrange distance only.",
            }

            # refresh plot
            self.on_plot_envelope()
            # mark only subpoints (safe)
            self.ax.plot([out50["subpoint_lon"]], [out50["subpoint_lat"]], marker="o", markersize=8, transform=ccrs.PlateCarree())
            self.ax.text(out50["subpoint_lon"] + 2, out50["subpoint_lat"] + 2, "P50 subpoint", transform=ccrs.PlateCarree(), fontsize=9)
            self.canvas.draw()

            self._log(f"P50 time: {out50['time_utc']} | {out50['time_ph']}")
            self._log(f"P50 descent time: {out50['descent']['descent_time_s']:.0f}s | downrange: {out50['descent']['downrange_km']:.0f} km")
            self._log(f"MSIS2.1 geomag={geomag} | CdA/m={cdaom} | dt={dt_step}s | max={max_sec}s")

        except Exception as e:
            messagebox.showerror("Run error", str(e))
            self._log(f"ERROR: {e}")

    def on_save_outputs(self):
        try:
            if not self.pred:
                raise RuntimeError("Nothing to save. Run 'Run MSIS2.1 Descent (Safe)' first.")
            norad = int(self.var_norad.get().strip())
            ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = os.path.join(self.out_dir, f"reentry_safe_{norad}_{ts}.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(self.pred, f, indent=2)
            self._log(f"Saved JSON: {out_path}")
            messagebox.showinfo("Saved", f"Saved:\n{out_path}")
        except Exception as e:
            messagebox.showerror("Save error", str(e))
            self._log(f"ERROR: {e}")


def main():
    app = ReentryGUI()
    app.mainloop()


if __name__ == "__main__":
    main()
