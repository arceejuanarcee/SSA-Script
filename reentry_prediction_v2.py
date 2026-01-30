#!/usr/bin/env python3
"""
Reentry Prediction & Assessment Tool (TIP + Sequential TLE + Monte Carlo Window + KML/JSON Outputs)

Design goal:
- Restructure the existing GUI script into a Digantara-like operational pipeline:
  1) Pre-processing: TLE filtering + adaptive drag/BC proxy from sequential TLEs
  2) Core simulation: SGP4 ground track + simple altitude-decay proxy to bracket reentry around TIP window
  3) Uncertainty modelling: Monte Carlo confidence window (percentiles) for time and impact subpoint
  4) Optional breakup proxy: fragment footprint KML (explicitly heuristic)
  5) Outputs: JSON summary + KMLs + PNG (map)

IMPORTANT:
- TIP window is still the primary operator-facing truth.
- The Monte Carlo layer here is an uncertainty model around TIP + drag proxy; it is NOT a full high-fidelity aerothermal reentry.
- This script focuses on "operational style + outputs" similar to Digantara's framework, not proprietary physics.

Requirements:
  pip install requests skyfield matplotlib python-dotenv cartopy numpy simplekml

Optional:
  pip install pymsis   (if you want your earlier MSIS proxy back; kept optional here)

Env vars (or GUI input):
  SPACE_TRACK_USERNAME
  SPACE_TRACK_PASSWORD
  NORAD_CAT_ID
  OUT_DIR=./reentry_out
"""

from __future__ import annotations

import os
import csv
import json
import time
import math
import random
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
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

import cartopy.crs as ccrs
import cartopy.feature as cfeature

from skyfield.api import EarthSatellite, load as sf_load

try:
    import simplekml
except Exception:
    simplekml = None


# -----------------------------
# Config
# -----------------------------
load_dotenv()
LOGIN_URL = "https://www.space-track.org/ajaxauth/login"
PH_TZ = dt.timezone(dt.timedelta(hours=8))


# -----------------------------
# Data models
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
    bstar: float  # TLE B* drag term


# -----------------------------
# Utils
# -----------------------------
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def iso_to_dt_tip(s: str) -> dt.datetime:
    # TIP: "YYYY-MM-DD HH:MM:SS" assumed UTC
    return dt.datetime.strptime(s.strip(), "%Y-%m-%d %H:%M:%S").replace(tzinfo=dt.timezone.utc)

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
        raise RuntimeError("Missing SPACE_TRACK_USERNAME / SPACE_TRACK_PASSWORD.")
    s = requests.Session()
    r = s.post(LOGIN_URL, data={"identity": username, "password": password}, timeout=30)
    r.raise_for_status()
    return s

def parse_tip_solutions(tip_json: list) -> List[TipSolution]:
    sols: List[TipSolution] = []
    for row in tip_json:
        sols.append(
            TipSolution(
                msg_epoch=row.get("MSG_EPOCH") or "",
                decay_epoch=row.get("DECAY_EPOCH") or "",
                rev=int(row["REV"]) if str(row.get("REV", "")).isdigit() else None,
                lat=float(row["LAT"]) if row.get("LAT") not in (None, "") else None,
                lon=float(row["LON"]) if row.get("LON") not in (None, "") else None,
                raw=row,
            )
        )

    def key(sol: TipSolution):
        try:
            return iso_to_dt_tip(sol.msg_epoch).timestamp()
        except Exception:
            return 0.0

    sols.sort(key=key, reverse=True)
    return sols

def fetch_tip(session: requests.Session, norad_id: int, tip_url_override: str = "") -> list:
    if tip_url_override.strip():
        url = tip_url_override.strip()
    else:
        url = (
            f"https://www.space-track.org/basicspacedata/query/class/tip/"
            f"NORAD_CAT_ID/{norad_id}/orderby/MSG_EPOCH%20desc/format/json"
        )
    r = retry_get(session, url)
    txt = r.text.strip()
    return r.json() if txt.startswith("[") else json.loads(txt)

def compute_tip_window(solutions: List[TipSolution], take_n: int = 10) -> Tuple[Optional[dt.datetime], Optional[dt.datetime], List[dt.datetime]]:
    decays: List[dt.datetime] = []
    for s in solutions[:take_n]:
        if s.decay_epoch:
            try:
                decays.append(iso_to_dt_tip(s.decay_epoch))
            except Exception:
                pass
    if not decays:
        return None, None, []
    return min(decays), max(decays), decays

def fetch_tle_history(session: requests.Session, norad_id: int, limit: int = 25) -> List[TLEPoint]:
    """
    Pull a recent sequence of TLEs to estimate an adaptive drag proxy.
    """
    url = (
        f"https://www.space-track.org/basicspacedata/query/class/gp/"
        f"NORAD_CAT_ID/{norad_id}/orderby/EPOCH%20desc/limit/{int(limit)}/format/json"
    )
    r = retry_get(session, url)
    data = r.json()
    if not isinstance(data, list) or not data:
        raise RuntimeError("No TLE history returned.")

    out: List[TLEPoint] = []
    for row in data:
        name = row.get("OBJECT_NAME") or f"NORAD {norad_id}"
        l1 = row.get("TLE_LINE1")
        l2 = row.get("TLE_LINE2")
        epoch = row.get("EPOCH")
        if not (l1 and l2 and epoch):
            continue

        # Space-Track EPOCH example: "2026-01-30 02:14:33"
        try:
            epoch_dt = dt.datetime.strptime(epoch.strip(), "%Y-%m-%d %H:%M:%S").replace(tzinfo=dt.timezone.utc)
        except Exception:
            # sometimes ISO-like
            epoch_dt = dt.datetime.fromisoformat(epoch.replace("Z", "+00:00")).astimezone(dt.timezone.utc)

        # Parse B* from TLE line 1 (columns 54-61-ish, scientific notation without 'E')
        # Safer approach: read substring and convert
        bstar = parse_bstar_from_tle_line1(l1)

        out.append(TLEPoint(epoch_utc=epoch_dt, name=name, l1=l1.strip(), l2=l2.strip(), bstar=bstar))

    # Sort oldest->newest for trend fitting
    out.sort(key=lambda x: x.epoch_utc)
    return out

def parse_bstar_from_tle_line1(l1: str) -> float:
    """
    Standard TLE B* format: columns 54-61 (8 chars) like ' 34123-4'
    which means +0.34123E-4.
    """
    try:
        s = l1[53:61].strip()  # 0-indexed slice; may vary but ok for most TLEs
        # Ensure sign exists
        if len(s) < 7:
            return float("nan")
        # split mantissa and exponent (last 2 chars are exponent with sign)
        mant = s[:-2]
        exp = s[-2:]
        # mantissa can include sign and decimal implied
        sign = 1.0
        if mant[0] == "-":
            sign = -1.0
            mant = mant[1:]
        elif mant[0] == "+":
            mant = mant[1:]

        # implied decimal: 0.mant
        mantissa = sign * float(f"0.{mant}")
        exponent = int(exp)  # includes sign, e.g. -4
        return mantissa * (10 ** exponent)
    except Exception:
        return float("nan")

def robust_bstar_stats(tles: List[TLEPoint]) -> Dict[str, float]:
    """
    Digantara-style 'Adaptive BC from sequential TLE data' proxy:
    we estimate a smoothed B* level and uncertainty from recent TLEs.
    (True BC estimation requires mass/area/Cd and a full density model.)
    """
    vals = np.array([x.bstar for x in tles if np.isfinite(x.bstar)], dtype=float)
    if len(vals) < 5:
        return {"bstar_med": float("nan"), "bstar_mad": float("nan"), "bstar_trend_per_day": float("nan")}

    # Robust center + spread
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med))) + 1e-30

    # Trend (linear) on last N points
    n = min(15, len(tles))
    xs = np.array([(tles[-n+i].epoch_utc - tles[-n].epoch_utc).total_seconds() / 86400 for i in range(n)], dtype=float)
    ys = np.array([tles[-n+i].bstar for i in range(n)], dtype=float)
    m = float(np.polyfit(xs, ys, 1)[0])  # per day
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
# Uncertainty model (Monte Carlo)
# -----------------------------
def mc_sample_reentry_time(
    wmin: dt.datetime,
    wmax: dt.datetime,
    bstar_stats: Dict[str, float],
    n: int = 1000,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Monte Carlo samples a reentry time inside the TIP window.

    The idea:
    - We keep the TIP window [wmin, wmax] as the bounding truth.
    - We bias where inside the window we land, based on:
      (a) B* level (higher drag proxy -> earlier)
      (b) B* trend (increasing -> earlier)
      (c) uncertainty from MAD

    Output:
    - array of UTC timestamps (POSIX seconds)
    """
    if seed is not None:
        np.random.seed(seed)

    width_s = max(1.0, (wmax - wmin).total_seconds())
    mid = wmin + dt.timedelta(seconds=width_s / 2)

    med = bstar_stats.get("bstar_med", float("nan"))
    mad = bstar_stats.get("bstar_mad", float("nan"))
    trend = bstar_stats.get("bstar_trend_per_day", float("nan"))

    # Normalize into a soft bias scalar in [-0.35, +0.35]
    # (tunable; chosen to stay conservative and remain within TIP bounds)
    bias = 0.0
    if np.isfinite(med):
        # Typical B* magnitude can vary widely; we use log scale.
        bias += -0.10 * np.tanh(math.log10(abs(med) + 1e-12) + 7.0)  # heuristic center
    if np.isfinite(trend):
        bias += -0.10 * np.tanh(trend * 5e4)  # trend sensitivity heuristic

    bias = float(np.clip(bias, -0.25, +0.25))

    # Spread: larger MAD -> flatter distribution
    flatness = 0.5
    if np.isfinite(mad) and np.isfinite(med) and abs(med) > 0:
        rel = mad / (abs(med) + 1e-30)
        flatness = float(np.clip(0.35 + 2.0 * rel, 0.35, 0.95))

    # Build a beta distribution over [0,1] and shift its mean by bias
    # Mean of Beta(a,b) = a/(a+b). We map to a,b.
    base_mean = 0.5 + bias
    base_mean = float(np.clip(base_mean, 0.10, 0.90))

    # Concentration controls peaked vs flat
    k = (1.0 - flatness) * 18.0 + 2.0  # in [2..20]
    a = base_mean * k
    b = (1.0 - base_mean) * k

    u = np.random.beta(a, b, size=int(n))
    ts = np.array([wmin.timestamp() + float(x) * width_s for x in u], dtype=float)
    return ts


# -----------------------------
# KML exporters
# -----------------------------
def export_kml_corridor(path: str, tracks: List[Dict[str, Any]]) -> None:
    """
    tracks: [{name: str, lats: [...], lons: [...]}]
    """
    if simplekml is None:
        raise RuntimeError("simplekml not installed. pip install simplekml")

    kml = simplekml.Kml()
    for tr in tracks:
        ls = kml.newlinestring(name=tr["name"])
        coords = list(zip(tr["lons"], tr["lats"]))
        ls.coords = coords
        ls.altitudemode = simplekml.AltitudeMode.clamptoground
        ls.extrude = 0

    kml.save(path)

def export_kml_footprint(path: str, center_lat: float, center_lon: float, radius_km: float, n: int = 72) -> None:
    """
    Simple circular footprint polygon (proxy).
    """
    if simplekml is None:
        raise RuntimeError("simplekml not installed. pip install simplekml")

    # Rough conversion: 1 deg lat ~ 111 km; lon scale by cos(lat)
    lat0 = math.radians(center_lat)
    dlat = radius_km / 111.0
    dlon = radius_km / (111.0 * max(0.2, math.cos(lat0)))

    pts = []
    for i in range(n + 1):
        ang = 2 * math.pi * i / n
        lat = center_lat + dlat * math.sin(ang)
        lon = center_lon + dlon * math.cos(ang)
        lon = ((lon + 180) % 360) - 180
        pts.append((lon, lat))

    kml = simplekml.Kml()
    pol = kml.newpolygon(name=f"Footprint ~{radius_km:.0f} km (proxy)")
    pol.outerboundaryis = pts
    pol.style.polystyle.fill = 0
    kml.save(path)


# -----------------------------
# GUI
# -----------------------------
class ReentryAssessmentGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Re-entry Prediction & Assessment Tool (TIP + Seq TLE + Monte Carlo)")
        self.geometry("1320x900")

        self.out_dir = os.getenv("OUT_DIR", "./reentry_out")
        ensure_dir(self.out_dir)

        # State
        self.tip_raw: Optional[list] = None
        self.solutions: List[TipSolution] = []
        self.window_min: Optional[dt.datetime] = None
        self.window_max: Optional[dt.datetime] = None

        self.tle_hist: List[TLEPoint] = []
        self.bstar_stats: Dict[str, float] = {}

        # UI vars
        self.var_user = tk.StringVar(value=os.getenv("SPACE_TRACK_USERNAME", ""))
        self.var_pass = tk.StringVar(value=os.getenv("SPACE_TRACK_PASSWORD", ""))
        self.var_norad = tk.StringVar(value=os.getenv("NORAD_CAT_ID", "66877"))
        self.var_tip_url = tk.StringVar(value=os.getenv("TIP_URL", ""))

        self.var_tip_rows = tk.StringVar(value="10")
        self.var_tle_hist = tk.StringVar(value="25")

        self.var_before = tk.StringVar(value="90")
        self.var_after = tk.StringVar(value="90")
        self.var_step = tk.StringVar(value="30")
        self.var_mid_tracks = tk.StringVar(value="6")

        self.var_mc_n = tk.StringVar(value="1500")
        self.var_conf_lo = tk.StringVar(value="10")
        self.var_conf_hi = tk.StringVar(value="90")

        # Breakup proxy knobs
        self.var_breakup_enable = tk.BooleanVar(value=False)
        self.var_breakup_radius_km = tk.StringVar(value="250")  # proxy footprint

        # Derived outputs
        self.latest_sat: Optional[EarthSatellite] = None
        self.assessment: Dict[str, Any] = {}

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

        def add_field(label: str, var: tk.StringVar, w: int = 7):
            ttk.Label(row2, text=label).pack(side=tk.LEFT)
            ttk.Entry(row2, textvariable=var, width=w).pack(side=tk.LEFT, padx=(6, 14))

        add_field("TIP rows", self.var_tip_rows, 7)
        add_field("TLE hist", self.var_tle_hist, 7)
        add_field("Before (min)", self.var_before, 7)
        add_field("After (min)", self.var_after, 7)
        add_field("Step (sec)", self.var_step, 7)
        add_field("Mid tracks", self.var_mid_tracks, 7)
        add_field("MC N", self.var_mc_n, 8)
        add_field("Conf lo %", self.var_conf_lo, 7)
        add_field("Conf hi %", self.var_conf_hi, 7)

        row3 = ttk.Frame(self, padding=(10, 0, 10, 10))
        row3.pack(side=tk.TOP, fill=tk.X)

        ttk.Checkbutton(row3, text="Enable breakup footprint (proxy)", variable=self.var_breakup_enable).pack(side=tk.LEFT)
        ttk.Label(row3, text="Footprint radius (km)").pack(side=tk.LEFT, padx=(12, 6))
        ttk.Entry(row3, textvariable=self.var_breakup_radius_km, width=7).pack(side=tk.LEFT)

        btns = ttk.Frame(self, padding=(10, 0, 10, 10))
        btns.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(btns, text="Fetch TIP + TLE History", command=self.on_fetch).pack(side=tk.LEFT)
        ttk.Button(btns, text="Run Assessment (MC + Corridor)", command=self.on_assess).pack(side=tk.LEFT, padx=8)
        ttk.Button(btns, text="Save Outputs (JSON/KML/PNG)…", command=self.on_save).pack(side=tk.LEFT, padx=8)

        self.status = tk.Text(self, height=14)
        self.status.pack(side=tk.BOTTOM, fill=tk.X)
        self._log("Ready. Fetch TIP+TLE history → Run Assessment → Save outputs.")

    def _build_plot(self):
        center = ttk.Frame(self, padding=10)
        center.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.fig = plt.Figure(figsize=(12.0, 5.8), dpi=100)
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

    def on_fetch(self):
        try:
            user = self.var_user.get().strip()
            pw = self.var_pass.get().strip()
            norad = self._get_int(self.var_norad, "NORAD_CAT_ID")
            tip_url_override = self.var_tip_url.get().strip()
            tip_rows = self._get_int(self.var_tip_rows, "TIP rows")
            tle_n = self._get_int(self.var_tle_hist, "TLE hist")

            self._log("Logging in to Space-Track…")
            session = spacetrack_login(user, pw)

            self._log("Fetching TIP…")
            self.tip_raw = fetch_tip(session, norad, tip_url_override)
            self.solutions = parse_tip_solutions(self.tip_raw)
            wmin, wmax, decays = compute_tip_window(self.solutions, take_n=tip_rows)
            self.window_min, self.window_max = wmin, wmax

            if not (wmin and wmax):
                raise RuntimeError("TIP did not yield a usable decay window (DECAY_EPOCH missing/unparsable).")

            self._log(f"TIP window: {dt_to_iso_z(wmin)} → {dt_to_iso_z(wmax)} (width {fmt_timedelta(wmax - wmin)})")
            self._log("Fetching sequential TLE history…")
            self.tle_hist = fetch_tle_history(session, norad, limit=tle_n)
            self.bstar_stats = robust_bstar_stats(self.tle_hist)

            # Build EarthSatellite from most recent TLE
            latest = self.tle_hist[-1]
            ts = sf_load.timescale()
            self.latest_sat = EarthSatellite(latest.l1, latest.l2, latest.name, ts)

            self._log(
                f"TLE hist loaded: {len(self.tle_hist)} | "
                f"B* median={self.bstar_stats.get('bstar_med', float('nan')):.3e} "
                f"MAD={self.bstar_stats.get('bstar_mad', float('nan')):.3e} "
                f"trend/day={self.bstar_stats.get('bstar_trend_per_day', float('nan')):.3e}"
            )

        except Exception as e:
            messagebox.showerror("Fetch error", str(e))
            self._log(f"ERROR: {e}")

    def _plot_track(self, lats: List[float], lons: List[float], linewidth: float, alpha: float):
        for seg_lat, seg_lon in split_dateline_segments(lats, lons):
            self.ax.plot(seg_lon, seg_lat, transform=ccrs.PlateCarree(), linewidth=linewidth, alpha=alpha)

    def on_assess(self):
        try:
            if not (self.latest_sat and self.window_min and self.window_max):
                raise RuntimeError("Fetch TIP + TLE history first.")

            before_min = self._get_int(self.var_before, "Before (min)")
            after_min = self._get_int(self.var_after, "After (min)")
            step_s = self._get_int(self.var_step, "Step (sec)")
            mid_tracks = max(0, min(20, self._get_int(self.var_mid_tracks, "Mid tracks")))

            mc_n = max(200, min(20000, self._get_int(self.var_mc_n, "MC N")))
            conf_lo = float(np.clip(self._get_float(self.var_conf_lo, "Conf lo %"), 0, 49))
            conf_hi = float(np.clip(self._get_float(self.var_conf_hi, "Conf hi %"), 51, 100))

            wmin = self.window_min
            wmax = self.window_max
            width = wmax - wmin
            t_mid = wmin + dt.timedelta(seconds=width.total_seconds() / 2)

            # Monte Carlo confidence window
            ts_samples = mc_sample_reentry_time(wmin, wmax, self.bstar_stats, n=mc_n)
            t_lo = dt.datetime.fromtimestamp(float(np.percentile(ts_samples, conf_lo)), tz=dt.timezone.utc)
            t_hi = dt.datetime.fromtimestamp(float(np.percentile(ts_samples, conf_hi)), tz=dt.timezone.utc)
            t_exp = dt.datetime.fromtimestamp(float(np.percentile(ts_samples, 50)), tz=dt.timezone.utc)

            # Impact subpoints (proxy: subpoint at sampled time; true impact point requires descent model)
            lat_exp, lon_exp = subpoint_at_time(self.latest_sat, t_exp)
            lat_lo, lon_lo = subpoint_at_time(self.latest_sat, t_lo)
            lat_hi, lon_hi = subpoint_at_time(self.latest_sat, t_hi)

            # Plot corridor envelope like your current script, but add MC confidence markers
            self._setup_map()

            # faint intermediates across TIP
            if mid_tracks > 0 and width.total_seconds() > 0:
                for i in range(1, mid_tracks + 1):
                    frac = i / (mid_tracks + 1)
                    t_i = wmin + dt.timedelta(seconds=width.total_seconds() * frac)
                    lats_i, lons_i, _ = groundtrack_corridor(self.latest_sat, t_i, before_min, after_min, step_s)
                    self._plot_track(lats_i, lons_i, linewidth=0.9, alpha=0.22)

            lats_min, lons_min, _ = groundtrack_corridor(self.latest_sat, wmin, before_min, after_min, step_s)
            lats_max, lons_max, _ = groundtrack_corridor(self.latest_sat, wmax, before_min, after_min, step_s)
            self._plot_track(lats_min, lons_min, linewidth=1.7, alpha=0.92)
            self._plot_track(lats_max, lons_max, linewidth=1.7, alpha=0.92)

            # markers: expected + conf bounds
            self.ax.plot([lon_exp], [lat_exp], marker="o", markersize=8, transform=ccrs.PlateCarree())
            self.ax.plot([lon_lo], [lat_lo], marker="o", markersize=6, transform=ccrs.PlateCarree())
            self.ax.plot([lon_hi], [lat_hi], marker="o", markersize=6, transform=ccrs.PlateCarree())
            self.ax.text(lon_exp + 2, lat_exp + 2, "Expected (P50)", transform=ccrs.PlateCarree(), fontsize=9)
            self.ax.text(lon_lo + 2, lat_lo - 2, f"P{conf_lo:.0f}", transform=ccrs.PlateCarree(), fontsize=9)
            self.ax.text(lon_hi + 2, lat_hi - 2, f"P{conf_hi:.0f}", transform=ccrs.PlateCarree(), fontsize=9)

            self.ax.set_title(
                f"{self.latest_sat.name} — TIP window {dt_to_iso_z(wmin)} → {dt_to_iso_z(wmax)} (width {fmt_timedelta(width)})\n"
                f"MC expected: {dt_to_iso_z(t_exp)} | {dt_to_iso_ph(t_exp)}  |  "
                f"Confidence: P{conf_lo:.0f}={dt_to_iso_z(t_lo)} … P{conf_hi:.0f}={dt_to_iso_z(t_hi)}"
            )

            self.canvas.draw()

            # Optional breakup footprint (proxy)
            breakup_enabled = bool(self.var_breakup_enable.get())
            footprint_radius_km = float(self._get_float(self.var_breakup_radius_km, "Footprint radius (km)"))

            self.assessment = {
                "generated_utc": dt_to_iso_z(dt.datetime.now(dt.timezone.utc)),
                "norad_id": int(self.var_norad.get().strip()),
                "tle_used_epoch_utc": dt_to_iso_z(self.tle_hist[-1].epoch_utc) if self.tle_hist else None,
                "tip_window": {
                    "start_utc": dt_to_iso_z(wmin),
                    "end_utc": dt_to_iso_z(wmax),
                    "width_sec": int(width.total_seconds()),
                },
                "adaptive_drag_proxy": {
                    "bstar_median": self.bstar_stats.get("bstar_med"),
                    "bstar_mad": self.bstar_stats.get("bstar_mad"),
                    "bstar_trend_per_day": self.bstar_stats.get("bstar_trend_per_day"),
                    "notes": "Adaptive drag proxy derived from sequential TLE B*; serves as uncertainty bias only (not true ballistic coefficient).",
                },
                "monte_carlo": {
                    "n": int(mc_n),
                    "confidence_percentiles": [conf_lo, 50.0, conf_hi],
                    "t_lo_utc": dt_to_iso_z(t_lo),
                    "t_p50_utc": dt_to_iso_z(t_exp),
                    "t_hi_utc": dt_to_iso_z(t_hi),
                },
                "predicted_impact_proxy": {
                    "t_p50_utc": dt_to_iso_z(t_exp),
                    "lat_deg": lat_exp,
                    "lon_deg": lon_exp,
                    "method": "Subpoint at P50 time (proxy). True impact requires descent/fragmentation physics.",
                },
                "confidence_impact_proxy": {
                    f"p{int(conf_lo)}": {"t_utc": dt_to_iso_z(t_lo), "lat_deg": lat_lo, "lon_deg": lon_lo},
                    f"p{int(conf_hi)}": {"t_utc": dt_to_iso_z(t_hi), "lat_deg": lat_hi, "lon_deg": lon_hi},
                },
                "breakup_proxy": {
                    "enabled": breakup_enabled,
                    "radius_km": footprint_radius_km if breakup_enabled else None,
                    "notes": "Simple footprint circle for ops visualization only (proxy).",
                },
                "corridor": {
                    "minutes_before": before_min,
                    "minutes_after": after_min,
                    "step_seconds": step_s,
                },
            }

            self._log(f"MC expected time: {dt_to_iso_z(t_exp)} | {dt_to_iso_ph(t_exp)}")
            self._log(f"Confidence window: P{conf_lo:.0f}={dt_to_iso_z(t_lo)} … P{conf_hi:.0f}={dt_to_iso_z(t_hi)}")
            self._log(f"Expected impact proxy: lat={lat_exp:.2f}, lon={lon_exp:.2f}")

        except Exception as e:
            messagebox.showerror("Assessment error", str(e))
            self._log(f"ERROR: {e}")

    def on_save(self):
        try:
            if not self.assessment:
                raise RuntimeError("Run Assessment first.")

            folder = filedialog.askdirectory(title="Select folder to save outputs")
            if not folder:
                return

            norad = int(self.var_norad.get().strip())
            stamp = dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")

            # JSON summary (Digantara-like output object)
            json_path = os.path.join(folder, f"assessment_{norad}_{stamp}.json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(self.assessment, f, indent=2)

            # PNG
            png_path = os.path.join(folder, f"corridor_{norad}_{stamp}.png")
            self.fig.savefig(png_path, dpi=220)

            # KML corridor + optional footprint
            if simplekml is not None and self.latest_sat and self.window_min and self.window_max:
                before_min = self._get_int(self.var_before, "Before (min)")
                after_min = self._get_int(self.var_after, "After (min)")
                step_s = self._get_int(self.var_step, "Step (sec)")

                # KML tracks for min/max and expected
                t_exp = dt.datetime.strptime(self.assessment["predicted_impact_proxy"]["t_p50_utc"], "%Y-%m-%d %H:%M:%SZ").replace(tzinfo=dt.timezone.utc)
                tracks = []
                for nm, tt in [("TIP_min", self.window_min), ("TIP_max", self.window_max), ("MC_P50", t_exp)]:
                    lats, lons, _ = groundtrack_corridor(self.latest_sat, tt, before_min, after_min, step_s)
                    tracks.append({"name": nm, "lats": lats, "lons": lons})

                kml_corr = os.path.join(folder, f"corridor_{norad}_{stamp}.kml")
                export_kml_corridor(kml_corr, tracks)

                # Footprint circle (proxy)
                if bool(self.var_breakup_enable.get()):
                    lat = float(self.assessment["predicted_impact_proxy"]["lat_deg"])
                    lon = float(self.assessment["predicted_impact_proxy"]["lon_deg"])
                    rad = float(self._get_float(self.var_breakup_radius_km, "Footprint radius (km)"))
                    kml_fp = os.path.join(folder, f"footprint_proxy_{norad}_{stamp}.kml")
                    export_kml_footprint(kml_fp, lat, lon, rad)

            messagebox.showinfo("Saved", "Outputs saved successfully.")
            self._log(f"Saved: {folder}")

        except Exception as e:
            messagebox.showerror("Save error", str(e))
            self._log(f"ERROR: {e}")


def main():
    app = ReentryAssessmentGUI()
    app.mainloop()

if __name__ == "__main__":
    main()
