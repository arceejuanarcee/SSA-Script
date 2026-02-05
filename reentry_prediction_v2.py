#!/usr/bin/env python3
"""
Reentry Prediction & Assessment Tool (TIP + Latest TLE) — EU SST-aligned outputs
Outputs aligned to EU SST "Product Viewer":
  - Re-entry Point (centre of window)  -> KML Point
  - Re-entry Ground Track              -> KML LineString
  - Re-entry Swath (±X km)             -> KML Polygon (buffer around ground track)

Key alignment choices (to match EU SST visuals/logic):
1) TIP "latest MSG_EPOCH batch only" is the operational truth.
2) Reentry epoch = centre (midpoint) of the TIP decay window.
3) Ground track is computed around the reentry epoch (configurable ± minutes).
4) Swath is a geodesic-ish buffer (left/right offsets) around that ground track, default ±100 km.

Notes:
- This is NOT a high-fidelity aerothermal model; it's a clean ops-style product generator.
- The swath buffering uses spherical Earth great-circle destination calculations (good for ops maps).

Requirements:
  pip install requests skyfield matplotlib python-dotenv cartopy numpy simplekml

Env vars (via .env or system env):
  SPACE_TRACK_USERNAME
  SPACE_TRACK_PASSWORD
Optional:
  NORAD_CAT_ID=66877
  OUT_DIR=./reentry_out
  TIP_URL=... (override full URL)
  TIP_LIMIT=200
"""

from __future__ import annotations

import os
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

import cartopy.crs as ccrs
import cartopy.feature as cfeature

from skyfield.api import EarthSatellite, load as sf_load

import simplekml


# -----------------------------
# Config
# -----------------------------
load_dotenv()
LOGIN_URL = "https://www.space-track.org/ajaxauth/login"
DEFAULT_TIP_LIMIT = int(os.getenv("TIP_LIMIT", "200"))
OUT_DIR = os.getenv("OUT_DIR", "./reentry_out")

EARTH_RADIUS_KM = 6371.0088
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
                msg_epoch=(row.get("MSG_EPOCH") or "").strip(),
                decay_epoch=(row.get("DECAY_EPOCH") or "").strip(),
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

def select_latest_tip_batch(solutions: List[TipSolution]) -> List[TipSolution]:
    if not solutions:
        return []
    newest = solutions[0].msg_epoch
    if not newest:
        return solutions[:1]
    return [s for s in solutions if s.msg_epoch == newest]

def fetch_tip(session: requests.Session, norad_id: int, tip_url_override: str = "", limit: int = DEFAULT_TIP_LIMIT) -> list:
    if tip_url_override.strip():
        url = tip_url_override.strip()
    else:
        url = (
            f"https://www.space-track.org/basicspacedata/query/class/tip/"
            f"NORAD_CAT_ID/{norad_id}/orderby/MSG_EPOCH%20desc/limit/{int(limit)}/format/json"
        )
    r = retry_get(session, url)
    txt = r.text.strip()
    return r.json() if txt.startswith("[") else json.loads(txt)

def fetch_latest_tle(session: requests.Session, norad_id: int) -> Tuple[str, str, str, dt.datetime]:
    """
    Returns (name, line1, line2, tle_epoch_utc) from latest GP entry.
    """
    url = (
        f"https://www.space-track.org/basicspacedata/query/class/gp/"
        f"NORAD_CAT_ID/{norad_id}/orderby/EPOCH%20desc/limit/1/format/json"
    )
    r = retry_get(session, url)
    data = r.json()
    if not isinstance(data, list) or not data:
        raise RuntimeError("No latest TLE JSON returned.")
    row = data[0]
    name = row.get("OBJECT_NAME") or f"NORAD {norad_id}"
    l1 = row.get("TLE_LINE1")
    l2 = row.get("TLE_LINE2")
    epoch_s = row.get("EPOCH")
    if not (l1 and l2 and epoch_s):
        raise RuntimeError("Latest TLE record missing fields.")
    tle_epoch = dt.datetime.strptime(epoch_s.strip(), "%Y-%m-%d %H:%M:%S").replace(tzinfo=dt.timezone.utc)
    return name, l1.strip(), l2.strip(), tle_epoch

def compute_tip_window_from_batch(batch: List[TipSolution]) -> Tuple[Optional[dt.datetime], Optional[dt.datetime], List[dt.datetime]]:
    decays: List[dt.datetime] = []
    for s in batch:
        if s.decay_epoch:
            try:
                decays.append(iso_to_dt_tip(s.decay_epoch))
            except Exception:
                pass
    if not decays:
        return None, None, []
    return min(decays), max(decays), decays

def normalize_lon(lon: float) -> float:
    return ((lon + 180) % 360) - 180

def split_dateline_segments(lats: List[float], lons: List[float], jump_deg: float = 180.0):
    if not lats or not lons or len(lats) != len(lons):
        return []
    segs = []
    cur_lat = [lats[0]]
    cur_lon = [lons[0]]
    for i in range(1, len(lats)):
        if abs(lons[i] - lons[i - 1]) > jump_deg:
            segs.append((cur_lat, cur_lon))
            cur_lat = [lats[i]]
            cur_lon = [lons[i]]
        else:
            cur_lat.append(lats[i])
            cur_lon.append(lons[i])
    segs.append((cur_lat, cur_lon))
    return segs


# -----------------------------
# Spherical navigation helpers
# -----------------------------
def bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Initial bearing from point1 to point2 (degrees from north).
    """
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dlon = math.radians(lon2 - lon1)

    y = math.sin(dlon) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlon)
    brng = math.degrees(math.atan2(y, x))
    return (brng + 360.0) % 360.0

def destination_point(lat: float, lon: float, bearing: float, distance_km: float) -> Tuple[float, float]:
    """
    Great-circle destination on a sphere.
    """
    ang = distance_km / EARTH_RADIUS_KM
    brng = math.radians(bearing)
    phi1 = math.radians(lat)
    lam1 = math.radians(lon)

    phi2 = math.asin(math.sin(phi1) * math.cos(ang) + math.cos(phi1) * math.sin(ang) * math.cos(brng))
    lam2 = lam1 + math.atan2(
        math.sin(brng) * math.sin(ang) * math.cos(phi1),
        math.cos(ang) - math.sin(phi1) * math.sin(phi2),
    )

    lat2 = math.degrees(phi2)
    lon2 = normalize_lon(math.degrees(lam2))
    return lat2, lon2


# -----------------------------
# Orbit / ground track
# -----------------------------
def groundtrack_for_span(
    sat: EarthSatellite,
    t_center: dt.datetime,
    span_minutes: int,
    step_seconds: int,
) -> Tuple[List[float], List[float], List[dt.datetime]]:
    """
    Ground track for [t_center - span_minutes, t_center + span_minutes]
    """
    ts = sf_load.timescale()
    start = t_center - dt.timedelta(minutes=span_minutes)
    end = t_center + dt.timedelta(minutes=span_minutes)

    times_dt: List[dt.datetime] = []
    cur = start
    while cur <= end:
        times_dt.append(cur)
        cur += dt.timedelta(seconds=step_seconds)

    t_sf = ts.from_datetimes(times_dt)
    sub = sat.at(t_sf).subpoint()

    lats = list(sub.latitude.degrees)
    lons_raw = list(sub.longitude.degrees)
    lons = [normalize_lon(x) for x in lons_raw]
    return lats, lons, times_dt

def subpoint_at_time(sat: EarthSatellite, t_utc: dt.datetime) -> Tuple[float, float]:
    ts = sf_load.timescale()
    t_sf = ts.from_datetime(t_utc)
    sub = sat.at(t_sf).subpoint()
    lat = float(sub.latitude.degrees)
    lon = normalize_lon(float(sub.longitude.degrees))
    return lat, lon


# -----------------------------
# Swath builder (±width_km around a polyline)
# -----------------------------
def build_swath_polygon(
    track_lats: List[float],
    track_lons: List[float],
    half_width_km: float,
) -> Tuple[List[float], List[float]]:
    """
    Build a single polygon (lat/lon lists) that buffers the track by ±half_width_km.
    Output polygon is closed (first point repeated at end).
    """
    if len(track_lats) < 3:
        raise ValueError("Track too short for swath polygon.")

    left_pts: List[Tuple[float, float]] = []
    right_pts: List[Tuple[float, float]] = []

    n = len(track_lats)
    for i in range(n):
        if i == 0:
            brng = bearing_deg(track_lats[i], track_lons[i], track_lats[i + 1], track_lons[i + 1])
        elif i == n - 1:
            brng = bearing_deg(track_lats[i - 1], track_lons[i - 1], track_lats[i], track_lons[i])
        else:
            b1 = bearing_deg(track_lats[i - 1], track_lons[i - 1], track_lats[i], track_lons[i])
            b2 = bearing_deg(track_lats[i], track_lons[i], track_lats[i + 1], track_lons[i + 1])
            # average bearing safely (vector average)
            x = math.cos(math.radians(b1)) + math.cos(math.radians(b2))
            y = math.sin(math.radians(b1)) + math.sin(math.radians(b2))
            brng = (math.degrees(math.atan2(y, x)) + 360.0) % 360.0

        left_bearing = (brng - 90.0) % 360.0
        right_bearing = (brng + 90.0) % 360.0

        latL, lonL = destination_point(track_lats[i], track_lons[i], left_bearing, half_width_km)
        latR, lonR = destination_point(track_lats[i], track_lons[i], right_bearing, half_width_km)

        left_pts.append((latL, lonL))
        right_pts.append((latR, lonR))

    # polygon: left forward + right backward
    poly = left_pts + right_pts[::-1] + [left_pts[0]]
    poly_lats = [p[0] for p in poly]
    poly_lons = [p[1] for p in poly]
    return poly_lats, poly_lons


# -----------------------------
# KML export (EU SST style: Point + Line + Swath)
# -----------------------------
def export_products_kml(
    out_path: str,
    obj_name: str,
    reentry_epoch_utc: dt.datetime,
    window_start_utc: dt.datetime,
    window_end_utc: dt.datetime,
    point_lat: float,
    point_lon: float,
    track_lats: List[float],
    track_lons: List[float],
    swath_poly_lats: List[float],
    swath_poly_lons: List[float],
    swath_half_width_km: float,
) -> None:
    kml = simplekml.Kml()
    kml.document.name = f"{obj_name} Reentry Products"

    desc = (
        f"Object: {obj_name}\n"
        f"Reentry epoch (centre of window): {dt_to_iso_z(reentry_epoch_utc)} | {dt_to_iso_ph(reentry_epoch_utc)}\n"
        f"Window start: {dt_to_iso_z(window_start_utc)}\n"
        f"Window end:   {dt_to_iso_z(window_end_utc)}\n"
        f"Swath: ±{swath_half_width_km:.0f} km\n"
        "CRS: WGS84 (EPSG:4326)\n"
    )

    fol = kml.newfolder(name="Products")

    # Point (centre of window)
    p = fol.newpoint(name="Re-entry Point (centre of window)", coords=[(point_lon, point_lat)])
    p.description = desc
    p.style.iconstyle.scale = 1.2
    p.style.iconstyle.icon.href = "http://maps.google.com/mapfiles/kml/shapes/target.png"

    # Ground track
    ls = fol.newlinestring(name="Re-entry Ground Track")
    ls.description = desc
    ls.coords = list(zip(track_lons, track_lats))
    ls.altitudemode = simplekml.AltitudeMode.clamptoground
    ls.extrude = 0
    ls.style.linestyle.width = 4
    ls.style.linestyle.color = simplekml.Color.rgb(0, 180, 0)  # green-ish

    # Swath polygon
    pol = fol.newpolygon(name=f"Re-entry Swath (±{int(round(swath_half_width_km))} km)")
    pol.description = desc
    pol.outerboundaryis = list(zip(swath_poly_lons, swath_poly_lats))
    pol.altitudemode = simplekml.AltitudeMode.clamptoground
    pol.style.linestyle.width = 2
    pol.style.linestyle.color = simplekml.Color.rgb(0, 120, 255)  # blue-ish outline
    pol.style.polystyle.color = simplekml.Color.changealphaint(70, simplekml.Color.rgb(0, 120, 255))  # semi-transparent fill

    kml.save(out_path)


# -----------------------------
# GUI
# -----------------------------
class ReentryProductsGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Reentry Products (TIP + TLE) — EU SST-aligned (Point + Track + Swath + KML)")
        self.geometry("1320x900")

        ensure_dir(OUT_DIR)

        # State
        self.tip_raw: Optional[list] = None
        self.solutions_all: List[TipSolution] = []
        self.solutions_latest: List[TipSolution] = []

        self.window_start: Optional[dt.datetime] = None
        self.window_end: Optional[dt.datetime] = None
        self.window_mid: Optional[dt.datetime] = None

        self.obj_name: str = ""
        self.tle_epoch: Optional[dt.datetime] = None
        self.latest_sat: Optional[EarthSatellite] = None

        # Cached products
        self.prod_point: Optional[Tuple[float, float]] = None
        self.prod_track: Tuple[List[float], List[float]] = ([], [])
        self.prod_swath: Tuple[List[float], List[float]] = ([], [])

        self.assessment: Dict[str, Any] = {}

        # UI vars
        self.var_user = tk.StringVar(value=os.getenv("SPACE_TRACK_USERNAME", ""))
        self.var_pass = tk.StringVar(value=os.getenv("SPACE_TRACK_PASSWORD", ""))
        self.var_norad = tk.StringVar(value=os.getenv("NORAD_CAT_ID", "66877"))
        self.var_tip_url = tk.StringVar(value=os.getenv("TIP_URL", ""))

        self.var_step = tk.StringVar(value="30")          # seconds
        self.var_span = tk.StringVar(value="90")          # minutes around reentry epoch
        self.var_swath = tk.StringVar(value="100")        # km (±)
        self.var_ph_focus = tk.BooleanVar(value=False)    # optional zoom toggle

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

        row2 = ttk.Frame(self, padding=(10, 0, 10, 10))
        row2.pack(side=tk.TOP, fill=tk.X)

        def add_field(label: str, var: tk.StringVar, w: int = 8):
            ttk.Label(row2, text=label).pack(side=tk.LEFT)
            ttk.Entry(row2, textvariable=var, width=w).pack(side=tk.LEFT, padx=(6, 14))

        add_field("Step (sec)", self.var_step, 8)
        add_field("Track span ± (min)", self.var_span, 10)
        add_field("Swath ± (km)", self.var_swath, 9)

        ttk.Checkbutton(row2, text="Philippines focus (zoom)", variable=self.var_ph_focus, command=self._apply_extent).pack(side=tk.LEFT, padx=12)

        btns = ttk.Frame(self, padding=(10, 0, 10, 10))
        btns.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(btns, text="Fetch TIP + Latest TLE", command=self.on_fetch).pack(side=tk.LEFT)
        ttk.Button(btns, text="Generate Products (Point + Track + Swath)", command=self.on_generate).pack(side=tk.LEFT, padx=8)
        ttk.Button(btns, text="Save Outputs (JSON/KML/PNG)…", command=self.on_save).pack(side=tk.LEFT, padx=8)

        self.status = tk.Text(self, height=14)
        self.status.pack(side=tk.BOTTOM, fill=tk.X)
        self._log("Ready. Fetch → Generate Products → Save KML for your colleague/QGIS.")

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
        self._apply_extent()

    def _apply_extent(self):
        if getattr(self, "ax", None) is None:
            return
        if self.var_ph_focus.get():
            self.ax.set_extent([115, 135, 0, 25], crs=ccrs.PlateCarree())
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

            if not self.solutions_latest:
                raise RuntimeError("No TIP solutions in latest batch.")

            wmin, wmax, decays = compute_tip_window_from_batch(self.solutions_latest)
            if not (wmin and wmax):
                raise RuntimeError("Latest TIP batch has no usable DECAY_EPOCH values to form a window.")

            self.window_start = wmin
            self.window_end = wmax
            self.window_mid = wmin + (wmax - wmin) / 2

            latest_msg_epoch = self.solutions_latest[0].msg_epoch
            self._log(f"Latest TIP MSG_EPOCH used: {latest_msg_epoch} | batch rows: {len(self.solutions_latest)}")
            self._log(f"Window start: {dt_to_iso_z(wmin)} | end: {dt_to_iso_z(wmax)} | width: {fmt_timedelta(wmax-wmin)}")
            self._log(f"Reentry epoch (centre of window): {dt_to_iso_z(self.window_mid)} | {dt_to_iso_ph(self.window_mid)}")

            self._log("Fetching latest TLE…")
            name, l1, l2, tle_epoch = fetch_latest_tle(session, norad)
            self.obj_name = name
            self.tle_epoch = tle_epoch

            ts = sf_load.timescale()
            self.latest_sat = EarthSatellite(l1, l2, name, ts)

            self._log(f"TLE used epoch: {dt_to_iso_z(tle_epoch)} | Object: {name}")

        except Exception as e:
            messagebox.showerror("Fetch error", str(e))
            self._log(f"ERROR: {e}")

    def _plot_track(self, lats: List[float], lons: List[float], linewidth: float, alpha: float, linestyle: str = "-"):
        for seg_lat, seg_lon in split_dateline_segments(lats, lons):
            self.ax.plot(seg_lon, seg_lat, transform=ccrs.PlateCarree(),
                         linewidth=linewidth, alpha=alpha, linestyle=linestyle)

    def on_generate(self):
        try:
            if not (self.latest_sat and self.window_start and self.window_end and self.window_mid):
                raise RuntimeError("Fetch TIP + latest TLE first.")

            step_s = max(5, min(600, self._get_int(self.var_step, "Step (sec)")))
            span_min = max(5, min(360, self._get_int(self.var_span, "Track span (min)")))
            swath_km = max(1.0, min(2000.0, self._get_float(self.var_swath, "Swath (km)")))

            re_t = self.window_mid

            # Re-entry point = subpoint at centre-of-window time (EU SST style)
            lat0, lon0 = subpoint_at_time(self.latest_sat, re_t)
            self.prod_point = (lat0, lon0)

            # Ground track around reentry epoch
            lats, lons, times_dt = groundtrack_for_span(self.latest_sat, re_t, span_minutes=span_min, step_seconds=step_s)
            self.prod_track = (lats, lons)

            # Swath polygon around that track (±swath_km)
            poly_lats, poly_lons = build_swath_polygon(lats, lons, half_width_km=swath_km)
            self.prod_swath = (poly_lats, poly_lons)

            # Plot (similar layering: swath, track, point)
            self._setup_map()

            # Swath (plot outline only; fill would need extra handling in matplotlib)
            for seg_lat, seg_lon in split_dateline_segments(poly_lats, poly_lons):
                self.ax.plot(seg_lon, seg_lat, transform=ccrs.PlateCarree(),
                             linewidth=1.5, alpha=0.9, linestyle="-")

            # Track
            self._plot_track(lats, lons, linewidth=2.2, alpha=0.95, linestyle="-")

            # Point
            self.ax.plot([lon0], [lat0], marker="o", markersize=7, transform=ccrs.PlateCarree(), linestyle="None")

            title = (
                f"{self.obj_name}\n"
                f"Reentry epoch (centre): {dt_to_iso_z(re_t)} | Window: {dt_to_iso_z(self.window_start)} → {dt_to_iso_z(self.window_end)} | "
                f"Swath: ±{int(round(swath_km))} km"
            )
            self.ax.set_title(title)
            self.canvas.draw()

            self.assessment = {
                "generated_utc": dt_to_iso_z(dt.datetime.now(dt.timezone.utc)),
                "norad_id": int(self.var_norad.get().strip()),
                "object_name": self.obj_name,
                "tle_used_epoch_utc": dt_to_iso_z(self.tle_epoch) if self.tle_epoch else None,
                "tip_latest_msg_epoch": self.solutions_latest[0].msg_epoch if self.solutions_latest else None,
                "tip_window": {
                    "start_utc": dt_to_iso_z(self.window_start),
                    "end_utc": dt_to_iso_z(self.window_end),
                    "width_sec": int((self.window_end - self.window_start).total_seconds()),
                    "centre_utc": dt_to_iso_z(self.window_mid),
                },
                "products": {
                    "reentry_point_centre_of_window": {"lat_deg": lat0, "lon_deg": lon0},
                    "ground_track": {
                        "span_minutes_each_side": span_min,
                        "step_seconds": step_s,
                        "num_points": len(lats),
                    },
                    "swath": {"half_width_km": swath_km, "method": "spherical left/right offsets from track bearings"},
                },
                "notes": [
                    "EU SST-aligned operational products: point=centre-of-window, track around point, swath=±width around track.",
                    "This is not a high-fidelity reentry physics model.",
                ],
            }

            self._log(f"Generated products. Point: lat={lat0:.4f}, lon={lon0:.4f}. Track points: {len(lats)}. Swath: ±{swath_km:.0f} km.")

        except Exception as e:
            messagebox.showerror("Generate error", str(e))
            self._log(f"ERROR: {e}")

    def on_save(self):
        try:
            if not (self.assessment and self.prod_point and self.prod_track[0] and self.prod_swath[0]):
                raise RuntimeError("Generate Products first.")

            folder = filedialog.askdirectory(title="Select folder to save outputs")
            if not folder:
                return

            norad = int(self.var_norad.get().strip())
            stamp = dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")

            # JSON
            json_path = os.path.join(folder, f"products_{norad}_{stamp}.json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(self.assessment, f, indent=2)

            # PNG
            png_path = os.path.join(folder, f"products_{norad}_{stamp}.png")
            self.fig.savefig(png_path, dpi=220)

            # KML (single file containing Point+Track+Swath)
            lat0, lon0 = self.prod_point
            lats, lons = self.prod_track
            poly_lats, poly_lons = self.prod_swath
            swath_km = float(self.assessment["products"]["swath"]["half_width_km"])

            kml_path = os.path.join(folder, f"products_{norad}_{stamp}.kml")
            export_products_kml(
                out_path=kml_path,
                obj_name=self.obj_name,
                reentry_epoch_utc=self.window_mid,  # centre-of-window
                window_start_utc=self.window_start,
                window_end_utc=self.window_end,
                point_lat=lat0,
                point_lon=lon0,
                track_lats=lats,
                track_lons=lons,
                swath_poly_lats=poly_lats,
                swath_poly_lons=poly_lons,
                swath_half_width_km=swath_km,
            )

            messagebox.showinfo("Saved", "Saved JSON + PNG + KML successfully.")
            self._log(f"Saved JSON: {json_path}")
            self._log(f"Saved PNG : {png_path}")
            self._log(f"Saved KML : {kml_path}")

        except Exception as e:
            messagebox.showerror("Save error", str(e))
            self._log(f"ERROR: {e}")


def main():
    app = ReentryProductsGUI()
    app.mainloop()


if __name__ == "__main__":
    main()
