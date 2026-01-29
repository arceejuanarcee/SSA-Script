#!/usr/bin/env python3
"""
Reentry Monitor GUI (TIP + TLE) — Map Envelope + NOAA + MSIS density proxy + "Nominal Impact Dot" + Expected Time (biased)

New in this version:
- Computes an "Expected Time" inside the TIP window by biasing the midpoint earlier/later
  based on MSIS density ratio vs quiet (proxy for drag intensity).
- Logs and displays:
  - Window start/end/mid
  - Expected time (UTC + Asia/Manila)
  - Bias explanation + density ratio

Notes:
- TIP window remains the primary truth. Expected time is an estimate to help operators.
- This does NOT make impact point “accurate”; it only biases within the given TIP window.

Requirements:
  pip install requests skyfield matplotlib python-dotenv cartopy pymsis numpy
"""

from __future__ import annotations

import os
import csv
import json
import time
import random
import datetime as dt
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any

import tkinter as tk
from tkinter import ttk, messagebox, filedialog

import requests
from dotenv import load_dotenv

import numpy as np

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

from skyfield.api import EarthSatellite, load as sf_load

import cartopy.crs as ccrs
import cartopy.feature as cfeature

from pymsis import msis  # NRLMSIS wrapper


# -----------------------------
# Load .env early
# -----------------------------
load_dotenv()

LOGIN_URL = "https://www.space-track.org/ajaxauth/login"

# NOAA SWPC endpoints (more stable for ops ingestion)
NOAA_F107 = "https://services.swpc.noaa.gov/json/f107_cm_flux.json"
NOAA_KP_TABLE = "https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json"
NOAA_PRED_A = "https://services.swpc.noaa.gov/json/predicted_fredericksburg_a_index.json"

PH_TZ = dt.timezone(dt.timedelta(hours=8))  # Asia/Manila


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
# Helpers
# -----------------------------
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def iso_to_dt_tip(s: str) -> dt.datetime:
    # TIP: "YYYY-MM-DD HH:MM:SS" (assume UTC)
    return dt.datetime.strptime(s.strip(), "%Y-%m-%d %H:%M:%S").replace(tzinfo=dt.timezone.utc)

def dt_to_iso_z(t: dt.datetime) -> str:
    t = t.astimezone(dt.timezone.utc)
    return t.strftime("%Y-%m-%d %H:%M:%SZ")

def dt_to_iso_ph(t: dt.datetime) -> str:
    t = t.astimezone(PH_TZ)
    return t.strftime("%Y-%m-%d %H:%M:%S (PH)")

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
                msg_epoch=row.get("MSG_EPOCH") or row.get("MSG_EPOCH ", "") or "",
                decay_epoch=row.get("DECAY_EPOCH") or "",
                rev=int(row["REV"]) if str(row.get("REV", "")).isdigit() else None,
                direction=row.get("DIRECTION"),
                lat=float(row["LAT"]) if row.get("LAT") not in (None, "") else None,
                lon=float(row["LON"]) if row.get("LON") not in (None, "") else None,
                incl=float(row.get("INCL")) if row.get("INCL") not in (None, "") else None,
                high_interest=row.get("HIGH_INTEREST"),
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
# NOAA SWPC pulls (fixed parsing)
# -----------------------------
def fetch_noaa_json(url: str) -> Optional[object]:
    try:
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None

def latest_kp() -> Optional[Tuple[str, float]]:
    """
    Uses NOAA "products" table:
      https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json
    Format: [header_row, data_row, data_row, ...]
    Header contains "time_tag" and "Kp".
    """
    data = fetch_noaa_json(NOAA_KP_TABLE)
    if not isinstance(data, list) or len(data) < 2:
        return None
    header = data[0]
    last = data[-1]
    try:
        t = last[header.index("time_tag")]
        kp = float(last[header.index("Kp")])
        return str(t), float(kp)
    except Exception:
        return None

def latest_f107() -> Optional[Tuple[str, float]]:
    data = fetch_noaa_json(NOAA_F107)
    if not isinstance(data, list) or not data:
        return None
    row = data[-1]
    if not isinstance(row, dict):
        return None
    t = row.get("time_tag") or row.get("time") or row.get("date")
    flux = row.get("flux") or row.get("f107") or row.get("value")
    if t is None or flux is None:
        return None
    try:
        return str(t), float(flux)
    except Exception:
        return None

def predicted_a_index_today() -> Optional[Tuple[str, float]]:
    """
    Uses:
      https://services.swpc.noaa.gov/json/predicted_fredericksburg_a_index.json
    Keys include: afred_1_day, afred_2_day, afred_3_day
    We'll take afred_1_day as the operational proxy.
    """
    data = fetch_noaa_json(NOAA_PRED_A)
    if not isinstance(data, list) or not data:
        return None
    for row in data:
        if not isinstance(row, dict):
            continue
        t = row.get("time_tag") or row.get("date")
        a = row.get("afred_1_day")
        if t is not None and a is not None:
            try:
                return str(t), float(a)
            except Exception:
                continue
    return None

def kp_to_ap_rough(kp: float) -> float:
    table = [
        (0.0, 0), (0.33, 2), (0.67, 3), (1.0, 4), (1.33, 5), (1.67, 6),
        (2.0, 7), (2.33, 9), (2.67, 12), (3.0, 15), (3.33, 18), (3.67, 22),
        (4.0, 27), (4.33, 32), (4.67, 39), (5.0, 48), (5.33, 56), (5.67, 67),
        (6.0, 80), (6.33, 94), (6.67, 111), (7.0, 132), (7.33, 154),
        (7.67, 179), (8.0, 207), (8.33, 236), (8.67, 300), (9.0, 400)
    ]
    best = min(table, key=lambda x: abs(x[0] - kp))
    return float(best[1])


# -----------------------------
# MSIS density proxy (pymsis-safe)
# -----------------------------
def msis_density_kg_m3(
    times: List[dt.datetime],
    lats_deg: List[float],
    lons_deg: List[float],
    alt_km: float,
    f107: float,
    ap: float
) -> np.ndarray:
    """
    Version-safe pymsis call using plural inputs:
      - f107s, f107as, aps
    Avoids: create_options() unexpected keyword argument 'ap' / 'f107a'
    """
    dates = np.array([np.datetime64(t.astimezone(dt.timezone.utc).replace(tzinfo=None)) for t in times])
    lats = np.array(lats_deg, dtype=float)
    lons = np.array(lons_deg, dtype=float)
    alts = np.full(len(times), float(alt_km), dtype=float)

    f107s = np.full(len(times), float(f107), dtype=float)
    f107as = np.full(len(times), float(f107), dtype=float)  # fallback if no 81-day centered value
    aps = np.full(len(times), float(ap), dtype=float)

    out = msis.run(
        dates,
        lons,
        lats,
        alts,
        f107s=f107s,
        f107as=f107as,
        aps=aps
    )
    return out[:, -1]  # total mass density (kg/m^3)


# -----------------------------
# Expected time estimation (bias inside TIP window)
# -----------------------------
def bias_fraction_from_density_ratio(ratio: float) -> Tuple[float, str]:
    """
    Returns:
      (bias_fraction, tag)
    bias_fraction is in [-0.35, +0.35] of TIP window width:
      negative -> earlier than mid
      positive -> later than mid
    """
    if not np.isfinite(ratio):
        return 0.0, "No bias (density ratio unavailable)"

    if ratio >= 2.5:
        return -0.30, "HIGH drag → bias earlier"
    if ratio >= 1.8:
        return -0.20, "ELEVATED drag → bias slightly earlier"
    if ratio >= 1.2:
        return -0.10, "MILDLY elevated drag → small earlier bias"
    if ratio >= 0.8:
        return 0.0, "NORMAL drag → no strong bias"
    if ratio >= 0.6:
        return +0.10, "LOW drag → bias slightly later"
    return +0.20, "VERY LOW drag → bias later"

def compute_expected_time_in_window(wmin: dt.datetime, wmax: dt.datetime, density_ratio: float) -> Tuple[dt.datetime, float, str]:
    width = wmax - wmin
    mid = wmin + (width / 2)
    frac, tag = bias_fraction_from_density_ratio(density_ratio)
    expected = mid + dt.timedelta(seconds=width.total_seconds() * frac)
    # clamp to window
    if expected < wmin:
        expected = wmin
    if expected > wmax:
        expected = wmax
    return expected, frac, tag


# -----------------------------
# GUI
# -----------------------------
class ReentryGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Reentry Monitor (TIP + TLE) — Envelope + Nominal Impact Dot + Expected Time")
        self.geometry("1280x860")

        self.out_dir = os.getenv("OUT_DIR", "./reentry_out")
        ensure_dir(self.out_dir)

        # Data state
        self.tip_raw: Optional[list] = None
        self.solutions: List[TipSolution] = []
        self.tle: Optional[Tuple[str, str, str]] = None
        self.window_min: Optional[dt.datetime] = None
        self.window_max: Optional[dt.datetime] = None

        # NOAA/MSIS state
        self.noaa_kp: Optional[Tuple[str, float]] = None
        self.noaa_f107: Optional[Tuple[str, float]] = None
        self.noaa_ap_pred: Optional[Tuple[str, float]] = None

        # Estimation state
        self.drag_density_info: Optional[str] = None
        self.last_density_ratio: Optional[float] = None
        self.expected_time_utc: Optional[dt.datetime] = None

        # UI vars
        self.var_user = tk.StringVar(value=os.getenv("SPACE_TRACK_USERNAME", ""))
        self.var_pass = tk.StringVar(value=os.getenv("SPACE_TRACK_PASSWORD", ""))
        self.var_norad = tk.StringVar(value=os.getenv("NORAD_CAT_ID", "66877"))
        self.var_tip_url = tk.StringVar(value=os.getenv("TIP_URL", ""))

        self.var_before = tk.StringVar(value="90")
        self.var_after = tk.StringVar(value="90")
        self.var_step = tk.StringVar(value="30")

        self.var_take_n = tk.StringVar(value="10")
        self.var_mid_tracks = tk.StringVar(value="6")

        # density knobs
        self.var_density_alt_km = tk.StringVar(value="200")
        self.var_density_samples = tk.StringVar(value="120")

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

        for label, var, w in [
            ("Before (min)", self.var_before, 7),
            ("After (min)", self.var_after, 7),
            ("Step (sec)", self.var_step, 7),
            ("TIP rows", self.var_take_n, 7),
            ("Mid tracks", self.var_mid_tracks, 7),
            ("MSIS alt (km)", self.var_density_alt_km, 7),
            ("Density samples", self.var_density_samples, 7),
        ]:
            ttk.Label(row2, text=label).pack(side=tk.LEFT)
            ttk.Entry(row2, textvariable=var, width=w).pack(side=tk.LEFT, padx=(6, 14))

        btns = ttk.Frame(self, padding=(10, 0, 10, 10))
        btns.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(btns, text="Fetch TIP + TLE", command=self.on_fetch).pack(side=tk.LEFT)
        ttk.Button(btns, text="Fetch NOAA (Kp/F10.7/A)", command=self.on_fetch_noaa).pack(side=tk.LEFT, padx=8)
        ttk.Button(btns, text="Plot + Expected Time", command=self.on_plot).pack(side=tk.LEFT, padx=8)
        ttk.Button(btns, text="Save Outputs…", command=self.on_save_outputs).pack(side=tk.LEFT, padx=8)

        self.status = tk.Text(self, height=14)
        self.status.pack(side=tk.BOTTOM, fill=tk.X)
        self._log("Ready. Fetch TIP+TLE → Fetch NOAA → Plot + Expected Time.")

    def _build_plot(self):
        center = ttk.Frame(self, padding=10)
        center.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.fig = plt.Figure(figsize=(11.8, 5.6), dpi=100)
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
        self.ax.set_title("Reentry corridor envelope (TIP window)")

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
            take_n = self._get_int(self.var_take_n, "TIP rows")

            self._log("Logging in to Space-Track…")
            session = spacetrack_login(user, pw)

            self._log("Fetching TIP…")
            self.tip_raw = fetch_tip(session, norad, tip_url_override)
            self.solutions = parse_tip_solutions(self.tip_raw)

            self._log("Fetching latest TLE…")
            self.tle = fetch_latest_tle(session, norad)

            wmin, wmax, decays = compute_tip_window(self.solutions, take_n=take_n)
            self.window_min, self.window_max = wmin, wmax

            hist_csv = history_path(self.out_dir, norad)
            prev = read_last_history_row(hist_csv)

            if wmin and wmax:
                width = wmax - wmin
                mid = wmin + (width / 2)
                row = {
                    "run_utc": dt_to_iso_z(dt.datetime.now(dt.timezone.utc)),
                    "norad_id": str(norad),
                    "tip_msg_epoch": (self.solutions[0].msg_epoch if self.solutions else ""),
                    "window_start_utc": dt_to_iso_z(wmin),
                    "window_end_utc": dt_to_iso_z(wmax),
                    "window_width_sec": str(int(width.total_seconds())),
                    "window_mid_utc": dt_to_iso_z(mid),
                    "tip_rows_used": str(take_n),
                    "decay_samples": str(len(decays)),
                }
                append_history_row(hist_csv, row)

                self._log(f"TIP window: {dt_to_iso_z(wmin)} → {dt_to_iso_z(wmax)} (width {fmt_timedelta(width)})")
                self._log(f"Window mid (UTC): {dt_to_iso_z(mid)} | {dt_to_iso_ph(mid)}")

                if prev and prev.get("window_mid_utc"):
                    try:
                        prev_mid = dt.datetime.strptime(prev["window_mid_utc"], "%Y-%m-%d %H:%M:%SZ").replace(tzinfo=dt.timezone.utc)
                        shift = mid - prev_mid
                        self._log(f"Shift vs previous mid: {fmt_timedelta(shift)} (positive = later)")
                    except Exception:
                        pass
            else:
                self._log("TIP returned entries but DECAY_EPOCH not usable to form a window.")

            self._log(f"TLE loaded: {self.tle[0] if self.tle else '(none)'} | TIP rows: {len(self.solutions)}")

        except Exception as e:
            messagebox.showerror("Fetch error", str(e))
            self._log(f"ERROR: {e}")

    def on_fetch_noaa(self):
        try:
            self.noaa_kp = latest_kp()
            self.noaa_f107 = latest_f107()
            self.noaa_ap_pred = predicted_a_index_today()

            self._log(f"NOAA Kp: {('unavailable' if not self.noaa_kp else f'{self.noaa_kp[1]:.2f} @ {self.noaa_kp[0]}')}")
            self._log(f"NOAA F10.7: {('unavailable' if not self.noaa_f107 else f'{self.noaa_f107[1]:.1f} sfu @ {self.noaa_f107[0]}')}")
            self._log(f"NOAA predicted A index: {('unavailable' if not self.noaa_ap_pred else f'{self.noaa_ap_pred[1]:.1f} @ {self.noaa_ap_pred[0]}')}")

        except Exception as e:
            messagebox.showerror("NOAA fetch error", str(e))
            self._log(f"ERROR: {e}")

    def _plot_track(self, lats: List[float], lons: List[float], linewidth: float, alpha: float):
        for seg_lat, seg_lon in split_dateline_segments(lats, lons):
            self.ax.plot(seg_lon, seg_lat, transform=ccrs.PlateCarree(), linewidth=linewidth, alpha=alpha)

    def _compute_drag_proxy(self, sat: EarthSatellite, t_ref: dt.datetime) -> Dict[str, Any]:
        """
        Returns dict with:
          - mean_now, mean_quiet, ratio
          - inputs (f107_now, ap_now, kp_now)
          - message (human readable)
        """
        alt_km = self._get_float(self.var_density_alt_km, "MSIS altitude (km)")
        n_samples = self._get_int(self.var_density_samples, "Density samples")
        n_samples = max(20, min(600, n_samples))

        f107_quiet = 70.0
        ap_quiet = 4.0

        f107_now = self.noaa_f107[1] if self.noaa_f107 else 110.0
        kp_now = self.noaa_kp[1] if self.noaa_kp else 3.0
        ap_now = self.noaa_ap_pred[1] if self.noaa_ap_pred else kp_to_ap_rough(kp_now)

        span_min = 20
        start = t_ref - dt.timedelta(minutes=span_min)
        end = t_ref + dt.timedelta(minutes=span_min)
        times = [start + (end - start) * (i / (n_samples - 1)) for i in range(n_samples)]

        ts = sf_load.timescale()
        t_sf = ts.from_datetimes(times)
        sub = sat.at(t_sf).subpoint()
        lats = list(sub.latitude.degrees)
        lons_raw = list(sub.longitude.degrees)
        lons = [((x + 180) % 360) - 180 for x in lons_raw]

        rho_now = msis_density_kg_m3(times, lats, lons, alt_km=alt_km, f107=f107_now, ap=ap_now)
        rho_quiet = msis_density_kg_m3(times, lats, lons, alt_km=alt_km, f107=f107_quiet, ap=ap_quiet)

        mean_now = float(np.nanmean(rho_now))
        mean_quiet = float(np.nanmean(rho_quiet))
        ratio = (mean_now / mean_quiet) if mean_quiet > 0 else float("nan")

        frac, tag = bias_fraction_from_density_ratio(ratio)

        msg = (
            f"MSIS density @ {alt_km:.0f} km: mean={mean_now:.3e} kg/m^3 | "
            f"ratio_vs_quiet={ratio:.2f} | bias={frac:+.2f}×window | {tag} | "
            f"inputs: F10.7={f107_now:.1f}, Ap≈{ap_now:.0f} (Kp={kp_now:.1f})"
        )

        return {
            "mean_now": mean_now,
            "mean_quiet": mean_quiet,
            "ratio": ratio,
            "bias_frac": frac,
            "tag": tag,
            "f107_now": f107_now,
            "ap_now": ap_now,
            "kp_now": kp_now,
            "message": msg,
        }

    def on_plot(self):
        try:
            if not self.tle or not self.window_min or not self.window_max:
                raise RuntimeError("Fetch TIP + TLE first (must have a valid TIP window).")

            before_min = self._get_int(self.var_before, "Before (min)")
            after_min = self._get_int(self.var_after, "After (min)")
            step_s = self._get_int(self.var_step, "Step (sec)")
            mid_tracks = self._get_int(self.var_mid_tracks, "Mid tracks")
            mid_tracks = max(0, min(20, mid_tracks))

            name, l1, l2 = self.tle
            ts = sf_load.timescale()
            sat = EarthSatellite(l1, l2, name, ts)

            wmin = self.window_min
            wmax = self.window_max
            width = wmax - wmin
            t_mid = wmin + (width / 2)

            self._setup_map()

            # Intermediate tracks (faint)
            if mid_tracks > 0 and width.total_seconds() > 0:
                for i in range(1, mid_tracks + 1):
                    frac = i / (mid_tracks + 1)
                    t_i = wmin + dt.timedelta(seconds=width.total_seconds() * frac)
                    lats_i, lons_i, _ = groundtrack_corridor(sat, t_i, before_min, after_min, step_s)
                    self._plot_track(lats_i, lons_i, linewidth=0.9, alpha=0.22)

            # Min/Max tracks (bold)
            lats_min, lons_min, _ = groundtrack_corridor(sat, wmin, before_min, after_min, step_s)
            lats_max, lons_max, _ = groundtrack_corridor(sat, wmax, before_min, after_min, step_s)
            self._plot_track(lats_min, lons_min, linewidth=1.7, alpha=0.92)
            self._plot_track(lats_max, lons_max, linewidth=1.7, alpha=0.92)

            # Compute density ratio + expected time bias (best-effort)
            density_ratio = float("nan")
            bias_frac = 0.0
            bias_tag = "No bias (MSIS not available)"
            density_msg = None

            try:
                drag = self._compute_drag_proxy(sat, t_mid)
                density_ratio = float(drag["ratio"])
                bias_frac = float(drag["bias_frac"])
                bias_tag = str(drag["tag"])
                density_msg = str(drag["message"])
                self.last_density_ratio = density_ratio
                self.drag_density_info = density_msg
                self._log(density_msg)
            except Exception as e:
                self._log(f"MSIS proxy skipped: {e}")

            expected_time, applied_frac, applied_tag = compute_expected_time_in_window(wmin, wmax, density_ratio)
            self.expected_time_utc = expected_time

            # Dots at: expected, mid, min/max
            lat_exp, lon_exp = subpoint_at_time(sat, expected_time)
            lat_mid, lon_mid = subpoint_at_time(sat, t_mid)
            lat_a, lon_a = subpoint_at_time(sat, wmin)
            lat_b, lon_b = subpoint_at_time(sat, wmax)

            # Plot dots (no custom colors)
            self.ax.plot([lon_exp], [lat_exp], marker="o", markersize=8, transform=ccrs.PlateCarree())
            self.ax.plot([lon_mid], [lat_mid], marker="o", markersize=6, transform=ccrs.PlateCarree())
            self.ax.plot([lon_a], [lat_a], marker="o", markersize=5, transform=ccrs.PlateCarree())
            self.ax.plot([lon_b], [lat_b], marker="o", markersize=5, transform=ccrs.PlateCarree())

            # Labels
            self.ax.text(lon_exp + 2, lat_exp + 2, "Expected time (biased)", transform=ccrs.PlateCarree(), fontsize=9)
            self.ax.text(lon_mid + 2, lat_mid - 2, "Mid-window", transform=ccrs.PlateCarree(), fontsize=9)

            # Title includes expected time
            self.ax.set_title(
                f"{name} — TIP window {dt_to_iso_z(wmin)} → {dt_to_iso_z(wmax)} (width {fmt_timedelta(width)})\n"
                f"Expected time (biased): {dt_to_iso_z(expected_time)} | {dt_to_iso_ph(expected_time)}"
            )
            self.canvas.draw()

            self._log(f"Expected time (UTC): {dt_to_iso_z(expected_time)} | {dt_to_iso_ph(expected_time)} | {applied_tag}")
            self._log(f"Expected dot: lat={lat_exp:.2f}, lon={lon_exp:.2f} (subpoint at expected time; NOT guaranteed impact)")
            self._log(f"Mid-window time: {dt_to_iso_z(t_mid)} | Mid dot: lat={lat_mid:.2f}, lon={lon_mid:.2f}")
            self._log(f"Min/Max times: {dt_to_iso_z(wmin)} and {dt_to_iso_z(wmax)}")
            self._log(f"Min/Max dots: ({lat_a:.2f},{lon_a:.2f}) and ({lat_b:.2f},{lon_b:.2f})")

        except Exception as e:
            messagebox.showerror("Plot error", str(e))
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
                tip_path = os.path.join(folder, f"tip_{norad}_{stamp}.json")
                with open(tip_path, "w", encoding="utf-8") as f:
                    json.dump(self.tip_raw, f, indent=2)

            if self.tle is not None:
                name, l1, l2 = self.tle
                tle_path = os.path.join(folder, f"tle_{norad}_{stamp}.txt")
                with open(tle_path, "w", encoding="utf-8") as f:
                    f.write(name + "\n" + l1 + "\n" + l2 + "\n")

            noaa_path = os.path.join(folder, f"noaa_{norad}_{stamp}.json")
            noaa_obj = {
                "kp_latest": self.noaa_kp,
                "f107_latest": self.noaa_f107,
                "predicted_a_index": self.noaa_ap_pred,
                "msis_drag_proxy": self.drag_density_info,
                "expected_time_utc": (dt_to_iso_z(self.expected_time_utc) if self.expected_time_utc else None),
                "expected_time_ph": (dt_to_iso_ph(self.expected_time_utc) if self.expected_time_utc else None),
                "sources": {"kp": NOAA_KP_TABLE, "f107": NOAA_F107, "pred_a": NOAA_PRED_A},
            }
            with open(noaa_path, "w", encoding="utf-8") as f:
                json.dump(noaa_obj, f, indent=2)

            png_path = os.path.join(folder, f"corridor_envelope_{norad}_{stamp}.png")
            self.fig.savefig(png_path, dpi=220)

            messagebox.showinfo("Saved", "Outputs saved successfully.")
            self._log(f"Saved outputs to: {folder}")

        except Exception as e:
            messagebox.showerror("Save error", str(e))
            self._log(f"ERROR: {e}")


def main():
    app = ReentryGUI()
    app.mainloop()


if __name__ == "__main__":
    main()
