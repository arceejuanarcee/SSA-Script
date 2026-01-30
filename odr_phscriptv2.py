#!/usr/bin/env python3
"""
Reentry Monitor GUI (TIP + TLE) — LATEST TIP ONLY + corridor envelope map
ENHANCED: PH zoom + export plotted paths to KML for GIS (QGIS/Google Earth)

What this version does:
- Fetch TIP + latest TLE from Space-Track
- Uses ONLY newest TIP MSG_EPOCH batch to compute decay window (min/max DECAY_EPOCH)
- Plots:
  - envelope min track (bold)
  - envelope max track (bold)
  - selected midpoint track (dotted)
  - optional intermediate tracks (faint dashed)
- Philippines Focus toggle: zoom to PH extent (115E–130E, 4N–22N)
- Exports the plotted tracks to KML:
  - envelope_min_*.kml
  - envelope_max_*.kml
  - selected_midpoint_*.kml

Requirements:
  pip install requests skyfield matplotlib python-dotenv cartopy

Env vars (via .env or system env):
  SPACE_TRACK_USERNAME
  SPACE_TRACK_PASSWORD

Optional env vars:
  OUT_DIR=./reentry_out
  NORAD_CAT_ID=66877
  TIP_URL=... (override full URL)
  TIP_LIMIT=200
"""

from __future__ import annotations

import os
import csv
import json
import time
import random
import datetime as dt
from dataclasses import dataclass
from typing import List, Optional, Tuple

import tkinter as tk
from tkinter import ttk, messagebox, filedialog

import requests
from dotenv import load_dotenv

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from skyfield.api import EarthSatellite, load as sf_load

import cartopy.crs as ccrs
import cartopy.feature as cfeature


# -----------------------------
# Load .env early
# -----------------------------
load_dotenv()

# -----------------------------
# Space-Track endpoint
# -----------------------------
LOGIN_URL = "https://www.space-track.org/ajaxauth/login"
DEFAULT_TIP_LIMIT = int(os.getenv("TIP_LIMIT", "200"))


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

def iso_to_dt(s: str) -> dt.datetime:
    return dt.datetime.strptime(s.strip(), "%Y-%m-%d %H:%M:%S").replace(tzinfo=dt.timezone.utc)

def dt_to_iso_z(t: dt.datetime) -> str:
    t = t.astimezone(dt.timezone.utc)
    return t.strftime("%Y-%m-%d %H:%M:%SZ")

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
                incl=float(row["INCL"]) if row.get("INCL") not in (None, "") else None,
                high_interest=row.get("HIGH_INTEREST"),
                raw=row,
            )
        )

    def key(sol: TipSolution):
        try:
            return iso_to_dt(sol.msg_epoch).timestamp()
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

def compute_tip_window_from_batch(solutions_latest_batch: List[TipSolution]) -> Tuple[Optional[dt.datetime], Optional[dt.datetime], List[dt.datetime]]:
    decays: List[dt.datetime] = []
    for s in solutions_latest_batch:
        if s.decay_epoch:
            try:
                decays.append(iso_to_dt(s.decay_epoch))
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

def export_track_kml(lats: List[float], lons: List[float], out_path: str, name: str, description: str = "") -> None:
    """
    Export a ground track as a KML LineString.
    KML coordinate order: lon,lat,alt
    """
    if not lats or not lons:
        raise ValueError("Empty track; nothing to export.")

    coords = "\n".join([f"{float(lon):.8f},{float(lat):.8f},0" for lat, lon in zip(lats, lons)])

    # Minimal, widely-compatible KML
    kml = f"""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Document>
    <Placemark>
      <name>{name}</name>
      <description>{description}</description>
      <Style>
        <LineStyle>
          <width>3</width>
        </LineStyle>
      </Style>
      <LineString>
        <tessellate>1</tessellate>
        <coordinates>
{coords}
        </coordinates>
      </LineString>
    </Placemark>
  </Document>
</kml>
"""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(kml)


# -----------------------------
# GUI
# -----------------------------
class ReentryGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Reentry Monitor (TIP + TLE) — Latest TIP Only + Envelope (PH Zoom + KML Export)")
        self.geometry("1240x820")

        self.out_dir = os.getenv("OUT_DIR", "./reentry_out")
        ensure_dir(self.out_dir)

        # State
        self.tip_raw: Optional[list] = None
        self.solutions_all: List[TipSolution] = []
        self.solutions_latest: List[TipSolution] = []
        self.tle: Optional[Tuple[str, str, str]] = None
        self.window_min: Optional[dt.datetime] = None
        self.window_max: Optional[dt.datetime] = None

        # Cached last computed tracks (for export)
        self.track_envelope_min: Tuple[List[float], List[float]] = ([], [])
        self.track_envelope_max: Tuple[List[float], List[float]] = ([], [])
        self.track_selected_mid: Tuple[List[float], List[float]] = ([], [])
        self.latest_msg_epoch_used: str = ""

        # UI vars
        self.var_user = tk.StringVar(value=os.getenv("SPACE_TRACK_USERNAME", ""))
        self.var_pass = tk.StringVar(value=os.getenv("SPACE_TRACK_PASSWORD", ""))
        self.var_norad = tk.StringVar(value=os.getenv("NORAD_CAT_ID", "66877"))
        self.var_tip_url = tk.StringVar(value=os.getenv("TIP_URL", ""))

        self.var_before = tk.StringVar(value="90")
        self.var_after = tk.StringVar(value="90")
        self.var_step = tk.StringVar(value="30")
        self.var_mid_tracks = tk.StringVar(value="5")

        self.var_ph_focus = tk.BooleanVar(value=True)

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

        ttk.Label(row2, text="Window Before (min)").pack(side=tk.LEFT)
        ttk.Entry(row2, textvariable=self.var_before, width=8).pack(side=tk.LEFT, padx=(6, 14))

        ttk.Label(row2, text="After (min)").pack(side=tk.LEFT)
        ttk.Entry(row2, textvariable=self.var_after, width=8).pack(side=tk.LEFT, padx=(6, 14))

        ttk.Label(row2, text="Step (sec)").pack(side=tk.LEFT)
        ttk.Entry(row2, textvariable=self.var_step, width=8).pack(side=tk.LEFT, padx=(6, 14))

        ttk.Label(row2, text="Intermediate tracks").pack(side=tk.LEFT)
        ttk.Entry(row2, textvariable=self.var_mid_tracks, width=8).pack(side=tk.LEFT, padx=(6, 14))

        toggles = ttk.Frame(self, padding=(10, 0, 10, 10))
        toggles.pack(side=tk.TOP, fill=tk.X)

        ttk.Checkbutton(
            toggles,
            text="Philippines Focus (zoom)",
            variable=self.var_ph_focus,
            command=self._apply_extent
        ).pack(side=tk.LEFT)

        btns = ttk.Frame(self, padding=(10, 0, 10, 10))
        btns.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(btns, text="Fetch TIP + TLE (latest TIP only)", command=self.on_fetch).pack(side=tk.LEFT)
        ttk.Button(btns, text="Plot Envelope", command=self.on_plot_envelope).pack(side=tk.LEFT, padx=8)
        ttk.Button(btns, text="Save Outputs…", command=self.on_save_outputs).pack(side=tk.LEFT, padx=8)
        ttk.Button(btns, text="Export Tracks to KML…", command=self.on_export_kml).pack(side=tk.LEFT, padx=8)

        self.status = tk.Text(self, height=10)
        self.status.pack(side=tk.BOTTOM, fill=tk.X)
        self._log("Ready. Use 'Export Tracks to KML…' then load in QGIS/Google Earth.")

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

            wmin, wmax, decays = compute_tip_window_from_batch(self.solutions_latest)
            self.window_min, self.window_max = wmin, wmax

            hist_csv = history_path(self.out_dir, norad)
            prev = read_last_history_row(hist_csv)

            now_utc = dt.datetime.now(dt.timezone.utc)
            latest_msg_epoch = self.solutions_latest[0].msg_epoch if self.solutions_latest else ""
            self.latest_msg_epoch_used = latest_msg_epoch

            hi = self.solutions_latest[0].high_interest if self.solutions_latest else ""
            incl = self.solutions_latest[0].incl if self.solutions_latest else ""
            direction = self.solutions_latest[0].direction if self.solutions_latest else ""

            if wmin and wmax:
                width = wmax - wmin
                mid = wmin + (width / 2)
                width_s = int(width.total_seconds())

                row = {
                    "run_utc": dt_to_iso_z(now_utc),
                    "norad_id": str(norad),
                    "tip_msg_epoch_used": latest_msg_epoch,
                    "latest_batch_count": str(len(self.solutions_latest)),
                    "window_start_utc": dt_to_iso_z(wmin),
                    "window_end_utc": dt_to_iso_z(wmax),
                    "window_width_sec": str(width_s),
                    "window_mid_utc": dt_to_iso_z(mid),
                    "hi": str(hi),
                    "incl": str(incl),
                    "direction": str(direction),
                    "decay_samples": str(len(decays)),
                    "tip_total_rows_fetched": str(len(self.solutions_all)),
                    "tip_limit": str(DEFAULT_TIP_LIMIT),
                }
                append_history_row(hist_csv, row)

                self._log(f"Latest TIP MSG_EPOCH used: {latest_msg_epoch} (batch rows: {len(self.solutions_latest)})")
                self._log(f"Decay window (latest batch): {dt_to_iso_z(wmin)} to {dt_to_iso_z(wmax)} (width {fmt_timedelta(width)})")
                self._log(f"Saved history: {hist_csv}")

                if prev and prev.get("window_mid_utc"):
                    try:
                        prev_mid = dt.datetime.strptime(prev["window_mid_utc"], "%Y-%m-%d %H:%M:%SZ").replace(tzinfo=dt.timezone.utc)
                        shift = mid - prev_mid
                        self._log(f"Shift vs previous mid: {fmt_timedelta(shift)} (positive = later)")
                    except Exception:
                        pass
            else:
                self._log(f"Latest TIP MSG_EPOCH: {latest_msg_epoch} (batch rows: {len(self.solutions_latest)})")
                self._log("But no valid DECAY_EPOCH values found in the latest batch to form a window.")

            self._log(f"TLE loaded: {self.tle[0] if self.tle else '(none)'} | TIP rows: {len(self.solutions_all)} | Latest-batch rows: {len(self.solutions_latest)}")

        except Exception as e:
            messagebox.showerror("Fetch error", str(e))
            self._log(f"ERROR: {e}")

    def _plot_track(self, lats: List[float], lons: List[float], linewidth: float, alpha: float, linestyle: str = "-"):
        for seg_lat, seg_lon in split_dateline_segments(lats, lons):
            self.ax.plot(seg_lon, seg_lat, transform=ccrs.PlateCarree(),
                         linewidth=linewidth, alpha=alpha, linestyle=linestyle)

    def on_plot_envelope(self):
        try:
            if not self.tle or not self.solutions_latest or not self.window_min or not self.window_max:
                raise RuntimeError("No latest-batch decay window available. Click 'Fetch TIP + TLE' first (and ensure latest TIP batch has DECAY_EPOCH).")

            before_min = self._get_int(self.var_before, "Window Before (min)")
            after_min = self._get_int(self.var_after, "Window After (min)")
            step_s = self._get_int(self.var_step, "Step (sec)")
            mid_tracks = self._get_int(self.var_mid_tracks, "Intermediate tracks")
            if mid_tracks < 0:
                mid_tracks = 0

            name, l1, l2 = self.tle
            ts = sf_load.timescale()
            sat = EarthSatellite(l1, l2, name, ts)

            wmin = self.window_min
            wmax = self.window_max
            width = wmax - wmin
            wmid = wmin + (width / 2)

            latest_msg_epoch = self.solutions_latest[0].msg_epoch if self.solutions_latest else ""
            self.latest_msg_epoch_used = latest_msg_epoch

            self._setup_map()

            # Intermediate tracks across the decay window (faint dashed)
            if mid_tracks > 0 and width.total_seconds() > 0:
                for i in range(1, mid_tracks + 1):
                    frac = i / (mid_tracks + 1)
                    t_i = wmin + dt.timedelta(seconds=width.total_seconds() * frac)
                    lats_i, lons_i, _ = groundtrack_corridor(sat, t_i, before_min, after_min, step_s)
                    self._plot_track(lats_i, lons_i, linewidth=0.9, alpha=0.25, linestyle="--")

            # Envelope boundary tracks (bold)
            lats_min, lons_min, _ = groundtrack_corridor(sat, wmin, before_min, after_min, step_s)
            lats_max, lons_max, _ = groundtrack_corridor(sat, wmax, before_min, after_min, step_s)
            self._plot_track(lats_min, lons_min, linewidth=1.7, alpha=0.92, linestyle="-")
            self._plot_track(lats_max, lons_max, linewidth=1.7, alpha=0.92, linestyle="-")

            # Selected midpoint track (dotted)
            sel_lats, sel_lons, _ = groundtrack_corridor(sat, wmid, before_min, after_min, step_s)
            self._plot_track(sel_lats, sel_lons, linewidth=2.0, alpha=0.95, linestyle=":")

            # Cache tracks for export
            self.track_envelope_min = (lats_min, lons_min)
            self.track_envelope_max = (lats_max, lons_max)
            self.track_selected_mid = (sel_lats, sel_lons)

            # Label near midpoint of selected track
            if sel_lats and sel_lons:
                idx = len(sel_lats) // 2
                self.ax.text(sel_lons[idx], sel_lats[idx], name.replace(" ", "_"),
                             transform=ccrs.PlateCarree(),
                             fontsize=10, color="red",
                             bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7, linewidth=0.0))

            self.ax.set_title(
                f"{name} — Latest TIP MSG_EPOCH: {latest_msg_epoch}\n"
                f"Decay window: {dt_to_iso_z(wmin)} to {dt_to_iso_z(wmax)} (width {fmt_timedelta(width)}) | Selected track: window midpoint"
            )
            self.canvas.draw()

            self._log("Envelope plotted (latest TIP only). Tracks cached for KML export.")

        except Exception as e:
            messagebox.showerror("Plot error", str(e))
            self._log(f"ERROR: {e}")

    def on_export_kml(self):
        try:
            if not self.tle:
                raise RuntimeError("No TLE loaded. Fetch + plot first.")
            (lats_min, lons_min) = self.track_envelope_min
            (lats_max, lons_max) = self.track_envelope_max
            (lats_sel, lons_sel) = self.track_selected_mid
            if not lats_min or not lats_max or not lats_sel:
                raise RuntimeError("No tracks cached yet. Click 'Plot Envelope' first.")

            folder = filedialog.askdirectory(title="Select folder to save KML tracks")
            if not folder:
                return

            norad = self._get_int(self.var_norad, "NORAD_CAT_ID")
            obj_name = self.tle[0]
            safe = obj_name.replace(" ", "_")
            stamp = dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")

            desc_common = (
                f"NORAD: {norad}\n"
                f"Object: {obj_name}\n"
                f"TIP MSG_EPOCH: {self.latest_msg_epoch_used}\n"
                f"Generated UTC: {dt_to_iso_z(dt.datetime.now(dt.timezone.utc))}\n"
                f"CRS: WGS84 (EPSG:4326)\n"
            )

            p_min = os.path.join(folder, f"envelope_min_{safe}_{norad}_{stamp}.kml")
            export_track_kml(lats_min, lons_min, p_min, name="envelope_min", description=desc_common + "Track: envelope_min")

            p_max = os.path.join(folder, f"envelope_max_{safe}_{norad}_{stamp}.kml")
            export_track_kml(lats_max, lons_max, p_max, name="envelope_max", description=desc_common + "Track: envelope_max")

            p_sel = os.path.join(folder, f"selected_midpoint_{safe}_{norad}_{stamp}.kml")
            export_track_kml(lats_sel, lons_sel, p_sel, name="selected_midpoint", description=desc_common + "Track: selected_midpoint")

            self._log(f"Saved KML: {p_min}")
            self._log(f"Saved KML: {p_max}")
            self._log(f"Saved KML: {p_sel}")
            messagebox.showinfo("Exported", "KML tracks exported successfully.\nLoad them in QGIS/Google Earth (WGS84).")

        except Exception as e:
            messagebox.showerror("Export error", str(e))
            self._log(f"ERROR: {e}")

    def on_save_outputs(self):
        try:
            if self.tip_raw is None and self.tle is None:
                raise RuntimeError("Nothing to save yet. Fetch TIP + TLE first.")

            folder = filedialog.askdirectory(title="Select folder to save outputs")
            if not folder:
                return

            norad = self._get_int(self.var_norad, "NORAD_CAT_ID")
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

            png_path = os.path.join(folder, f"corridor_envelope_PHzoom_{norad}_{stamp}.png")
            self.fig.savefig(png_path, dpi=220)
            self._log(f"Saved: {png_path}")

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
