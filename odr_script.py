#!/usr/bin/env python3
"""
Reentry Monitor GUI (TIP + TLE) — with convergence tracking + corridor envelope map

What’s new in this updated version:
- Loads .env automatically (SPACE_TRACK_USERNAME / SPACE_TRACK_PASSWORD)
- Saves TIP history to CSV (for convergence tracking)
- Computes a decay time window (earliest/latest) from the newest TIP solutions
- Shows convergence metrics (window width + shift from previous run)
- Plots an "envelope" on a world map:
  - Earliest TIP track (bold)
  - Latest TIP track (bold)
  - Several intermediate tracks (faint) to visualize uncertainty

Requirements:
  pip install requests skyfield matplotlib python-dotenv cartopy

Env vars (via .env or system env):
  SPACE_TRACK_USERNAME
  SPACE_TRACK_PASSWORD

Optional env vars:
  OUT_DIR=./reentry_out
  NORAD_CAT_ID=66877
  TIP_URL=... (override)
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
    # TIP times typically: "YYYY-MM-DD HH:MM:SS"
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

def split_dateline_segments(lats: List[float], lons: List[float], jump_deg: float = 180.0):
    """Split a track so map plotting doesn't draw a huge line across the dateline."""
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
    """Take the newest N TIP rows and compute min/max decay epoch."""
    decays: List[dt.datetime] = []
    for s in solutions[:take_n]:
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


# -----------------------------
# GUI
# -----------------------------
class ReentryGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Reentry Monitor (TIP + TLE) — Convergence + Map Envelope")
        self.geometry("1240x760")

        # Defaults
        self.out_dir = os.getenv("OUT_DIR", "./reentry_out")
        ensure_dir(self.out_dir)

        # State
        self.tip_raw: Optional[list] = None
        self.solutions: List[TipSolution] = []
        self.tle: Optional[Tuple[str, str, str]] = None
        self.window_min: Optional[dt.datetime] = None
        self.window_max: Optional[dt.datetime] = None

        # UI vars
        self.var_user = tk.StringVar(value=os.getenv("SPACE_TRACK_USERNAME", ""))
        self.var_pass = tk.StringVar(value=os.getenv("SPACE_TRACK_PASSWORD", ""))
        self.var_norad = tk.StringVar(value=os.getenv("NORAD_CAT_ID", "66877"))
        self.var_tip_url = tk.StringVar(value=os.getenv("TIP_URL", ""))

        self.var_before = tk.StringVar(value="90")
        self.var_after = tk.StringVar(value="90")
        self.var_step = tk.StringVar(value="30")

        self.var_take_n = tk.StringVar(value="10")     # how many TIP rows to compute window
        self.var_mid_tracks = tk.StringVar(value="5")  # intermediate tracks inside window

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

        ttk.Label(row2, text="TIP rows for window").pack(side=tk.LEFT)
        ttk.Entry(row2, textvariable=self.var_take_n, width=8).pack(side=tk.LEFT, padx=(6, 14))

        ttk.Label(row2, text="Intermediate tracks").pack(side=tk.LEFT)
        ttk.Entry(row2, textvariable=self.var_mid_tracks, width=8).pack(side=tk.LEFT, padx=(6, 14))

        btns = ttk.Frame(self, padding=(10, 0, 10, 10))
        btns.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(btns, text="Fetch TIP + TLE", command=self.on_fetch).pack(side=tk.LEFT)
        ttk.Button(btns, text="Plot Envelope (min–max)", command=self.on_plot_envelope).pack(side=tk.LEFT, padx=8)
        ttk.Button(btns, text="Save Outputs…", command=self.on_save_outputs).pack(side=tk.LEFT, padx=8)

        self.status = tk.Text(self, height=9)
        self.status.pack(side=tk.BOTTOM, fill=tk.X)
        self._log("Ready. (This version logs convergence history + plots min–max envelope on a world map.)")

    def _build_plot(self):
        center = ttk.Frame(self, padding=10)
        center.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.fig = plt.Figure(figsize=(11.5, 5.4), dpi=100)
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
        self.ax.set_title("Ground-track envelope around TIP decay window")

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
            take_n = self._get_int(self.var_take_n, "TIP rows for window")

            self._log("Logging in to Space-Track…")
            session = spacetrack_login(user, pw)

            self._log("Fetching TIP…")
            self.tip_raw = fetch_tip(session, norad, tip_url_override)
            self.solutions = parse_tip_solutions(self.tip_raw)

            self._log("Fetching latest TLE…")
            self.tle = fetch_latest_tle(session, norad)

            # Compute window
            wmin, wmax, decays = compute_tip_window(self.solutions, take_n=take_n)
            self.window_min, self.window_max = wmin, wmax

            # Convergence metrics + history
            hist_csv = history_path(self.out_dir, norad)
            prev = read_last_history_row(hist_csv)

            now_utc = dt.datetime.now(dt.timezone.utc)
            msg_epoch = self.solutions[0].msg_epoch if self.solutions else ""
            hi = self.solutions[0].high_interest if self.solutions else ""
            incl = self.solutions[0].incl if self.solutions else ""
            direction = self.solutions[0].direction if self.solutions else ""

            if wmin and wmax:
                width = wmax - wmin
                mid = wmin + (width / 2)
                width_s = int(width.total_seconds())
                row = {
                    "run_utc": dt_to_iso_z(now_utc),
                    "norad_id": str(norad),
                    "tip_msg_epoch": msg_epoch,
                    "window_start_utc": dt_to_iso_z(wmin),
                    "window_end_utc": dt_to_iso_z(wmax),
                    "window_width_sec": str(width_s),
                    "window_mid_utc": dt_to_iso_z(mid),
                    "hi": str(hi),
                    "incl": str(incl),
                    "direction": str(direction),
                    "tip_rows_used": str(take_n),
                    "decay_samples": str(len(decays)),
                }
                append_history_row(hist_csv, row)

                self._log(f"TIP window (from newest {take_n} rows): {dt_to_iso_z(wmin)} to {dt_to_iso_z(wmax)} (width {fmt_timedelta(width)})")
                self._log(f"Saved history: {hist_csv}")

                if prev and prev.get("window_mid_utc"):
                    try:
                        prev_mid = dt.datetime.strptime(prev["window_mid_utc"], "%Y-%m-%d %H:%M:%SZ").replace(tzinfo=dt.timezone.utc)
                        shift = mid - prev_mid
                        self._log(f"Shift vs previous mid: {fmt_timedelta(shift)} (positive = later)")
                    except Exception:
                        pass
            else:
                self._log("TIP returned entries but no valid DECAY_EPOCH values found to form a window.")

            self._log(f"TLE loaded: {self.tle[0] if self.tle else '(none)'} | TIP rows: {len(self.solutions)}")

        except Exception as e:
            messagebox.showerror("Fetch error", str(e))
            self._log(f"ERROR: {e}")

    def _plot_track(self, lats: List[float], lons: List[float], linewidth: float, alpha: float):
        for seg_lat, seg_lon in split_dateline_segments(lats, lons):
            self.ax.plot(seg_lon, seg_lat, transform=ccrs.PlateCarree(), linewidth=linewidth, alpha=alpha)

    def on_plot_envelope(self):
        try:
            if not self.tle or not self.solutions or not self.window_min or not self.window_max:
                raise RuntimeError("No decay window available. Click 'Fetch TIP + TLE' first (and ensure DECAY_EPOCH exists).")

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

            self._setup_map()

            # Intermediate tracks across the decay window (faint)
            if mid_tracks > 0 and width.total_seconds() > 0:
                for i in range(1, mid_tracks + 1):
                    frac = i / (mid_tracks + 1)
                    t_i = wmin + dt.timedelta(seconds=width.total_seconds() * frac)
                    lats_i, lons_i, _ = groundtrack_corridor(sat, t_i, before_min, after_min, step_s)
                    self._plot_track(lats_i, lons_i, linewidth=0.9, alpha=0.25)

            # Plot min and max (bold)
            lats_min, lons_min, _ = groundtrack_corridor(sat, wmin, before_min, after_min, step_s)
            lats_max, lons_max, _ = groundtrack_corridor(sat, wmax, before_min, after_min, step_s)

            self._plot_track(lats_min, lons_min, linewidth=1.6, alpha=0.9)
            self._plot_track(lats_max, lons_max, linewidth=1.6, alpha=0.9)

            self.ax.set_title(
                f"{name} — TIP decay window: {dt_to_iso_z(wmin)} to {dt_to_iso_z(wmax)} (width {fmt_timedelta(width)})"
            )
            self.canvas.draw()

            self._log("Envelope plotted: min & max TIP window (bold) + intermediate tracks (faint).")

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

            norad = self._get_int(self.var_norad, "NORAD_CAT_ID")
            stamp = dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")

            # TIP JSON
            if self.tip_raw is not None:
                tip_path = os.path.join(folder, f"tip_{norad}_{stamp}.json")
                with open(tip_path, "w", encoding="utf-8") as f:
                    json.dump(self.tip_raw, f, indent=2)
                self._log(f"Saved: {tip_path}")

            # TLE
            if self.tle is not None:
                name, l1, l2 = self.tle
                tle_path = os.path.join(folder, f"tle_{norad}_{stamp}.txt")
                with open(tle_path, "w", encoding="utf-8") as f:
                    f.write(name + "\n" + l1 + "\n" + l2 + "\n")
                self._log(f"Saved: {tle_path}")

            # Plot PNG
            png_path = os.path.join(folder, f"corridor_envelope_{norad}_{stamp}.png")
            self.fig.savefig(png_path, dpi=200)
            self._log(f"Saved: {png_path}")

            # Copy history CSV as well (if exists)
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
