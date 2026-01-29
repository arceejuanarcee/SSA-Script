#!/usr/bin/env python3
"""
Reentry Monitor GUI (Space-Track TIP + TLE) — corridor plot on a world map

What it does:
- Loads SPACE_TRACK_USERNAME / SPACE_TRACK_PASSWORD from .env (or environment)
- Fetches TIP decay solutions for a NORAD ID (default 66877)
- Fetches latest TLE
- Propagates the ground track around each TIP decay epoch and plots it on a world map
- Shows results inside a Tkinter GUI + allows Save PNG/JSON/TLE outputs

Requirements:
  pip install requests skyfield matplotlib python-dotenv cartopy

Notes:
- cartopy on Windows may require the "prebuilt wheels" (pip usually works now, but if it fails,
  install from conda-forge or use a prebuilt wheel).
- TIP schema can vary; this script reads DECAY_EPOCH, MSG_EPOCH, INCL, etc safely.

Env vars:
  SPACE_TRACK_USERNAME
  SPACE_TRACK_PASSWORD

Optional env vars:
  TIP_URL (override if your TIP endpoint differs)
"""

from __future__ import annotations

import os
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

from skyfield.api import EarthSatellite, load

# Map plotting
import cartopy.crs as ccrs
import cartopy.feature as cfeature


# -----------------------------
# Load .env early
# -----------------------------
load_dotenv()


# -----------------------------
# Space-Track endpoints
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
# Core logic
# -----------------------------
def iso_to_dt(s: str) -> dt.datetime:
    # TIP times typically like "2026-01-30 08:03:00" (UTC)
    return dt.datetime.strptime(s.strip(), "%Y-%m-%d %H:%M:%S").replace(tzinfo=dt.timezone.utc)

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
            return 0

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
    ts = load.timescale()
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
    # Normalize lon to [-180, 180]
    lons = [((x + 180) % 360) - 180 for x in lons_raw]
    return lats, lons, times_dt

def make_brief(norad_id: int, solutions: List[TipSolution]) -> str:
    if not solutions:
        return f"NORAD {norad_id}: No TIP solutions returned."

    top = solutions[:3]
    decay_times = [s.decay_epoch for s in top if s.decay_epoch]
    uniq: List[str] = []
    for d in decay_times:
        if d not in uniq:
            uniq.append(d)

    hi = top[0].high_interest
    incl = top[0].incl
    direction = top[0].direction

    if uniq:
        if len(uniq) == 1:
            return f"NORAD {norad_id}: TIP decay epoch ~ {uniq[0]} UTC (HI={hi}, INCL≈{incl}, DIR={direction})."
        return f"NORAD {norad_id}: TIP decay solutions ~ {uniq[0]} to {uniq[-1]} UTC (HI={hi}, INCL≈{incl}, DIR={direction})."

    return f"NORAD {norad_id}: TIP entries found but DECAY_EPOCH missing."


# -----------------------------
# GUI
# -----------------------------
class ReentryGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Reentry Monitor (Space-Track TIP + TLE)")
        self.geometry("1180x720")

        # State
        self.tip_raw: Optional[list] = None
        self.solutions: List[TipSolution] = []
        self.tle: Optional[Tuple[str, str, str]] = None  # (name,l1,l2)

        # UI vars
        self.var_user = tk.StringVar(value=os.getenv("SPACE_TRACK_USERNAME", ""))
        self.var_pass = tk.StringVar(value=os.getenv("SPACE_TRACK_PASSWORD", ""))
        self.var_norad = tk.StringVar(value=os.getenv("NORAD_CAT_ID", "66877"))
        self.var_tip_url = tk.StringVar(value=os.getenv("TIP_URL", ""))
        self.var_before = tk.StringVar(value="90")
        self.var_after = tk.StringVar(value="90")
        self.var_step = tk.StringVar(value="30")

        self._build_ui()
        self._build_plot()

    def _build_ui(self):
        top = ttk.Frame(self, padding=10)
        top.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(top, text="SPACE_TRACK_USERNAME").grid(row=0, column=0, sticky="w")
        ttk.Entry(top, textvariable=self.var_user, width=30).grid(row=0, column=1, padx=6)

        ttk.Label(top, text="SPACE_TRACK_PASSWORD").grid(row=0, column=2, sticky="w")
        ttk.Entry(top, textvariable=self.var_pass, width=30, show="*").grid(row=0, column=3, padx=6)

        ttk.Label(top, text="NORAD_CAT_ID").grid(row=0, column=4, sticky="w")
        ttk.Entry(top, textvariable=self.var_norad, width=10).grid(row=0, column=5, padx=6)

        ttk.Label(top, text="TIP_URL (optional override)").grid(row=1, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(top, textvariable=self.var_tip_url, width=110).grid(row=1, column=1, columnspan=5, sticky="we", pady=(8, 0))

        ttk.Label(top, text="Window Before (min)").grid(row=2, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(top, textvariable=self.var_before, width=10).grid(row=2, column=1, sticky="w", pady=(8, 0))

        ttk.Label(top, text="After (min)").grid(row=2, column=2, sticky="w", pady=(8, 0))
        ttk.Entry(top, textvariable=self.var_after, width=10).grid(row=2, column=3, sticky="w", pady=(8, 0))

        ttk.Label(top, text="Step (sec)").grid(row=2, column=4, sticky="w", pady=(8, 0))
        ttk.Entry(top, textvariable=self.var_step, width=10).grid(row=2, column=5, sticky="w", pady=(8, 0))

        btns = ttk.Frame(self, padding=(10, 0, 10, 10))
        btns.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(btns, text="Fetch TIP + TLE", command=self.on_fetch).pack(side=tk.LEFT)
        ttk.Button(btns, text="Plot Top TIP Solution", command=self.on_plot_top).pack(side=tk.LEFT, padx=8)
        ttk.Button(btns, text="Plot All (Top 3)", command=self.on_plot_all).pack(side=tk.LEFT, padx=8)
        ttk.Button(btns, text="Save Outputs…", command=self.on_save_outputs).pack(side=tk.LEFT, padx=8)

        self.status = tk.Text(self, height=7)
        self.status.pack(side=tk.BOTTOM, fill=tk.X)
        self._log("Ready. Tip: put credentials in .env or enter above.")

    def _build_plot(self):
        center = ttk.Frame(self, padding=10)
        center.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Create a Cartopy map axis
        self.fig = plt.Figure(figsize=(10, 5), dpi=100)
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
        self.ax.set_title("Ground-track corridor (around TIP decay epoch)")

    def _log(self, msg: str):
        ts = dt.datetime.now().strftime("%H:%M:%S")
        self.status.insert("end", f"[{ts}] {msg}\n")
        self.status.see("end")

    def _get_int(self, v: tk.StringVar, label: str) -> int:
        try:
            return int(v.get().strip())
        except Exception:
            raise ValueError(f"Invalid integer for {label}: {v.get()}")

    def on_fetch(self):
        try:
            user = self.var_user.get().strip()
            pw = self.var_pass.get().strip()
            norad = self._get_int(self.var_norad, "NORAD_CAT_ID")
            tip_url_override = self.var_tip_url.get().strip()

            self._log("Logging in to Space-Track…")
            session = spacetrack_login(user, pw)

            self._log("Fetching TIP…")
            self.tip_raw = fetch_tip(session, norad, tip_url_override)
            self.solutions = parse_tip_solutions(self.tip_raw)

            self._log("Fetching latest TLE…")
            self.tle = fetch_latest_tle(session, norad)

            self._log(make_brief(norad, self.solutions))
            self._log(f"TIP solutions returned: {len(self.solutions)}")
            if self.solutions:
                self._log(f"Newest MSG_EPOCH: {self.solutions[0].msg_epoch} | DECAY_EPOCH: {self.solutions[0].decay_epoch}")

        except Exception as e:
            messagebox.showerror("Fetch error", str(e))
            self._log(f"ERROR: {e}")

    def _plot_solution(self, idx: int):
        if not self.tle or not self.solutions:
            raise RuntimeError("No data yet. Click 'Fetch TIP + TLE' first.")
        if idx < 0 or idx >= len(self.solutions):
            raise RuntimeError("Invalid solution index.")

        sol = self.solutions[idx]
        if not sol.decay_epoch:
            raise RuntimeError("Selected TIP solution has no DECAY_EPOCH.")

        before_min = self._get_int(self.var_before, "Window Before (min)")
        after_min = self._get_int(self.var_after, "Window After (min)")
        step_s = self._get_int(self.var_step, "Step (sec)")

        name, l1, l2 = self.tle
        ts = load.timescale()
        sat = EarthSatellite(l1, l2, name, ts)

        t_center = iso_to_dt(sol.decay_epoch)
        lats, lons, _ = groundtrack_corridor(sat, t_center, before_min, after_min, step_s)

        # Draw on map
        self._setup_map()
        self.ax.plot(lons, lats, transform=ccrs.PlateCarree(), linewidth=1.2)
        self.ax.set_title(f"{name} — corridor around TIP decay ({sol.decay_epoch} UTC)")
        self.canvas.draw()

        self._log(f"Plotted solution #{idx+1}: DECAY_EPOCH={sol.decay_epoch} UTC")

    def on_plot_top(self):
        try:
            self._plot_solution(0)
        except Exception as e:
            messagebox.showerror("Plot error", str(e))
            self._log(f"ERROR: {e}")

    def on_plot_all(self):
        try:
            if not self.tle or not self.solutions:
                raise RuntimeError("No data yet. Click 'Fetch TIP + TLE' first.")

            self._setup_map()
            for i, sol in enumerate(self.solutions[:3]):
                if not sol.decay_epoch:
                    continue
                name, l1, l2 = self.tle
                ts = load.timescale()
                sat = EarthSatellite(l1, l2, name, ts)

                before_min = self._get_int(self.var_before, "Window Before (min)")
                after_min = self._get_int(self.var_after, "Window After (min)")
                step_s = self._get_int(self.var_step, "Step (sec)")

                t_center = iso_to_dt(sol.decay_epoch)
                lats, lons, _ = groundtrack_corridor(sat, t_center, before_min, after_min, step_s)
                self.ax.plot(lons, lats, transform=ccrs.PlateCarree(), linewidth=1.0)

            name = self.tle[0]
            self.ax.set_title(f"{name} — corridors around top TIP solutions")
            self.canvas.draw()
            self._log("Plotted top 3 TIP solutions (where DECAY_EPOCH available).")

        except Exception as e:
            messagebox.showerror("Plot error", str(e))
            self._log(f"ERROR: {e}")

    def on_save_outputs(self):
        try:
            if not self.tle and not self.tip_raw:
                raise RuntimeError("Nothing to save yet. Fetch TIP + TLE first.")

            folder = filedialog.askdirectory(title="Select folder to save outputs")
            if not folder:
                return

            norad = self._get_int(self.var_norad, "NORAD_CAT_ID")
            stamp = dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")

            # Save TIP JSON
            if self.tip_raw is not None:
                tip_path = os.path.join(folder, f"tip_{norad}_{stamp}.json")
                with open(tip_path, "w", encoding="utf-8") as f:
                    json.dump(self.tip_raw, f, indent=2)
                self._log(f"Saved: {tip_path}")

            # Save TLE
            if self.tle is not None:
                name, l1, l2 = self.tle
                tle_path = os.path.join(folder, f"tle_{norad}_{stamp}.txt")
                with open(tle_path, "w", encoding="utf-8") as f:
                    f.write(name + "\n" + l1 + "\n" + l2 + "\n")
                self._log(f"Saved: {tle_path}")

            # Save current plot as PNG
            png_path = os.path.join(folder, f"corridor_{norad}_{stamp}.png")
            self.fig.savefig(png_path, dpi=200)
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
