#!/usr/bin/env python3
"""
Reentry Monitor (Space-Track TIP + TLE) — best-effort corridor plot
- Fetch TIP predicted decay epochs (may be multiple solutions)
- Fetch latest TLE
- Propagate TLE around each decay epoch and plot/print ground-track corridor
- Alerts + JSON/PNG outputs

Requirements:
  pip install requests skyfield matplotlib

Env vars:
  SPACETRACK_USER
  SPACETRACK_PASS
"""

from __future__ import annotations

import os
import sys
import json
import time
import math
import random
import datetime as dt
from dataclasses import dataclass
from typing import List, Optional, Tuple

import requests

from skyfield.api import EarthSatellite, load
import matplotlib
matplotlib.use("Agg")  # safe for servers
import matplotlib.pyplot as plt


# -----------------------------
# Config
# -----------------------------
NORAD_CAT_ID = int(os.getenv("NORAD_CAT_ID", "66877"))
OUT_DIR = os.getenv("OUT_DIR", "./reentry_out")
PLOT = os.getenv("PLOT", "1") == "1"

# Time window (minutes) around each TIP decay epoch for corridor plotting
WINDOW_MIN_BEFORE = int(os.getenv("WINDOW_MIN_BEFORE", "90"))
WINDOW_MIN_AFTER  = int(os.getenv("WINDOW_MIN_AFTER",  "90"))
STEP_SECONDS = int(os.getenv("STEP_SECONDS", "30"))

# Space-Track endpoints
LOGIN_URL = "https://www.space-track.org/ajaxauth/login"

# NOTE: You MUST set this to your working TIP query.
# Common patterns (examples only; adjust to your org’s working query):
# TIP_URL = f"https://www.space-track.org/basicspacedata/query/class/tip/NORAD_CAT_ID/{NORAD_CAT_ID}/orderby/MSG_EPOCH%20desc/format/json"
TIP_URL = os.getenv(
    "TIP_URL",
    f"https://www.space-track.org/basicspacedata/query/class/tip/NORAD_CAT_ID/{NORAD_CAT_ID}/orderby/MSG_EPOCH%20desc/format/json"
)

# Latest TLE for the object
TLE_URL = (
    f"https://www.space-track.org/basicspacedata/query/class/gp/"
    f"NORAD_CAT_ID/{NORAD_CAT_ID}/orderby/EPOCH%20desc/limit/1/format/tle"
)

# Basic throttling to avoid hammering Space-Track
STATE_FILE = os.getenv("STATE_FILE", os.path.join(OUT_DIR, f".state_{NORAD_CAT_ID}.json"))
MIN_SECONDS_BETWEEN_RUNS = int(os.getenv("MIN_SECONDS_BETWEEN_RUNS", "900"))  # 15 min


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
def ensure_out_dir() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

def load_state() -> dict:
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def save_state(state: dict) -> None:
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)

def should_throttle() -> bool:
    state = load_state()
    last = state.get("last_run_unix", 0)
    now = int(time.time())
    return (now - last) < MIN_SECONDS_BETWEEN_RUNS

def mark_run() -> None:
    state = load_state()
    state["last_run_unix"] = int(time.time())
    save_state(state)

def iso_to_dt(s: str) -> dt.datetime:
    # TIP times typically look like: "2026-01-30 08:03:00"
    # Treat as UTC unless your TIP explicitly includes TZ.
    return dt.datetime.strptime(s.strip(), "%Y-%m-%d %H:%M:%S").replace(tzinfo=dt.timezone.utc)

def retry_get(session: requests.Session, url: str, tries: int = 5, timeout: int = 30) -> requests.Response:
    last_exc = None
    for i in range(tries):
        try:
            r = session.get(url, timeout=timeout)
            if r.status_code in (429, 500, 502, 503, 504):
                # backoff
                sleep_s = (2 ** i) + random.random()
                time.sleep(sleep_s)
                continue
            r.raise_for_status()
            return r
        except Exception as e:
            last_exc = e
            time.sleep((2 ** i) + random.random())
    raise RuntimeError(f"GET failed after retries: {url}") from last_exc

def spacetrack_login() -> requests.Session:
    user = os.getenv("SPACE_TRACK_USERNAME", "")
    pw = os.getenv("SPACE_TRACK_PASSWORD", "")
    if not user or not pw:
        raise RuntimeError("Missing SPACETRACK_USER / SPACETRACK_PASS environment variables.")

    s = requests.Session()
    r = s.post(LOGIN_URL, data={"identity": user, "password": pw}, timeout=30)
    r.raise_for_status()
    return s

def parse_tip_solutions(tip_json: list) -> List[TipSolution]:
    sols: List[TipSolution] = []
    for row in tip_json:
        # fields vary depending on TIP schema; safely read
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
    # Sort newest MSG_EPOCH first if possible
    def key(sol: TipSolution):
        try:
            return iso_to_dt(sol.msg_epoch).timestamp()
        except Exception:
            return 0
    sols.sort(key=key, reverse=True)
    return sols

def fetch_latest_tle(session: requests.Session) -> Tuple[str, str, str]:
    r = retry_get(session, TLE_URL)
    lines = [ln.strip() for ln in r.text.splitlines() if ln.strip()]
    if len(lines) < 2:
        raise RuntimeError("TLE fetch returned insufficient lines.")
    if lines[0].startswith("1 ") and lines[1].startswith("2 "):
        name = f"NORAD {NORAD_CAT_ID}"
        l1, l2 = lines[0], lines[1]
    else:
        # name + two lines
        name = lines[0]
        l1, l2 = lines[1], lines[2]
    return name, l1, l2

def groundtrack_corridor(
    sat: EarthSatellite,
    t_center: dt.datetime,
    minutes_before: int,
    minutes_after: int,
    step_s: int
) -> Tuple[List[float], List[float], List[dt.datetime]]:
    ts = load.timescale()
    start = t_center - dt.timedelta(minutes=minutes_before)
    end = t_center + dt.timedelta(minutes=minutes_after)

    times_dt: List[dt.datetime] = []
    lats: List[float] = []
    lons: List[float] = []

    cur = start
    while cur <= end:
        times_dt.append(cur)
        cur += dt.timedelta(seconds=step_s)

    t_sf = ts.from_datetimes(times_dt)
    geoc = sat.at(t_sf)
    sub = geoc.subpoint()
    lat_deg = sub.latitude.degrees
    lon_deg = sub.longitude.degrees

    # Normalize lon to [-180, 180]
    lon_norm = []
    for x in lon_deg:
        y = ((x + 180) % 360) - 180
        lon_norm.append(y)

    return list(lat_deg), lon_norm, times_dt

def plot_corridor(lats: List[float], lons: List[float], title: str, outpath: str) -> None:
    plt.figure(figsize=(12, 5))
    plt.title(title)
    plt.xlabel("Longitude (deg)")
    plt.ylabel("Latitude (deg)")
    plt.xlim(-180, 180)
    plt.ylim(-90, 90)
    plt.grid(True, linewidth=0.5)
    plt.plot(lons, lats, linewidth=1.0)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

def brief_from_tip(solutions: List[TipSolution]) -> str:
    if not solutions:
        return f"NORAD {NORAD_CAT_ID}: No TIP solutions returned."

    # Take the newest N solutions (often TIP provides multiple decay estimates)
    top = solutions[:3]
    decay_times = []
    for s in top:
        if s.decay_epoch:
            decay_times.append(s.decay_epoch)

    uniq = []
    for d in decay_times:
        if d not in uniq:
            uniq.append(d)

    hi = top[0].high_interest
    incl = top[0].incl
    direction = top[0].direction

    if uniq:
        if len(uniq) == 1:
            win = uniq[0]
            return (f"NORAD {NORAD_CAT_ID}: TIP indicates predicted decay epoch ~ {win} UTC "
                    f"(HI={hi}, INCL≈{incl}, DIR={direction}). Continue monitoring; location uncertainty remains high.")
        else:
            return (f"NORAD {NORAD_CAT_ID}: TIP provides multiple decay solutions around "
                    f"{uniq[0]} to {uniq[-1]} UTC (HI={hi}, INCL≈{incl}, DIR={direction}). Continue monitoring; refine closer to event.")
    return f"NORAD {NORAD_CAT_ID}: TIP returned entries but no DECAY_EPOCH fields were found."


# -----------------------------
# Main
# -----------------------------
def main() -> int:
    ensure_out_dir()

    if should_throttle():
        print(f"[THROTTLE] Last run was < {MIN_SECONDS_BETWEEN_RUNS}s ago. Exiting.")
        return 0

    session = spacetrack_login()

    tip_resp = retry_get(session, TIP_URL)
    tip_json = tip_resp.json() if tip_resp.text.strip().startswith("[") else json.loads(tip_resp.text)

    solutions = parse_tip_solutions(tip_json)
    print(brief_from_tip(solutions))

    # Save raw TIP snapshot
    tip_path = os.path.join(OUT_DIR, f"tip_{NORAD_CAT_ID}_{int(time.time())}.json")
    with open(tip_path, "w", encoding="utf-8") as f:
        json.dump(tip_json, f, indent=2)
    print(f"[OK] Saved TIP snapshot: {tip_path}")

    # Fetch latest TLE
    name, l1, l2 = fetch_latest_tle(session)
    print(f"[OK] Latest TLE loaded: {name}")
    tle_path = os.path.join(OUT_DIR, f"tle_{NORAD_CAT_ID}_{int(time.time())}.txt")
    with open(tle_path, "w", encoding="utf-8") as f:
        f.write(name + "\n" + l1 + "\n" + l2 + "\n")
    print(f"[OK] Saved TLE: {tle_path}")

    # Build satellite
    ts = load.timescale()
    sat = EarthSatellite(l1, l2, name, ts)

    # For each of top TIP solutions, generate a corridor plot
    if PLOT:
        for idx, sol in enumerate(solutions[:3], start=1):
            if not sol.decay_epoch:
                continue
            t_center = iso_to_dt(sol.decay_epoch)
            lats, lons, _ = groundtrack_corridor(
                sat,
                t_center=t_center,
                minutes_before=WINDOW_MIN_BEFORE,
                minutes_after=WINDOW_MIN_AFTER,
                step_s=STEP_SECONDS
            )
            title = f"{name} — Ground-track corridor around TIP decay ({sol.decay_epoch} UTC)"
            out_png = os.path.join(OUT_DIR, f"corridor_{NORAD_CAT_ID}_{idx}.png")
            plot_corridor(lats, lons, title, out_png)
            print(f"[OK] Wrote corridor plot: {out_png}")

    mark_run()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
