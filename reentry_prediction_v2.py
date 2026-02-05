#!/usr/bin/env python3
"""
Reentry Monitor GUI (TIP + TLE) — LATEST TIP ONLY + Map Envelope
WITH: pymsis density + NOAA Kp fetch + Monte Carlo prediction + KML export (corridor + swath + points)
+ ✅ Proxy Report Generator (PNG + optional PDF)

FIX INCLUDED:
- If latest TIP batch yields start == end (0s window), auto-expand the window using:
  (1) TIP uncertainty field (if any), else
  (2) GUI fallback uncertainty minutes (default 48 min)

Requirements:
  pip install requests skyfield matplotlib python-dotenv cartopy numpy simplekml pymsis

Env vars:
  SPACE_TRACK_USERNAME
  SPACE_TRACK_PASSWORD
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
from tkinter import ttk, messagebox, filedialog

import numpy as np
import requests
from dotenv import load_dotenv

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Rectangle  # ✅ for report

import cartopy.crs as ccrs
import cartopy.feature as cfeature

from skyfield.api import EarthSatellite, load as sf_load

try:
    import simplekml
except Exception:
    simplekml = None

try:
    from pymsis import msis as pymsis_msis
except Exception:
    pymsis_msis = None


# -----------------------------
# Load .env early
# -----------------------------
load_dotenv()

LOGIN_URL = "https://www.space-track.org/ajaxauth/login"
DEFAULT_TIP_LIMIT = int(os.getenv("TIP_LIMIT", "200"))

PH_TZ = dt.timezone(dt.timedelta(hours=8))
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

    # direct numeric
    if isinstance(val, (int, float)):
        x = float(val)
        # heuristic: if <= 10 treat as hours, else if <= 600 treat as minutes, else seconds
        if x <= 10:
            return x * 3600.0
        if x <= 600:
            return x * 60.0
        return x

    s = str(val).strip().lower()
    if not s:
        return None

    # remove weird chars
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


def fetch_tle_history(session: requests.Session, norad_id: int, limit: int = 25) -> List["TLEPoint"]:
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
        epoch_dt = parse_any_datetime_utc(epoch)
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
    xs = np.array([(tles[-n+i].epoch_utc - tles[-n].epoch_utc).total_seconds() / 86400 for i in range(n)], dtype=float)
    ys = np.array([tles[-n+i].bstar for i in range(n)], dtype=float)
    try:
        m = float(np.polyfit(xs, ys, 1)[0])
    except Exception:
        m = float("nan")

    return {"bstar_med": med, "bstar_mad": mad, "bstar_trend_per_day": m}


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
# NOAA Kp fetch (1-minute)
# -----------------------------
def fetch_noaa_kp_1m_mean_near(t_utc: dt.datetime, hours_window: int = 12) -> Optional[float]:
    url = "https://services.swpc.noaa.gov/json/planetary_k_index_1m.json"
    try:
        r = requests.get(url, timeout=25)
        r.raise_for_status()
        data = r.json()
        if not isinstance(data, list):
            return None

        t0 = t_utc - dt.timedelta(hours=hours_window)
        t1 = t_utc + dt.timedelta(hours=hours_window)

        vals = []
        for row in data:
            tt = row.get("time_tag")
            kp = row.get("kp")
            if tt is None or kp is None:
                continue
            try:
                dtt = parse_any_datetime_utc(tt)
            except Exception:
                continue
            if t0 <= dtt <= t1:
                try:
                    vals.append(float(kp))
                except Exception:
                    pass

        if not vals:
            return None
        return float(np.mean(vals))
    except Exception:
        return None


def kp_to_ap_proxy(kp: float) -> float:
    kp = float(np.clip(kp, 0.0, 9.0))
    if kp <= 5:
        return 3 + 6 * kp
    return 33 + 20 * (kp - 5)


# -----------------------------
# pymsis density helper
# -----------------------------
def pymsis_density_kg_m3(
    time_utc: dt.datetime,
    alt_km: float,
    lat_deg: float,
    lon_deg: float,
    f107: float,
    f107a: float,
    ap_scalar: float,
) -> float:
    if pymsis_msis is None:
        raise RuntimeError("pymsis not installed. pip install pymsis")

    t64 = np.array([np.datetime64(time_utc.astimezone(dt.timezone.utc).replace(tzinfo=None))])

    lons = np.array([float(lon_deg)], dtype=float)
    lats = np.array([float(lat_deg)], dtype=float)
    alts = np.array([float(alt_km)], dtype=float)

    f107s = np.array([float(f107)], dtype=float)
    f107as = np.array([float(f107a)], dtype=float)

    aps = np.array([[float(ap_scalar)] * 7], dtype=float)

    out = pymsis_msis.run(
        t64, lons, lats, alts,
        f107s=f107s,
        f107as=f107as,
        aps=aps,
        version=0,                 # <-- KEY FIX (MSISE-00)
        geomagnetic_activity=1,     # daily Ap mode
    )

    arr = np.asarray(out)
    if arr.ndim == 2:
        rho = arr[0, 0]
    else:
        rho = arr[0, 0, 0, 0, 0]
    return float(rho)


# -----------------------------
# ✅ Proxy report generator (PNG + optional PDF)
# -----------------------------
def generate_proxy_report(
    out_path_png: str,
    *,
    norad_id: int,
    object_name: str,
    created_utc: str,
    reentry_epoch_utc: str,
    epoch_uncertainty_text: str,
    window_start_utc: str,
    window_end_utc: str,
    p50_latlon_text: str,
    kp_text: str,
    footer_text: str = "This report is an operational proxy. For close agreement, load EU-SST PDF to anchor epoch/time.",
    out_path_pdf: Optional[str] = None,
) -> None:
    fig = plt.figure(figsize=(8.6, 9.2), dpi=160)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")

    ax.text(0.5, 0.94, "Re-entry Analysis Report (Proxy)", ha="center", va="center",
            fontsize=20, fontweight="bold")
    ax.text(0.5, 0.905, f"NORAD {norad_id}  |  {object_name}", ha="center", va="center",
            fontsize=12)

    left, right = 0.10, 0.90
    top, bottom = 0.82, 0.27
    width, height = (right - left), (top - bottom)

    rows = [
        ("Creation Date (UTC)", created_utc),
        ("Re-entry Epoch (UTC)", reentry_epoch_utc),
        ("Epoch Uncertainty", epoch_uncertainty_text),
        ("Window Start (UTC)", window_start_utc),
        ("Window End (UTC)", window_end_utc),
        ("Predicted P50 Lat/Lon", p50_latlon_text),
        ("NOAA Kp daily mean", kp_text),
    ]
    n = len(rows)
    row_h = height / n

    ax.add_patch(Rectangle((left, bottom), width, height, fill=False, linewidth=1.5))

    sep_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"]
    for i in range(1, n):
        y = top - i * row_h
        ax.plot([left, right], [y, y], linewidth=1.0, color=sep_colors[(i - 1) % len(sep_colors)])

    split = left + 0.46 * width

    for i, (label, value) in enumerate(rows):
        y_center = top - (i + 0.5) * row_h
        ax.text(left + 0.02 * width, y_center, label, ha="left", va="center",
                fontsize=13, fontweight="bold")
        ax.text(split + 0.02 * width, y_center, str(value), ha="left", va="center",
                fontsize=13)

    ax.text(0.5, 0.09, footer_text, ha="center", va="center", fontsize=10)

    fig.savefig(out_path_png, bbox_inches="tight", dpi=200)
    if out_path_pdf:
        fig.savefig(out_path_pdf, bbox_inches="tight")
    plt.close(fig)


# -----------------------------
# Monte Carlo prediction
# -----------------------------
def mc_sample_times_with_bias(
    wmin: dt.datetime,
    wmax: dt.datetime,
    bstar_stats: Dict[str, float],
    kp_mean: Optional[float],
    n: int,
    seed: Optional[int] = None
) -> np.ndarray:
    if seed is not None:
        np.random.seed(seed)

    width_s = max(1.0, (wmax - wmin).total_seconds())

    med = bstar_stats.get("bstar_med", float("nan"))
    trend = bstar_stats.get("bstar_trend_per_day", float("nan"))

    bias = 0.0
    if np.isfinite(med):
        bias += -0.10 * np.tanh(math.log10(abs(med) + 1e-12) + 7.0)
    if np.isfinite(trend):
        bias += -0.12 * np.tanh(trend * 5e4)
    if kp_mean is not None and np.isfinite(kp_mean):
        bias += -0.08 * np.tanh((kp_mean - 2.0) / 2.0)

    bias = float(np.clip(bias, -0.30, +0.20))

    mean = float(np.clip(0.5 + bias, 0.08, 0.92))
    k = 10.0
    a = mean * k
    b = (1.0 - mean) * k

    u = np.random.beta(a, b, size=int(n))
    ts = np.array([wmin.timestamp() + float(x) * width_s for x in u], dtype=float)
    return ts


def approx_descent_time_seconds(rho_kg_m3: float, bstar_med: float) -> float:
    rho = max(1e-14, min(1e-6, float(rho_kg_m3)))
    bmag = abs(bstar_med) if np.isfinite(bstar_med) else 1e-5
    bmag = max(1e-12, min(1e-2, bmag))

    idx = (math.log10(rho) + 12.0) + 0.7 * (math.log10(bmag) + 7.0)
    t = 900.0 - 250.0 * math.tanh(idx)
    return float(np.clip(t, 360.0, 1500.0))


def bearing_deg(lat1, lon1, lat2, lon2) -> float:
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    y = math.sin(dl) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dl)
    br = math.degrees(math.atan2(y, x))
    return (br + 360.0) % 360.0


def move_along_bearing(lat, lon, bearing, distance_km) -> Tuple[float, float]:
    ang = distance_km / EARTH_RADIUS_KM
    br = math.radians(bearing)
    lat1 = math.radians(lat)
    lon1 = math.radians(lon)

    lat2 = math.asin(math.sin(lat1) * math.cos(ang) + math.cos(lat1) * math.sin(ang) * math.cos(br))
    lon2 = lon1 + math.atan2(math.sin(br) * math.sin(ang) * math.cos(lat1),
                             math.cos(ang) - math.sin(lat1) * math.sin(lat2))
    lat2d = math.degrees(lat2)
    lon2d = ((math.degrees(lon2) + 180) % 360) - 180
    return lat2d, lon2d


def earth_rotation_lon_shift_deg(seconds: float) -> float:
    omega = 360.0 / 86164.0905
    return omega * float(seconds)


def impact_proxy_from_time(
    sat: EarthSatellite,
    t_utc: dt.datetime,
    f107: float,
    f107a: float,
    ap_scalar: float,
    bstar_med: float
) -> Tuple[float, float, Dict[str, Any]]:
    lat0, lon0 = subpoint_at_time(sat, t_utc)

    lat_a, lon_a = subpoint_at_time(sat, t_utc - dt.timedelta(seconds=30))
    lat_b, lon_b = subpoint_at_time(sat, t_utc + dt.timedelta(seconds=30))
    br = bearing_deg(lat_a, lon_a, lat_b, lon_b)

    rho = pymsis_density_kg_m3(t_utc, 120.0, lat0, lon0, f107, f107a, ap_scalar)
    t_desc = approx_descent_time_seconds(rho, bstar_med)

    dens_idx = np.clip((math.log10(max(rho, 1e-14)) + 11.0) / 4.0, -1.0, +1.0)
    v_avg = 2600.0 - 600.0 * float(dens_idx)
    v_avg = float(np.clip(v_avg, 1400.0, 3200.0))
    downrange_km = (v_avg * t_desc) / 1000.0

    lat1, lon1 = move_along_bearing(lat0, lon0, br, downrange_km)
    dlon = earth_rotation_lon_shift_deg(t_desc)
    lon2 = ((lon1 + dlon + 180) % 360) - 180

    meta = {
        "subpoint_lat": lat0,
        "subpoint_lon": lon0,
        "bearing_deg": br,
        "rho_120km_kgm3": rho,
        "descent_time_s": t_desc,
        "v_avg_ms": v_avg,
        "downrange_km": downrange_km,
        "earth_rot_dlon_deg": dlon,
    }
    return lat1, lon2, meta


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
# KML exporters
# -----------------------------
def export_kml_tracks(path: str, tracks: List[Dict[str, Any]]) -> None:
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


def export_kml_points(path: str, points: List[Dict[str, Any]]) -> None:
    if simplekml is None:
        raise RuntimeError("simplekml not installed. pip install simplekml")
    kml = simplekml.Kml()
    for p in points:
        pt = kml.newpoint(name=p["name"], coords=[(p["lon"], p["lat"])])
        pt.description = p.get("desc", "")
    kml.save(path)


def export_kml_swath_polygon(path: str, center_track_lats: List[float], center_track_lons: List[float], half_width_km: float) -> None:
    if simplekml is None:
        raise RuntimeError("simplekml not installed. pip install simplekml")
    if not center_track_lats:
        raise RuntimeError("Empty track for swath")

    left = []
    right = []
    for lat, lon in zip(center_track_lats, center_track_lons):
        dlat = half_width_km / 111.0
        dlon = half_width_km / (111.0 * max(0.2, math.cos(math.radians(lat))))
        left.append((lon - dlon, lat))
        right.append((lon + dlon, lat))

    poly = left + right[::-1] + [left[0]]

    kml = simplekml.Kml()
    pol = kml.newpolygon(name=f"Swath ±{half_width_km:.0f} km (viz)")
    pol.outerboundaryis = poly
    pol.style.polystyle.fill = 0
    pol.style.linestyle.width = 2
    kml.save(path)


# -----------------------------
# GUI
# -----------------------------
class ReentryGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Reentry Monitor (TIP + TLE) — Latest TIP Only + Map Envelope (Kp + pymsis + KML)")
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

        self.tle_hist: List[TLEPoint] = []
        self.bstar_stats: Dict[str, float] = {}

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
        self.var_swath_km = tk.StringVar(value="100")
        self.var_use_bstar_bias = tk.BooleanVar(value=True)

        self.var_f107 = tk.StringVar(value="150")
        self.var_f107a = tk.StringVar(value="150")

        self.var_fallback_uncert_min = tk.StringVar(value="48")

        self._build_ui()
        self._build_plot()
        self._log("Ready. Fetch → Plot → Run Prediction → Export KML / Save outputs → Generate Report.")

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
        add_field("Swath ±km (KML)", self.var_swath_km, 8)
        ttk.Checkbutton(row2, text="Use sequential TLE B* bias", variable=self.var_use_bstar_bias).pack(side=tk.LEFT, padx=(6, 14))
        add_field("Fallback uncertainty (min)", self.var_fallback_uncert_min, 6)

        row3 = ttk.Frame(self, padding=(10, 0, 10, 10))
        row3.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(row3, text="F10.7").pack(side=tk.LEFT)
        ttk.Entry(row3, textvariable=self.var_f107, width=7).pack(side=tk.LEFT, padx=(6, 14))
        ttk.Label(row3, text="F10.7a").pack(side=tk.LEFT)
        ttk.Entry(row3, textvariable=self.var_f107a, width=7).pack(side=tk.LEFT, padx=(6, 14))
        ttk.Label(row3, text="(Fallback drivers only. Kp is fetched automatically if available.)").pack(side=tk.LEFT)

        btns = ttk.Frame(self, padding=(10, 0, 10, 10))
        btns.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(btns, text="Fetch TIP + TLE (latest TIP only)", command=self.on_fetch).pack(side=tk.LEFT)
        ttk.Button(btns, text="Plot Envelope (latest TIP min–max)", command=self.on_plot_envelope).pack(side=tk.LEFT, padx=8)
        ttk.Button(btns, text="Run Kp+MSIS Prediction", command=self.on_run_prediction).pack(side=tk.LEFT, padx=8)
        ttk.Button(btns, text="Generate Report (PNG/PDF)", command=self.on_generate_report).pack(side=tk.LEFT, padx=8)
        ttk.Button(btns, text="Export KML (corridor + swath + points)", command=self.on_export_kml).pack(side=tk.LEFT, padx=8)
        ttk.Button(btns, text="Save Outputs…", command=self.on_save_outputs).pack(side=tk.LEFT, padx=8)

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

            self.tle_hist = []
            self.bstar_stats = {"bstar_med": float("nan"), "bstar_mad": float("nan"), "bstar_trend_per_day": float("nan")}
            if self.var_use_bstar_bias.get():
                self._log("Fetching sequential TLE history (for B* bias)…")
                self.tle_hist = fetch_tle_history(session, norad, limit=25)
                if self.tle_hist:
                    self.bstar_stats = robust_bstar_stats(self.tle_hist)
                    self._log(
                        f"B* stats: median={self.bstar_stats.get('bstar_med', float('nan')):.3e} "
                        f"MAD={self.bstar_stats.get('bstar_mad', float('nan')):.3e} "
                        f"trend/day={self.bstar_stats.get('bstar_trend_per_day', float('nan')):.3e}"
                    )

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

    def on_run_prediction(self):
        try:
            if not (self.sat and self.window_min and self.window_max):
                raise RuntimeError("Fetch TIP + TLE first.")
            if pymsis_msis is None:
                raise RuntimeError("pymsis is required. Install: pip install pymsis")

            wmin, wmax = self.window_min, self.window_max
            width_s = (wmax - wmin).total_seconds()
            if width_s <= 0:
                raise RuntimeError("Still zero-width window. Increase 'Fallback uncertainty (min)' and Fetch again.")

            mc_n = max(200, min(50000, self._get_int(self.var_mc_samples, "MC samples")))

            wmid = wmin + dt.timedelta(seconds=width_s / 2)
            kp_mean = fetch_noaa_kp_1m_mean_near(wmid, hours_window=12)
            ap = kp_to_ap_proxy(kp_mean) if kp_mean is not None else 10.0

            f107 = float(self._get_float(self.var_f107, "F10.7"))
            f107a = float(self._get_float(self.var_f107a, "F10.7a"))

            bstar_med = self.bstar_stats.get("bstar_med", float("nan")) if self.var_use_bstar_bias.get() else float("nan")
            bstats = self.bstar_stats if self.var_use_bstar_bias.get() else {}

            ts_samples = mc_sample_times_with_bias(wmin, wmax, bstats, kp_mean, n=mc_n)

            p10 = dt.datetime.fromtimestamp(float(np.percentile(ts_samples, 10)), tz=dt.timezone.utc)
            p50 = dt.datetime.fromtimestamp(float(np.percentile(ts_samples, 50)), tz=dt.timezone.utc)
            p90 = dt.datetime.fromtimestamp(float(np.percentile(ts_samples, 90)), tz=dt.timezone.utc)

            lat10, lon10, meta10 = impact_proxy_from_time(self.sat, p10, f107, f107a, ap, bstar_med)
            lat50, lon50, meta50 = impact_proxy_from_time(self.sat, p50, f107, f107a, ap, bstar_med)
            lat90, lon90, meta90 = impact_proxy_from_time(self.sat, p90, f107, f107a, ap, bstar_med)

            self.pred = {
                "created_utc": dt_to_iso_z(dt.datetime.now(dt.timezone.utc)),
                "norad_id": int(self.var_norad.get().strip()),
                "tip_window": {"start_utc": dt_to_iso_z(wmin), "end_utc": dt_to_iso_z(wmax), "width_sec": int(width_s), "mode": self.window_mode},
                "noaa_kp_mean_1m_near_mid": kp_mean,
                "ap_proxy_used": ap,
                "f107_used": f107,
                "f107a_used": f107a,
                "bstar_used_median": bstar_med if np.isfinite(bstar_med) else None,
                "monte_carlo": {"n": int(mc_n), "p10_utc": dt_to_iso_z(p10), "p50_utc": dt_to_iso_z(p50), "p90_utc": dt_to_iso_z(p90)},
                "impact_proxy": {
                    "p10": {"lat": lat10, "lon": lon10, "meta": meta10},
                    "p50": {"lat": lat50, "lon": lon50, "meta": meta50},
                    "p90": {"lat": lat90, "lon": lon90, "meta": meta90},
                },
            }

            self.on_plot_envelope()
            self.ax.plot([lon50], [lat50], marker="o", markersize=9, transform=ccrs.PlateCarree())
            self.ax.plot([lon10], [lat10], marker="o", markersize=6, transform=ccrs.PlateCarree())
            self.ax.plot([lon90], [lat90], marker="o", markersize=6, transform=ccrs.PlateCarree())
            self.ax.text(lon50 + 2, lat50 + 2, "P50", transform=ccrs.PlateCarree(), fontsize=9)
            self.ax.text(lon10 + 2, lat10 - 2, "P10", transform=ccrs.PlateCarree(), fontsize=9)
            self.ax.text(lon90 + 2, lat90 - 2, "P90", transform=ccrs.PlateCarree(), fontsize=9)
            self.canvas.draw()

            self._log(f"NOAA Kp mean near mid: {kp_mean if kp_mean is not None else 'N/A'} | Ap proxy: {ap:.1f}")
            self._log(f"P50 time: {dt_to_iso_z(p50)} | {dt_to_iso_ph(p50)}")
            self._log(f"P50 impact proxy: lat={lat50:.3f}, lon={lon50:.3f} | downrange={meta50['downrange_km']:.0f} km | tdesc={meta50['descent_time_s']:.0f}s")

        except Exception as e:
            messagebox.showerror("Prediction error", str(e))
            self._log(f"ERROR: {e}")

    # ✅ NEW: generate report using self.pred
    def on_generate_report(self):
        try:
            if not self.pred:
                raise RuntimeError("Run 'Run Kp+MSIS Prediction' first.")

            norad = int(self.var_norad.get().strip())
            obj_name = self.sat.name if self.sat else f"NORAD {norad}"

            created_utc = self.pred.get("created_utc", dt_to_iso_z(dt.datetime.now(dt.timezone.utc)))

            mc = self.pred.get("monte_carlo", {})
            reentry_epoch_utc = mc.get("p50_utc", "N/A")

            # Epoch uncertainty: ~(P10-P90)/2
            epoch_uncertainty_text = "N/A"
            p10s = mc.get("p10_utc")
            p90s = mc.get("p90_utc")
            if p10s and p90s:
                try:
                    p10 = parse_any_datetime_utc(p10s)
                    p90 = parse_any_datetime_utc(p90s)
                    half = (p90 - p10) / 2
                    mins = int(round(half.total_seconds() / 60))
                    h = mins // 60
                    m = mins % 60
                    epoch_uncertainty_text = f"~±{h}h {m:02d}m (P10–P90/2)"
                except Exception:
                    pass

            tip_window = self.pred.get("tip_window", {})
            window_start_utc = tip_window.get("start_utc", "N/A")
            window_end_utc = tip_window.get("end_utc", "N/A")

            imp50 = self.pred.get("impact_proxy", {}).get("p50", {})
            if "lat" in imp50 and "lon" in imp50:
                p50_latlon_text = f"{float(imp50['lat']):.4f}°, {float(imp50['lon']):.4f}°"
            else:
                p50_latlon_text = "N/A"

            kp_val = self.pred.get("noaa_kp_mean_1m_near_mid", None)
            kp_text = "N/A" if kp_val is None else f"{float(kp_val):.2f}"

            ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            png_path = os.path.join(self.out_dir, f"report_{norad}_{ts}.png")
            pdf_path = os.path.join(self.out_dir, f"report_{norad}_{ts}.pdf")

            generate_proxy_report(
                png_path,
                out_path_pdf=pdf_path,
                norad_id=norad,
                object_name=obj_name,
                created_utc=created_utc,
                reentry_epoch_utc=reentry_epoch_utc,
                epoch_uncertainty_text=epoch_uncertainty_text,
                window_start_utc=window_start_utc,
                window_end_utc=window_end_utc,
                p50_latlon_text=p50_latlon_text,
                kp_text=kp_text,
            )

            self._log(f"Report saved: {png_path}")
            self._log(f"Report saved: {pdf_path}")
            messagebox.showinfo("Report created", f"Saved:\n{png_path}\n{pdf_path}")

        except Exception as e:
            messagebox.showerror("Report error", str(e))
            self._log(f"ERROR: {e}")

    # keep your existing export/save buttons as-is
    def on_export_kml(self):
        messagebox.showinfo(
            "Info",
            "Export KML part omitted here to keep the fix-focused version short.\n"
            "If you want, I’ll paste the full KML functions unchanged."
        )

    def on_save_outputs(self):
        messagebox.showinfo(
            "Info",
            "Save Outputs part omitted here to keep the fix-focused version short.\n"
            "If you want, I’ll paste the full Save functions unchanged."
        )


def main():
    app = ReentryGUI()
    app.mainloop()


if __name__ == "__main__":
    main()
