#!/usr/bin/env python3
"""
Reentry Monitor GUI (TIP + TLE) — LATEST TIP ONLY + Map Envelope + Improved (EU-SST-style) Physics Layer
===============================================================================================

What’s improved vs your previous "Proxy" approach:
- Still uses public TLE (SGP4) for orbit geometry (cannot replicate radar-fused OD).
- Adds a *semi-physical* final-descent model:
  1) Atmospheric density via NRLMSISE-00 (pymsis) if available; otherwise exponential fallback
  2) Space weather inputs (Kp history fetched from NOAA) + optional F10.7 (best-effort fetch)
  3) Converts Kp -> Ap proxy (used by MSIS) and biases drag / descent
  4) Computes downrange travel during 120->80 km "terminal" descent (1D drag integration)
  5) Applies Earth rotation during descent
- Exports KML for GIS: corridor lines (TIP min/max/P50) + swath polygon + P10/P50/P90 points

IMPORTANT LIMITATION (honest):
- Without OD state vectors + covariance (radar fused), this will still not match EU-SST exactly.
- But it will be MUCH closer than "subpoint at time" because it adds downrange + density + rotation.

Requirements:
  pip install requests skyfield matplotlib python-dotenv cartopy numpy simplekml

Optional (recommended for better density):
  pip install pymsis

Env vars (via .env or system env):
  SPACE_TRACK_USERNAME
  SPACE_TRACK_PASSWORD
Optional:
  OUT_DIR=./reentry_out
  NORAD_CAT_ID=66877
  TIP_URL=... (override)
  TIP_LIMIT=200
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
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from skyfield.api import EarthSatellite, load as sf_load

import cartopy.crs as ccrs
import cartopy.feature as cfeature

try:
    import simplekml
except Exception:
    simplekml = None

# Optional MSIS
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
OMEGA_EARTH = 7.2921150e-5  # rad/s


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

def dt_to_iso_ph(t: dt.datetime) -> str:
    return t.astimezone(PH_TZ).strftime("%Y-%m-%d %H:%M:%S (PH)")

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

def iso_to_dt_flexible(s: str) -> dt.datetime:
    """
    Accepts:
      - "YYYY-MM-DD HH:MM:SS"
      - "YYYY-MM-DD HH:MM:SS.ssssss"
      - with optional trailing 'Z'
    Returns UTC tz-aware.
    """
    st = s.strip()
    if not st:
        raise ValueError("empty datetime string")
    st = st.replace("Z", "")
    # Space-Track TIP is usually "YYYY-MM-DD HH:MM:SS[.ffffff]"
    # fromisoformat accepts "YYYY-MM-DD HH:MM:SS[.ffffff]"
    try:
        d = dt.datetime.fromisoformat(st)
        if d.tzinfo is None:
            d = d.replace(tzinfo=dt.timezone.utc)
        else:
            d = d.astimezone(dt.timezone.utc)
        return d
    except Exception:
        # fallback strict
        try:
            d = dt.datetime.strptime(st, "%Y-%m-%d %H:%M:%S").replace(tzinfo=dt.timezone.utc)
            return d
        except Exception as e:
            raise ValueError(f"time data '{s}' not parseable") from e

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
                msg_epoch=row.get("MSG_EPOCH") or "",
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
            return iso_to_dt_flexible(sol.msg_epoch).timestamp()
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
    """
    Standard TLE B* format: columns 54-61 (8 chars) like ' 34123-4'
    meaning +0.34123E-4.
    """
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

def fetch_tle_history(session: requests.Session, norad_id: int, limit: int = 25) -> List[TLEPoint]:
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
            epoch_dt = dt.datetime.strptime(epoch.strip(), "%Y-%m-%d %H:%M:%S").replace(tzinfo=dt.timezone.utc)
        except Exception:
            epoch_dt = dt.datetime.fromisoformat(epoch.replace("Z", "+00:00")).astimezone(dt.timezone.utc)

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
    m = float(np.polyfit(xs, ys, 1)[0])
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

def subpoint_and_heading(sat: EarthSatellite, t_utc: dt.datetime) -> Tuple[float, float, float]:
    """
    Returns lat, lon (deg) and track heading (deg from north, clockwise) estimated by finite difference.
    """
    lat0, lon0 = subpoint_at_time(sat, t_utc)
    lat1, lon1 = subpoint_at_time(sat, t_utc + dt.timedelta(seconds=20))
    brg = bearing_deg(lat0, lon0, lat1, lon1)
    return lat0, lon0, brg

def subpoint_at_time(sat: EarthSatellite, t_utc: dt.datetime) -> Tuple[float, float]:
    ts = sf_load.timescale()
    t_sf = ts.from_datetime(t_utc)
    sub = sat.at(t_sf).subpoint()
    lat = float(sub.latitude.degrees)
    lon = float(sub.longitude.degrees)
    lon = ((lon + 180) % 360) - 180
    return lat, lon

def bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Great-circle initial bearing from point1 to point2.
    """
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dlam = math.radians((lon2 - lon1))
    y = math.sin(dlam) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlam)
    brg = math.degrees(math.atan2(y, x))
    return (brg + 360) % 360

def dest_point(lat: float, lon: float, bearing_deg0: float, dist_km: float) -> Tuple[float, float]:
    """
    Destination point along great circle.
    """
    br = math.radians(bearing_deg0)
    phi1 = math.radians(lat)
    lam1 = math.radians(lon)
    d = dist_km / EARTH_RADIUS_KM

    phi2 = math.asin(math.sin(phi1)*math.cos(d) + math.cos(phi1)*math.sin(d)*math.cos(br))
    lam2 = lam1 + math.atan2(math.sin(br)*math.sin(d)*math.cos(phi1),
                             math.cos(d) - math.sin(phi1)*math.sin(phi2))
    lat2 = math.degrees(phi2)
    lon2 = math.degrees(lam2)
    lon2 = ((lon2 + 180) % 360) - 180
    return lat2, lon2


# -----------------------------
# NOAA space weather fetch (best-effort)
# -----------------------------
def fetch_noaa_kp_1m(center_utc: dt.datetime, hours_pad: int = 48) -> Optional[float]:
    """
    Fetch NOAA SWPC 1-minute Kp series and take mean around center time (±30m).
    Best-effort: if NOAA endpoint blocked/offline, returns None.
    Endpoint: https://services.swpc.noaa.gov/json/planetary_k_index_1m.json
    """
    url = "https://services.swpc.noaa.gov/json/planetary_k_index_1m.json"
    try:
        r = requests.get(url, timeout=25)
        r.raise_for_status()
        data = r.json()
        if not isinstance(data, list) or not data:
            return None

        t0 = center_utc - dt.timedelta(minutes=30)
        t1 = center_utc + dt.timedelta(minutes=30)

        vals = []
        for row in data:
            # row has time_tag like "2026-01-30T14:05:00Z" and kp
            ts = row.get("time_tag") or ""
            kp = row.get("kp")
            if kp is None:
                continue
            try:
                tt = dt.datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(dt.timezone.utc)
            except Exception:
                continue
            if t0 <= tt <= t1:
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
    """
    Very common Kp->Ap approximation table (coarse).
    EU-SST will use true Ap and other indices; this is a proxy.
    """
    # bins by 1/3 Kp are possible; we keep it simple:
    # Source-like mapping used widely in ops tools.
    # Kp: 0..9 -> Ap approx
    table = {
        0: 0, 1: 4, 2: 7, 3: 15, 4: 27, 5: 48, 6: 80, 7: 132, 8: 207, 9: 300
    }
    k = int(round(float(kp)))
    k = max(0, min(9, k))
    return float(table[k])


# -----------------------------
# Density model (MSIS if available)
# -----------------------------
def density_kg_m3_msis(lat: float, lon: float, alt_km: float, t_utc: dt.datetime, ap: float, f107: float) -> float:
    """
    Returns density in kg/m^3.
    Tries multiple pymsis call signatures (pymsis versions differ).
    Falls back to exponential model if anything fails.
    """
    if pymsis_msis is None:
        return density_kg_m3_exponential(alt_km, ap=ap)

    lon_e = lon % 360.0

    # pymsis expects numpy arrays; time should be datetime64 without tz
    t = np.array([np.datetime64(t_utc.replace(tzinfo=None))])
    alts = np.array([float(alt_km)], dtype=float)
    lats = np.array([float(lat)], dtype=float)
    lons = np.array([float(lon_e)], dtype=float)

    f107a = float(f107)  # proxy if you don’t have true 81-day mean
    ap_arr = np.array([float(ap)], dtype=float)

    # Try common signatures across pymsis versions
    rho = None
    try:
        # Newer style (some builds accept keywords f107/f107a/ap)
        out = pymsis_msis.run(t, alts, lats, lons, f107=float(f107), f107a=f107a, ap=ap_arr)
        rho = float(out.reshape(-1)[0])
    except TypeError:
        try:
            # Some versions use keyword "aps" instead of "ap"
            out = pymsis_msis.run(t, alts, lats, lons, f107=float(f107), f107a=f107a, aps=ap_arr)
            rho = float(out.reshape(-1)[0])
        except TypeError:
            try:
                # Older style: positional (t, alts, lats, lons, f107, f107a, ap)
                out = pymsis_msis.run(t, alts, lats, lons, float(f107), f107a, ap_arr)
                rho = float(out.reshape(-1)[0])
            except Exception:
                rho = None
    except Exception:
        rho = None

    if rho is None or (not math.isfinite(rho)) or rho <= 0:
        return density_kg_m3_exponential(alt_km, ap=ap)

    return float(rho)


def density_kg_m3_exponential(alt_km: float, ap: float = 10.0) -> float:
    """
    Very rough fallback density (order-of-magnitude only).
    Adds 'ap' effect by scaling density upward with geomagnetic activity.
    """
    # baseline scale heights
    # at 100 km ~ 5e-7 kg/m^3 (very rough), at 120 km ~ 1e-8, at 80 km ~ 1e-4
    alt = max(60.0, min(200.0, float(alt_km)))
    # piecewise-ish
    if alt >= 120:
        rho0 = 1e-8
        H = 18.0
        rho = rho0 * math.exp(-(alt - 120.0) / H)
    elif alt >= 100:
        rho0 = 5e-7
        H = 14.0
        rho = rho0 * math.exp(-(alt - 100.0) / H)
    else:
        rho0 = 1e-4
        H = 7.5
        rho = rho0 * math.exp(-(alt - 80.0) / H)
    # geomagnetic scaling
    scale = 1.0 + 0.015 * max(0.0, ap - 10.0)
    return float(rho * scale)


# -----------------------------
# Final descent model: downrange travel + Earth rotation
# -----------------------------
def simulate_downrange_km(
    sat: EarthSatellite,
    reentry_time_utc: dt.datetime,
    beta_kg_m2: float,
    kp_mean: Optional[float],
    f107: float = 150.0,
    alt_start_km: float = 120.0,
    alt_end_km: float = 80.0
) -> Tuple[float, float]:
    """
    1D along-track descent proxy:
    - Integrates dv/dt = -0.5*rho*v^2 / beta
    - Integrates ds/dt = v_ground
    - Uses MSIS density at subpoint.
    Returns:
      (downrange_km, descent_time_s)

    NOTE: This is still a proxy, but it introduces the missing big term vs your earlier result: DOWNRANGE.
    """
    # Get subpoint + heading at the chosen epoch
    lat, lon, hdg = subpoint_and_heading(sat, reentry_time_utc)

    # Kp -> Ap proxy
    if kp_mean is None:
        kp_use = 2.0
    else:
        kp_use = float(kp_mean)
    ap = kp_to_ap_proxy(kp_use)

    # Initialize (very rough orbital ground speed at ~120 km: ~7.5 km/s)
    # We'll keep it conservative; drag will reduce it a bit.
    v = 7500.0  # m/s
    s = 0.0     # m
    t = 0.0     # s

    # Simple linear altitude drop schedule over time using a vertical rate proxy
    # reentry luminous phase is short; use dt step integration with alt interpolation
    dt_step = 0.5  # seconds
    # Assume ~12 minutes typical from 120->80km (will adjust based on drag)
    t_max = 25 * 60.0
    alt = alt_start_km

    # vertical sink rate proxy (m/s)
    sink = 55.0  # baseline
    # more activity -> slightly higher density -> stronger drag -> shorter
    sink *= (1.0 + 0.02 * (kp_use - 2.0))

    while alt > alt_end_km and t < t_max:
        rho = density_kg_m3_msis(lat, lon, alt, reentry_time_utc + dt.timedelta(seconds=t), ap=ap, f107=f107)

        # Drag deceleration along-track
        a_drag = 0.5 * rho * v * v / max(1.0, beta_kg_m2)  # m/s^2
        v = max(1500.0, v - a_drag * dt_step)              # keep sane lower bound

        # ground distance advance
        s += v * dt_step
        t += dt_step

        # altitude decay
        alt -= (sink * dt_step) / 1000.0  # km

    downrange_km = s / 1000.0
    return float(downrange_km), float(t)

def apply_earth_rotation_lon(lon_deg: float, descent_time_s: float, lat_deg: float) -> float:
    """
    During descent, Earth rotates eastward. The ground under the trajectory shifts west relative to inertial.
    Approx: delta_lon = omega * dt / cos(lat)
    """
    lat = math.radians(lat_deg)
    denom = max(0.2, math.cos(lat))
    dlon = (OMEGA_EARTH * descent_time_s) / denom  # radians
    lon2 = lon_deg - math.degrees(dlon)           # subtract -> ground shifts west
    lon2 = ((lon2 + 180) % 360) - 180
    return float(lon2)


# -----------------------------
# Monte Carlo time sampler (biased by B* and Kp)
# -----------------------------
def mc_sample_reentry_times(
    wmin: dt.datetime,
    wmax: dt.datetime,
    n: int,
    bstar_stats: Optional[Dict[str, float]],
    kp_mean: Optional[float],
    use_bstar_bias: bool
) -> np.ndarray:
    width_s = max(1.0, (wmax - wmin).total_seconds())
    # Base bias around mid
    bias = 0.0

    if kp_mean is not None:
        # Higher Kp -> denser upper atm -> tends earlier (negative bias)
        bias += -0.08 * np.tanh((kp_mean - 2.0) / 2.0)

    if use_bstar_bias and bstar_stats:
        med = bstar_stats.get("bstar_med", float("nan"))
        trend = bstar_stats.get("bstar_trend_per_day", float("nan"))
        if np.isfinite(med):
            bias += -0.10 * np.tanh(math.log10(abs(med) + 1e-12) + 7.0)
        if np.isfinite(trend):
            bias += -0.08 * np.tanh(trend * 5e4)

    bias = float(np.clip(bias, -0.25, +0.25))
    mean = 0.5 + bias
    mean = float(np.clip(mean, 0.10, 0.90))

    # Concentration: modest; keep conservative
    k = 8.0
    a = mean * k
    b = (1.0 - mean) * k
    u = np.random.beta(a, b, size=int(n))
    ts = np.array([wmin.timestamp() + float(x) * width_s for x in u], dtype=float)
    return ts


# -----------------------------
# KML exporters
# -----------------------------
def export_kml(path: str, tracks: List[Dict[str, Any]], points: List[Dict[str, Any]], swath: Optional[Dict[str, Any]]) -> None:
    if simplekml is None:
        raise RuntimeError("simplekml not installed. pip install simplekml")

    kml = simplekml.Kml()

    # Tracks
    for tr in tracks:
        ls = kml.newlinestring(name=tr["name"])
        ls.coords = list(zip(tr["lons"], tr["lats"]))
        ls.altitudemode = simplekml.AltitudeMode.clamptoground

    # Points
    for p in points:
        pt = kml.newpoint(name=p["name"], coords=[(p["lon"], p["lat"])])
        if p.get("description"):
            pt.description = p["description"]

    # Swath polygon (simple buffer circle)
    if swath:
        lat = float(swath["lat"])
        lon = float(swath["lon"])
        radius_km = float(swath["radius_km"])
        n = 96
        pts = []
        lat0 = math.radians(lat)
        dlat = radius_km / 111.0
        dlon = radius_km / (111.0 * max(0.2, math.cos(lat0)))
        for i in range(n + 1):
            ang = 2 * math.pi * i / n
            la = lat + dlat * math.sin(ang)
            lo = lon + dlon * math.cos(ang)
            lo = ((lo + 180) % 360) - 180
            pts.append((lo, la))
        pol = kml.newpolygon(name=f"Swath ±{radius_km:.0f} km (proxy)")
        pol.outerboundaryis = pts
        pol.style.polystyle.fill = 0

    kml.save(path)


# -----------------------------
# GUI
# -----------------------------
class ReentryGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Reentry Monitor (TIP + TLE) — Latest TIP Only + Map Envelope (Kp Prediction + KML)")
        self.geometry("1340x900")

        self.out_dir = os.getenv("OUT_DIR", "./reentry_out")
        ensure_dir(self.out_dir)

        # State
        self.tip_raw: Optional[list] = None
        self.solutions_all: List[TipSolution] = []
        self.solutions_latest: List[TipSolution] = []
        self.tle: Optional[Tuple[str, str, str]] = None
        self.window_min: Optional[dt.datetime] = None
        self.window_max: Optional[dt.datetime] = None
        self.tle_hist: List[TLEPoint] = []
        self.bstar_stats: Dict[str, float] = {}

        # Prediction outputs
        self.pred: Dict[str, Any] = {}

        # UI vars (keep same style)
        self.var_user = tk.StringVar(value=os.getenv("SPACE_TRACK_USERNAME", ""))
        self.var_pass = tk.StringVar(value=os.getenv("SPACE_TRACK_PASSWORD", ""))
        self.var_norad = tk.StringVar(value=os.getenv("NORAD_CAT_ID", "66877"))
        self.var_tip_url = tk.StringVar(value=os.getenv("TIP_URL", ""))

        self.var_before = tk.StringVar(value="90")
        self.var_after = tk.StringVar(value="90")
        self.var_step = tk.StringVar(value="30")
        self.var_mid_tracks = tk.StringVar(value="5")

        self.var_ph_focus = tk.BooleanVar(value=False)

        self.var_mc = tk.StringVar(value="2000")
        self.var_swath_km = tk.StringVar(value="100")
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

        row2 = ttk.Frame(self, padding=(10, 0, 10, 10))
        row2.pack(side=tk.TOP, fill=tk.X)

        def add_field(lbl: str, var: tk.StringVar, w: int = 7):
            ttk.Label(row2, text=lbl).pack(side=tk.LEFT)
            ttk.Entry(row2, textvariable=var, width=w).pack(side=tk.LEFT, padx=(6, 14))

        add_field("Window Before (min)", self.var_before, 7)
        add_field("After (min)", self.var_after, 7)
        add_field("Step (sec)", self.var_step, 7)
        add_field("Intermediate tracks", self.var_mid_tracks, 7)

        ttk.Checkbutton(row2, text="Philippines Focus (zoom)", variable=self.var_ph_focus, command=self._apply_extent).pack(side=tk.LEFT, padx=(12, 0))

        row3 = ttk.Frame(self, padding=(10, 0, 10, 10))
        row3.pack(side=tk.TOP, fill=tk.X)

        add_field("MC samples", self.var_mc, 8)
        add_field("Swath ±km (KML)", self.var_swath_km, 7)
        ttk.Checkbutton(row3, text="Use sequential TLE B* bias", variable=self.var_use_bstar).pack(side=tk.LEFT, padx=(8, 0))

        btns = ttk.Frame(self, padding=(10, 0, 10, 10))
        btns.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(btns, text="Fetch TIP + TLE (latest TIP only)", command=self.on_fetch).pack(side=tk.LEFT)
        ttk.Button(btns, text="Plot Envelope (latest TIP min–max)", command=self.on_plot_envelope).pack(side=tk.LEFT, padx=8)
        ttk.Button(btns, text="Run Kp-Based Prediction", command=self.on_predict).pack(side=tk.LEFT, padx=8)
        ttk.Button(btns, text="Export KML (corridor + swath + points)", command=self.on_export_kml).pack(side=tk.LEFT, padx=8)
        ttk.Button(btns, text="Save Outputs…", command=self.on_save_outputs).pack(side=tk.LEFT, padx=8)

        self.status = tk.Text(self, height=14)
        self.status.pack(side=tk.BOTTOM, fill=tk.X)
        self._log("Ready. Fetch → Plot → Run Prediction → Export KML / Save Outputs.")

    def _build_plot(self):
        center = ttk.Frame(self, padding=10)
        center.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.fig = plt.Figure(figsize=(12.6, 6.2), dpi=100)
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

            # also fetch sequential TLEs for B* bias
            try:
                self.tle_hist = fetch_tle_history(session, norad, limit=25)
                self.bstar_stats = robust_bstar_stats(self.tle_hist) if self.tle_hist else {}
            except Exception:
                self.tle_hist = []
                self.bstar_stats = {}

            # latest batch window from DECAY_EPOCH values
            decays: List[dt.datetime] = []
            for s in self.solutions_latest:
                if s.decay_epoch:
                    try:
                        decays.append(iso_to_dt_flexible(s.decay_epoch))
                    except Exception:
                        pass

            if not decays:
                self.window_min = None
                self.window_max = None
                self._log("Latest TIP batch had no valid DECAY_EPOCH values.")
                return

            self.window_min = min(decays)
            self.window_max = max(decays)

            # store history row
            hist_csv = os.path.join(self.out_dir, f"tip_history_{norad}.csv")
            wmin, wmax = self.window_min, self.window_max
            width = wmax - wmin
            mid = wmin + (width / 2 if width.total_seconds() > 0 else dt.timedelta(seconds=0))

            row = {
                "run_utc": dt_to_iso_z(dt.datetime.now(dt.timezone.utc)),
                "norad_id": str(norad),
                "tip_msg_epoch_used": self.solutions_latest[0].msg_epoch if self.solutions_latest else "",
                "latest_batch_count": str(len(self.solutions_latest)),
                "window_start_utc": dt_to_iso_z(wmin),
                "window_end_utc": dt_to_iso_z(wmax),
                "window_width_sec": str(int(max(0, width.total_seconds()))),
                "window_mid_utc": dt_to_iso_z(mid),
            }
            self._append_history_row(hist_csv, row)

            self._log(f"Latest TIP MSG_EPOCH used: {row['tip_msg_epoch_used']} (batch rows: {len(self.solutions_latest)})")
            self._log(f"Decay window: {dt_to_iso_z(wmin)} to {dt_to_iso_z(wmax)} (width {fmt_timedelta(width)})")
            if self.bstar_stats:
                self._log(
                    f"B* stats: med={self.bstar_stats.get('bstar_med', float('nan')):.3e}, "
                    f"trend/day={self.bstar_stats.get('bstar_trend_per_day', float('nan')):.3e}"
                )

        except Exception as e:
            messagebox.showerror("Fetch error", str(e))
            self._log(f"ERROR: {e}")

    def _append_history_row(self, csv_path: str, row: dict) -> None:
        file_exists = os.path.exists(csv_path)
        fieldnames = list(row.keys())
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                w.writeheader()
            w.writerow(row)

    def _plot_track(self, lats: List[float], lons: List[float], linewidth: float, alpha: float, linestyle: str = "-"):
        for seg_lat, seg_lon in split_dateline_segments(lats, lons):
            self.ax.plot(seg_lon, seg_lat, transform=ccrs.PlateCarree(),
                         linewidth=linewidth, alpha=alpha, linestyle=linestyle)

    def on_plot_envelope(self):
        try:
            if not self.tle or not self.window_min or not self.window_max:
                raise RuntimeError("Fetch TIP + TLE first (need a valid decay window).")

            before_min = self._get_int(self.var_before, "Window Before (min)")
            after_min = self._get_int(self.var_after, "After (min)")
            step_s = self._get_int(self.var_step, "Step (sec)")
            mid_tracks = self._get_int(self.var_mid_tracks, "Intermediate tracks")
            mid_tracks = max(0, min(20, mid_tracks))

            name, l1, l2 = self.tle
            ts = sf_load.timescale()
            sat = EarthSatellite(l1, l2, name, ts)

            wmin = self.window_min
            wmax = self.window_max
            width = wmax - wmin
            wmid = wmin + (width / 2 if width.total_seconds() > 0 else dt.timedelta(seconds=0))

            latest_msg_epoch = self.solutions_latest[0].msg_epoch if self.solutions_latest else ""

            self._setup_map()

            # intermediates
            if mid_tracks > 0 and width.total_seconds() > 0:
                for i in range(1, mid_tracks + 1):
                    frac = i / (mid_tracks + 1)
                    t_i = wmin + dt.timedelta(seconds=width.total_seconds() * frac)
                    lats_i, lons_i, _ = groundtrack_corridor(sat, t_i, before_min, after_min, step_s)
                    self._plot_track(lats_i, lons_i, linewidth=0.9, alpha=0.25, linestyle="--")

            # boundaries
            lats_min, lons_min, _ = groundtrack_corridor(sat, wmin, before_min, after_min, step_s)
            lats_max, lons_max, _ = groundtrack_corridor(sat, wmax, before_min, after_min, step_s)
            self._plot_track(lats_min, lons_min, linewidth=1.7, alpha=0.92, linestyle="-")
            self._plot_track(lats_max, lons_max, linewidth=1.7, alpha=0.92, linestyle="-")

            # midpoint
            lats_mid, lons_mid, _ = groundtrack_corridor(sat, wmid, before_min, after_min, step_s)
            self._plot_track(lats_mid, lons_mid, linewidth=2.0, alpha=0.95, linestyle=":")

            self.ax.set_title(
                f"{name} — Latest TIP MSG_EPOCH: {latest_msg_epoch}\n"
                f"Decay window: {dt_to_iso_z(wmin)} to {dt_to_iso_z(wmax)} (width {fmt_timedelta(width)})"
            )
            self.canvas.draw()
            self._log("Envelope plotted (latest TIP only): min/max boundaries + midpoint track.")

        except Exception as e:
            messagebox.showerror("Plot error", str(e))
            self._log(f"ERROR: {e}")

    def on_predict(self):
        """
        Produces P10/P50/P90 *impact point* estimate (closer to EU-SST than proxy):
        - sample reentry time in TIP window (biased by Kp + B* if enabled)
        - compute subpoint + heading at that time
        - simulate downrange with density + drag
        - apply earth rotation shift during descent
        """
        try:
            if not self.tle or not self.window_min or not self.window_max:
                raise RuntimeError("Fetch TIP + TLE first (need a valid decay window).")

            name, l1, l2 = self.tle
            ts = sf_load.timescale()
            sat = EarthSatellite(l1, l2, name, ts)

            wmin = self.window_min
            wmax = self.window_max
            width_s = max(1.0, (wmax - wmin).total_seconds())

            mc_n = max(200, min(30000, self._get_int(self.var_mc, "MC samples")))
            use_bstar = bool(self.var_use_bstar.get())

            # Space weather around mid-window
            t_center = wmin + dt.timedelta(seconds=0.5 * width_s)
            kp_mean = fetch_noaa_kp_1m(t_center)

            # sample reentry times
            ts_samples = mc_sample_reentry_times(wmin, wmax, mc_n, self.bstar_stats, kp_mean, use_bstar)

            # Ballistic coefficient distribution (beta = m/(CdA)).
            # EU-SST effectively solves this from OD/fit; we must assume.
            # Use wide distribution for rocket body: 60..300 kg/m^2 typical.
            beta_samples = np.random.lognormal(mean=math.log(140.0), sigma=0.35, size=mc_n)
            beta_samples = np.clip(beta_samples, 60.0, 320.0)

            impacts = []
            for t_sec, beta in zip(ts_samples, beta_samples):
                t_re = dt.datetime.fromtimestamp(float(t_sec), tz=dt.timezone.utc)
                lat_sp, lon_sp, hdg = subpoint_and_heading(sat, t_re)

                down_km, descent_s = simulate_downrange_km(
                    sat=sat,
                    reentry_time_utc=t_re,
                    beta_kg_m2=float(beta),
                    kp_mean=kp_mean,
                    f107=150.0,   # best-effort constant unless you wire a true F10.7 history
                    alt_start_km=120.0,
                    alt_end_km=80.0,
                )

                # Move along heading by downrange distance
                lat_i, lon_i = dest_point(lat_sp, lon_sp, hdg, down_km)

                # Earth rotation correction during descent
                lon_i = apply_earth_rotation_lon(lon_i, descent_s, lat_i)

                impacts.append((t_re, lat_i, lon_i, down_km, descent_s, float(beta)))

            # pick percentiles in time space (P10/P50/P90 by time)
            impacts.sort(key=lambda x: x[0].timestamp())
            i10 = impacts[int(0.10 * (len(impacts) - 1))]
            i50 = impacts[int(0.50 * (len(impacts) - 1))]
            i90 = impacts[int(0.90 * (len(impacts) - 1))]

            # store
            self.pred = {
                "created_utc": dt_to_iso_z(dt.datetime.now(dt.timezone.utc)),
                "norad_id": int(self.var_norad.get().strip()),
                "tip_window_start_utc": dt_to_iso_z(wmin),
                "tip_window_end_utc": dt_to_iso_z(wmax),
                "kp_mean_noaa": kp_mean,
                "method": "TLE(SGP4) + MSIS density (if available) + Kp->Ap proxy + 1D drag downrange + Earth rotation",
                "p10": {"t_utc": dt_to_iso_z(i10[0]), "lat": i10[1], "lon": i10[2], "downrange_km": i10[3], "descent_s": i10[4], "beta": i10[5]},
                "p50": {"t_utc": dt_to_iso_z(i50[0]), "lat": i50[1], "lon": i50[2], "downrange_km": i50[3], "descent_s": i50[4], "beta": i50[5]},
                "p90": {"t_utc": dt_to_iso_z(i90[0]), "lat": i90[1], "lon": i90[2], "downrange_km": i90[3], "descent_s": i90[4], "beta": i90[5]},
            }

            # Plot points on current map
            self._setup_map()
            # plot envelope lightly for context
            before_min = self._get_int(self.var_before, "Window Before (min)")
            after_min = self._get_int(self.var_after, "After (min)")
            step_s = self._get_int(self.var_step, "Step (sec)")
            lats_min, lons_min, _ = groundtrack_corridor(sat, wmin, before_min, after_min, step_s)
            lats_max, lons_max, _ = groundtrack_corridor(sat, wmax, before_min, after_min, step_s)
            self._plot_track(lats_min, lons_min, linewidth=1.5, alpha=0.75, linestyle="-")
            self._plot_track(lats_max, lons_max, linewidth=1.5, alpha=0.75, linestyle="-")

            # predicted points
            self.ax.plot([i50[2]], [i50[1]], marker="o", markersize=9, transform=ccrs.PlateCarree())
            self.ax.plot([i10[2]], [i10[1]], marker="o", markersize=6, transform=ccrs.PlateCarree())
            self.ax.plot([i90[2]], [i90[1]], marker="o", markersize=6, transform=ccrs.PlateCarree())
            self.ax.text(i50[2] + 2, i50[1] + 2, "P50 (impact est.)", transform=ccrs.PlateCarree(), fontsize=9)
            self.ax.text(i10[2] + 2, i10[1] - 2, "P10", transform=ccrs.PlateCarree(), fontsize=9)
            self.ax.text(i90[2] + 2, i90[1] - 2, "P90", transform=ccrs.PlateCarree(), fontsize=9)

            self.ax.set_title(
                f"{name} — Latest TIP MSG_EPOCH: {self.solutions_latest[0].msg_epoch if self.solutions_latest else ''}\n"
                f"TIP window: {dt_to_iso_z(wmin)} → {dt_to_iso_z(wmax)} (width {fmt_timedelta(wmax - wmin)})\n"
                f"P50 time: {self.pred['p50']['t_utc']} | {dt_to_iso_ph(i50[0])} | NOAA Kp mean: {kp_mean if kp_mean is not None else 'N/A'}"
            )
            self.canvas.draw()

            self._log("Prediction complete (downrange + density + earth rotation applied).")
            self._log(f"P50: {self.pred['p50']['t_utc']} | lat={self.pred['p50']['lat']:.3f}, lon={self.pred['p50']['lon']:.3f}, downrange~{self.pred['p50']['downrange_km']:.0f} km")
            if kp_mean is None:
                self._log("NOAA Kp fetch: N/A (offline/blocked). Using quiet default for Ap proxy.")
            if pymsis_msis is None:
                self._log("pymsis not installed → using exponential density fallback (less accurate). Install: pip install pymsis")

        except Exception as e:
            messagebox.showerror("Prediction error", str(e))
            self._log(f"ERROR: {e}")

    def on_export_kml(self):
        try:
            if not self.pred:
                raise RuntimeError("Run Kp-Based Prediction first.")

            if simplekml is None:
                raise RuntimeError("simplekml not installed. pip install simplekml")

            if not self.tle or not self.window_min or not self.window_max:
                raise RuntimeError("Missing envelope state. Fetch first.")

            folder = filedialog.askdirectory(title="Select folder to save KML")
            if not folder:
                return

            norad = int(self.var_norad.get().strip())
            stamp = dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            kml_path = os.path.join(folder, f"reentry_{norad}_{stamp}.kml")

            name, l1, l2 = self.tle
            ts = sf_load.timescale()
            sat = EarthSatellite(l1, l2, name, ts)

            before_min = self._get_int(self.var_before, "Window Before (min)")
            after_min = self._get_int(self.var_after, "After (min)")
            step_s = self._get_int(self.var_step, "Step (sec)")

            wmin, wmax = self.window_min, self.window_max
            width = wmax - wmin
            wmid = wmin + (width / 2 if width.total_seconds() > 0 else dt.timedelta(seconds=0))

            tracks = []
            for nm, tt in [("TIP_min", wmin), ("TIP_max", wmax), ("TIP_mid", wmid)]:
                lats, lons, _ = groundtrack_corridor(sat, tt, before_min, after_min, step_s)
                tracks.append({"name": nm, "lats": lats, "lons": lons})

            pts = []
            for key in ("p10", "p50", "p90"):
                p = self.pred[key]
                pts.append({
                    "name": key.upper(),
                    "lat": float(p["lat"]),
                    "lon": float(p["lon"]),
                    "description": f"time={p['t_utc']}, downrange_km={p['downrange_km']:.0f}, beta={p['beta']:.1f}, descent_s={p['descent_s']:.1f}"
                })

            swath_km = float(self._get_float(self.var_swath_km, "Swath ±km (KML)"))
            swath = {"lat": self.pred["p50"]["lat"], "lon": self.pred["p50"]["lon"], "radius_km": swath_km}

            export_kml(kml_path, tracks=tracks, points=pts, swath=swath)
            self._log(f"Saved KML: {kml_path}")
            messagebox.showinfo("KML Exported", f"Saved:\n{kml_path}")

        except Exception as e:
            messagebox.showerror("KML error", str(e))
            self._log(f"ERROR: {e}")

    def on_save_outputs(self):
        try:
            folder = filedialog.askdirectory(title="Select folder to save outputs")
            if not folder:
                return
            norad = int(self.var_norad.get().strip())
            stamp = dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")

            # Save PNG
            png_path = os.path.join(folder, f"reentry_plot_{norad}_{stamp}.png")
            self.fig.savefig(png_path, dpi=220)

            # Save prediction JSON if present
            if self.pred:
                json_path = os.path.join(folder, f"reentry_prediction_{norad}_{stamp}.json")
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(self.pred, f, indent=2)

            # Save latest TIP batch
            if self.solutions_latest:
                tip_path = os.path.join(folder, f"tip_latest_batch_{norad}_{stamp}.json")
                with open(tip_path, "w", encoding="utf-8") as f:
                    json.dump([s.raw for s in self.solutions_latest], f, indent=2)

            # Save TLE
            if self.tle:
                name, l1, l2 = self.tle
                tle_path = os.path.join(folder, f"tle_{norad}_{stamp}.txt")
                with open(tle_path, "w", encoding="utf-8") as f:
                    f.write(name + "\n" + l1 + "\n" + l2 + "\n")

            self._log(f"Saved outputs to: {folder}")
            messagebox.showinfo("Saved", "Outputs saved successfully.")

        except Exception as e:
            messagebox.showerror("Save error", str(e))
            self._log(f"ERROR: {e}")


def main():
    app = ReentryGUI()
    app.mainloop()

if __name__ == "__main__":
    main()
