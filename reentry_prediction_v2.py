#!/usr/bin/env python3
"""
Reentry Assessment (TIP + Seq TLE + NOAA Kp bias) -> KML/JSON outputs

What this script DOES (operationally useful):
- Uses latest TIP batch (newest MSG_EPOCH) to define the reentry time window.
- Uses sequential TLEs (B* proxy) + NOAA planetary Kp to BIAS where inside the TIP window
  the reentry time likely falls (Monte Carlo).
- Converts sampled reentry times to "impact proxy" using SGP4 subpoint-at-time.
- Exports KML for GIS:
    * Centerline at expected (P50) time over +/- minutes around reentry
    * Swath edges (±swath_km) as two lines (approx)
    * Placemark points: Expected (P50), P_lo, P_hi

What it does NOT do (physics honesty):
- Not a high-fidelity descent / breakup / fragment dispersion model.
- Kp affects drag environment, but "where debris lands" still remains uncertain and
  should be treated as a time-uncertainty -> along-track uncertainty proxy.

Requirements:
  pip install requests python-dotenv numpy skyfield simplekml

Env vars (or fill in GUI-less config below):
  SPACE_TRACK_USERNAME
  SPACE_TRACK_PASSWORD
Optional:
  OUT_DIR=./reentry_out
  NORAD_CAT_ID=66877
  TIP_LIMIT=200
"""

from __future__ import annotations

import os
import json
import time
import math
import random
import datetime as dt
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import requests
from dotenv import load_dotenv
from skyfield.api import EarthSatellite, load as sf_load

import simplekml


# -----------------------------
# Config
# -----------------------------
load_dotenv()

LOGIN_URL = "https://www.space-track.org/ajaxauth/login"
DEFAULT_TIP_LIMIT = int(os.getenv("TIP_LIMIT", "200"))

NOAA_KP_OBS_URL = "https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json"
# (Optional) 1-minute estimated Kp for "right now" context:
NOAA_KP_1M_URL = "https://services.swpc.noaa.gov/products/noaa-estimated-planetary-k-index-1-minute.json"

OUT_DIR = os.getenv("OUT_DIR", "./reentry_out")
NORAD_CAT_ID = int(os.getenv("NORAD_CAT_ID", "66877"))

# Corridor drawing settings (for KML)
CORRIDOR_MIN_BEFORE = 90   # minutes before expected time for drawing the path
CORRIDOR_MIN_AFTER = 90    # minutes after expected time for drawing the path
CORRIDOR_STEP_S = 30       # seconds step for ground track sampling
SWATH_KM = 100.0           # swath half-width (EU-SST-like ±100km)


# -----------------------------
# Data models
# -----------------------------
@dataclass
class TipSolution:
    msg_epoch: str
    decay_epoch: str
    rev: Optional[int]
    raw: dict


@dataclass
class TLEPoint:
    epoch_utc: dt.datetime
    name: str
    l1: str
    l2: str
    bstar: float


# -----------------------------
# Time parsing (robust)
# -----------------------------
def parse_dt_any(s: str) -> dt.datetime:
    """
    Handles:
      - 'YYYY-MM-DD HH:MM:SS'
      - 'YYYY-MM-DDTHH:MM:SS'
      - with fractional seconds
      - with optional trailing 'Z'
    Returns UTC tz-aware datetime.
    """
    s = (s or "").strip()
    if not s:
        raise ValueError("empty datetime string")

    # Normalize Z
    if s.endswith("Z"):
        s = s[:-1]

    # Replace space/T variations: we'll try multiple formats
    fmts = [
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
    ]
    for f in fmts:
        try:
            return dt.datetime.strptime(s, f).replace(tzinfo=dt.timezone.utc)
        except Exception:
            pass

    # Last resort: fromisoformat (handles fractional; may include offset)
    try:
        x = dt.datetime.fromisoformat(s)
        if x.tzinfo is None:
            x = x.replace(tzinfo=dt.timezone.utc)
        return x.astimezone(dt.timezone.utc)
    except Exception as e:
        raise ValueError(f"Unrecognized datetime: {s}") from e


def dt_to_iso_z(t: dt.datetime) -> str:
    return t.astimezone(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")


# -----------------------------
# Space-Track helpers
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
        raise RuntimeError("Missing SPACE_TRACK_USERNAME / SPACE_TRACK_PASSWORD.")
    s = requests.Session()
    r = s.post(LOGIN_URL, data={"identity": username, "password": password}, timeout=30)
    r.raise_for_status()
    return s


def fetch_tip_latest_batch(session: requests.Session, norad_id: int, limit: int = DEFAULT_TIP_LIMIT) -> List[TipSolution]:
    """
    Fetch TIP rows, sort by MSG_EPOCH desc, then keep ONLY newest MSG_EPOCH batch.
    """
    url = (
        f"https://www.space-track.org/basicspacedata/query/class/tip/"
        f"NORAD_CAT_ID/{norad_id}/orderby/MSG_EPOCH%20desc/limit/{int(limit)}/format/json"
    )
    r = retry_get(session, url)
    data = r.json()
    if not isinstance(data, list) or not data:
        raise RuntimeError("No TIP data returned.")

    sols: List[TipSolution] = []
    for row in data:
        sols.append(
            TipSolution(
                msg_epoch=(row.get("MSG_EPOCH") or "").strip(),
                decay_epoch=(row.get("DECAY_EPOCH") or "").strip(),
                rev=int(row["REV"]) if str(row.get("REV", "")).isdigit() else None,
                raw=row,
            )
        )

    # Sort newest MSG_EPOCH first
    sols.sort(key=lambda s: parse_dt_any(s.msg_epoch).timestamp() if s.msg_epoch else 0.0, reverse=True)
    newest = sols[0].msg_epoch
    if not newest:
        return sols[:1]
    return [s for s in sols if s.msg_epoch == newest]


def compute_tip_window(latest_batch: List[TipSolution]) -> Tuple[dt.datetime, dt.datetime]:
    decays: List[dt.datetime] = []
    for s in latest_batch:
        if s.decay_epoch:
            try:
                decays.append(parse_dt_any(s.decay_epoch))
            except Exception:
                pass
    if not decays:
        raise RuntimeError("Latest TIP batch has no parsable DECAY_EPOCH values.")
    return min(decays), max(decays)


def parse_bstar_from_tle_line1(l1: str) -> float:
    """
    TLE B* is typically in cols 54-61 (1-indexed); common parsing uses slice [53:61].
    Example: ' 34123-4' => 0.34123e-4
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
        exponent = int(exp)  # like -4
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
        raise RuntimeError("No TLE history returned.")

    out: List[TLEPoint] = []
    for row in data:
        name = row.get("OBJECT_NAME") or f"NORAD {norad_id}"
        l1 = row.get("TLE_LINE1")
        l2 = row.get("TLE_LINE2")
        epoch = row.get("EPOCH")
        if not (l1 and l2 and epoch):
            continue

        epoch_dt = parse_dt_any(epoch)
        bstar = parse_bstar_from_tle_line1(l1)

        out.append(TLEPoint(epoch_utc=epoch_dt, name=name, l1=l1.strip(), l2=l2.strip(), bstar=bstar))

    out.sort(key=lambda x: x.epoch_utc)  # oldest -> newest
    if len(out) < 2:
        raise RuntimeError("Insufficient TLE history after parsing.")
    return out


def robust_bstar_stats(tles: List[TLEPoint]) -> Dict[str, float]:
    vals = np.array([x.bstar for x in tles if np.isfinite(x.bstar)], dtype=float)
    if len(vals) < 5:
        return {"bstar_med": float("nan"), "bstar_mad": float("nan"), "bstar_trend_per_day": float("nan")}

    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med))) + 1e-30

    n = min(15, len(tles))
    t0 = tles[-n].epoch_utc
    xs = np.array([(tles[-n+i].epoch_utc - t0).total_seconds() / 86400 for i in range(n)], dtype=float)
    ys = np.array([tles[-n+i].bstar for i in range(n)], dtype=float)
    m = float(np.polyfit(xs, ys, 1)[0])
    return {"bstar_med": med, "bstar_mad": mad, "bstar_trend_per_day": m}


# -----------------------------
# NOAA Kp fetch + mapping
# -----------------------------
def fetch_noaa_kp_observed() -> List[Dict[str, Any]]:
    r = requests.get(NOAA_KP_OBS_URL, timeout=30)
    r.raise_for_status()
    data = r.json()
    if not isinstance(data, list) or len(data) < 2:
        raise RuntimeError("Unexpected NOAA Kp JSON format.")
    # First row is header in SWPC products; subsequent rows are values
    header = data[0]
    rows = data[1:]
    # Build dicts
    out = []
    for row in rows:
        d = {header[i]: row[i] for i in range(min(len(header), len(row)))}
        out.append(d)
    return out


def kp_at_time(kp_rows: List[Dict[str, Any]], t_utc: dt.datetime) -> Optional[float]:
    """
    Find the nearest observed 3-hour Kp to a given UTC time.
    NOAA 'time_tag' is ISO-ish.
    """
    best = None
    best_dt = None
    for r in kp_rows:
        tt = r.get("time_tag") or r.get("time") or r.get("timestamp")
        kv = r.get("kp_index") if "kp_index" in r else r.get("kp")
        if tt is None or kv is None:
            continue
        try:
            dtt = parse_dt_any(str(tt))
            kpf = float(kv)
        except Exception:
            continue
        if best_dt is None or abs((dtt - t_utc).total_seconds()) < abs((best_dt - t_utc).total_seconds()):
            best_dt = dtt
            best = kpf
    return best


def kp_drag_multiplier(kp: float) -> float:
    """
    Heuristic: map Kp into a drag activity multiplier.
    Keep conservative so we DON'T overclaim.
    """
    # Kp 0..9 -> multiplier ~ 0.9 .. 1.35
    kp = float(np.clip(kp, 0.0, 9.0))
    return 0.90 + 0.05 * kp


# -----------------------------
# Orbit -> ground track
# -----------------------------
def groundtrack(
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
# Monte Carlo: time sampling inside TIP window (biased by B* + Kp)
# -----------------------------
def mc_sample_reentry_times(
    wmin: dt.datetime,
    wmax: dt.datetime,
    bstar_stats: Dict[str, float],
    kp_value: Optional[float],
    n: int = 2000,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Output: array of POSIX seconds within [wmin, wmax].

    Bias knobs:
    - Increasing drag activity (higher Kp) -> earlier within the window.
    - Higher B* median / increasing trend -> earlier.
    """
    if seed is not None:
        np.random.seed(seed)

    width_s = max(1.0, (wmax - wmin).total_seconds())

    med = bstar_stats.get("bstar_med", float("nan"))
    mad = bstar_stats.get("bstar_mad", float("nan"))
    trend = bstar_stats.get("bstar_trend_per_day", float("nan"))

    # Base bias in [-0.25, +0.25] where negative -> earlier
    bias = 0.0

    if np.isfinite(med):
        bias += -0.08 * np.tanh(math.log10(abs(med) + 1e-12) + 7.0)
    if np.isfinite(trend):
        bias += -0.10 * np.tanh(trend * 5e4)

    if kp_value is not None and np.isfinite(kp_value):
        mult = kp_drag_multiplier(kp_value)
        # mult 0.9..1.35 -> map to approx bias -0.06..+0.02 (earlier when higher)
        bias += float(np.clip(-(mult - 1.0) * 0.20, -0.10, 0.05))

    bias = float(np.clip(bias, -0.25, +0.25))

    # Flatness based on MAD (bigger MAD -> flatter distribution)
    flatness = 0.55
    if np.isfinite(mad) and np.isfinite(med) and abs(med) > 0:
        rel = mad / (abs(med) + 1e-30)
        flatness = float(np.clip(0.35 + 2.0 * rel, 0.35, 0.95))

    base_mean = float(np.clip(0.5 + bias, 0.10, 0.90))
    k = (1.0 - flatness) * 18.0 + 2.0  # 2..20
    a = base_mean * k
    b = (1.0 - base_mean) * k

    u = np.random.beta(a, b, size=int(n))
    ts = np.array([wmin.timestamp() + float(x) * width_s for x in u], dtype=float)
    return ts


# -----------------------------
# Swath edges (approx offset)
# -----------------------------
def bearing_deg(lat1, lon1, lat2, lon2) -> float:
    """Initial bearing from point1->point2."""
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dlon = math.radians(lon2 - lon1)
    y = math.sin(dlon) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlon)
    brng = math.degrees(math.atan2(y, x))
    return (brng + 360.0) % 360.0


def offset_point(lat, lon, bearing, offset_km) -> Tuple[float, float]:
    """
    Move from (lat,lon) along 'bearing' by offset_km on a sphere (great-circle).
    """
    R = 6371.0088
    d = offset_km / R
    br = math.radians(bearing)
    phi1 = math.radians(lat)
    lam1 = math.radians(lon)

    phi2 = math.asin(math.sin(phi1) * math.cos(d) + math.cos(phi1) * math.sin(d) * math.cos(br))
    lam2 = lam1 + math.atan2(math.sin(br) * math.sin(d) * math.cos(phi1), math.cos(d) - math.sin(phi1) * math.sin(phi2))

    lat2 = math.degrees(phi2)
    lon2 = math.degrees(lam2)
    lon2 = ((lon2 + 180) % 360) - 180
    return lat2, lon2


def make_swath_edges(lats: List[float], lons: List[float], swath_km: float) -> Tuple[List[float], List[float], List[float], List[float]]:
    """
    Returns (left_lats, left_lons, right_lats, right_lons) approximated by perpendicular offsets.
    """
    n = len(lats)
    if n < 2:
        return [], [], [], []

    left_lats, left_lons = [], []
    right_lats, right_lons = [], []

    for i in range(n):
        if i == n - 1:
            br = bearing_deg(lats[i-1], lons[i-1], lats[i], lons[i])
        else:
            br = bearing_deg(lats[i], lons[i], lats[i+1], lons[i+1])

        left_br = (br - 90.0) % 360.0
        right_br = (br + 90.0) % 360.0

        laL, loL = offset_point(lats[i], lons[i], left_br, swath_km)
        laR, loR = offset_point(lats[i], lons[i], right_br, swath_km)

        left_lats.append(laL); left_lons.append(loL)
        right_lats.append(laR); right_lons.append(loR)

    return left_lats, left_lons, right_lats, right_lons


# -----------------------------
# KML export
# -----------------------------
def kml_color_aabbggrr(a, bb, gg, rr) -> str:
    """KML uses aabbggrr hex."""
    return f"{a:02x}{bb:02x}{gg:02x}{rr:02x}"


def add_linestring(kml: simplekml.Kml, name: str, lats: List[float], lons: List[float], color: str, width: int = 3) -> None:
    ls = kml.newlinestring(name=name)
    ls.coords = list(zip(lons, lats))
    ls.altitudemode = simplekml.AltitudeMode.clamptoground
    ls.style.linestyle.color = color
    ls.style.linestyle.width = width


def add_point(kml: simplekml.Kml, name: str, lat: float, lon: float, color: str, scale: float = 1.2) -> None:
    p = kml.newpoint(name=name, coords=[(lon, lat)])
    p.style.iconstyle.color = color
    p.style.iconstyle.scale = scale


# -----------------------------
# Main pipeline
# -----------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    user = os.getenv("SPACE_TRACK_USERNAME", "").strip()
    pw = os.getenv("SPACE_TRACK_PASSWORD", "").strip()
    if not user or not pw:
        raise RuntimeError("Set SPACE_TRACK_USERNAME and SPACE_TRACK_PASSWORD in env or .env")

    print("[1] Login Space-Track")
    s = spacetrack_login(user, pw)

    print("[2] Fetch latest TIP batch")
    tip_batch = fetch_tip_latest_batch(s, NORAD_CAT_ID, limit=DEFAULT_TIP_LIMIT)
    msg_epoch = tip_batch[0].msg_epoch if tip_batch else ""
    wmin, wmax = compute_tip_window(tip_batch)
    print(f"    Latest TIP MSG_EPOCH: {msg_epoch}")
    print(f"    TIP window: {dt_to_iso_z(wmin)} -> {dt_to_iso_z(wmax)}  (width {(wmax-wmin)})")

    print("[3] Fetch TLE history")
    tles = fetch_tle_history(s, NORAD_CAT_ID, limit=25)
    stats = robust_bstar_stats(tles)
    latest = tles[-1]
    print(f"    Latest TLE epoch: {dt_to_iso_z(latest.epoch_utc)} | name={latest.name}")
    print(f"    B*: med={stats['bstar_med']:.3e} mad={stats['bstar_mad']:.3e} trend/day={stats['bstar_trend_per_day']:.3e}")

    ts = sf_load.timescale()
    sat = EarthSatellite(latest.l1, latest.l2, latest.name, ts)

    print("[4] Fetch NOAA observed planetary Kp and pick nearest to TLE epoch")
    kp_rows = fetch_noaa_kp_observed()
    kp_val = kp_at_time(kp_rows, latest.epoch_utc)
    print(f"    Kp near TLE epoch: {kp_val if kp_val is not None else 'N/A'}")

    print("[5] Monte Carlo sample reentry time inside TIP window (biased by B* + Kp)")
    samples = mc_sample_reentry_times(wmin, wmax, stats, kp_val, n=2500)
    p_lo = 10.0
    p_hi = 90.0
    t_lo = dt.datetime.fromtimestamp(float(np.percentile(samples, p_lo)), tz=dt.timezone.utc)
    t_p50 = dt.datetime.fromtimestamp(float(np.percentile(samples, 50.0)), tz=dt.timezone.utc)
    t_hi = dt.datetime.fromtimestamp(float(np.percentile(samples, p_hi)), tz=dt.timezone.utc)

    lat_lo, lon_lo = subpoint_at_time(sat, t_lo)
    lat_p50, lon_p50 = subpoint_at_time(sat, t_p50)
    lat_hi, lon_hi = subpoint_at_time(sat, t_hi)

    print(f"    P{p_lo:.0f}: {dt_to_iso_z(t_lo)}  ->  ({lat_lo:.2f},{lon_lo:.2f})")
    print(f"    P50 : {dt_to_iso_z(t_p50)} ->  ({lat_p50:.2f},{lon_p50:.2f})")
    print(f"    P{p_hi:.0f}: {dt_to_iso_z(t_hi)}  ->  ({lat_hi:.2f},{lon_hi:.2f})")

    print("[6] Build expected corridor centerline and swath edges for KML")
    lats, lons, times_dt = groundtrack(sat, t_p50, CORRIDOR_MIN_BEFORE, CORRIDOR_MIN_AFTER, CORRIDOR_STEP_S)
    left_lats, left_lons, right_lats, right_lons = make_swath_edges(lats, lons, SWATH_KM)

    stamp = dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    base = f"{NORAD_CAT_ID}_{stamp}"

    print("[7] Write outputs (JSON + KML)")
    assessment = {
        "generated_utc": dt_to_iso_z(dt.datetime.now(dt.timezone.utc)),
        "norad_id": NORAD_CAT_ID,
        "latest_tip_msg_epoch": msg_epoch,
        "tip_window": {"start_utc": dt_to_iso_z(wmin), "end_utc": dt_to_iso_z(wmax)},
        "tle_used": {"epoch_utc": dt_to_iso_z(latest.epoch_utc), "name": latest.name},
        "drag_proxies": {
            "bstar_median": stats.get("bstar_med"),
            "bstar_mad": stats.get("bstar_mad"),
            "bstar_trend_per_day": stats.get("bstar_trend_per_day"),
            "kp_near_tle_epoch": kp_val,
            "kp_source": NOAA_KP_OBS_URL,
            "notes": "Kp and B* used only to bias reentry time inside TIP window (not a full physics model).",
        },
        "monte_carlo": {
            "n": int(len(samples)),
            "percentiles": {"p10": dt_to_iso_z(t_lo), "p50": dt_to_iso_z(t_p50), "p90": dt_to_iso_z(t_hi)},
            "impact_proxy": {
                "p10": {"lat": lat_lo, "lon": lon_lo},
                "p50": {"lat": lat_p50, "lon": lon_p50},
                "p90": {"lat": lat_hi, "lon": lon_hi},
                "method": "SGP4 subpoint at sampled reentry time.",
            },
        },
        "kml": {
            "centerline_minutes_before": CORRIDOR_MIN_BEFORE,
            "centerline_minutes_after": CORRIDOR_MIN_AFTER,
            "step_seconds": CORRIDOR_STEP_S,
            "swath_km": SWATH_KM,
        },
    }

    json_path = os.path.join(OUT_DIR, f"assessment_{base}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(assessment, f, indent=2)

    # KML colors (aabbggrr)
    YELLOW = kml_color_aabbggrr(255, 0, 255, 255)   # ff00ffff
    GREEN  = kml_color_aabbggrr(255, 0, 255, 0)     # ff00ff00
    RED    = kml_color_aabbggrr(255, 0, 0, 255)     # ff0000ff
    ORANGE = kml_color_aabbggrr(255, 0, 165, 255)   # ff00a5ff

    kml = simplekml.Kml()
    add_linestring(kml, f"Reentry Ground Track (P50) - Centerline", lats, lons, color=YELLOW, width=4)
    if left_lats and right_lats:
        add_linestring(kml, f"Reentry Swath Edge (+{SWATH_KM:.0f} km)", right_lats, right_lons, color=GREEN, width=3)
        add_linestring(kml, f"Reentry Swath Edge (-{SWATH_KM:.0f} km)", left_lats, left_lons, color=GREEN, width=3)

    add_point(kml, f"Expected (P50) {dt_to_iso_z(t_p50)}", lat_p50, lon_p50, color=RED, scale=1.4)
    add_point(kml, f"P10 {dt_to_iso_z(t_lo)}", lat_lo, lon_lo, color=ORANGE, scale=1.1)
    add_point(kml, f"P90 {dt_to_iso_z(t_hi)}", lat_hi, lon_hi, color=ORANGE, scale=1.1)

    kml_path = os.path.join(OUT_DIR, f"corridor_{base}.kml")
    kml.save(kml_path)

    print(f"\nDONE ✅")
    print(f"- JSON: {json_path}")
    print(f"- KML : {kml_path}")
    print(f"\nTip: Open the KML in QGIS/Google Earth. The swath is an approximate ±{SWATH_KM:.0f} km envelope.")


if __name__ == "__main__":
    main()
