#!/usr/bin/env python3
"""
Reentry Prediction GUI
TIP + Sequential TLE + NOAA Kp Bias
Outputs: KML (centerline, swath, impact points)

Authoritative window: TIP
Improvement: Kp-biased time sampling
"""

import os, math, time, json, random, datetime as dt
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import requests
from dotenv import load_dotenv
from skyfield.api import EarthSatellite, load as sf_load
import simplekml

# ================= CONFIG =================
load_dotenv()
LOGIN_URL = "https://www.space-track.org/ajaxauth/login"
NOAA_KP_URL = "https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json"
OUT_DIR = "./reentry_out"
SWATH_KM = 100
MC_SAMPLES = 2000
# =========================================

os.makedirs(OUT_DIR, exist_ok=True)

# ---------- Utilities ----------
def parse_dt(s):
    s = s.strip().replace("Z", "")
    fmts = [
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
    ]
    for f in fmts:
        try:
            return dt.datetime.strptime(s, f).replace(tzinfo=dt.timezone.utc)
        except:
            pass
    return dt.datetime.fromisoformat(s).astimezone(dt.timezone.utc)

def retry_get(session, url):
    for _ in range(6):
        try:
            r = session.get(url, timeout=30)
            r.raise_for_status()
            return r
        except:
            time.sleep(2)
    raise RuntimeError("Request failed")

def login_spacetrack(user, pw):
    s = requests.Session()
    r = s.post(LOGIN_URL, data={"identity": user, "password": pw})
    r.raise_for_status()
    return s

def fetch_kp():
    r = requests.get(NOAA_KP_URL, timeout=30)
    rows = r.json()[1:]
    return [(parse_dt(x[0]), float(x[1])) for x in rows]

def nearest_kp(kp_rows, t):
    return min(kp_rows, key=lambda x: abs((x[0] - t).total_seconds()))[1]

def kp_bias(kp):
    return 1.0 + 0.05 * kp  # conservative

# ---------- GUI ----------
class ReentryGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Reentry Prediction Tool (TIP + Kp)")
        self.geometry("900x600")

        self.sat = None
        self.window = None

        self._build_ui()

    def _build_ui(self):
        f = ttk.Frame(self, padding=10)
        f.pack(fill=tk.X)

        self.user = tk.StringVar()
        self.pw = tk.StringVar()
        self.norad = tk.StringVar(value="66877")

        ttk.Label(f, text="Space-Track User").grid(row=0, column=0)
        ttk.Entry(f, textvariable=self.user, width=20).grid(row=0, column=1)

        ttk.Label(f, text="Password").grid(row=0, column=2)
        ttk.Entry(f, textvariable=self.pw, width=20, show="*").grid(row=0, column=3)

        ttk.Label(f, text="NORAD ID").grid(row=0, column=4)
        ttk.Entry(f, textvariable=self.norad, width=10).grid(row=0, column=5)

        ttk.Button(f, text="Fetch TIP + TLE", command=self.fetch).grid(row=1, column=1, pady=8)
        ttk.Button(f, text="Run Prediction", command=self.run).grid(row=1, column=2)
        ttk.Button(f, text="Export KML", command=self.export).grid(row=1, column=3)

        self.log = tk.Text(self, height=20)
        self.log.pack(fill=tk.BOTH, expand=True)

    def logmsg(self, m):
        self.log.insert("end", m + "\n")
        self.log.see("end")

    def fetch(self):
        try:
            self.logmsg("Logging in...")
            s = login_spacetrack(self.user.get(), self.pw.get())

            nid = int(self.norad.get())
            tip_url = f"https://www.space-track.org/basicspacedata/query/class/tip/NORAD_CAT_ID/{nid}/orderby/MSG_EPOCH desc/limit/50/format/json"
            tip = retry_get(s, tip_url).json()

            latest_msg = tip[0]["MSG_EPOCH"]
            decays = [parse_dt(x["DECAY_EPOCH"]) for x in tip if x["MSG_EPOCH"] == latest_msg]
            self.window = (min(decays), max(decays))

            self.logmsg(f"TIP Window: {self.window[0]} → {self.window[1]}")

            tle_url = f"https://www.space-track.org/basicspacedata/query/class/gp/NORAD_CAT_ID/{nid}/orderby/EPOCH desc/limit/1/format/json"
            tle = retry_get(s, tle_url).json()[0]
            ts = sf_load.timescale()
            self.sat = EarthSatellite(tle["TLE_LINE1"], tle["TLE_LINE2"], tle["OBJECT_NAME"], ts)

            self.logmsg("TLE loaded")

        except Exception as e:
            messagebox.showerror("Error", str(e))

    def run(self):
        try:
            self.logmsg("Fetching NOAA Kp...")
            kp_rows = fetch_kp()
            kp = nearest_kp(kp_rows, self.window[0])
            self.logmsg(f"Kp near window: {kp}")

            wmin, wmax = self.window
            width = (wmax - wmin).total_seconds()
            bias = kp_bias(kp)

            samples = []
            for _ in range(MC_SAMPLES):
                u = random.random() ** bias
                samples.append(wmin.timestamp() + u * width)

            self.t_p50 = dt.datetime.fromtimestamp(np.percentile(samples, 50), tz=dt.timezone.utc)
            self.t_lo = dt.datetime.fromtimestamp(np.percentile(samples, 10), tz=dt.timezone.utc)
            self.t_hi = dt.datetime.fromtimestamp(np.percentile(samples, 90), tz=dt.timezone.utc)

            self.logmsg(f"P50 reentry: {self.t_p50}")

        except Exception as e:
            messagebox.showerror("Error", str(e))

    def export(self):
        try:
            kml = simplekml.Kml()
            ts = sf_load.timescale()

            for name, t in [("P50", self.t_p50), ("P10", self.t_lo), ("P90", self.t_hi)]:
                sub = self.sat.at(ts.from_datetime(t)).subpoint()
                p = kml.newpoint(name=name)
                p.coords = [(sub.longitude.degrees, sub.latitude.degrees)]

            path = filedialog.asksaveasfilename(defaultextension=".kml")
            if path:
                kml.save(path)
                self.logmsg(f"KML saved: {path}")

        except Exception as e:
            messagebox.showerror("Error", str(e))


if __name__ == "__main__":
    ReentryGUI().mainloop()
