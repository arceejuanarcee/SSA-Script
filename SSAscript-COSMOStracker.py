#!/usr/bin/env python3
"""
Script to fetch TIP data from Space-Track and filter for NORAD_CAT_ID = 6073
"""
import os
import requests
import json
from dotenv import load_dotenv

# Load credentials from .env file
dotenv_path = ".env"
load_dotenv = load_dotenv
load_dotenv(dotenv_path)

USERNAME = os.getenv("SPACE_TRACK_USERNAME")
PASSWORD = os.getenv("SPACE_TRACK_PASSWORD")

# API endpoints
LOGIN_URL = "https://www.space-track.org/ajaxauth/login"
TIP_URL = (
    "https://www.space-track.org/basicspacedata/query/class/tip/"
    "DECAY_EPOCH/%3E2024-05-01/format/json/emptyresult/show"
)

# Create a session
session = requests.Session()


def login():
    """Log in to Space-Track"""
    payload = {"identity": USERNAME, "password": PASSWORD}
    r = session.post(LOGIN_URL, data=payload)
    if r.status_code == 200:
        print("Login successful.")
        return True
    else:
        print(f"Login failed ({r.status_code}): {r.text}")
        return False


def fetch_tip_data():
    """Fetch TIP data as JSON"""
    r = session.get(TIP_URL)
    if r.status_code == 200:
        return r.json()
    else:
        print(f"Failed to fetch TIP data ({r.status_code})")
        return []


def filter_norad_6073(tip_data):
    """Filter list of TIP records for NORAD_CAT_ID = 6073"""
    matches = [obj for obj in tip_data if str(obj.get("NORAD_CAT_ID")) == "6073"]
    return matches


def main():
    if not login():
        return

    tip_data = fetch_tip_data()
    if not tip_data:
        print("No TIP data to process.")
        return

    results = filter_norad_6073(tip_data)
    if results:
        print(json.dumps(results, indent=2))
    else:
        print("No records found for NORAD_CAT_ID = 6073.")


if __name__ == "__main__":
    main()
