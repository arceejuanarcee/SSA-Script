# Author: Arcee Juan, Project Technical Assistant IV, SMCOD

import os
import requests
import json
import xmltodict
import csv
from dotenv import load_dotenv

# Load credentials from .env file (recommended for security)
load_dotenv()

USERNAME = os.getenv("SPACE_TRACK_USERNAME")
PASSWORD = os.getenv("SPACE_TRACK_PASSWORD")

# API URLs
LOGIN_URL = "https://www.space-track.org/ajaxauth/login"
SATCAT_URL = "https://www.space-track.org/basicspacedata/query/class/satcat/orderby/NORAD_CAT_ID%20desc/format/json/emptyresult/show"
TIP_URL = "https://www.space-track.org/basicspacedata/query/class/tip/DECAY_EPOCH/%3E2024-05-01/LAT/5--22/LON/111--128/NEXT_REPORT/%3C3/orderby/DECAY_EPOCH%20asc/format/json/emptyresult/show"

# Microsoft Teams Webhook (Replace with your actual webhook URL)
TEAMS_WEBHOOK_URL = "https://prod-35.southeastasia.logic.azure.com:443/workflows/71f9f90a83214555b09ec0412648bdc4/triggers/manual/paths/invoke?api-version=2016-06-01&sp=%2Ftriggers%2Fmanual%2Frun&sv=1.0&sig=_hh7aBNPqZmtA3zEqwSaXBwKOyxsWc4W3Bprrh8vj7o"

# Storage file for tracking last known NORAD IDs
LAST_KNOWN_DEBRIS_FILE = "last_known_debris.json"

# Create a session object
session = requests.Session()

def login():
    """Logs in to Space-Track and returns True if successful."""
    login_payload = {"identity": USERNAME, "password": PASSWORD}
    
    response = session.post(LOGIN_URL, data=login_payload)
    
    if response.status_code == 200:
        print("Login successful.")
        return True
    else:
        print(f"Login failed: {response.status_code}, {response.text}")
        return False

def fetch_data(url, filename):
    """Fetches data using the active session and saves it to a file."""
    data_response = session.get(url)

    if data_response.status_code == 200:
        print(f"Data retrieved successfully from {url}")
        with open(filename, "w") as file:
            json.dump(data_response.json(), file, indent=4)
        print(f"Data saved to {filename}.")
        return data_response.json()
    else:
        print(f"Failed to retrieve data from {url}: {data_response.status_code}")
        return None

def send_teams_notification(norad_id, debris_type, lat, lon, next_report, decay_epoch, high_interest):
    """Sends a Microsoft Teams notification using an Adaptive Card for structured reporting."""
    status = "High Interest Object" if high_interest else "Low Interest Object"
    payload = {
        "type": "message",
        "attachments": [
            {
                "contentType": "application/vnd.microsoft.card.adaptive",
                "content": {
                    "type": "AdaptiveCard",
                    "body": [
                        {"type": "TextBlock", "size": "Medium", "weight": "Bolder", "text": f"New Orbital Debris Detected - {status}"},
                        {"type": "TextBlock", "text": f"**NORAD ID:** {norad_id}", "wrap": True},
                        {"type": "TextBlock", "text": f"**Debris Type:** {debris_type}", "wrap": True},
                        {"type": "TextBlock", "text": f"**Latitude:** {lat}", "wrap": True},
                        {"type": "TextBlock", "text": f"**Longitude:** {lon}", "wrap": True},
                        {"type": "TextBlock", "text": f"**Next Report:** {next_report}", "wrap": True},
                        {"type": "TextBlock", "text": f"**Decay Date:** {decay_epoch}", "wrap": True},
                        {"type": "TextBlock", "text": f"Status: {status}", "wrap": True},
                        {"type": "TextBlock", "text": "Sourced from space-track.org", "wrap": True, "size": "Small", "color": "Accent"}
                    ],
                    "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
                    "version": "1.2"
                }
            }
        ]
    }
    
    headers = {"Content-Type": "application/json"}
    requests.post(TEAMS_WEBHOOK_URL, data=json.dumps(payload), headers=headers)

def check_for_new_debris():
    """Checks the CSV files for new debris and sends alerts."""
    last_known = {"high_interest": [], "non_high_interest": []}
    if os.path.exists(LAST_KNOWN_DEBRIS_FILE):
        with open(LAST_KNOWN_DEBRIS_FILE, "r") as file:
            last_known = json.load(file)

    new_high_interest = []
    new_non_high_interest = []

    # Process high-interest debris
    if os.path.exists("OrbitalDebrisY.csv"):
        with open("OrbitalDebrisY.csv", "r") as file:
            reader = csv.reader(file)
            next(reader)  # Skip header
            for row in reader:
                norad_id, next_report, lat, lon, debris_type, decay_epoch = row
                if int(norad_id) not in last_known["high_interest"]:
                    send_teams_notification(norad_id, debris_type, lat, lon, next_report, decay_epoch, high_interest=True)
                    new_high_interest.append(int(norad_id))

    # Process low-interest debris (previously "non-high interest")
    if os.path.exists("OrbitalDebrisN.csv"):
        with open("OrbitalDebrisN.csv", "r") as file:
            reader = csv.reader(file)
            next(reader)  # Skip header
            for row in reader:
                norad_id, next_report, lat, lon, debris_type, decay_epoch = row
                if int(norad_id) not in last_known["non_high_interest"]:
                    send_teams_notification(norad_id, debris_type, lat, lon, next_report, decay_epoch, high_interest=False)
                    new_non_high_interest.append(int(norad_id))

    # Update last known debris
    last_known["high_interest"].extend(new_high_interest)
    last_known["non_high_interest"].extend(new_non_high_interest)
    with open(LAST_KNOWN_DEBRIS_FILE, "w") as file:
        json.dump(last_known, file, indent=4)

def main():
    if login():
        fetch_data(TIP_URL, "tip.json")
        fetch_data(SATCAT_URL, "satcat.json")
        check_for_new_debris()

if __name__ == "__main__":
    main()
