#!/usr/bin/env python3
# Author: Arcee Juan, Project Technical Assistant IV, SMCOD

import os
import requests
import json
import csv
from dotenv import load_dotenv

# Load credentials from .env file
load_dotenv()

USERNAME = os.getenv("SPACE_TRACK_USERNAME")
PASSWORD = os.getenv("SPACE_TRACK_PASSWORD")

# API Endpoints
LOGIN_URL = "https://www.space-track.org/ajaxauth/login"
SATCAT_URL = "https://www.space-track.org/basicspacedata/query/class/satcat/orderby/NORAD_CAT_ID%20desc/format/json/emptyresult/show"
TIP_URL = "https://www.space-track.org/basicspacedata/query/class/tip/DECAY_EPOCH/%3E2024-05-01/format/json/emptyresult/show"

# Microsoft Teams Webhook URL
TEAMS_WEBHOOK_URL = "https://prod-35.southeastasia.logic.azure.com:443/workflows/..."  # Replace with actual URL

session = requests.Session()

def login():
    payload = {"identity": USERNAME, "password": PASSWORD}
    response = session.post(LOGIN_URL, data=payload)
    if response.status_code == 200:
        print("Login successful.")
        return True
    else:
        print(f"Login failed ({response.status_code}): {response.text}")
        return False

def fetch_json_data(url):
    response = session.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Failed to fetch data from {url}: {response.status_code}")
        return []

def send_teams_notification(norad_id, debris_type, lat, lon, next_report, decay_epoch):
    """Sends a Teams notification (no interest level tagging)."""
    payload = {
        "type": "message",
        "attachments": [{
            "contentType": "application/vnd.microsoft.card.adaptive",
            "content": {
                "type": "AdaptiveCard",
                "body": [
                    {"type": "TextBlock", "size": "Medium", "weight": "Bolder", "text": f"TIP Record for NORAD ID {norad_id}"},
                    {"type": "TextBlock", "text": f"**Debris Type:** {debris_type}", "wrap": True},
                    {"type": "TextBlock", "text": f"**Latitude:** {lat}", "wrap": True},
                    {"type": "TextBlock", "text": f"**Longitude:** {lon}", "wrap": True},
                    {"type": "TextBlock", "text": f"**Next Report:** {next_report}", "wrap": True},
                    {"type": "TextBlock", "text": f"**Decay Date:** {decay_epoch}", "wrap": True}
                ],
                "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
                "version": "1.2"
            }
        }]
    }
    headers = {"Content-Type": "application/json"}
    try:
        response = requests.post(TEAMS_WEBHOOK_URL, data=json.dumps(payload), headers=headers)
        if response.status_code == 200:
            print(f"Notification sent for NORAD ID {norad_id}.")
        else:
            print(f"Teams notification failed: {response.status_code} - {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"Exception while sending Teams notification: {e}")

def process_norad_6073(tip_data, satcat_data):
    satcat_dict = {int(entry.get("NORAD_CAT_ID", -1)): entry.get("OBJECT_TYPE", "Unknown") 
                   for entry in satcat_data}

    filtered = [obj for obj in tip_data if str(obj.get("NORAD_CAT_ID")) == "6073"]
    if not filtered:
        print("No TIP records found for NORAD_CAT_ID = 6073.")
        return

    headers = ["NORAD ID", "Next Report", "Latitude", "Longitude", "Debris Type", "Decay Date"]
    results = []

    for obj in filtered:
        try:
            lat = float(obj.get("LAT", 0.0))
            lon = float(obj.get("LON", 0.0))
            if lon > 180:
                lon -= 360
        except:
            lat, lon = 0.0, 0.0

        norad_id = 6073
        next_report = obj.get("NEXT_REPORT", "N/A")
        decay_epoch = obj.get("DECAY_EPOCH", "N/A")
        debris_type = satcat_dict.get(norad_id, "Unknown")

        results.append([norad_id, next_report, lat, lon, debris_type, decay_epoch])
        send_teams_notification(norad_id, debris_type, lat, lon, next_report, decay_epoch)

    with open("Filtered_NORAD_6073.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(results)
    print("Filtered results written to Filtered_NORAD_6073.csv.")

def main():
    if not login():
        return
    tip_data = fetch_json_data(TIP_URL)
    satcat_data = fetch_json_data(SATCAT_URL)
    if tip_data and satcat_data:
        process_norad_6073(tip_data, satcat_data)

if __name__ == "__main__":
    main()
