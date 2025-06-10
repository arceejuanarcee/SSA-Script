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

# Define the EEZ polygon (list of [lat, lon] coordinates)
EEZ_POLYGON = [
    [20, 118],
    [20, 127],
    [4.75, 127],
    [4.75, 119.583],
    [7.667, 119.583],
    [7.667, 116],
    [10, 118],
    [20, 118]
]

# Create a session object
session = requests.Session()

def is_point_in_polygon(lat, lon, polygon):
    """
    Determines if a point (lat, lon) is inside a polygon defined by a list of [lat, lon] points.
    Uses the ray-casting algorithm.
    """
    # Convert polygon coordinates to (lon, lat) tuples
    poly = [(point[1], point[0]) for point in polygon]
    x, y = lon, lat
    inside = False
    n = len(poly)
    p1x, p1y = poly[0]
    for i in range(1, n + 1):
        p2x, p2y = poly[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if p1y != p2y:
                    xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                else:
                    xinters = p1x
                if p1x == p2x or x <= xinters:
                    inside = not inside
        p1x, p1y = p2x, p2y
    return inside

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
        content_type = data_response.headers.get("Content-Type", "")
        if "application/json" in content_type:
            data = data_response.json()
            with open(filename, "w") as file:
                json.dump(data, file, indent=4)
            print(f"Data saved to {filename}.")
            return data
        elif "application/xml" in content_type or "text/xml" in content_type:
            xml_data = data_response.text
            json_data = json.loads(json.dumps(xmltodict.parse(xml_data)))
            with open(filename, "w") as file:
                json.dump(json_data, file, indent=4)
            print(f"XML data converted and saved as JSON to {filename}.")
            return json_data
        else:
            print(f"Unexpected data format received from {url}")
            return None
    else:
        print(f"Failed to retrieve data from {url}: {data_response.status_code}")
        return None

def process_orbital_debris(tip_data, satcat_data):
    """
    Processes TIP data, identifies debris objects, sends notifications, and saves results to CSV files.
    A debris object is considered 'high interest' if it is flagged as such by the API or if it falls
    within the specified EEZ polygon.
    """
    if not tip_data or not isinstance(tip_data, list):
        print("No valid TIP data found.")
        return
    if not satcat_data or not isinstance(satcat_data, list):
        print("No valid SATCAT data found.")
        return

    # Create a lookup dictionary for SATCAT data
    satcat_dict = {int(entry.get("NORAD_CAT_ID", -1)): entry.get("OBJECT_TYPE", "Unknown") 
                   for entry in satcat_data}

    debris_high = []
    debris_low = []

    # Process each debris object from TIP data
    for obj in tip_data:
        try:
            norad_id = int(obj.get("NORAD_CAT_ID", -1))
        except ValueError:
            continue

        # Convert latitude and longitude to floats
        try:
            lat = float(obj.get("LAT", "N/A"))
        except (ValueError, TypeError):
            lat = 0.0
        try:
            lon = float(obj.get("LON", "N/A"))
        except (ValueError, TypeError):
            lon = 0.0

        # Adjust longitude if needed (convert from [0,360] to [-180,180])
        if lon > 180:
            lon = (-1) * (lon - 180)

        next_report = obj.get("NEXT_REPORT", "N/A")
        decay_epoch = obj.get("DECAY_EPOCH", "N/A")
        object_type = satcat_dict.get(norad_id, "Unknown")

        # Determine high interest: flagged by API OR within the EEZ polygon
        api_high_interest = (obj.get("HIGH_INTEREST") == "Y")
        eez_high_interest = is_point_in_polygon(lat, lon, EEZ_POLYGON)
        high_interest = api_high_interest or eez_high_interest

        debris_entry = [norad_id, next_report, lat, lon, object_type, decay_epoch]

        if high_interest:
            debris_high.append(debris_entry)
        else:
            debris_low.append(debris_entry)

        # Send Teams notification with the determined interest level
        send_teams_notification(norad_id, object_type, lat, lon, next_report, decay_epoch, high_interest)

    # Save results to CSV files
    headers = ["NORAD ID", "Next Report", "Latitude", "Longitude", "Debris Type", "Decay Date"]

    with open("OrbitalDebrisHigh.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(debris_high)
    print("High-interest debris saved to OrbitalDebrisHigh.csv.")

    with open("OrbitalDebrisLow.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(debris_low)
    print("Low-interest debris saved to OrbitalDebrisLow.csv.")

def send_teams_notification(norad_id, debris_type, lat, lon, next_report, decay_epoch, high_interest):
    """Sends a Microsoft Teams notification including the NEXT_REPORT field."""
    status = "High Interest" if high_interest else "Low Interest"
    title = f"New Orbital Debris Detected - {status}"
    payload = {
        "type": "message",
        "attachments": [
            {
                "contentType": "application/vnd.microsoft.card.adaptive",
                "content": {
                    "type": "AdaptiveCard",
                    "body": [
                        {"type": "TextBlock", "size": "Medium", "weight": "Bolder", "text": title},
                        {"type": "TextBlock", "text": f"**NORAD ID:** {norad_id}", "wrap": True},
                        {"type": "TextBlock", "text": f"**Debris Type:** {debris_type}", "wrap": True},
                        {"type": "TextBlock", "text": f"**Latitude:** {lat}", "wrap": True},
                        {"type": "TextBlock", "text": f"**Longitude:** {lon}", "wrap": True},
                        {"type": "TextBlock", "text": f"**Next Report:** {next_report}", "wrap": True},
                        {"type": "TextBlock", "text": f"**Decay Date:** {decay_epoch}", "wrap": True}
                    ],
                    "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
                    "version": "1.2"
                }
            }
        ]
    }
    headers = {"Content-Type": "application/json"}
    try:
        response = requests.post(TEAMS_WEBHOOK_URL, data=json.dumps(payload), headers=headers, timeout=30)
        if response.status_code == 200:
            print(f"Notification sent for NORAD ID {norad_id}.")
        else:
            print(f"Sending notification for NORAD ID {norad_id}: {response.status_code} - {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"Failed to send Teams notification: {str(e)}")

def main():
    if login():
        tip_data = fetch_data(TIP_URL, "tip.json")
        satcat_data = fetch_data(SATCAT_URL, "satcat.json")
        if tip_data and satcat_data:
            process_orbital_debris(tip_data, satcat_data)

if __name__ == "__main__":
    main()
