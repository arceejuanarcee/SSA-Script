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

        # Check content type to handle XML responses
        content_type = data_response.headers.get("Content-Type", "")

        if "application/json" in content_type:
            # Save JSON directly
            with open(filename, "w") as file:
                json.dump(data_response.json(), file, indent=4)
            print(f"Data saved to {filename}.")
            return data_response.json()
        elif "application/xml" in content_type or "text/xml" in content_type:
            # Convert XML to JSON
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
    """Processes TIP data, identifies high-interest objects, and saves results to CSV files."""
    if not tip_data or not isinstance(tip_data, list):
        print("No valid TIP data found.")
        return
    
    if not satcat_data or not isinstance(satcat_data, list):
        print("No valid SATCAT data found.")
        return

    # Normalize SATCAT dictionary for easy lookup
    satcat_dict = {int(entry.get("NORAD_CAT_ID", -1)): entry.get("OBJECT_TYPE", "Unknown") for entry in satcat_data}

    debrisY = []
    debrisN = []

    # Detect and categorize debris
    for obj in tip_data:
        norad_id = int(obj.get("NORAD_CAT_ID", -1))
        lat = obj.get("LAT", "N/A")
        lon = obj.get("LON", "N/A")
        next_report = obj.get("NEXT_REPORT", "N/A")
        decay_epoch = obj.get("DECAY_EPOCH", "N/A")
        object_type = satcat_dict.get(norad_id, "Unknown")

        # Convert LON [0,360] to LON [-180,180]
        lon = float(lon) if lon != "N/A" else 0
        if lon > 180:
            lon = (-1) * (lon - 180)

        debris_entry = [norad_id, next_report, lat, lon, object_type, decay_epoch]

        if obj.get("HIGH_INTEREST") == "Y":
            debrisY.append(debris_entry)
        else:
            debrisN.append(debris_entry)

    # Save data to CSV
    headers = ["NORAD ID", "Next Report", "Latitude", "Longitude", "Debris Type", "Decay Date"]

    with open("OrbitalDebrisY.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(debrisY)
    print("High-interest debris saved to OrbitalDebrisY.csv.")

    with open("OrbitalDebrisN.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(debrisN)
    print("Non-high-interest debris saved to OrbitalDebrisN.csv.")

def main():
    """Automates login, data retrieval, and orbital debris processing."""
    if login():
        satcat_data = fetch_data(SATCAT_URL, "satcat.json")
        tip_data = fetch_data(TIP_URL, "tip.json")

        if tip_data and satcat_data:
            process_orbital_debris(tip_data, satcat_data)

if __name__ == "__main__":
    main()
