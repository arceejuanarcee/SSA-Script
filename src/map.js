// Initialize Map
const map = L.map("map").setView([12.8797, 121.7740], 5);

// Add Tile Layer
L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    attribution: "&copy; OpenStreetMap contributors",
}).addTo(map);

// Create Marker Groups
const highInterestLayer = L.layerGroup().addTo(map);
const lowInterestLayer = L.layerGroup().addTo(map);

// Function to Add Markers to Map
function addMarkers(data, layer, color) {
    data.forEach((entry) => {
        const marker = L.marker([entry.Latitude, entry.Longitude], {
            icon: L.divIcon({
                className: "custom-marker",
                html: `<div style="background-color: ${color}; width: 12px; height: 12px; border-radius: 50%;"></div>`,
            }),
        }).bindPopup(`<b>Lat:</b> ${entry.Latitude} <br> <b>Lon:</b> ${entry.Longitude}`);

        marker.addTo(layer);
    });
}

// Toggle Controls
document.getElementById("toggleHighInterest").addEventListener("change", function () {
    if (this.checked) {
        map.addLayer(highInterestLayer);
    } else {
        map.removeLayer(highInterestLayer);
    }
});

document.getElementById("toggleLowInterest").addEventListener("change", function () {
    if (this.checked) {
        map.addLayer(lowInterestLayer);
    } else {
        map.removeLayer(lowInterestLayer);
    }
});
