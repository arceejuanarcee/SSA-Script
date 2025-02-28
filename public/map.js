// Initialize Map with Philippine Centering
const map = L.map("map", { 
    center: [12.8797, 121.7740], 
    zoom: 6, 
    zoomControl: true,
    scrollWheelZoom: true 
});

// **Use Google Satellite Tiles for Clarity**
L.tileLayer("https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}", {
    attribution: "&copy; Google Maps"
}).addTo(map);

// **Ensure the map resizes properly**
function fixMapSize() {
    setTimeout(() => {
        map.invalidateSize();
    }, 500);
}

// Call the function to fix the map on load
fixMapSize();

// **Fix Tile Rendering on Resize**
window.addEventListener("resize", fixMapSize);

// **Define Philippine Territorial Zone**
const phTerritorialZone = [
    [21.5, 117.0], [21.5, 127.0], [4.5, 127.0], [4.5, 117.0], [21.5, 117.0]
];

L.polygon(phTerritorialZone, {
    color: "red",
    fillColor: "transparent",
    weight: 2
}).addTo(map).bindPopup("Philippine Territorial Zone");

// **Create Marker Groups**
const highInterestLayer = L.layerGroup().addTo(map);
const lowInterestLayer = L.layerGroup().addTo(map);

// **Function to Add Markers to Map**
function addMarkers(data, layer, color) {
    data.forEach((entry) => {
        // Parse latitude and longitude as numbers to ensure they're valid
        const lat = parseFloat(entry.Latitude);
        const lng = parseFloat(entry.Longitude);
        
        // Check if coordinates are valid numbers
        if (!isNaN(lat) && !isNaN(lng)) {
            const marker = L.marker([lat, lng], {
                icon: L.divIcon({
                    className: "custom-marker",
                    html: `<div style="background-color: ${color}; width: 14px; height: 14px; border-radius: 50%;"></div>`,
                }),
            });
            
            // Create a popup with all available data
            let popupContent = '<div class="marker-popup">';
            Object.keys(entry).forEach(key => {
                popupContent += `<b>${key}:</b> ${entry[key]}<br>`;
            });
            popupContent += '</div>';
            
            marker.bindPopup(popupContent);
            marker.addTo(layer);
        } else {
            console.warn("Invalid coordinates:", entry);
        }
    });
}

// **Toggle Controls for High & Low Interest Objects**
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

// **Load CSV Data**
function loadCSV(file, callback) {
    console.log(`Loading CSV file: ${file}`);
    Papa.parse(file, {
        download: true,
        header: true,
        skipEmptyLines: true,
        complete: function (results) {
            console.log(`CSV loaded: ${file}`, results.data.length + " records found");
            callback(results.data);
        },
        error: function(error) {
            console.error(`Error loading CSV file ${file}:`, error);
        }
    });
}

// **Load High-Interest Objects**
loadCSV("OrbitalDebrisY.csv", function (data) {
    console.log("Adding high interest markers", data.length);
    addMarkers(data, highInterestLayer, "red");
});

// **Load Low-Interest Objects**
loadCSV("OrbitalDebrisN.csv", function (data) {
    console.log("Adding low interest markers", data.length);
    addMarkers(data, lowInterestLayer, "blue");
});