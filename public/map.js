document.addEventListener("DOMContentLoaded", function () {
    // Initialize the map centered on the Philippines
    const map = L.map("map", {
      center: [12.8797, 121.7740],
      zoom: 6,
      zoomControl: true,
      scrollWheelZoom: true,
    });
  
    // Use OpenStreetMap tiles
    L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
      attribution: "&copy; OpenStreetMap contributors"
    }).addTo(map);
  
    // Helper function to fix the map size if the container resizes
    function fixMapSize() {
      setTimeout(() => {
        map.invalidateSize();
      }, 300);
    }
    fixMapSize();
    window.addEventListener("resize", fixMapSize);
  
    // Define polygon for Philippine territorial zone
    const phTerritorialZone = [
        [20,118],
        [20,127],
        [4.75,127],
        [4.75,119.583],
        [7.667,119.583],
        [7.667,116],
        [10,118],
        [20,118]
    ];
  
    L.polygon(phTerritorialZone, {
      color: "red",
      fillColor: "transparent",
      weight: 2
    })
      .addTo(map)
      .bindPopup("Philippine Territorial Zone");
  
    // Create layer groups for markers
    const highInterestLayer = L.layerGroup().addTo(map);
    const lowInterestLayer = L.layerGroup().addTo(map);
  
    // Function to add markers to the specified layer
    function addMarkers(data, layer, color) {
      data.forEach((entry) => {
        const lat = parseFloat(entry.Latitude);
        const lng = parseFloat(entry.Longitude);
  
        if (!isNaN(lat) && !isNaN(lng)) {
          const marker = L.marker([lat, lng], {
            icon: L.divIcon({
              className: "custom-marker",
              html: `<div style="background-color: ${color}; width: 14px; height: 14px; border-radius: 50%;"></div>`
            })
          });
  
          // Create a popup with all available data
          let popupContent = '<div class="marker-popup">';
          Object.keys(entry).forEach((key) => {
            popupContent += `<b>${key}:</b> ${entry[key]}<br>`;
          });
          popupContent += "</div>";
  
          marker.bindPopup(popupContent);
          marker.addTo(layer);
        } else {
          console.warn("Invalid coordinates:", entry);
        }
      });
    }
  
    // Toggle event for High-Interest Objects
    const toggleHighInterest = document.getElementById("toggleHighInterest");
    if (toggleHighInterest) {
      toggleHighInterest.addEventListener("change", function () {
        if (this.checked) {
          map.addLayer(highInterestLayer);
        } else {
          map.removeLayer(highInterestLayer);
        }
      });
    }
  
    // Toggle event for Low-Interest Objects
    const toggleLowInterest = document.getElementById("toggleLowInterest");
    if (toggleLowInterest) {
      toggleLowInterest.addEventListener("change", function () {
        if (this.checked) {
          map.addLayer(lowInterestLayer);
        } else {
          map.removeLayer(lowInterestLayer);
        }
      });
    }
  
    // CSV loader using Papa Parse
    function loadCSV(file, callback) {
      console.log(`Loading CSV file: ${file}`);
      Papa.parse(file, {
        download: true,
        header: true,
        skipEmptyLines: true,
        complete: function (results) {
          console.log(`${file} loaded, ${results.data.length} records found`);
          callback(results.data);
        },
        error: function (error) {
          console.error(`Error loading CSV file ${file}:`, error);
        }
      });
    }
  
    // Load High-Interest Objects
    loadCSV("/OrbitalDebrisY.csv", function (data) {
      console.log("Adding high interest markers", data.length);
      addMarkers(data, highInterestLayer, "red");
    });
  
    // Load Low-Interest Objects
    loadCSV("/OrbitalDebrisN.csv", function (data) {
      console.log("Adding low interest markers", data.length);
      addMarkers(data, lowInterestLayer, "blue");
    });
  });
  