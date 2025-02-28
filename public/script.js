// Function to Load CSV Data
function loadCSV(file, callback) {
    Papa.parse(file, {
        download: true,
        header: true,
        skipEmptyLines: true,
        complete: function (results) {
            callback(results.data);
        },
    });
}

// Load High-Interest Objects
loadCSV("/OrbitalDebrisY.csv", function (data) {
    addMarkers(data, highInterestLayer, "red");
});

// Load Low-Interest Objects
loadCSV("/OrbitalDebrisN.csv", function (data) {
    addMarkers(data, lowInterestLayer, "blue");
});
