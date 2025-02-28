import { defineConfig } from "vite";
import { resolve } from "path";

export default defineConfig({
  root: "src",
  server: {
    port: 3000,
    open: true,
  },
  build: {
    outDir: "../dist",
    rollupOptions: {
      input: {
        main: resolve(__dirname, "src/index.html"),
      },
    },
  },
  plugins: [
    {
      name: "copy-csv-to-root",
      apply: "build",
      closeBundle() {
        const fs = require("fs");
        fs.copyFileSync("OrbitalDebrisY.csv", "dist/OrbitalDebrisY.csv");
        fs.copyFileSync("OrbitalDebrisN.csv", "dist/OrbitalDebrisN.csv");
      },
    },
  ],
});
