import { defineConfig } from 'vite';
import { resolve } from 'path';
import { viteStaticCopy } from 'vite-plugin-static-copy'; // FIXED IMPORT

export default defineConfig({
  root: 'src',
  server: {
    port: 3000,
    open: true,
  },
  build: {
    outDir: '../dist',
    rollupOptions: {
      input: {
        main: resolve(__dirname, 'src/index.html'),
      },
    },
  },
  plugins: [
    viteStaticCopy({
      targets: [
        {
          src: 'src/map.js',  // Source file
          dest: ''  // Copies it directly to dist/
        }
      ]
    })
  ]
});
