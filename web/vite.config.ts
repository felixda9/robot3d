import { defineConfig } from "vite";

// Where the Python backend (`uv run scripts/serve.py`) listens, as host:port.
// 127.0.0.1 rather than "localhost": Node may resolve localhost to IPv6 (::1),
// but uvicorn listens on IPv4 by default.
const backend = process.env.ROBOT3D_BACKEND ?? "127.0.0.1:8000";

// The page always talks to the host it was loaded from: ws://<host>/ws and
// http://<host>/api. In development Vite (port 5173) forwards both to the
// backend, so the frontend code never needs to know the backend's address.
const proxy = {
  "/ws": { target: `ws://${backend}`, ws: true },
  "/api": { target: `http://${backend}` },
};

export default defineConfig({
  server: { port: 5173, strictPort: true, proxy },
  preview: { port: 4173, strictPort: true, proxy },
  // three.js alone is ~550 kB minified; that's fine for a local app.
  build: { chunkSizeWarningLimit: 1000 },
});
