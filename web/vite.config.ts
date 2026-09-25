import { defineConfig } from "vite";

// Where the Python backend (`uv run scripts/serve.py`) listens. 127.0.0.1
// rather than "localhost": Node may resolve localhost to IPv6 (::1), but
// uvicorn listens on IPv4 by default.
const backend = process.env.ROBOT3D_BACKEND ?? "ws://127.0.0.1:8000";

// The page always connects to ws://<the host it was loaded from>/ws. In
// development Vite (port 5173) forwards that WebSocket to the backend, so
// the frontend code never needs to know the backend's address.
const proxy = { "/ws": { target: backend, ws: true } };

export default defineConfig({
  server: { port: 5173, strictPort: true, proxy },
  preview: { port: 4173, strictPort: true, proxy },
  // three.js alone is ~550 kB minified; that's fine for a local app.
  build: { chunkSizeWarningLimit: 1000 },
});
