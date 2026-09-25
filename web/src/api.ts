// Typed client for the dashboard's HTTP API (types: protocol.ts, "HTTP API").
// Same origin in both setups: Vite proxies /api to the backend in dev.
import type { EvaluateResponse, RunDetail, RunSummary, ScalarsResponse } from "./protocol";

async function request<T>(method: "GET" | "POST", path: string): Promise<T> {
  const response = await fetch(path, { method });
  if (!response.ok) {
    const body = (await response.json().catch(() => null)) as { detail?: string } | null;
    throw new Error(body?.detail ?? `${method} ${path}: HTTP ${response.status}`);
  }
  return (await response.json()) as T; // trusted: our server, contract-tested types
}

const enc = encodeURIComponent;

export const api = {
  runs: () => request<RunSummary[]>("GET", "/api/runs"),
  run: (run: string) => request<RunDetail>("GET", `/api/runs/${enc(run)}`),
  scalars: (run: string, tags: string[]) =>
    request<ScalarsResponse>("GET", `/api/runs/${enc(run)}/scalars?tags=${enc(tags.join(","))}`),
  evaluate: (run: string) => request<EvaluateResponse>("POST", `/api/runs/${enc(run)}/evaluate`),
};
