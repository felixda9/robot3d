// Training dashboard: runs list, training curves (several runs overlaid),
// checkpoint table with evaluations, and "Watch" to replay a checkpoint in
// the simulator. Data comes from the HTTP API (api.ts); a running run is
// polled so its curves grow live.
import "./dashboard.css";
import { api } from "../api";
import type { CheckpointInfo, RunDetail, RunSummary, ScalarSeries } from "../protocol";
import { formatSteps, LineChart, type ChartOptions, type ChartSeries } from "./linechart";

const SERIES_SLOTS = 8; // categorical palette size (dashboard.css --series-1..8)
const POLL_MS = 10_000; // runs list + curves of running runs
const EVAL_POLL_MS = 2_000; // while checkpoints are being evaluated

interface CurveChart extends ChartOptions {
  tag: string;
}

const fixed = (digits: number) => (v: number) => v.toFixed(digits);

// One chart per TensorBoard tag; every checked run is a line on it.
const CURVES: CurveChart[] = [
  {
    tag: "rollout/ep_rew_mean",
    title: "Episode reward",
    subtitle: "Total reward per 20 s episode, averaged over recent episodes. Higher is better.",
    format: fixed(0),
  },
  {
    tag: "episode/speed_mps",
    title: "Forward speed",
    subtitle: "Average speed per episode in training (with exploration noise), m/s.",
    format: (v) => `${v.toFixed(2)} m/s`,
    tickFormat: fixed(1),
  },
  {
    tag: "episode/fell",
    title: "Falls",
    subtitle: "Share of episodes that ended with the robot falling over.",
    format: (v) => `${(v * 100).toFixed(0)}%`,
    yMin: 0,
    yMax: 1,
  },
  {
    tag: "time/fps",
    title: "Training speed",
    subtitle: "Environment steps per second, all parallel environments together.",
    format: (v) => `${Math.round(v).toLocaleString()} steps/s`,
    tickFormat: (v) => formatSteps(v),
    yMin: 0,
  },
  {
    tag: "train/std",
    title: "Exploration noise",
    subtitle: "Spread of the policy's random action noise. Shrinks as it grows confident.",
    format: fixed(3),
    tickFormat: fixed(2),
    yMin: 0,
  },
  {
    tag: "train/approx_kl",
    title: "Policy change per update",
    subtitle: "How far each update moved the policy (approx. KL, log scale). Spikes = too big a step.",
    format: fixed(4),
    tickFormat: (v) => String(v),
    log: true,
  },
  {
    tag: "train/explained_variance",
    title: "Value function fit",
    subtitle: "How well the critic predicts future reward (1 = perfectly). Dips = it lost track.",
    format: fixed(2),
    tickFormat: fixed(1),
    yMax: 1,
  },
];

// Reward terms of the selected run, one line per term (fixed order = fixed colors).
const REWARD_TERMS = ["forward", "upright", "energy", "smoothness", "fall"];

export interface DashboardOptions {
  /** Replay a checkpoint in the simulator. */
  onWatch(run: string, checkpoint: string): void;
}

export class Dashboard {
  private readonly onWatch: DashboardOptions["onWatch"];
  private readonly runList: HTMLElement;
  private readonly detail: HTMLElement;
  private readonly charts = new Map<string, LineChart>();
  private readonly termsChart: LineChart;

  private runs: RunSummary[] = [];
  /** Checked runs -> their color slot (1..8). A run keeps its color while checked. */
  private readonly colors = new Map<string, number>();
  private selected: string | null = null;
  private selectedDetail: RunDetail | null = null;
  private readonly curves = new Map<string, Map<string, ScalarSeries>>(); // run -> tag -> series
  private pollTimer: number | undefined;
  private evalTimer: number | undefined;
  private visible = false;

  constructor(root: HTMLElement, options: DashboardOptions) {
    this.onWatch = options.onWatch;
    root.innerHTML = `
      <aside class="runs">
        <h2>Runs</h2>
        <p class="hint">Check runs to compare them on the charts. Click a name for details.</p>
        <div class="run-list"></div>
      </aside>
      <main class="run-detail">
        <div class="detail-head"></div>
        <div class="tiles"></div>
        <div class="charts"></div>
        <section class="checkpoints"></section>
        <details class="settings"><summary>Training settings</summary><div class="settings-body"></div></details>
      </main>`;
    this.runList = root.querySelector(".run-list")!;
    this.detail = root.querySelector(".run-detail")!;
    const grid = root.querySelector<HTMLElement>(".charts")!;
    for (const curve of CURVES) this.charts.set(curve.tag, new LineChart(grid, curve));
    this.termsChart = new LineChart(grid, {
      title: "Reward terms",
      subtitle: "What the selected run is rewarded (+) or penalized (−) for, average per step.",
      format: fixed(3),
      tickFormat: fixed(2),
    });
  }

  show(): void {
    this.visible = true;
    void this.refresh();
    this.pollTimer = window.setInterval(() => void this.refresh(), POLL_MS);
  }

  hide(): void {
    this.visible = false;
    window.clearInterval(this.pollTimer);
    window.clearTimeout(this.evalTimer);
  }

  // ------------------------------------------------------------------ data

  private async refresh(): Promise<void> {
    try {
      this.runs = await api.runs();
    } catch (e) {
      this.runList.textContent = `Can't reach the server: ${(e as Error).message}`;
      return;
    }
    const names = new Set(this.runs.map((r) => r.name));
    for (const name of [...this.colors.keys()]) if (!names.has(name)) this.colors.delete(name);
    if (this.selected === null || !names.has(this.selected)) {
      this.selected = this.runs[0]?.name ?? null; // newest run first
      if (this.selected) this.check(this.selected);
    }
    this.renderRunList();
    // Curves: fetch runs we don't have yet, and refresh running ones.
    const stale = this.runs.filter(
      (r) => this.colors.has(r.name) && (!this.curves.has(r.name) || r.status === "running"),
    );
    await Promise.all([...stale.map((r) => this.loadCurves(r.name)), this.loadDetail()]);
    this.renderCharts();
  }

  private async loadCurves(run: string): Promise<void> {
    const tags = [...CURVES.map((c) => c.tag), ...REWARD_TERMS.map((t) => `reward/${t}`)];
    for (const chart of this.charts.values()) chart.setLoading(true);
    try {
      const response = await api.scalars(run, tags);
      this.curves.set(run, new Map(response.series.map((s) => [s.tag, s])));
    } finally {
      for (const chart of this.charts.values()) chart.setLoading(false);
    }
  }

  private async loadDetail(): Promise<void> {
    if (this.selected === null) {
      this.selectedDetail = null;
    } else {
      this.selectedDetail = await api.run(this.selected);
      if (this.selectedDetail.evaluating > 0 && this.visible) {
        window.clearTimeout(this.evalTimer);
        this.evalTimer = window.setTimeout(() => void this.loadDetail(), EVAL_POLL_MS);
      }
    }
    this.renderDetail();
  }

  private check(run: string): void {
    if (this.colors.has(run)) return;
    const used = new Set(this.colors.values());
    const slot = Array.from({ length: SERIES_SLOTS }, (_, i) => i + 1).find((s) => !used.has(s));
    if (slot !== undefined) this.colors.set(run, slot); // max 8 runs overlaid
  }

  private async toggle(run: string, on: boolean): Promise<void> {
    if (on) this.check(run);
    else this.colors.delete(run);
    this.renderRunList();
    if (on && !this.curves.has(run)) await this.loadCurves(run);
    this.renderCharts();
  }

  private async select(run: string): Promise<void> {
    this.selected = run;
    await this.toggle(run, true);
    await this.loadDetail();
  }

  // ---------------------------------------------------------------- render

  private renderRunList(): void {
    if (this.runs.length === 0) {
      this.runList.innerHTML = `<p class="hint">No runs yet. Start one with <code>uv run scripts/train.py</code>.</p>`;
      return;
    }
    this.runList.replaceChildren(
      ...this.runs.map((run) => {
        const row = document.createElement("div");
        row.className = "run-row" + (run.name === this.selected ? " selected" : "");
        const box = document.createElement("input");
        box.type = "checkbox";
        box.checked = this.colors.has(run.name);
        box.title = "Show on the charts";
        box.disabled = !box.checked && this.colors.size >= SERIES_SLOTS;
        box.addEventListener("change", () => void this.toggle(run.name, box.checked));
        const swatch = document.createElement("i");
        swatch.className = "swatch";
        const slot = this.colors.get(run.name);
        if (slot) swatch.style.background = `var(--series-${slot})`;
        const name = document.createElement("button");
        name.className = "run-name";
        name.textContent = run.name;
        name.addEventListener("click", () => void this.select(run.name));
        const meta = document.createElement("div");
        meta.className = "run-meta";
        meta.append(statusPill(run), document.createTextNode(` ${formatSteps(run.steps_done)} steps`));
        const text = document.createElement("div");
        text.append(name, meta);
        row.append(box, swatch, text);
        return row;
      }),
    );
  }

  private renderCharts(): void {
    const checked = this.runs.filter((r) => this.colors.has(r.name));
    for (const curve of CURVES) {
      const series: ChartSeries[] = [];
      for (const run of checked) {
        const s = this.curves.get(run.name)?.get(curve.tag);
        if (s) series.push(toSeries(run.name, run.name, this.colors.get(run.name)!, s));
      }
      this.charts.get(curve.tag)!.setSeries(series);
    }
    const terms = this.selected ? this.curves.get(this.selected) : undefined;
    this.termsChart.setSeries(
      REWARD_TERMS.flatMap((term, i) => {
        const s = terms?.get(`reward/${term}`);
        return s ? [toSeries(term, term, i + 1, s)] : [];
      }),
    );
    this.termsChart.element.querySelector("h3")!.textContent = this.selected
      ? `Reward terms: ${this.selected}`
      : "Reward terms";
  }

  private renderDetail(): void {
    const head = this.detail.querySelector<HTMLElement>(".detail-head")!;
    const tiles = this.detail.querySelector<HTMLElement>(".tiles")!;
    const table = this.detail.querySelector<HTMLElement>(".checkpoints")!;
    const settings = this.detail.querySelector<HTMLElement>(".settings-body")!;
    const d = this.selectedDetail;
    if (d === null) {
      head.replaceChildren();
      tiles.replaceChildren();
      table.replaceChildren();
      settings.replaceChildren();
      return;
    }
    const s = d.summary;
    const title = document.createElement("h2");
    title.textContent = s.name;
    const meta = document.createElement("p");
    meta.append(statusPill(s), document.createTextNode(` ${s.robot} · ${s.n_envs} parallel environments · started ${s.started.replace("T", " ")}`));
    head.replaceChildren(title, meta);

    const best = bestCheckpoint(d.checkpoints);
    const seconds = duration(s.started, s.finished);
    tiles.replaceChildren(
      tile("Steps", `${formatSteps(s.steps_done)}`, `of ${formatSteps(s.total_steps)} planned`),
      tile("Duration", seconds === null ? "–" : formatDuration(seconds), s.finished ? "" : "so far"),
      tile(
        "Throughput",
        seconds ? `${formatSteps(s.steps_done / seconds)}/s` : "–",
        "environment steps per second",
      ),
      best
        ? tile("Best checkpoint", `${best.evaluation!.speed.toFixed(2)} m/s`, `at ${formatSteps(best.steps)} steps, no noise`)
        : tile("Best checkpoint", "–", "evaluate checkpoints to find it"),
    );

    table.replaceChildren(this.checkpointTable(d, best));

    settings.replaceChildren(
      ...d.settings.map((setting) => {
        const row = document.createElement("div");
        const key = document.createElement("span");
        key.textContent = `${setting.group}.${setting.key}`;
        const value = document.createElement("code");
        value.textContent = setting.value;
        row.append(key, value);
        return row;
      }),
    );
  }

  private checkpointTable(d: RunDetail, best: CheckpointInfo | null): HTMLElement {
    const wrap = document.createElement("div");
    const head = document.createElement("div");
    head.className = "section-head";
    const h = document.createElement("h3");
    h.textContent = "Checkpoints";
    const missing = d.checkpoints.filter((c) => c.evaluation === null).length;
    const button = document.createElement("button");
    button.textContent =
      d.evaluating > 0
        ? `Evaluating… ${d.evaluating} left`
        : missing === 0
          ? "All evaluated"
          : `Evaluate ${missing} checkpoint${missing === 1 ? "" : "s"}`;
    button.disabled = d.evaluating > 0 || missing === 0;
    button.title = "Run each checkpoint for 5 episodes without exploration noise (a few seconds each)";
    button.addEventListener("click", async () => {
      button.disabled = true;
      await api.evaluate(d.summary.name);
      await this.loadDetail();
    });
    const note = document.createElement("p");
    note.className = "hint";
    note.textContent = "Measured headless without exploration noise, 5 episodes each. The newest checkpoint isn't always the best.";
    head.append(h, button);
    wrap.append(head, note);

    const table = document.createElement("table");
    table.className = "data-table";
    const header = table.createTHead().insertRow();
    for (const label of ["Checkpoint", "Speed", "Distance", "Falls", "Mean return", ""]) {
      const th = document.createElement("th");
      th.textContent = label;
      header.append(th);
    }
    const body = table.createTBody();
    for (const c of [...d.checkpoints].reverse()) {
      const row = body.insertRow();
      if (c === best) row.className = "best";
      const name = row.insertCell();
      name.textContent = `${preciseSteps(c.steps)} steps`;
      name.title = `${c.steps.toLocaleString()} steps (${c.name})`;
      if (c === best) {
        const pill = document.createElement("span");
        pill.className = "pill best-pill";
        pill.textContent = "best";
        name.append(" ", pill);
      }
      const e = c.evaluation;
      row.insertCell().textContent = e ? `${e.speed.toFixed(2)} m/s` : "–";
      row.insertCell().textContent = e ? `${e.distance.toFixed(1)} m` : "–";
      row.insertCell().textContent = e ? `${e.falls}/${e.episodes}` : "–";
      row.insertCell().textContent = e ? e.mean_return.toFixed(0) : "–";
      const watch = document.createElement("button");
      watch.textContent = "Watch";
      watch.title = "Replay this checkpoint in the simulator";
      watch.addEventListener("click", () => this.onWatch(d.summary.name, c.name));
      row.insertCell().append(watch);
    }
    wrap.append(table);
    return wrap;
  }
}

// ---------------------------------------------------------------- helpers

/** Table precision: "10.01M" vs "10.00M" (formatSteps would show both as "10M"). */
function preciseSteps(steps: number): string {
  return steps >= 1e6 ? `${(steps / 1e6).toFixed(2)}M` : formatSteps(steps);
}

function toSeries(key: string, label: string, slot: number, s: ScalarSeries): ChartSeries {
  return { key, label, color: `var(--series-${slot})`, steps: s.steps, values: s.values };
}

function bestCheckpoint(checkpoints: CheckpointInfo[]): CheckpointInfo | null {
  let best: CheckpointInfo | null = null;
  for (const c of checkpoints) {
    if (c.evaluation && (!best || c.evaluation.mean_return > best.evaluation!.mean_return)) best = c;
  }
  return best;
}

function statusPill(run: RunSummary): HTMLElement {
  const pill = document.createElement("span");
  pill.className = `pill status-${run.status}`;
  pill.textContent = run.status;
  return pill;
}

function tile(label: string, value: string, note: string): HTMLElement {
  const el = document.createElement("div");
  el.className = "tile";
  const l = document.createElement("div");
  l.className = "tile-label";
  l.textContent = label;
  const v = document.createElement("div");
  v.className = "tile-value";
  v.textContent = value;
  const n = document.createElement("div");
  n.className = "tile-note";
  n.textContent = note;
  el.append(l, v, n);
  return el;
}

function duration(started: string, finished: string): number | null {
  const start = Date.parse(started);
  const end = finished ? Date.parse(finished) : Date.now();
  return Number.isNaN(start) ? null : Math.max(0, (end - start) / 1000);
}

function formatDuration(seconds: number): string {
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = Math.floor(seconds % 60);
  return h > 0 ? `${h}h ${m}m` : `${m}m ${s}s`;
}
