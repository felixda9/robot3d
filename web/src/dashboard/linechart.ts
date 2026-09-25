// A small SVG line chart for training curves: several series (runs or reward
// terms) over training steps on one y axis, with a snapping crosshair
// tooltip, a legend for 2+ series, keyboard access, and a table view.
// Specs follow the dataviz skill: 2px lines, hairline solid grid, 8px dots
// with a 2px surface ring, text in text colors (never the series color).

const SVG_NS = "http://www.w3.org/2000/svg";
const PLOT_HEIGHT = 170;
const TOP = 10;
const X_AXIS = 24;
const LEFT = 58;
const RIGHT = 16;
const HEIGHT = TOP + PLOT_HEIGHT + X_AXIS;

export interface ChartSeries {
  key: string;
  label: string;
  /** CSS color, e.g. "var(--series-1)". */
  color: string;
  steps: number[];
  values: number[];
}

export interface ChartOptions {
  title: string;
  subtitle: string;
  /** Value format for the tooltip and table. */
  format: (value: number) => string;
  /** Y tick format (default: format). */
  tickFormat?: (value: number) => string;
  /** Log y axis, for quantities spanning decades (e.g. KL divergence). */
  log?: boolean;
  /** Fixed y range ends (e.g. 0..1 for a fraction); otherwise from the data. */
  yMin?: number;
  yMax?: number;
}

export class LineChart {
  readonly element: HTMLElement;
  private readonly opts: ChartOptions;
  private readonly legend: HTMLElement;
  private readonly plot: HTMLElement;
  private readonly svg: SVGSVGElement;
  private readonly tooltip: HTMLElement;
  private readonly details: HTMLDetailsElement;
  private series: ChartSeries[] = [];
  private allSteps: number[] = [];
  private width = 520;
  private x = (_step: number) => 0;
  private y = (_value: number) => 0;
  private hoverIndex: number | null = null;

  constructor(parent: HTMLElement, opts: ChartOptions) {
    this.opts = opts;
    this.element = document.createElement("section");
    this.element.className = "chart";
    this.element.innerHTML = `
      <header><h3></h3><p></p></header>
      <div class="chart-legend"></div>
      <div class="chart-plot" tabindex="0"><div class="chart-tooltip" hidden></div></div>
      <details class="chart-table"><summary>Show data</summary><div class="table-wrap"></div></details>`;
    this.element.querySelector("h3")!.textContent = opts.title;
    this.element.querySelector("header p")!.textContent = opts.subtitle;
    this.legend = this.element.querySelector(".chart-legend")!;
    this.plot = this.element.querySelector(".chart-plot")!;
    this.tooltip = this.element.querySelector(".chart-tooltip")!;
    this.details = this.element.querySelector("details")!;
    this.plot.setAttribute("aria-label", `${opts.title} chart. Use arrow keys to read values.`);
    this.svg = document.createElementNS(SVG_NS, "svg");
    this.svg.setAttribute("height", String(HEIGHT));
    this.plot.prepend(this.svg);
    parent.append(this.element);

    this.plot.addEventListener("pointermove", (e) => this.hoverAt(e.clientX));
    this.plot.addEventListener("pointerleave", () => this.setHover(null));
    this.plot.addEventListener("focus", () => this.setHover(this.allSteps.length - 1));
    this.plot.addEventListener("blur", () => this.setHover(null));
    this.plot.addEventListener("keydown", (e) => this.onKey(e));
    this.details.addEventListener("toggle", () => this.renderTable());
    new ResizeObserver(() => {
      const width = Math.round(this.plot.clientWidth);
      if (width > 0 && width !== this.width) {
        this.width = width;
        this.render();
      }
    }).observe(this.plot);
  }

  setSeries(series: ChartSeries[]): void {
    this.series = series.filter((s) => s.steps.length > 0);
    this.allSteps = [...new Set(this.series.flatMap((s) => s.steps))].sort((a, b) => a - b);
    this.hoverIndex = null;
    this.render();
  }

  /** Refetch keeps the frame: dim the old render instead of blanking it. */
  setLoading(loading: boolean): void {
    this.element.classList.toggle("loading", loading);
  }

  // ------------------------------------------------------------------ render

  private render(): void {
    const { opts, width } = this;
    this.svg.setAttribute("width", String(width));
    this.svg.setAttribute("viewBox", `0 0 ${width} ${HEIGHT}`);
    this.svg.replaceChildren();
    this.renderLegend();
    this.renderTable();
    this.tooltip.hidden = true;

    const values = this.series.flatMap((s) => s.values).filter((v) => !opts.log || v > 0);
    if (values.length === 0) {
      this.svg.append(svgText(width / 2, TOP + PLOT_HEIGHT / 2, "No data yet", "chart-empty", "middle"));
      return;
    }

    // Scales. x: training steps from 0 to the last point (ticks only inside
    // it, so a 10M run doesn't get an empty stretch to 15M); y: data range
    // rounded out to nice ticks.
    const xMax = Math.max(...this.allSteps, 1);
    const xTicks = niceTicks(0, xMax, Math.max(2, Math.floor(width / 110))).filter((t) => t <= xMax);
    const plotW = width - LEFT - RIGHT;
    this.x = (s) => LEFT + (s / xMax) * plotW;

    let yTicks: number[];
    if (opts.log) {
      const lo = Math.floor(Math.log10(Math.min(...values)));
      const hi = Math.ceil(Math.log10(Math.max(...values)));
      yTicks = range(lo, Math.max(hi, lo + 1)).map((e) => 10 ** e);
      const [a, b] = [Math.log10(yTicks[0]), Math.log10(yTicks[yTicks.length - 1])];
      this.y = (v) => TOP + PLOT_HEIGHT - ((Math.log10(v) - a) / (b - a)) * PLOT_HEIGHT;
    } else {
      let lo = opts.yMin ?? Math.min(...values);
      let hi = opts.yMax ?? Math.max(...values);
      if (hi - lo < 1e-9) [lo, hi] = [lo - 1, hi + 1];
      yTicks = niceTicks(lo, hi, 4);
      const a = opts.yMin ?? yTicks[0];
      const b = opts.yMax ?? yTicks[yTicks.length - 1];
      yTicks = yTicks.filter((t) => t >= a - 1e-9 && t <= b + 1e-9);
      this.y = (v) => TOP + PLOT_HEIGHT - ((v - a) / (b - a)) * PLOT_HEIGHT;
    }

    // Grid + axes: solid hairlines, recessive; the baseline a step stronger.
    const tickFormat = opts.tickFormat ?? opts.format;
    for (const t of yTicks) {
      const y = snap(this.y(t));
      this.svg.append(svgLine(LEFT, y, width - RIGHT, y, "chart-grid"));
      this.svg.append(svgText(LEFT - 8, y + 4, tickFormat(t), "chart-tick", "end"));
    }
    const base = snap(TOP + PLOT_HEIGHT);
    this.svg.append(svgLine(LEFT, base, width - RIGHT, base, "chart-axis"));
    for (const t of xTicks) {
      this.svg.append(svgText(this.x(t), base + 17, formatSteps(t), "chart-tick", "middle"));
    }

    // Lines (2px, round joins). Log scale skips points <= 0.
    for (const s of this.series) {
      let d = "";
      let pen = "M";
      s.steps.forEach((step, i) => {
        const v = s.values[i];
        if (opts.log && v <= 0) {
          pen = "M";
          return;
        }
        d += `${pen}${this.x(step).toFixed(1)},${this.y(v).toFixed(1)}`;
        pen = "L";
      });
      const path = document.createElementNS(SVG_NS, "path");
      path.setAttribute("d", d);
      path.setAttribute("class", "chart-line");
      path.style.stroke = s.color;
      this.svg.append(path);
    }

    // Crosshair layer (drawn on hover).
    const cross = document.createElementNS(SVG_NS, "g");
    cross.setAttribute("class", "chart-cross");
    this.svg.append(cross);
    if (this.hoverIndex !== null) this.setHover(this.hoverIndex);
  }

  private renderLegend(): void {
    // A single series needs no legend: the title already says what it is.
    this.legend.hidden = this.series.length < 2;
    this.legend.replaceChildren(
      ...this.series.map((s) => {
        const item = document.createElement("span");
        const key = document.createElement("i");
        key.style.background = s.color;
        item.append(key, document.createTextNode(s.label));
        return item;
      }),
    );
  }

  private renderTable(): void {
    if (!this.details.open) return;
    const wrap = this.details.querySelector(".table-wrap")!;
    if (this.allSteps.length === 0) {
      wrap.textContent = "No data yet.";
      return;
    }
    const table = document.createElement("table");
    const head = table.createTHead().insertRow();
    for (const label of ["Step", ...this.series.map((s) => s.label)]) {
      const th = document.createElement("th");
      th.textContent = label;
      head.append(th);
    }
    const body = table.createTBody();
    const last = this.allSteps[this.allSteps.length - 1];
    for (let k = 0; k <= 10; k++) {
      const step = nearest(this.allSteps, (last * k) / 10);
      const row = body.insertRow();
      row.insertCell().textContent = formatSteps(step);
      for (const s of this.series) {
        const i = nearestIndex(s.steps, step);
        row.insertCell().textContent = this.near(s, i, step) ? this.opts.format(s.values[i]) : "–";
      }
    }
    wrap.replaceChildren(table);
  }

  // ------------------------------------------------------------------- hover

  private hoverAt(clientX: number): void {
    if (this.allSteps.length === 0) return;
    const px = clientX - this.svg.getBoundingClientRect().left;
    const xMax = this.allSteps[this.allSteps.length - 1];
    const plotW = this.width - LEFT - RIGHT;
    const step = ((px - LEFT) / plotW) * Math.max(xMax, 1);
    this.setHover(nearestIndex(this.allSteps, Math.max(0, step)));
  }

  private onKey(e: KeyboardEvent): void {
    if (this.allSteps.length === 0) return;
    const n = this.allSteps.length;
    const i = this.hoverIndex ?? n - 1;
    const jump = e.shiftKey ? 10 : 1;
    const moves: Record<string, number> = {
      ArrowLeft: i - jump,
      ArrowRight: i + jump,
      Home: 0,
      End: n - 1,
    };
    if (e.key in moves) {
      e.preventDefault();
      this.setHover(Math.min(n - 1, Math.max(0, moves[e.key])));
    } else if (e.key === "Escape") {
      this.setHover(null);
    }
  }

  /** Crosshair at allSteps[index]: hairline, a dot per series, one tooltip listing every series. */
  private setHover(index: number | null): void {
    this.hoverIndex = index;
    const cross = this.svg.querySelector(".chart-cross");
    if (!cross) return;
    cross.replaceChildren();
    if (index === null || this.allSteps.length === 0) {
      this.tooltip.hidden = true;
      return;
    }
    const step = this.allSteps[index];
    const x = snap(this.x(step));
    cross.append(svgLine(x, TOP, x, TOP + PLOT_HEIGHT, "chart-crosshair"));

    const rows: { series: ChartSeries; value: number }[] = [];
    for (const s of this.series) {
      const i = nearestIndex(s.steps, step);
      if (!this.near(s, i, step)) continue;
      const value = s.values[i];
      rows.push({ series: s, value });
      if (this.opts.log && value <= 0) continue;
      const dot = document.createElementNS(SVG_NS, "circle");
      dot.setAttribute("cx", this.x(s.steps[i]).toFixed(1));
      dot.setAttribute("cy", this.y(value).toFixed(1));
      dot.setAttribute("r", "4");
      dot.setAttribute("class", "chart-dot");
      dot.style.fill = s.color;
      cross.append(dot);
    }

    // Values lead, labels follow; line keys, not boxes. textContent only.
    const header = document.createElement("div");
    header.className = "tt-head";
    header.textContent = `${formatSteps(step)} steps`;
    const lines = rows.map(({ series, value }) => {
      const row = document.createElement("div");
      row.className = "tt-row";
      const key = document.createElement("i");
      key.style.background = series.color;
      const strong = document.createElement("strong");
      strong.textContent = this.opts.format(value);
      const label = document.createElement("span");
      label.textContent = series.label;
      row.append(key, strong, label);
      return row;
    });
    this.tooltip.replaceChildren(header, ...lines);
    this.tooltip.hidden = false;
    const flip = x > this.width * 0.6;
    this.tooltip.style.left = flip ? "" : `${x + 12}px`;
    this.tooltip.style.right = flip ? `${this.width - x + 12}px` : "";
  }

  /** Does series `s` have data close to `step` (not just its nearest point far away)? */
  private near(s: ChartSeries, i: number, step: number): boolean {
    const span = this.allSteps[this.allSteps.length - 1] || 1;
    return i >= 0 && Math.abs(s.steps[i] - step) <= span * 0.02;
  }
}

// -------------------------------------------------------------------- helpers

export function formatSteps(steps: number): string {
  if (steps >= 1e6) return `${trim(steps / 1e6)}M`;
  if (steps >= 1e3) return `${trim(steps / 1e3)}k`;
  return String(Math.round(steps));
}

function trim(x: number): string {
  return x >= 10 ? x.toFixed(0) : x.toFixed(x % 1 === 0 ? 0 : 2).replace(/0$/, "");
}

/** Round, human-friendly tick values (steps of 1, 2, 2.5 or 5 x 10^n) covering lo..hi. */
function niceTicks(lo: number, hi: number, count: number): number[] {
  const raw = (hi - lo) / count;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw) ?? 10 * mag;
  const start = Math.floor(lo / step) * step;
  const end = Math.ceil(hi / step) * step;
  const ticks = [];
  for (let t = start; t <= end + step / 2; t += step) ticks.push(Number(t.toPrecision(12)));
  return ticks;
}

function range(from: number, to: number): number[] {
  return Array.from({ length: to - from + 1 }, (_, i) => from + i);
}

function nearestIndex(sorted: number[], value: number): number {
  if (sorted.length === 0) return -1;
  let lo = 0;
  let hi = sorted.length - 1;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (sorted[mid] < value) lo = mid + 1;
    else hi = mid;
  }
  return lo > 0 && value - sorted[lo - 1] < sorted[lo] - value ? lo - 1 : lo;
}

function nearest(sorted: number[], value: number): number {
  return sorted[nearestIndex(sorted, value)];
}

/** Hairlines look crisp on whole pixels + 0.5. */
function snap(v: number): number {
  return Math.round(v) + 0.5;
}

function svgLine(x1: number, y1: number, x2: number, y2: number, cls: string): SVGLineElement {
  const line = document.createElementNS(SVG_NS, "line");
  for (const [k, v] of Object.entries({ x1, y1, x2, y2 })) line.setAttribute(k, String(v));
  line.setAttribute("class", cls);
  return line;
}

function svgText(x: number, y: number, text: string, cls: string, anchor: string): SVGTextElement {
  const el = document.createElementNS(SVG_NS, "text");
  el.setAttribute("x", String(x));
  el.setAttribute("y", String(y));
  el.setAttribute("class", cls);
  el.setAttribute("text-anchor", anchor);
  el.textContent = text;
  return el;
}
