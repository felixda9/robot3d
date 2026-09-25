import type { ActuatorInfo, FrameMessage, KeyframeInfo, SceneMessage } from "./protocol";

const DEG_PER_RAD = 180 / Math.PI;
/** Slider resolution: half a degree, in radians (the protocol's unit). */
const SLIDER_STEP = 0.5 / DEG_PER_RAD;
/**
 * After the user moves a slider, ignore the server's value for that slider
 * this long. Otherwise a frame sent before our command arrived would make the
 * thumb jump back for a moment.
 */
const LOCAL_EDIT_HOLD_MS = 300;
/** A motor at >= 95% of its torque limit is "saturated": it can't push harder. */
const SATURATED = 0.95;
/**
 * Seconds for a pose preset: all motor targets glide there together (see
 * SetCtrlCommand.duration). Jumping instantly can flip the robot (e.g.
 * crouch -> tall); tests/test_quadruped.py checks every transition at this value.
 */
const PRESET_DURATION = 0.8;

/** Send motor targets; duration 0 = at once (see SetCtrlCommand). */
export type SetCtrl = (ctrl: Record<string, number>, duration: number) => void;

interface MotorRow {
  actuator: ActuatorInfo;
  slider: HTMLInputElement;
  /** Marker on the slider track at the joint's ACTUAL angle. */
  marker: HTMLElement;
  target: HTMLElement;
  actual: HTMLElement;
  torqueFill: HTMLElement;
  torqueLimit: number;
  dragging: boolean;
  lastLocalEdit: number;
}

/**
 * One slider per motor. Our motors are position-controlled: a slider sets the
 * TARGET angle, the motor's PD controller then pulls the joint toward it,
 * limited by its torque. The white marker shows where the joint actually is,
 * so you can watch it follow, or fall short under load.
 *
 * The UI shows degrees; everything sent to the server is radians.
 */
export class MotorPanel {
  private readonly list: HTMLElement;
  private readonly presetBar: HTMLElement;
  private readonly setCtrl: SetCtrl;
  private rows: MotorRow[] = [];
  private keyframes: KeyframeInfo[] = [];
  private presetButtons: HTMLButtonElement[] = [];

  constructor(root: HTMLElement, setCtrl: SetCtrl) {
    this.setCtrl = setCtrl;
    root.innerHTML = `
      <div class="panel-header">Motors <span class="hint">target angle</span></div>
      <div class="presets"></div>
      <div class="motor-list"></div>
      <div class="legend">
        <span><i class="legend-thumb"></i>target</span>
        <span><i class="legend-marker"></i>actual</span>
        <span><i class="legend-torque"></i>torque</span>
      </div>`;
    this.presetBar = root.querySelector(".presets")!;
    this.list = root.querySelector(".motor-list")!;
    window.addEventListener("pointerup", () => this.endDrags());
    window.addEventListener("pointercancel", () => this.endDrags());
  }

  get presetCount(): number {
    return this.keyframes.length;
  }

  /** Build the sliders and pose buttons for a new scene. */
  load(scene: SceneMessage): void {
    this.rows = []; // createRow() adds the new ones
    this.list.replaceChildren(...scene.actuators.map((a) => this.createRow(a)));
    this.keyframes = scene.keyframes;
    this.presetButtons = scene.keyframes.map((keyframe, i) => {
      const button = document.createElement("button");
      button.innerHTML = i < 9 ? `<kbd>${i + 1}</kbd> ${keyframe.name}` : keyframe.name;
      button.title = `Pose preset "${keyframe.name}" (sets all motor targets)`;
      button.addEventListener("click", () => {
        this.applyPreset(i);
        button.blur(); // so Space doesn't "click" it again (Space = play/pause)
      });
      return button;
    });
    this.presetBar.replaceChildren(...this.presetButtons);
  }

  /** Send all motor targets of pose preset `index`; false if there is none. */
  applyPreset(index: number): boolean {
    const keyframe = this.keyframes[index];
    if (keyframe === undefined) return false;
    const ctrl: Record<string, number> = {};
    this.rows.forEach((row, i) => (ctrl[row.actuator.name] = keyframe.ctrl[i]));
    this.setCtrl(ctrl, PRESET_DURATION);
    return true;
  }

  setEnabled(enabled: boolean): void {
    for (const row of this.rows) row.slider.disabled = !enabled;
    for (const button of this.presetButtons) button.disabled = !enabled;
  }

  /** Show a frame's targets, actual angles, and torques. */
  update(frame: FrameMessage): void {
    const now = performance.now();
    this.rows.forEach((row, i) => {
      const [min, max] = row.actuator.ctrl_range;
      const userIsEditing = row.dragging || now - row.lastLocalEdit < LOCAL_EDIT_HOLD_MS;
      if (!userIsEditing) {
        row.slider.value = String(frame.ctrl[i]);
        setText(row.target, degrees(frame.ctrl[i]));
      }
      const angle = frame.joint_pos[i];
      setText(row.actual, degrees(angle));
      row.marker.style.setProperty("--frac", String(clamp01((angle - min) / (max - min))));

      // Torque bar grows from the center: right = positive, left = negative.
      const torque = frame.torque[i];
      const frac = Math.min(Math.abs(torque) / row.torqueLimit, 1);
      row.torqueFill.style.width = `${50 * frac}%`;
      row.torqueFill.style.left = torque >= 0 ? "50%" : `${50 - 50 * frac}%`;
      row.torqueFill.classList.toggle("saturated", frac >= SATURATED);
      row.torqueFill.parentElement!.title = `torque ${torque.toFixed(2)} N·m`;
    });

    // Highlight the preset whose targets match the current ones, if any.
    this.keyframes.forEach((keyframe, k) => {
      const active = keyframe.ctrl.every((c, i) => Math.abs(c - frame.ctrl[i]) < 1e-3);
      this.presetButtons[k].classList.toggle("active", active);
    });
  }

  private createRow(actuator: ActuatorInfo): HTMLElement {
    const [min, max] = actuator.ctrl_range;
    const [forceMin, forceMax] = actuator.force_range;
    const el = document.createElement("div");
    el.className = "motor";
    el.innerHTML = `
      <span class="motor-name"></span>
      <div class="motor-slider">
        <input type="range" />
        <div class="motor-actual"></div>
      </div>
      <span class="motor-values"><span class="target"></span><span class="actual"></span></span>
      <div class="motor-torque"><div class="fill"></div></div>`;
    el.querySelector(".motor-name")!.textContent = actuator.name.replace(/_/g, " ");
    el.title = `${actuator.name}: ${degrees(min)} to ${degrees(max)}`;

    const slider = el.querySelector("input")!;
    slider.min = String(min);
    slider.max = String(max);
    slider.step = String(SLIDER_STEP);

    const row: MotorRow = {
      actuator,
      slider,
      marker: el.querySelector(".motor-actual")!,
      target: el.querySelector(".target")!,
      actual: el.querySelector(".actual")!,
      torqueFill: el.querySelector(".motor-torque .fill")!,
      // Unlimited motors ([0, 0]) get a nominal 10 N·m scale for the bar.
      torqueLimit: Math.max(Math.abs(forceMin), Math.abs(forceMax)) || 10,
      dragging: false,
      lastLocalEdit: 0,
    };
    slider.addEventListener("pointerdown", () => (row.dragging = true));
    slider.addEventListener("input", () => {
      row.lastLocalEdit = performance.now();
      const value = Number(slider.value);
      setText(row.target, degrees(value));
      // At once: the user sets the speed by how fast they drag. (Clicking far
      // along the track jumps, which can knock the robot over. That's physics.)
      this.setCtrl({ [actuator.name]: value }, 0);
    });
    this.rows.push(row);
    return el;
  }

  private endDrags(): void {
    const now = performance.now();
    for (const row of this.rows) {
      if (row.dragging) {
        row.dragging = false;
        row.lastLocalEdit = now;
      }
    }
  }
}

function degrees(radians: number): string {
  return `${Math.round(radians * DEG_PER_RAD)}°`;
}

function clamp01(x: number): number {
  return Math.min(Math.max(x, 0), 1);
}

/** Only touch the DOM when the text actually changes (this runs 60x/s). */
function setText(el: HTMLElement, text: string): void {
  if (el.textContent !== text) el.textContent = text;
}
