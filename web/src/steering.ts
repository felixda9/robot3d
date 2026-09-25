// Steering the walk policy (milestone 7) with the keyboard or a gamepad.
//   keyboard: W/S or Up/Down = forward/back, A/D or Left/Right = turn,
//             Q/E = sideways left/right (held keys; release = stop)
//   gamepad:  left stick = forward/back + sideways, right stick = turn
// Sends SetCommandCommand whenever the command changes (at most ~20 per
// second), scaled to the walker's command_limits. The server clamps too.
import type { ClientMessage, StatusMessage } from "./protocol";

const SEND_INTERVAL_MS = 50;
const DEADZONE = 0.15;
const KEYS: Record<string, [number, number, number]> = {
  // [forward, sideways, turn] direction per key
  w: [1, 0, 0],
  arrowup: [1, 0, 0],
  s: [-1, 0, 0],
  arrowdown: [-1, 0, 0],
  a: [0, 0, 1],
  arrowleft: [0, 0, 1],
  d: [0, 0, -1],
  arrowright: [0, 0, -1],
  q: [0, 1, 0],
  e: [0, -1, 0],
};

export class Steering {
  private readonly held = new Set<string>();
  private readonly send: (message: ClientMessage) => void;
  private status: StatusMessage | null = null;
  private last = "0,0,0";
  private lastSent = 0;
  private enabled = () => false;
  /** Called with the current command, for a display. */
  onCommand: (forward: number, sideways: number, turn: number) => void = () => {};

  constructor(send: (message: ClientMessage) => void, enabled: () => boolean) {
    this.send = send;
    this.enabled = enabled;
    window.addEventListener("keydown", (e) => {
      const key = e.key.toLowerCase();
      if (!(key in KEYS) || e.ctrlKey || e.metaKey || e.altKey || !this.active()) return;
      e.preventDefault(); // arrows would scroll
      this.held.add(key);
    });
    window.addEventListener("keyup", (e) => this.held.delete(e.key.toLowerCase()));
    window.addEventListener("blur", () => this.held.clear());
    const tick = () => {
      this.update();
      requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  }

  setStatus(status: StatusMessage): void {
    this.status = status;
  }

  /** Steering applies in Walk mode, with a steerable walk policy driving. */
  active(): boolean {
    const s = this.status;
    return s !== null && s.steerable && s.policy_active && s.mode === "walk" && this.enabled();
  }

  private update(): void {
    if (!this.active()) {
      this.held.clear();
      return;
    }
    let [forward, sideways, turn] = [0, 0, 0];
    for (const key of this.held) {
      const [f, s, t] = KEYS[key];
      forward += f;
      sideways += s;
      turn += t;
    }
    for (const pad of navigator.getGamepads?.() ?? []) {
      if (!pad) continue;
      const axis = (i: number) => (Math.abs(pad.axes[i] ?? 0) > DEADZONE ? pad.axes[i] : 0);
      forward += -axis(1); // stick up = forward
      sideways += -axis(0); // stick right = to the robot's right (-y)
      turn += -axis(2); // right stick right = turn right (negative yaw rate)
    }
    const clamp = (v: number) => Math.max(-1, Math.min(1, v));
    const [maxForward, maxBackward, maxSideways, maxTurn] = this.status!.command_limits;
    const command: [number, number, number] = [
      clamp(forward) * (forward >= 0 ? maxForward : maxBackward),
      clamp(sideways) * maxSideways,
      clamp(turn) * maxTurn,
    ];
    const key = command.map((v) => v.toFixed(2)).join(",");
    const now = performance.now();
    if (key !== this.last && now - this.lastSent > SEND_INTERVAL_MS) {
      this.last = key;
      this.lastSent = now;
      this.send({ type: "set_command", forward: command[0], sideways: command[1], turn: command[2] });
      this.onCommand(...command);
    }
  }
}
