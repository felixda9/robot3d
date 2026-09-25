import "./style.css";
import { Connection, defaultSocketUrl } from "./connection";
import { MotorPanel } from "./motors";
import type { ServerMessage, StatusMessage } from "./protocol";
import { Viewer } from "./viewer";

function element<T extends HTMLElement>(id: string): T {
  const el = document.getElementById(id);
  if (el === null) throw new Error(`missing #${id} in index.html`);
  return el as T;
}

const ui = {
  viewport: element<HTMLDivElement>("viewport"),
  dot: element<HTMLSpanElement>("connection-dot"),
  robotName: element<HTMLSpanElement>("robot-name"),
  playPause: element<HTMLButtonElement>("play-pause"),
  reset: element<HTMLButtonElement>("reset"),
  follow: element<HTMLInputElement>("follow"),
  policyBox: element<HTMLDivElement>("policy-box"),
  policyName: element<HTMLDivElement>("policy-name"),
  policyToggle: element<HTMLButtonElement>("policy-toggle"),
  simTime: element<HTMLSpanElement>("sim-time"),
  streamFps: element<HTMLSpanElement>("stream-fps"),
  motors: element<HTMLDivElement>("motors"),
  help: element<HTMLDivElement>("help"),
};

const viewer = new Viewer(ui.viewport);
const motors = new MotorPanel(ui.motors, (ctrl, duration) =>
  connection.send({ type: "set_ctrl", ctrl, duration }),
);
let status: StatusMessage = { type: "status", paused: false, policy: "", policy_active: false };
let presetCount = 0;
let framesThisSecond = 0;
viewer.setFollow(ui.follow.checked);

const connection = new Connection(defaultSocketUrl(), {
  onMessage: handleMessage,
  onConnectedChange(connected) {
    ui.dot.classList.toggle("connected", connected);
    ui.playPause.disabled = !connected;
    ui.reset.disabled = !connected;
    ui.policyToggle.disabled = !connected;
    motors.setConnected(connected);
    if (!connected) ui.robotName.textContent = "disconnected, retrying…";
  },
});

function handleMessage(message: ServerMessage): void {
  // `message.type` tells TypeScript which message this is, so inside each
  // case `message` has exactly that message's fields.
  switch (message.type) {
    case "scene":
      viewer.loadScene(message);
      motors.load(message);
      ui.robotName.textContent = message.robot;
      presetCount = message.keyframes.length;
      updateHelp();
      break;
    case "frame":
      viewer.applyFrame(message);
      motors.update(message);
      ui.simTime.textContent = `t = ${message.time.toFixed(2)} s`;
      framesThisSecond++;
      break;
    case "status":
      applyStatus(message);
      break;
    case "error":
      console.error("server:", message.message);
      break;
    default: {
      // Exhaustiveness check: if protocol.ts gains a new message type, this
      // line stops compiling until it is handled above.
      const unhandled: never = message;
      console.warn("unknown message", unhandled);
    }
  }
}

function applyStatus(next: StatusMessage): void {
  status = next;
  ui.playPause.textContent = status.paused ? "Play" : "Pause";
  ui.policyBox.hidden = status.policy === "";
  ui.policyBox.classList.toggle("active", status.policy_active);
  ui.policyName.textContent = status.policy;
  ui.policyToggle.textContent = status.policy_active ? "Take manual control" : "Let the policy drive";
  motors.setLocked(status.policy_active);
  updateHelp();
}

function togglePlay(): void {
  connection.send({ type: status.paused ? "play" : "pause" });
}

function reset(): void {
  connection.send({ type: "reset" });
}

function togglePolicy(): void {
  if (status.policy !== "") connection.send({ type: "use_policy", active: !status.policy_active });
}

function setFollow(on: boolean): void {
  ui.follow.checked = on;
  viewer.setFollow(on);
}

// blur(): a focused button would also react to Space, toggling twice.
for (const [button, action] of [
  [ui.playPause, togglePlay],
  [ui.reset, reset],
  [ui.policyToggle, togglePolicy],
] as const) {
  button.addEventListener("click", () => {
    action();
    button.blur();
  });
}
ui.follow.addEventListener("change", () => {
  setFollow(ui.follow.checked);
  ui.follow.blur();
});

// Keyboard shortcuts. Modifier combos (Ctrl+R etc.) stay with the browser.
window.addEventListener("keydown", (event) => {
  if (event.ctrlKey || event.metaKey || event.altKey) return;
  const key = event.key.toLowerCase();
  if (key === " ") {
    event.preventDefault(); // don't scroll or press a focused button
    if (!event.repeat) togglePlay();
  } else if (key === "r") {
    reset();
  } else if (key === "f") {
    setFollow(!ui.follow.checked);
  } else if (key === "p") {
    togglePolicy();
  } else if (/^[1-9]$/.test(key)) {
    motors.applyPreset(Number(key) - 1);
  }
});

function updateHelp(): void {
  const parts = ["<kbd>Space</kbd> play/pause", "<kbd>R</kbd> reset", "<kbd>F</kbd> follow"];
  if (status.policy !== "") parts.push("<kbd>P</kbd> policy/manual");
  if (!status.policy_active && presetCount > 0) {
    parts.push(`<kbd>${presetCount > 1 ? `1–${Math.min(presetCount, 9)}` : "1"}</kbd> poses`);
  }
  parts.push("Mouse: left rotate, right pan, wheel zoom");
  ui.help.innerHTML = parts.join(" · ");
}

setInterval(() => {
  ui.streamFps.textContent = `${framesThisSecond} fps`;
  framesThisSecond = 0;
}, 1000);
