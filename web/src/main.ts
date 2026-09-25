import "./style.css";
import { Connection, defaultSocketUrl } from "./connection";
import { MotorPanel } from "./motors";
import type { ServerMessage } from "./protocol";
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
  simTime: element<HTMLSpanElement>("sim-time"),
  streamFps: element<HTMLSpanElement>("stream-fps"),
  motors: element<HTMLDivElement>("motors"),
  help: element<HTMLDivElement>("help"),
};

const viewer = new Viewer(ui.viewport);
const motors = new MotorPanel(ui.motors, (ctrl, duration) =>
  connection.send({ type: "set_ctrl", ctrl, duration }),
);
let paused = false;
let framesThisSecond = 0;

const connection = new Connection(defaultSocketUrl(), {
  onMessage: handleMessage,
  onConnectedChange(connected) {
    ui.dot.classList.toggle("connected", connected);
    ui.playPause.disabled = !connected;
    ui.reset.disabled = !connected;
    motors.setEnabled(connected);
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
      updateHelp(message.keyframes.length);
      break;
    case "frame":
      viewer.applyFrame(message);
      motors.update(message);
      ui.simTime.textContent = `t = ${message.time.toFixed(2)} s`;
      framesThisSecond++;
      break;
    case "status":
      paused = message.paused;
      ui.playPause.textContent = paused ? "Play" : "Pause";
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

function togglePlay(): void {
  connection.send({ type: paused ? "play" : "pause" });
}

function reset(): void {
  connection.send({ type: "reset" });
}

// blur(): a focused button would also react to Space, toggling twice.
ui.playPause.addEventListener("click", () => {
  togglePlay();
  ui.playPause.blur();
});
ui.reset.addEventListener("click", () => {
  reset();
  ui.reset.blur();
});

// Keyboard shortcuts. Modifier combos (Ctrl+R etc.) stay with the browser.
window.addEventListener("keydown", (event) => {
  if (event.ctrlKey || event.metaKey || event.altKey) return;
  if (event.key === " ") {
    event.preventDefault(); // don't scroll or press a focused button
    if (!event.repeat) togglePlay();
  } else if (event.key === "r" || event.key === "R") {
    reset();
  } else if (/^[1-9]$/.test(event.key)) {
    motors.applyPreset(Number(event.key) - 1);
  }
});

function updateHelp(presets: number): void {
  const poses = presets > 1 ? `1–${Math.min(presets, 9)}` : "1";
  ui.help.innerHTML =
    `<kbd>Space</kbd> play/pause · <kbd>R</kbd> reset · <kbd>${poses}</kbd> poses · ` +
    "Mouse: left rotate, right pan, wheel zoom";
}

setInterval(() => {
  ui.streamFps.textContent = `${framesThisSecond} fps`;
  framesThisSecond = 0;
}, 1000);
