import "./style.css";
import { Connection, defaultSocketUrl } from "./connection";
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
};

const viewer = new Viewer(ui.viewport);
let paused = false;
let framesThisSecond = 0;

const connection = new Connection(defaultSocketUrl(), {
  onMessage: handleMessage,
  onConnectedChange(connected) {
    ui.dot.classList.toggle("connected", connected);
    ui.playPause.disabled = !connected;
    ui.reset.disabled = !connected;
    if (!connected) ui.robotName.textContent = "disconnected, retrying…";
  },
});

function handleMessage(message: ServerMessage): void {
  // `message.type` tells TypeScript which message this is, so inside each
  // case `message` has exactly that message's fields.
  switch (message.type) {
    case "scene":
      viewer.loadScene(message);
      ui.robotName.textContent = message.robot;
      break;
    case "frame":
      viewer.applyFrame(message);
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

ui.playPause.addEventListener("click", () => connection.send({ type: paused ? "play" : "pause" }));
ui.reset.addEventListener("click", () => connection.send({ type: "reset" }));

setInterval(() => {
  ui.streamFps.textContent = `${framesThisSecond} fps`;
  framesThisSecond = 0;
}, 1000);
