// Mouse interaction with the robot, to test how well it copes:
//   drag a part      -> grab it and pull it along (GrabCommand, a spring)
//   double-click one -> push it away from the camera (PushCommand)
// Dragging anywhere else still rotates the camera. The physics happens on
// the server; this only says which part, which spot, and where to.
import * as THREE from "three";
import type { ClientMessage } from "./protocol";
import type { GrabView, Pick, Viewer } from "./viewer";

/** Pixels the mouse must move before a press on the robot becomes a grab (so clicks and double-clicks don't grab). */
const DRAG_THRESHOLD = 4;

interface Grab extends GrabView {
  geom: number;
  /** The mouse moves on this plane: through the grabbed spot, facing the camera. */
  plane: THREE.Plane;
}

export class RobotMouse {
  /** Pressed on a part but not moved yet. */
  private pressed: { pick: Pick; x: number; y: number } | null = null;
  private grab: Grab | null = null;
  private sendQueued = false;
  private readonly viewer: Viewer;
  private readonly send: (message: ClientMessage) => void;
  /** Newtons, from the panel's slider. */
  private readonly pushForce: () => number;

  constructor(viewer: Viewer, send: (message: ClientMessage) => void, pushForce: () => number) {
    this.viewer = viewer;
    this.send = send;
    this.pushForce = pushForce;
    const canvas = viewer.canvas;
    // Capture phase on the parent runs before OrbitControls' own listener on
    // the canvas, so a press on the robot never starts a camera rotation.
    canvas.parentElement!.addEventListener("pointerdown", (e) => this.onDown(e), { capture: true });
    canvas.addEventListener("pointermove", (e) => this.onMove(e));
    canvas.addEventListener("pointerup", () => this.letGo());
    canvas.addEventListener("pointercancel", () => this.letGo());
    canvas.addEventListener("dblclick", (e) => this.onDoubleClick(e));
    window.addEventListener("blur", () => this.letGo());
  }

  private onDown(event: PointerEvent): void {
    if (event.button !== 0) return; // left button only; right still pans
    const pick = this.viewer.pick(event.clientX, event.clientY);
    if (pick === null) return; // not on the robot: let the camera rotate
    this.viewer.setOrbitEnabled(false);
    this.viewer.canvas.setPointerCapture(event.pointerId); // keep getting moves outside the canvas
    this.pressed = { pick, x: event.clientX, y: event.clientY };
  }

  private onMove(event: PointerEvent): void {
    if (this.pressed !== null) {
      const { pick, x, y } = this.pressed;
      if (Math.hypot(event.clientX - x, event.clientY - y) < DRAG_THRESHOLD) return;
      this.pressed = null;
      pick.mesh.updateMatrixWorld();
      this.grab = {
        geom: pick.geom,
        mesh: pick.mesh,
        local: pick.mesh.worldToLocal(pick.point.clone()),
        target: pick.point.clone(),
        plane: new THREE.Plane().setFromNormalAndCoplanarPoint(this.viewer.cameraDirection(), pick.point),
      };
      this.viewer.showGrab(this.grab);
    }
    if (this.grab === null) return;
    const ray = this.viewer.ray(event.clientX, event.clientY);
    if (ray.intersectPlane(this.grab.plane, this.grab.target) === null) return; // looking along the plane
    this.queueGrabMessage();
  }

  /** At most one grab message per animation frame, however fast the mouse reports. */
  private queueGrabMessage(): void {
    if (this.sendQueued) return;
    this.sendQueued = true;
    requestAnimationFrame(() => {
      this.sendQueued = false;
      const grab = this.grab;
      if (grab === null) return;
      this.send({ type: "grab", geom: grab.geom, point: vec3(grab.local), target: vec3(grab.target) });
    });
  }

  private letGo(): void {
    if (this.grab !== null) {
      this.send({ type: "release" });
      this.viewer.showGrab(null);
      this.grab = null;
    }
    this.pressed = null;
    this.viewer.setOrbitEnabled(true);
  }

  private onDoubleClick(event: MouseEvent): void {
    const pick = this.viewer.pick(event.clientX, event.clientY);
    if (pick === null) return;
    // Push away from the camera, horizontally: a shove, not a stomp. Looking
    // (almost) straight down, push along the view instead.
    const direction = this.viewer.ray(event.clientX, event.clientY).direction;
    const horizontal = new THREE.Vector3(direction.x, direction.y, 0);
    if (horizontal.length() > 0.2) direction.copy(horizontal);
    direction.normalize();
    pick.mesh.updateMatrixWorld();
    const local = pick.mesh.worldToLocal(pick.point.clone());
    this.send({ type: "push", geom: pick.geom, point: vec3(local), direction: vec3(direction), force: this.pushForce() });
    this.viewer.flashPush(pick.point, direction);
  }
}

function vec3(v: THREE.Vector3): [number, number, number] {
  return [v.x, v.y, v.z];
}
