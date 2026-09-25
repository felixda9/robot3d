import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { createGeomMesh, setPose } from "./geoms";
import type { CameraInfo, FrameMessage, SceneMessage } from "./protocol";

// MuJoCo's world is z-up; three.js defaults to y-up. Rather than converting
// every pose, we make three.js z-up too. This must run before any camera or
// controls are created, since they read DEFAULT_UP when constructed.
THREE.Object3D.DEFAULT_UP.set(0, 0, 1);

// Colors echo MuJoCo's default look (haze = the model's <rgba haze>).
const HAZE = new THREE.Color().setRGB(0.15, 0.25, 0.35, THREE.SRGBColorSpace);
const SKY_TOP = new THREE.Color().setRGB(0.3, 0.5, 0.7, THREE.SRGBColorSpace);
const SKY_BOTTOM = new THREE.Color().setRGB(0.02, 0.03, 0.05, THREE.SRGBColorSpace);
const FLOOR = new THREE.Color().setRGB(0.13, 0.2, 0.28, THREE.SRGBColorSpace);

/** Direction from the sun toward the scene. The light follows the robot. */
const SUN_OFFSET = new THREE.Vector3(1.5, -1.0, 4.0);
/** Shadows are computed in a box this many meters around the robot. */
const SHADOW_EXTENT = 2.5;

/**
 * The 3D view: draws a MuJoCo scene and moves its geoms to each frame's poses.
 * Physics never runs here; the browser only displays what the server sends.
 */
export class Viewer {
  private readonly renderer: THREE.WebGLRenderer;
  private readonly scene = new THREE.Scene();
  private readonly camera: THREE.PerspectiveCamera;
  private readonly controls: OrbitControls;
  private readonly sun: THREE.DirectionalLight;
  private readonly sky: THREE.Mesh;
  private readonly ground: THREE.Group;

  /** Meshes of the current MuJoCo scene (rebuilt on every SceneMessage). */
  private readonly geomRoot = new THREE.Group();
  /** Meshes in FrameMessage order (null = geom not drawn, e.g. a hidden group). */
  private frameMeshes: (THREE.Mesh | null)[] = [];
  /** Center of the moving geoms; the sun, shadows and grid follow it. */
  private readonly focus = new THREE.Vector3();
  /** Follow camera: where `focus` was when the camera last moved with it. */
  private readonly followAnchor = new THREE.Vector3();
  private following = false;
  private currentRobot: string | null = null;

  constructor(container: HTMLElement) {
    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFShadowMap;
    container.appendChild(this.renderer.domElement);

    this.camera = new THREE.PerspectiveCamera(45, 1, 0.01, 1000);
    // Same mouse mapping as MuJoCo's viewer: left = rotate, right = pan, wheel = zoom.
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.12;
    this.controls.maxPolarAngle = 0.495 * Math.PI; // don't orbit below the floor

    this.scene.fog = new THREE.Fog(HAZE, 8, 40);
    this.scene.add(this.geomRoot);

    // Lighting: soft sky/ground fill + a "headlight" riding on the camera
    // (like MuJoCo's viewer) + a sun that casts shadows.
    this.scene.add(new THREE.HemisphereLight(0xdde8ff, 0x303840, 1.3));
    const headlight = new THREE.DirectionalLight(0xffffff, 0.8);
    headlight.position.set(0, 0, 1); // in camera space: from behind the viewer
    this.camera.add(headlight);
    this.scene.add(this.camera); // so the headlight (a camera child) is part of the scene

    this.sun = new THREE.DirectionalLight(0xffffff, 2.2);
    this.sun.castShadow = true;
    this.sun.shadow.mapSize.set(2048, 2048);
    const shadowCam = this.sun.shadow.camera;
    shadowCam.left = shadowCam.bottom = -SHADOW_EXTENT;
    shadowCam.right = shadowCam.top = SHADOW_EXTENT;
    shadowCam.near = 0.1;
    shadowCam.far = 20;
    this.sun.shadow.bias = -0.0005;
    this.sun.shadow.normalBias = 0.01;
    this.scene.add(this.sun, this.sun.target);

    this.sky = createSky();
    this.scene.add(this.sky);
    this.ground = createGround();
    this.ground.visible = false;
    this.scene.add(this.ground);

    new ResizeObserver(() => this.resize(container)).observe(container);
    this.resize(container);
    this.renderer.setAnimationLoop(() => this.render());
  }

  /** Replace everything with a new MuJoCo scene. */
  loadScene(scene: SceneMessage): void {
    this.clearGeoms();
    const meshes = new Map<number, THREE.Mesh>();
    this.ground.visible = false;

    for (const geom of scene.geoms) {
      if (geom.group > 2) continue; // MuJoCo's viewer hides groups 3-5 by default
      const mesh = createGeomMesh(geom);
      if (mesh === null) {
        // Infinite plane: our ground (floor + grid) stands in for it.
        this.ground.visible = true;
        this.ground.position.z = geom.pos[2];
        continue;
      }
      meshes.set(geom.id, mesh);
      this.geomRoot.add(mesh);
    }
    this.frameMeshes = scene.frame_geoms.map((id) => meshes.get(id) ?? null);

    // Start from MuJoCo's default viewpoint, but keep the user's camera when
    // merely reconnecting to the same robot.
    if (scene.robot !== this.currentRobot) {
      this.currentRobot = scene.robot;
      this.setCamera(scene.camera);
    }
    this.followAnchor.copy(this.focus);
  }

  /** Move every dynamic geom to its pose in this frame. */
  applyFrame(frame: FrameMessage): void {
    const count = this.frameMeshes.length;
    this.focus.set(0, 0, 0);
    for (let i = 0; i < count; i++) {
      const mesh = this.frameMeshes[i];
      if (mesh !== null) setPose(mesh, frame.xpos, frame.xmat, i);
      this.focus.x += frame.xpos[3 * i];
      this.focus.y += frame.xpos[3 * i + 1];
      this.focus.z += frame.xpos[3 * i + 2];
    }
    if (count > 0) this.focus.divideScalar(count);
  }

  /** Stop drawing while the view is hidden (e.g. the dashboard is open). */
  setActive(active: boolean): void {
    this.renderer.setAnimationLoop(active ? () => this.render() : null);
  }

  /**
   * Follow camera: the camera moves along with the robot (horizontally only,
   * so it doesn't bob with every step); you can still orbit and zoom.
   */
  setFollow(on: boolean): void {
    this.following = on;
    this.followAnchor.copy(this.focus); // start from here, no jump
  }

  /** Place the camera like MuJoCo's free camera (azimuth/elevation/distance). */
  private setCamera(cam: CameraInfo): void {
    const azimuth = THREE.MathUtils.degToRad(cam.azimuth);
    const elevation = THREE.MathUtils.degToRad(cam.elevation);
    // MuJoCo's camera looks along this direction; it sits `distance` behind lookat.
    const forward = new THREE.Vector3(
      Math.cos(elevation) * Math.cos(azimuth),
      Math.cos(elevation) * Math.sin(azimuth),
      Math.sin(elevation),
    );
    const lookat = new THREE.Vector3(...cam.lookat);
    this.camera.position.copy(lookat).addScaledVector(forward, -cam.distance);
    this.camera.fov = cam.fovy;
    this.camera.updateProjectionMatrix();
    this.controls.target.copy(lookat);
    this.controls.update();
  }

  private render(): void {
    if (this.following) {
      const dx = this.focus.x - this.followAnchor.x;
      const dy = this.focus.y - this.followAnchor.y;
      this.camera.position.x += dx;
      this.camera.position.y += dy;
      this.controls.target.x += dx;
      this.controls.target.y += dy;
      this.followAnchor.copy(this.focus);
    }
    this.controls.update();
    // The sun (and its shadow box) and the grid follow the robot, so shadows
    // work anywhere and the ground looks endless.
    this.sun.target.position.copy(this.focus);
    this.sun.position.copy(this.focus).add(SUN_OFFSET);
    // Snap to whole meters so the grid lines don't slide along with the robot.
    this.ground.position.x = Math.round(this.focus.x);
    this.ground.position.y = Math.round(this.focus.y);
    this.sky.position.copy(this.camera.position); // the sky is "infinitely" far away
    this.renderer.render(this.scene, this.camera);
  }

  private resize(container: HTMLElement): void {
    const { clientWidth: w, clientHeight: h } = container;
    if (w === 0 || h === 0) return;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  private clearGeoms(): void {
    for (const child of [...this.geomRoot.children]) {
      const mesh = child as THREE.Mesh;
      mesh.geometry.dispose();
      (mesh.material as THREE.Material).dispose();
      this.geomRoot.remove(mesh);
    }
    this.frameMeshes = [];
  }
}

/** Floor plane + grid (0.2 m minor lines, 1 m major lines), in the xy plane. */
function createGround(): THREE.Group {
  const group = new THREE.Group();

  const floor = new THREE.Mesh(
    new THREE.PlaneGeometry(400, 400),
    new THREE.MeshStandardMaterial({ color: FLOOR, roughness: 0.9, metalness: 0 }),
  );
  floor.receiveShadow = true;
  group.add(floor);

  const minor = new THREE.GridHelper(30, 150, 0x5a7084, 0x5a7084);
  const major = new THREE.GridHelper(60, 60, 0x9fb3c4, 0x9fb3c4);
  for (const [grid, opacity, z] of [
    [minor, 0.18, 0.0005],
    [major, 0.35, 0.001],
  ] as const) {
    grid.rotation.x = Math.PI / 2; // GridHelper lies in xz; rotate it into xy
    grid.position.z = z; // just above the floor to avoid flickering
    const material = grid.material as THREE.LineBasicMaterial;
    material.transparent = true;
    material.opacity = opacity;
    material.depthWrite = false;
    group.add(grid);
  }
  return group;
}

/** A large sphere around the camera, colored by a vertical gradient. */
function createSky(): THREE.Mesh {
  const material = new THREE.ShaderMaterial({
    side: THREE.BackSide,
    depthWrite: false,
    uniforms: {
      top: { value: SKY_TOP },
      horizon: { value: HAZE },
      bottom: { value: SKY_BOTTOM },
    },
    vertexShader: /* glsl */ `
      varying vec3 vDirection;
      void main() {
        vDirection = normalize(position);
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
      }`,
    fragmentShader: /* glsl */ `
      uniform vec3 top;
      uniform vec3 horizon;
      uniform vec3 bottom;
      varying vec3 vDirection;
      void main() {
        float h = normalize(vDirection).z; // world z = up
        vec3 color = h > 0.0 ? mix(horizon, top, pow(h, 0.5)) : mix(horizon, bottom, pow(-h, 0.35));
        gl_FragColor = vec4(color, 1.0);
        #include <colorspace_fragment>
      }`,
  });
  const sky = new THREE.Mesh(new THREE.SphereGeometry(500, 32, 16), material);
  sky.renderOrder = -1;
  sky.frustumCulled = false;
  return sky;
}
