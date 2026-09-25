import * as THREE from "three";
import type { GeomInfo, Mat3, Vec3 } from "./protocol";

/**
 * Build a three.js mesh for one MuJoCo geom, in the geom's LOCAL frame.
 * Its world pose is set separately (setPose) from MuJoCo's xpos/xmat.
 *
 * MuJoCo sizes are half-sizes (see GeomInfo in protocol.ts); three.js
 * geometries take full sizes. MuJoCo capsules and cylinders run along their
 * local z axis, three.js ones along y, so we rotate those.
 *
 * Returns null for an infinite plane (the viewer draws its own ground).
 */
export function createGeomMesh(geom: GeomInfo): THREE.Mesh | null {
  const geometry = createGeometry(geom);
  if (geometry === null) return null;

  const [r, g, b, a] = geom.rgba;
  const material = new THREE.MeshStandardMaterial({
    // MuJoCo colors are display (sRGB) colors; tell three.js so they show up
    // on screen as the same color.
    color: new THREE.Color().setRGB(r, g, b, THREE.SRGBColorSpace),
    roughness: 0.55,
    metalness: 0.05,
    transparent: a < 1,
    opacity: a,
  });

  const mesh = new THREE.Mesh(geometry, material);
  mesh.name = geom.name || `geom${geom.id}`;
  mesh.castShadow = geom.type !== "plane";
  mesh.receiveShadow = true;
  setPose(mesh, geom.pos, geom.mat);
  return mesh;
}

function createGeometry(geom: GeomInfo): THREE.BufferGeometry | null {
  const [s0, s1, s2] = geom.size;
  switch (geom.type) {
    case "plane":
      if (s0 === 0 || s1 === 0) return null; // infinite plane = the ground
      return new THREE.PlaneGeometry(2 * s0, 2 * s1); // lies in local xy, like MuJoCo's
    case "sphere":
      return new THREE.SphereGeometry(s0, 32, 16);
    case "ellipsoid":
      return new THREE.SphereGeometry(1, 32, 16).scale(s0, s1, s2);
    case "capsule":
      // height = length of the straight middle part = 2 * MuJoCo's half-length.
      return new THREE.CapsuleGeometry(s0, 2 * s1, 8, 24).rotateX(Math.PI / 2);
    case "cylinder":
      return new THREE.CylinderGeometry(s0, s0, 2 * s1, 32).rotateX(Math.PI / 2);
    case "box":
      return new THREE.BoxGeometry(2 * s0, 2 * s1, 2 * s2);
  }
}

const rotation = new THREE.Matrix4();

/**
 * Place an object at MuJoCo position `pos` with rotation matrix `mat`.
 * `offset` selects geom number `offset` inside the flat arrays of a FrameMessage.
 */
export function setPose(
  object: THREE.Object3D,
  pos: Vec3 | number[],
  mat: Mat3 | number[],
  offset = 0,
): void {
  const p = 3 * offset;
  const m = 9 * offset;
  object.position.set(pos[p], pos[p + 1], pos[p + 2]);
  // Matrix4.set() takes its arguments row by row, and MuJoCo's xmat is
  // row-major, so the numbers go in as they come.
  rotation.set(
    mat[m], mat[m + 1], mat[m + 2], 0,
    mat[m + 3], mat[m + 4], mat[m + 5], 0,
    mat[m + 6], mat[m + 7], mat[m + 8], 0,
    0, 0, 0, 1,
  );
  object.quaternion.setFromRotationMatrix(rotation);
}
