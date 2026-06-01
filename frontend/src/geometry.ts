import * as THREE from "three";
import type { Intrinsics } from "./api";

/** Overlay colours. GT = green, prediction = magenta. Axes stay RGB = XYZ. */
export const COLORS = {
  gt: "#2fcc4f",
  pred: "#ff35d4",
  axisX: "#ff5555",
  axisY: "#46e05a",
  axisZ: "#5b8cff",
} as const;

/** Coordinate-triad arm length, metres (matches fp_viz.py). */
export const AXIS_LEN_M = 0.04;

/**
 * World -> OpenCV camera extrinsic (metres) for the GST_Scene Zivid camera:
 * T_world2cvcam = D_FLIP @ DEFAULT_WORLD2CAM.T — identical to the gateway's
 * extrinsic (tools/sdg_to_gt_overlay.py) and emitted by tools/usd_scene_to_glb.py
 * as public/meshes/scene_props.json. The static scene props (table, arm) are
 * exported in USD *world* coordinates, so this matrix is their "pose": they are
 * placed by the very same poseToMatrix(F · T) path used for the per-frame parts,
 * just with the world->cam extrinsic instead of a per-object T_cam_obj.
 * Keep in sync with scene_props.json if the scene camera ever changes.
 */
export const T_WORLD2CVCAM: number[][] = [
  [-0.99939083, -0.00665914, 0.03425829, 0.41262822],
  [0.0, 0.98162718, 0.190809, -0.2963418],
  [-0.0348995, 0.19069276, -0.9810292, 1.07804259],
  [0.0, 0.0, 0.0, 1.0],
];

export type Kind = "gt" | "pred";

/** A renderable object for either viewer (prediction or ground truth). */
export interface VizObject {
  key: string; // `${kind}:${id}`
  kind: Kind;
  id: number;
  class: string;
  conf?: number;
  /** 4x4 row-major, OpenCV camera frame, metres (mesh -> cam). */
  T: number[][];
  /** base64 PNG instance mask (full frame, 255 = object). */
  mask?: string;
}

export function hexToRgb(hex: string): [number, number, number] {
  const h = hex.replace("#", "");
  return [
    parseInt(h.slice(0, 2), 16),
    parseInt(h.slice(2, 4), 16),
    parseInt(h.slice(4, 6), 16),
  ];
}

/**
 * OpenCV camera frame (x right, y down, +z forward) -> three.js (y up,
 * -z forward). F = rotation PI about X = diag(1,-1,-1,1), a proper rotation.
 */
export function makeF(): THREE.Matrix4 {
  return new THREE.Matrix4().makeRotationX(Math.PI);
}

/** Build a three.js Matrix4 from a row-major 4x4 pose, premultiplied by F. */
export function poseToMatrix(T: number[][]): THREE.Matrix4 {
  const m = new THREE.Matrix4();
  m.set(
    T[0][0], T[0][1], T[0][2], T[0][3],
    T[1][0], T[1][1], T[1][2], T[1][3],
    T[2][0], T[2][1], T[2][2], T[2][3],
    T[3][0], T[3][1], T[3][2], T[3][3]
  );
  return makeF().multiply(m);
}

/** Object-origin translation of a pose, mapped into three.js space. */
export function poseTranslationThree(T: number[][]): THREE.Vector3 {
  return new THREE.Vector3(T[0][3], T[1][3], T[2][3]).applyMatrix4(makeF());
}

/** Centroid (three.js space) of a set of objects, for camera framing. */
export function centroidThree(objs: VizObject[]): THREE.Vector3 {
  const c = new THREE.Vector3();
  if (objs.length === 0) return new THREE.Vector3(0, 0, -1.05);
  for (const o of objs) c.add(poseTranslationThree(o.T));
  return c.multiplyScalar(1 / objs.length);
}

/**
 * Centre of the plane the parts rest on, in three.js space: the object-cluster
 * centroid in the table plane (x,y under `up`) dropped to the parts' resting
 * height along `up`. Height uses a low percentile of object-origin projections
 * (robust to a stray high/low pose) rather than the table mesh's tallest vertex,
 * which would catch overhead workstation structure and float the grid metres up.
 */
export function restingPlaneThree(
  objs: VizObject[],
  up: THREE.Vector3
): THREE.Vector3 {
  if (objs.length === 0) return new THREE.Vector3(0, 0, -1.05);
  const origins = objs.map((o) => poseTranslationThree(o.T));
  const heights = origins.map((p) => p.dot(up)).sort((a, b) => a - b);
  const h = heights[Math.floor(0.1 * (heights.length - 1))]; // ~10th percentile
  const c = new THREE.Vector3();
  for (const p of origins) c.add(p);
  c.multiplyScalar(1 / origins.length);
  // Replace the along-`up` component of the centroid with the resting height.
  return c.add(up.clone().multiplyScalar(h - c.dot(up)));
}

export interface Pt2 {
  u: number;
  v: number;
}

/**
 * Project an object-local coordinate triad through pose T (row-major, OpenCV
 * cam frame) and intrinsics K to image pixels. `ok` is false if any point is
 * behind the camera.
 */
export function projectTriad(
  T: number[][],
  K: Intrinsics,
  len = AXIS_LEN_M
): { o: Pt2; x: Pt2; y: Pt2; z: Pt2; ok: boolean } {
  const pts = [
    [0, 0, 0],
    [len, 0, 0],
    [0, len, 0],
    [0, 0, len],
  ];
  const out: Pt2[] = [];
  let ok = true;
  for (const p of pts) {
    const xc = T[0][0] * p[0] + T[0][1] * p[1] + T[0][2] * p[2] + T[0][3];
    const yc = T[1][0] * p[0] + T[1][1] * p[1] + T[1][2] * p[2] + T[1][3];
    const zc = T[2][0] * p[0] + T[2][1] * p[1] + T[2][2] * p[2] + T[2][3];
    if (zc <= 1e-6) ok = false;
    out.push({ u: (K.fx * xc) / zc + K.cx, v: (K.fy * yc) / zc + K.cy });
  }
  return { o: out[0], x: out[1], y: out[2], z: out[3], ok };
}
