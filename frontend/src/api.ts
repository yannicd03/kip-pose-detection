export const GATEWAY_URL: string =
  (import.meta.env.VITE_GATEWAY_URL as string | undefined) ||
  "http://localhost:8000";

export interface Intrinsics {
  fx: number;
  fy: number;
  cx: number;
  cy: number;
}

export interface Instance {
  id: number;
  class: string;
  conf: number;
  /** 4x4 row-major matrix, OpenCV camera frame, metres (mesh->cam). */
  T_cam_obj: number[][];
  /** base64 PNG instance mask (full frame, 255=object). Optional. */
  mask_b64?: string;
}

/**
 * Ground-truth instance for an overlay bundle. Carries both the GT pose (from
 * the simulator's bbox_3d) and, optionally, the GT instance mask, so the viewer
 * can pair pose<->mask per object exactly like a prediction.
 */
export interface GtInstance {
  id: number;
  class: string;
  T_cam_obj: number[][];
  mask_b64?: string;
}

export interface GtBundle {
  instances: GtInstance[];
}

export interface PointCloud {
  xyz: number[][];
  rgb: number[][];
}

export interface Timings {
  seg_ms: number;
  pose_ms: number;
  num_posed: number;
}

export interface PredictResponse {
  width: number;
  height: number;
  K: Intrinsics;
  seg_source?: string;
  num_detections?: number;
  instances: Instance[];
  pointcloud: PointCloud | null;
  timings?: Timings;
}

/**
 * Available pipelines (segmentation source -> FoundationPose). Add a future
 * segmentation model here AND register its service URL in the gateway's
 * INFER_SOURCES. `gt` is special: it sends supplied ground-truth masks.
 */
export interface SegSource {
  id: string;
  label: string;
  /** true if this source needs a ground-truth mask bundle supplied by the client. */
  needsGtMasks?: boolean;
}

export const SEG_SOURCES: SegSource[] = [
  { id: "yolo", label: "YOLO26n-seg (infer masks)" },
  { id: "sam3", label: "SAM3 (promptable concept masks)" },
  { id: "gt", label: "Ground-truth masks (sim)", needsGtMasks: true },
];

/**
 * Default SAM3 concept prompts, one per class. Shown (editable) in the UI when
 * the SAM3 segmentation source is selected and sent as a per-request override;
 * must mirror the sam3-svc SAM3_PROMPTS defaults. Note: the prompts find
 * armature instances well but do NOT reliably separate kurz from lang — SAM3
 * class labels are approximate (see sam3-svc/app.py).
 */
export const SAM3_DEFAULT_PROMPTS: Record<string, string> = {
  anker_kurz: "short metal motor armature part",
  anker_lang: "long metal motor armature part",
};

/**
 * Pose estimator selection — a stage *separate* from the segmentation source
 * above. The seg source decides how masks are produced; the pose source decides
 * which 6-DoF estimator consumes them. Add a future estimator here AND register
 * its service URL in the gateway's POSE_SOURCES. `rgbOnly` sources run without a
 * depth map (no depth uploaded, no depth forwarded, no Kabsch tail).
 */
export interface PoseSource {
  id: string;
  label: string;
  /** true if this estimator runs RGB-only (no depth required or consumed). */
  rgbOnly?: boolean;
}

export const POSE_SOURCES: PoseSource[] = [
  { id: "foundationpose", label: "FoundationPose (RGB-D)" },
  { id: "gigapose_rgbd", label: "GigaPose (RGB-D, 5-hypo + Kabsch)" },
  { id: "gigapose_rgb", label: "GigaPose (RGB-only, 5-hypo)", rgbOnly: true },
];

export interface PredictParams {
  rgb: File;
  /** Optional: RGB-only pose sources (e.g. gigapose_rgb) run without depth. */
  depth: File | null;
  intrinsics: Intrinsics;
  iterations: number;
  topN?: number;
  wantPointcloud: boolean;
  segSource: string;
  /** Pose estimator id (POSE_SOURCES). Separate from segSource. */
  poseSource: string;
  /** JSON string of the GT mask bundle, required when segSource needs GT masks. */
  gtMasks?: string;
  /** Per-class SAM3 concept prompts, sent only for the sam3 seg source. */
  segPrompts?: Record<string, string>;
}

export async function predict(params: PredictParams): Promise<PredictResponse> {
  const {
    rgb,
    depth,
    intrinsics,
    iterations,
    topN,
    wantPointcloud,
    segSource,
    poseSource,
    gtMasks,
    segPrompts,
  } = params;
  const form = new FormData();
  form.append("rgb", rgb);
  // RGB-only pose sources (e.g. gigapose_rgb) run without depth.
  const rgbOnly = POSE_SOURCES.find((s) => s.id === poseSource)?.rgbOnly;
  if (depth && !rgbOnly) {
    form.append("depth", depth);
  }
  form.append("fx", String(intrinsics.fx));
  form.append("fy", String(intrinsics.fy));
  form.append("cx", String(intrinsics.cx));
  form.append("cy", String(intrinsics.cy));
  form.append("iterations", String(iterations));
  if (topN !== undefined && topN !== null && !Number.isNaN(topN)) {
    form.append("top_n", String(topN));
  }
  form.append("want_pointcloud", String(wantPointcloud));
  form.append("seg_source", segSource);
  form.append("pose_source", poseSource);
  if (gtMasks) {
    form.append("gt_masks", gtMasks);
  }
  if (segPrompts && Object.keys(segPrompts).length > 0) {
    form.append("seg_prompts", JSON.stringify(segPrompts));
  }

  const res = await fetch(`${GATEWAY_URL}/predict`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`Gateway ${res.status}: ${text || res.statusText}`);
  }
  return (await res.json()) as PredictResponse;
}

/** Fetch a static asset (from /public) as a File for use as a form input. */
export async function fetchAsFile(
  url: string,
  filename: string,
  type: string
): Promise<File> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Failed to fetch ${url}: ${res.status}`);
  const blob = await res.blob();
  return new File([blob], filename, { type });
}

export interface ConvertGtParams {
  bbox3d: File;
  intrinsics: Intrinsics;
  /** Camera extrinsic. Omit to use the gateway's default GST_Scene extrinsic. */
  world2cam?: File | null;
  instance?: File | null;
  instanceLabels?: File | null;
}

/**
 * Convert a raw Isaac-Sim SDG frame (bbox_3d + optional instance/labels +
 * world2cam) into the viewer's GT overlay bundle, server-side. The gateway does
 * the USD->OpenCV flip / 0.001-unscale / world->cam transform and returns pure
 * T_cam_obj, so the viewer stays convention-agnostic.
 */
export async function convertRawGtOverlay(p: ConvertGtParams): Promise<GtBundle> {
  const form = new FormData();
  form.append("bbox_3d", p.bbox3d);
  if (p.world2cam) form.append("world2cam", p.world2cam);
  form.append("fx", String(p.intrinsics.fx));
  form.append("fy", String(p.intrinsics.fy));
  form.append("cx", String(p.intrinsics.cx));
  form.append("cy", String(p.intrinsics.cy));
  if (p.instance) form.append("instance", p.instance);
  if (p.instanceLabels) form.append("instance_labels", p.instanceLabels);

  const res = await fetch(`${GATEWAY_URL}/gt_overlay`, { method: "POST", body: form });
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`Gateway ${res.status}: ${text || res.statusText}`);
  }
  const data = (await res.json()) as GtBundle;
  if (!data || !Array.isArray(data.instances)) {
    throw new Error("gateway returned no instances");
  }
  return data;
}

/** Optionally load a GT pose+mask overlay bundle (e.g. the bundled sim sample). */
export async function fetchGtBundle(url: string): Promise<GtBundle | null> {
  try {
    const res = await fetch(url);
    if (!res.ok) return null;
    const data = (await res.json()) as GtBundle;
    if (!data || !Array.isArray(data.instances)) return null;
    return data;
  } catch {
    return null;
  }
}
