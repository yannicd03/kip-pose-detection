import { useCallback, useMemo, useState } from "react";
import Controls from "./components/Controls";
import Viewer3D from "./components/Viewer";
import Viewer2D from "./components/Viewer2D";
import {
  convertRawGtOverlay,
  fetchAsFile,
  fetchGtBundle,
  predict,
  POSE_SOURCES,
  SEG_SOURCES,
  type GtBundle,
  type Intrinsics,
  type PredictResponse,
} from "./api";
import type { VizObject } from "./geometry";

export type ViewMode = "2d" | "3d";

export interface Layers {
  // 2D
  backdrop: boolean;
  predMask: boolean;
  gtMask: boolean;
  // 3D
  grid: boolean;
  pointcloud: boolean;
  table: boolean;
  arm: boolean;
  predMesh: boolean;
  gtMesh: boolean;
  // shared (axes appear in both modes)
  predAxes: boolean;
  gtAxes: boolean;
}

const DEFAULT_LAYERS: Layers = {
  backdrop: true,
  predMask: true,
  gtMask: true,
  grid: true,
  pointcloud: true,
  // Table on by default (context, sits below the parts); arm off by default —
  // it occludes ~60% of the frame from the camera angle (see ADR-008), opt-in.
  table: true,
  arm: false,
  predMesh: true,
  gtMesh: true,
  predAxes: true,
  gtAxes: true,
};

const DEFAULT_INTRINSICS: Intrinsics = {
  fx: 1322.667,
  fy: 1322.667,
  cx: 640,
  cy: 360,
};

export default function App() {
  const [mode, setMode] = useState<ViewMode>("2d");

  const [rgbFile, setRgbFile] = useState<File | null>(null);
  const [depthFile, setDepthFile] = useState<File | null>(null);
  const [intrinsics, setIntrinsics] = useState<Intrinsics>(DEFAULT_INTRINSICS);
  const [iterations, setIterations] = useState<number>(5);
  const [topN, setTopN] = useState<string>("");
  const [wantPointcloud, setWantPointcloud] = useState<boolean>(false);
  const [segSource, setSegSource] = useState<string>("yolo");
  // Pose estimator — separate stage from the segmentation source above.
  const [poseSource, setPoseSource] = useState<string>("foundationpose");
  // GT mask bundle for the "gt" *segmentation source* (feeds the pose stage).
  const [gtMasks, setGtMasks] = useState<string | null>(null);
  // GT *overlay* bundle (poses + masks), shown as a reference, never inferred.
  const [gtBundle, setGtBundle] = useState<GtBundle | null>(null);

  const [result, setResult] = useState<PredictResponse | null>(null);
  const [busy, setBusy] = useState<boolean>(false);
  const [status, setStatus] = useState<string>("Idle.");

  const [layers, setLayers] = useState<Layers>(DEFAULT_LAYERS);
  // Per-object hidden set, keyed `pred:<id>` / `gt:<id>`. Default: all visible.
  const [hidden, setHidden] = useState<Set<string>>(new Set());

  const rgbUrl = useMemo(
    () => (rgbFile ? URL.createObjectURL(rgbFile) : null),
    [rgbFile]
  );

  // Image dimensions + intrinsics for the viewers. Prefer the run result
  // (authoritative), fall back to the form intrinsics + a default size.
  const W = result?.width ?? 1280;
  const H = result?.height ?? 720;
  const K = result?.K ?? intrinsics;

  // Build the renderable object lists (prediction + ground truth).
  const predObjects = useMemo<VizObject[]>(
    () =>
      (result?.instances ?? [])
        .filter((i) => Array.isArray(i.T_cam_obj))
        .map((i) => ({
          key: `pred:${i.id}`,
          kind: "pred" as const,
          id: i.id,
          class: i.class,
          conf: i.conf,
          T: i.T_cam_obj,
          mask: i.mask_b64,
        })),
    [result]
  );
  const gtObjects = useMemo<VizObject[]>(
    () =>
      (gtBundle?.instances ?? [])
        .filter((i) => Array.isArray(i.T_cam_obj))
        .map((i) => ({
          key: `gt:${i.id}`,
          kind: "gt" as const,
          id: i.id,
          class: i.class,
          T: i.T_cam_obj,
          mask: i.mask_b64,
        })),
    [gtBundle]
  );
  const allObjects = useMemo(
    () => [...gtObjects, ...predObjects],
    [gtObjects, predObjects]
  );

  const toggleHidden = useCallback((key: string) => {
    setHidden((prev) => {
      const next = new Set(prev);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });
  }, []);

  const setMany = useCallback((keys: string[], hide: boolean) => {
    setHidden((prev) => {
      const next = new Set(prev);
      for (const k of keys) {
        if (hide) next.add(k);
        else next.delete(k);
      }
      return next;
    });
  }, []);

  const loadSample = useCallback(async () => {
    setStatus("Loading sample...");
    try {
      const [rgb, depth, gt, gtPoses] = await Promise.all([
        fetchAsFile("/samples/sample_rgb.png", "sample_rgb.png", "image/png"),
        fetchAsFile(
          "/samples/sample_depth_mm.png",
          "sample_depth_mm.png",
          "image/png"
        ),
        fetch("/samples/sample_gt_masks.json").then((r) => (r.ok ? r.text() : null)),
        fetchGtBundle("/samples/sample_gt_poses.json"),
      ]);
      setRgbFile(rgb);
      setDepthFile(depth);
      setGtMasks(gt);
      setGtBundle(gtPoses);
      setIntrinsics({ fx: 1322.667, fy: 1322.667, cx: 640, cy: 360 });
      setResult(null);
      setHidden(new Set());
      setStatus(
        gtPoses
          ? `Sample loaded (incl. GT overlay: ${gtPoses.instances.length} objects). Ready to run.`
          : "Sample loaded. Ready to run."
      );
    } catch (e) {
      setStatus(`Error loading sample: ${(e as Error).message}`);
    }
  }, []);

  // Manual RGB/depth uploads invalidate the sample's GT (masks + overlay only
  // match the sample frame). Re-upload a GT overlay after the new image.
  const onRgb = useCallback((f: File | null) => {
    setRgbFile(f);
    setGtMasks(null);
    setGtBundle(null);
  }, []);
  const onDepth = useCallback((f: File | null) => {
    setDepthFile(f);
  }, []);

  const onGtOverlay = useCallback((f: File | null) => {
    if (!f) {
      setGtBundle(null);
      return;
    }
    const reader = new FileReader();
    reader.onload = () => {
      try {
        const data = JSON.parse(String(reader.result));
        const instances = Array.isArray(data) ? data : data.instances;
        if (!Array.isArray(instances)) throw new Error("no 'instances' array");
        setGtBundle({ instances });
        setHidden(new Set());
        setStatus(`Loaded GT overlay: ${instances.length} objects from ${f.name}.`);
      } catch (e) {
        setStatus(`Invalid GT overlay JSON: ${(e as Error).message}`);
      }
    };
    reader.readAsText(f);
  }, []);

  // Convert a raw SDG frame (bbox_3d + optional instance/labels + world2cam) to
  // the GT overlay, server-side. Uses the current intrinsics as K.
  const onConvertRawGt = useCallback(
    async (files: {
      bbox3d: File | null;
      world2cam: File | null;
      instance: File | null;
      instanceLabels: File | null;
    }) => {
      if (!files.bbox3d) {
        setStatus("Raw GT conversion needs at least bbox_3d.json.");
        return;
      }
      setBusy(true);
      setStatus("Converting raw SDG frame to GT overlay...");
      try {
        const bundle = await convertRawGtOverlay({
          bbox3d: files.bbox3d,
          world2cam: files.world2cam,
          intrinsics,
          instance: files.instance,
          instanceLabels: files.instanceLabels,
        });
        setGtBundle(bundle);
        setHidden(new Set());
        const withMask = bundle.instances.filter((i) => i.mask_b64).length;
        setStatus(
          `Converted GT overlay: ${bundle.instances.length} objects ` +
            `(${withMask} with masks).`
        );
      } catch (e) {
        setStatus(`GT conversion failed: ${(e as Error).message}`);
      } finally {
        setBusy(false);
      }
    },
    [intrinsics]
  );

  const run = useCallback(async () => {
    const pose = POSE_SOURCES.find((s) => s.id === poseSource);
    // Depth is optional for RGB-only pose estimators (e.g. GigaPose RGB-only).
    if (!rgbFile) {
      setStatus("Please provide an RGB image.");
      return;
    }
    if (!pose?.rgbOnly && !depthFile) {
      setStatus(
        "Please provide a depth image, or pick the GigaPose RGB-only estimator."
      );
      return;
    }
    const src = SEG_SOURCES.find((s) => s.id === segSource);
    if (src?.needsGtMasks && !gtMasks) {
      setStatus(
        "Ground-truth masks unavailable. They exist only for the sim sample — " +
          "click “Load sample”, or pick the YOLO pipeline."
      );
      return;
    }
    setBusy(true);
    setStatus(
      `Running ${src?.label ?? segSource} → ${pose?.label ?? poseSource} pipeline...`
    );
    const t0 = performance.now();
    try {
      const parsedTopN = topN.trim() === "" ? undefined : Number(topN);
      const res = await predict({
        rgb: rgbFile,
        // RGB-only estimators ignore depth even if one was loaded.
        depth: pose?.rgbOnly ? null : depthFile,
        intrinsics,
        iterations,
        topN: parsedTopN,
        wantPointcloud,
        segSource,
        poseSource,
        gtMasks: src?.needsGtMasks ? gtMasks ?? undefined : undefined,
      });
      const dt = ((performance.now() - t0) / 1000).toFixed(2);
      setResult(res);
      const t = res.timings;
      const split = t
        ? ` [${src?.label ?? segSource} ${(t.seg_ms / 1000).toFixed(2)}s + ` +
          `${pose?.label ?? poseSource} ${(t.pose_ms / 1000).toFixed(2)}s` +
          (t.num_posed > 0 ? `, ${(t.pose_ms / t.num_posed).toFixed(0)}ms/obj` : "") +
          `]`
        : "";
      setStatus(
        `Done in ${dt}s — ${src?.label ?? segSource} → ${pose?.label ?? poseSource}: ` +
          `${res.num_detections ?? "?"} detection(s) → ${res.instances.length} posed, ` +
          `${res.width}x${res.height}.${split}`
      );
    } catch (e) {
      setStatus(`Error: ${(e as Error).message}`);
    } finally {
      setBusy(false);
    }
  }, [
    rgbFile,
    depthFile,
    intrinsics,
    iterations,
    topN,
    wantPointcloud,
    segSource,
    poseSource,
    gtMasks,
  ]);

  return (
    <div className="app">
      <Controls
        mode={mode}
        setMode={setMode}
        segSource={segSource}
        setSegSource={setSegSource}
        poseSource={poseSource}
        setPoseSource={setPoseSource}
        intrinsics={intrinsics}
        setIntrinsics={setIntrinsics}
        iterations={iterations}
        setIterations={setIterations}
        topN={topN}
        setTopN={setTopN}
        wantPointcloud={wantPointcloud}
        setWantPointcloud={setWantPointcloud}
        onRgb={onRgb}
        onDepth={onDepth}
        onGtOverlay={onGtOverlay}
        onConvertRawGt={onConvertRawGt}
        rgbName={rgbFile?.name ?? null}
        depthName={depthFile?.name ?? null}
        gtName={gtBundle ? `${gtBundle.instances.length} GT objects` : null}
        onLoadSample={loadSample}
        onRun={run}
        busy={busy}
        status={status}
        predObjects={predObjects}
        gtObjects={gtObjects}
        hidden={hidden}
        toggleHidden={toggleHidden}
        setMany={setMany}
        layers={layers}
        setLayers={setLayers}
      />
      <div className="stage">
        {allObjects.length === 0 ? (
          <div className="viewer">
            {mode === "2d" && layers.backdrop && rgbUrl && (
              <img className="backdrop" src={rgbUrl} alt="RGB input" />
            )}
            <div className="placeholder">
              {rgbUrl
                ? "Press Run for predictions, or load a GT overlay."
                : "Load inputs (or the sample) to begin."}
            </div>
          </div>
        ) : mode === "2d" ? (
          <Viewer2D
            W={W}
            H={H}
            K={K}
            objects={allObjects}
            hidden={hidden}
            rgbUrl={rgbUrl}
            show={{
              backdrop: layers.backdrop,
              predMask: layers.predMask,
              predAxes: layers.predAxes,
              gtMask: layers.gtMask,
              gtAxes: layers.gtAxes,
            }}
          />
        ) : (
          <Viewer3D
            W={W}
            H={H}
            K={K}
            objects={allObjects}
            hidden={hidden}
            pointcloud={result?.pointcloud ?? null}
            show={{
              grid: layers.grid,
              pointcloud: layers.pointcloud,
              table: layers.table,
              arm: layers.arm,
              predMesh: layers.predMesh,
              predAxes: layers.predAxes,
              gtMesh: layers.gtMesh,
              gtAxes: layers.gtAxes,
            }}
          />
        )}
      </div>
    </div>
  );
}
