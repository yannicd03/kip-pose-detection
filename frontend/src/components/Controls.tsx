import { useState } from "react";
import type { Layers, ViewMode } from "../App";
import { POSE_SOURCES, SEG_SOURCES, type Intrinsics } from "../api";
import { COLORS, type VizObject } from "../geometry";
import InfoTip from "./InfoTip";

const INTRINSIC_TIPS: Record<keyof Intrinsics, string> = {
  fx: "Focal length along X, in pixels. From camera calibration. Sim sample: 1322.667.",
  fy: "Focal length along Y, in pixels. Equals fx for square pixels. Sim sample: 1322.667.",
  cx: "Principal point X (optical centre), in pixels. Usually image width / 2 (sample: 640).",
  cy: "Principal point Y (optical centre), in pixels. Usually image height / 2 (sample: 360).",
};

interface Props {
  mode: ViewMode;
  setMode: (m: ViewMode) => void;
  segSource: string;
  setSegSource: (s: string) => void;
  segPrompts: Record<string, string>;
  setSegPrompts: (p: Record<string, string>) => void;
  poseSource: string;
  setPoseSource: (s: string) => void;
  intrinsics: Intrinsics;
  setIntrinsics: (i: Intrinsics) => void;
  iterations: number;
  setIterations: (n: number) => void;
  topN: string;
  setTopN: (s: string) => void;
  wantPointcloud: boolean;
  setWantPointcloud: (b: boolean) => void;
  onRgb: (f: File | null) => void;
  onDepth: (f: File | null) => void;
  onGtOverlay: (f: File | null) => void;
  onConvertRawGt: (files: {
    bbox3d: File | null;
    world2cam: File | null;
    instance: File | null;
    instanceLabels: File | null;
  }) => void;
  rgbName: string | null;
  depthName: string | null;
  gtName: string | null;
  onLoadSample: () => void;
  onRun: () => void;
  busy: boolean;
  status: string;
  predObjects: VizObject[];
  gtObjects: VizObject[];
  hidden: Set<string>;
  toggleHidden: (key: string) => void;
  setMany: (keys: string[], hide: boolean) => void;
  layers: Layers;
  setLayers: (l: Layers) => void;
}

const INTRINSIC_KEYS: (keyof Intrinsics)[] = ["fx", "fy", "cx", "cy"];

export default function Controls(props: Props) {
  const {
    mode,
    setMode,
    segSource,
    setSegSource,
    segPrompts,
    setSegPrompts,
    poseSource,
    setPoseSource,
    intrinsics,
    setIntrinsics,
    iterations,
    setIterations,
    topN,
    setTopN,
    wantPointcloud,
    setWantPointcloud,
    onRgb,
    onDepth,
    onGtOverlay,
    onConvertRawGt,
    rgbName,
    depthName,
    gtName,
    onLoadSample,
    onRun,
    busy,
    status,
    predObjects,
    gtObjects,
    hidden,
    toggleHidden,
    setMany,
    layers,
    setLayers,
  } = props;

  // Local file selections for the raw-SDG -> GT conversion sub-form.
  const [rawBbox, setRawBbox] = useState<File | null>(null);
  const [rawWorld2cam, setRawWorld2cam] = useState<File | null>(null);
  const [rawInstance, setRawInstance] = useState<File | null>(null);
  const [rawLabels, setRawLabels] = useState<File | null>(null);
  const [showConvert, setShowConvert] = useState(false);

  const poseRgbOnly =
    POSE_SOURCES.find((s) => s.id === poseSource)?.rgbOnly ?? false;

  const layerBox = (key: keyof Layers, label: string, tip: string) => (
    <label className="checkbox" key={key}>
      <input
        type="checkbox"
        checked={layers[key]}
        onChange={(e) => setLayers({ ...layers, [key]: e.target.checked })}
      />
      <span>{label}</span>
      <InfoTip text={tip} />
    </label>
  );

  const objectList = (title: string, objs: VizObject[], color: string) => {
    const keys = objs.map((o) => o.key);
    const anyVisible = keys.some((k) => !hidden.has(k));
    return (
      <section>
        <h2 className="listhead">
          <span>
            {title} ({objs.length})
          </span>
          {objs.length > 0 && (
            <button
              className="linkbtn"
              onClick={() => setMany(keys, anyVisible)}
            >
              {anyVisible ? "hide all" : "show all"}
            </button>
          )}
        </h2>
        {objs.length === 0 ? (
          <p className="hint">None.</p>
        ) : (
          <ul className="objlist">
            {objs.map((o) => (
              <li key={o.key}>
                <input
                  type="checkbox"
                  checked={!hidden.has(o.key)}
                  onChange={() => toggleHidden(o.key)}
                />
                <span className="dot" style={{ background: color }} />
                <span className="cls">{o.class}</span>
                {o.conf != null && (
                  <span className="conf">{(o.conf * 100).toFixed(1)}%</span>
                )}
              </li>
            ))}
          </ul>
        )}
      </section>
    );
  };

  return (
    <aside className="panel">
      <h1>KIP Pose Viewer</h1>

      <div className="modeswitch" role="tablist">
        <button
          className={mode === "2d" ? "active" : ""}
          onClick={() => setMode("2d")}
        >
          2D overlay
        </button>
        <button
          className={mode === "3d" ? "active" : ""}
          onClick={() => setMode("3d")}
        >
          3D scene
        </button>
      </div>

      <section>
        <h2>Pipeline</h2>
        <label className="field">
          <span>
            Segmentation source
            <InfoTip text="Stage 1 — how object masks are produced. 'YOLO26n-seg' runs the trained detector; 'Ground-truth' feeds the sim's exact masks (sim sample only). This is the MASK source only; it is independent of the pose estimator chosen below, and separate from the GT overlay further down." />
          </span>
          <select value={segSource} onChange={(e) => setSegSource(e.target.value)}>
            {SEG_SOURCES.map((s) => (
              <option key={s.id} value={s.id}>
                {s.label}
              </option>
            ))}
          </select>
        </label>
        {segSource === "sam3" &&
          Object.entries(segPrompts).map(([cls, prompt]) => (
            <label className="field" key={cls}>
              <span>
                SAM3 prompt — {cls}
                <InfoTip
                  text={
                    "Concept prompt (short noun phrase) SAM3 uses to find every " +
                    `instance it labels '${cls}'. Sent per request; blank falls ` +
                    "back to the service default. Caveat: prompts retrieve " +
                    "armatures reliably but do NOT separate kurz from lang — " +
                    "treat SAM3 class labels as approximate."
                  }
                />
              </span>
              <input
                type="text"
                value={prompt}
                placeholder="e.g. small metal object"
                onChange={(e) =>
                  setSegPrompts({ ...segPrompts, [cls]: e.target.value })
                }
              />
            </label>
          ))}
        <label className="field">
          <span>
            Pose estimator
            <InfoTip text="Stage 2 — which 6-DoF estimator consumes the masks above. 'FoundationPose' and 'GigaPose (RGB-D)' both use depth; 'GigaPose (RGB-only)' runs without depth (no depth upload needed, no Kabsch refinement). This is a SEPARATE stage from the segmentation source — the two combine freely." />
          </span>
          <div
            style={{ display: "flex", alignItems: "center", gap: 6 }}
          >
            <select
              value={poseSource}
              onChange={(e) => setPoseSource(e.target.value)}
              style={{ flex: 1 }}
            >
              {POSE_SOURCES.map((s) => (
                <option key={s.id} value={s.id}>
                  {s.label}
                </option>
              ))}
            </select>
            {poseRgbOnly && (
              <span
                style={{
                  fontSize: 11,
                  fontWeight: 600,
                  whiteSpace: "nowrap",
                  color: "#1d4ed8",
                  background: "#dbeafe",
                  borderRadius: 4,
                  padding: "2px 6px",
                }}
              >
                RGB-only
              </span>
            )}
          </div>
        </label>
      </section>

      <section>
        <h2>Inputs</h2>
        <label className="field">
          <span>
            RGB image (PNG/JPG)
            <InfoTip text="Colour image of the scene. Used for detection and as the 2D backdrop." />
          </span>
          <input
            type="file"
            accept="image/png,image/jpeg"
            onChange={(e) => onRgb(e.target.files?.[0] ?? null)}
          />
          {rgbName && <span className="hint">{rgbName}</span>}
        </label>
        <label className="field">
          <span>
            Depth (uint16 mm PNG)
            <InfoTip text="Per-pixel depth as a 16-bit PNG in millimetres. FoundationPose needs metric depth — a normalized or 8-bit map gives wrong poses." />
          </span>
          <input
            type="file"
            accept="image/png"
            onChange={(e) => onDepth(e.target.files?.[0] ?? null)}
          />
          {depthName && <span className="hint">{depthName}</span>}
        </label>
        <label className="field">
          <span>
            Ground-truth overlay (JSON)
            <InfoTip text='Optional reference overlay shown alongside predictions (never sent to the pipeline). Schema: {"instances":[{"id","class","T_cam_obj":[[4x4 row-major, OpenCV cam frame, metres]],"mask_b64":"<PNG base64, optional>"}]}. Upload AFTER the RGB image.' />
          </span>
          <input
            type="file"
            accept="application/json,.json"
            onChange={(e) => onGtOverlay(e.target.files?.[0] ?? null)}
          />
          {gtName && <span className="hint">{gtName}</span>}
        </label>

        <button
          className="linkbtn convert-toggle"
          onClick={() => setShowConvert((s) => !s)}
        >
          {showConvert ? "▾ " : "▸ "}Convert a raw SDG frame
          <InfoTip text="Upload raw Isaac-Sim output (bbox_3d + optional instance PNG / instance_labels + world2cam_view) and the gateway converts it to the T_cam_obj overlay above. Uses the intrinsics below as K." />
        </button>
        {showConvert && (
          <div className="convert-box">
            <label className="field">
              <span>bbox_3d_*.json (required)</span>
              <input
                type="file"
                accept="application/json,.json"
                onChange={(e) => setRawBbox(e.target.files?.[0] ?? null)}
              />
            </label>
            <label className="field">
              <span>
                world2cam_view.txt (optional)
                <InfoTip text="Camera extrinsic (4x4 USD world→camera). Defaults to the GST_Scene Zivid extrinsic, which matches the bundled poc500 frames and most SDG runs against that scene. Upload one only to override for a different scene/camera." />
              </span>
              <input
                type="file"
                accept=".txt,text/plain"
                onChange={(e) => setRawWorld2cam(e.target.files?.[0] ?? null)}
              />
            </label>
            <label className="field">
              <span>
                instance_*.png (optional)
                <InfoTip text="Per-object instance segmentation. Needed to draw GT masks and to filter to visible objects. Without it you get poses only, for every bbox_3d row." />
              </span>
              <input
                type="file"
                accept="image/png"
                onChange={(e) => setRawInstance(e.target.files?.[0] ?? null)}
              />
            </label>
            <label className="field">
              <span>
                instance_labels_*.json (optional)
                <InfoTip text="Maps instance pixel-ids to objects for robust mask↔pose linkage. Without it, linkage falls back to 3D-centroid projection." />
              </span>
              <input
                type="file"
                accept="application/json,.json"
                onChange={(e) => setRawLabels(e.target.files?.[0] ?? null)}
              />
            </label>
            <button
              className="secondary"
              disabled={busy || !rawBbox}
              onClick={() =>
                onConvertRawGt({
                  bbox3d: rawBbox,
                  world2cam: rawWorld2cam,
                  instance: rawInstance,
                  instanceLabels: rawLabels,
                })
              }
            >
              Convert &amp; load
            </button>
            <span className="hint">Converts using the intrinsics below as K.</span>
          </div>
        )}

        <button className="secondary" onClick={onLoadSample} disabled={busy}>
          Load sample
        </button>
        <span className="hint">
          Bundled sim frame: RGB + depth + intrinsics + GT masks + GT overlay.
        </span>
      </section>

      <section>
        <h2>
          Intrinsics
          <InfoTip text="Pinhole camera parameters. They must match the camera that captured the depth map, or back-projection and poses will be wrong." />
        </h2>
        <div className="grid2">
          {INTRINSIC_KEYS.map((k) => (
            <label className="field" key={k}>
              <span>
                {k}
                <InfoTip text={INTRINSIC_TIPS[k]} />
              </span>
              <input
                type="number"
                step="0.001"
                value={intrinsics[k]}
                onChange={(e) =>
                  setIntrinsics({ ...intrinsics, [k]: Number(e.target.value) })
                }
              />
            </label>
          ))}
        </div>
      </section>

      <section>
        <h2>Parameters</h2>
        <div className="grid2">
          <label className="field">
            <span>
              iterations
              <InfoTip text="FoundationPose pose-refinement iterations per object. More = slightly more accurate but slower. Default 5." />
            </span>
            <input
              type="number"
              min={1}
              value={iterations}
              onChange={(e) => setIterations(Number(e.target.value))}
            />
          </label>
          <label className="field">
            <span>
              top_n
              <InfoTip text="Pose only the N highest-confidence detections (blank = all). Each posed instance adds ~2–3 s." />
            </span>
            <input
              type="number"
              min={1}
              placeholder="all"
              value={topN}
              onChange={(e) => setTopN(e.target.value)}
            />
          </label>
        </div>
        <label className="checkbox">
          <input
            type="checkbox"
            checked={wantPointcloud}
            onChange={(e) => setWantPointcloud(e.target.checked)}
          />
          <span>Request point cloud</span>
          <InfoTip text="Also return the depth map back-projected into a 3D point cloud (3D mode). Adds response size." />
        </label>
      </section>

      <button className="primary" onClick={onRun} disabled={busy}>
        {busy ? "Running..." : "Run"}
      </button>
      <p className="status">{status}</p>

      <section>
        <h2>Layers — {mode === "2d" ? "2D overlay" : "3D scene"}</h2>
        {mode === "2d" ? (
          <>
            {layerBox("backdrop", "Image backdrop", "Show the input RGB photo behind the overlay.")}
            <div className="layergroup">
              <span className="grouplabel">
                <span className="dot" style={{ background: COLORS.pred }} /> Prediction
              </span>
              {layerBox("predMask", "Masks", "Predicted instance masks (translucent magenta).")}
              {layerBox("predAxes", "Axes", "Predicted pose triad (X red, Y green, Z blue), solid.")}
            </div>
            <div className="layergroup">
              <span className="grouplabel">
                <span className="dot" style={{ background: COLORS.gt }} /> Ground truth
              </span>
              {layerBox("gtMask", "Masks", "Ground-truth instance masks (translucent green).")}
              {layerBox("gtAxes", "Axes", "Ground-truth pose triad (RGB = XYZ), dashed.")}
            </div>
          </>
        ) : (
          <>
            {layerBox("grid", "Grid", "Reference grid at the scene centre for orientation.")}
            {layerBox("pointcloud", "Point cloud", "Depth map back-projected as 3D points. Requires 'Request point cloud'.")}
            <div className="layergroup">
              <span className="grouplabel">Scene props</span>
              {layerBox("table", "Table / carts", "Static Basiswagen carts the parts rest on (decimated from GST_Scene.usd), placed in world coordinates. Solid background context.")}
              {layerBox("arm", "Robot arm", "Static NEURA LARA5 arm (decimated from GST_Scene.usd). Off by default: from the camera angle it occludes ~60% of the frame (ADR-008). Orbit to see around it.")}
            </div>
            <div className="layergroup">
              <span className="grouplabel">
                <span className="dot" style={{ background: COLORS.pred }} /> Prediction
              </span>
              {layerBox("predMesh", "Mesh", "Predicted-pose CAD mesh (semi-transparent magenta).")}
              {layerBox("predAxes", "Axes", "Predicted pose triad at each object.")}
            </div>
            <div className="layergroup">
              <span className="grouplabel">
                <span className="dot" style={{ background: COLORS.gt }} /> Ground truth
              </span>
              {layerBox("gtMesh", "Mesh", "Ground-truth-pose CAD mesh (semi-transparent green).")}
              {layerBox("gtAxes", "Axes", "Ground-truth pose triad at each object.")}
            </div>
          </>
        )}
      </section>

      {objectList("Predictions", predObjects, COLORS.pred)}
      {objectList("Ground truth", gtObjects, COLORS.gt)}
    </aside>
  );
}
