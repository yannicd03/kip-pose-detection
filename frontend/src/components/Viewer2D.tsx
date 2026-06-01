import type { Intrinsics } from "../api";
import { COLORS, projectTriad, type VizObject } from "../geometry";
import TintedMask from "./TintedMask";

export interface Show2D {
  backdrop: boolean;
  predMask: boolean;
  predAxes: boolean;
  gtMask: boolean;
  gtAxes: boolean;
}

/**
 * 2D overlay: the RGB backdrop with per-object instance masks (translucent,
 * green = GT / magenta = pred) and projected coordinate triads (RGB = XYZ;
 * GT dashed, prediction solid — same convention as fp_viz.py). No mesh.
 *
 * The SVG uses the image's pixel space as its viewBox and the same
 * letterboxing (`xMidYMid meet`) as the backdrop's `object-fit: contain`, so
 * overlay and photo stay pixel-aligned at any container size.
 */
export default function Viewer2D({
  W,
  H,
  K,
  objects,
  show,
  hidden,
  rgbUrl,
}: {
  W: number;
  H: number;
  K: Intrinsics;
  objects: VizObject[];
  show: Show2D;
  hidden: Set<string>;
  rgbUrl: string | null;
}) {
  const visible = objects.filter((o) => !hidden.has(o.key));

  return (
    <div className="viewer">
      {show.backdrop && rgbUrl && (
        <img className="backdrop" src={rgbUrl} alt="RGB input" />
      )}
      <svg
        className="overlay2d"
        viewBox={`0 0 ${W} ${H}`}
        preserveAspectRatio="xMidYMid meet"
      >
        {/* masks first (under the triads) */}
        {visible.map((o) => {
          const on = o.kind === "gt" ? show.gtMask : show.predMask;
          if (!on || !o.mask) return null;
          return (
            <TintedMask
              key={`${o.key}:mask`}
              mask_b64={o.mask}
              color={o.kind === "gt" ? COLORS.gt : COLORS.pred}
              W={W}
              H={H}
            />
          );
        })}

        {/* coordinate triads on top */}
        {visible.map((o) => {
          const on = o.kind === "gt" ? show.gtAxes : show.predAxes;
          if (!on) return null;
          const t = projectTriad(o.T, K);
          if (!t.ok) return null;
          const dash = o.kind === "gt" ? "6,4" : undefined;
          return (
            <g key={`${o.key}:axes`} strokeWidth={2.2} fill="none">
              <line x1={t.o.u} y1={t.o.v} x2={t.x.u} y2={t.x.v} stroke={COLORS.axisX} strokeDasharray={dash} />
              <line x1={t.o.u} y1={t.o.v} x2={t.y.u} y2={t.y.v} stroke={COLORS.axisY} strokeDasharray={dash} />
              <line x1={t.o.u} y1={t.o.v} x2={t.z.u} y2={t.z.v} stroke={COLORS.axisZ} strokeDasharray={dash} />
              <circle cx={t.o.u} cy={t.o.v} r={2.6} fill="#fff" stroke="none" />
            </g>
          );
        })}
      </svg>
    </div>
  );
}
