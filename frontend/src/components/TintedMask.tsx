import { useEffect, useState } from "react";
import { hexToRgb } from "../geometry";

/**
 * Recolours a white-on-black instance-mask PNG into a translucent coloured
 * <image> for the 2D SVG overlay. Object pixels (luminance > 127) become
 * `color` at `opacity`; background becomes fully transparent. The tinted
 * data-URL is produced once per (mask, color) via an offscreen canvas.
 */
export default function TintedMask({
  mask_b64,
  color,
  W,
  H,
  opacity = 0.45,
}: {
  mask_b64: string;
  color: string;
  W: number;
  H: number;
  opacity?: number;
}) {
  const [url, setUrl] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    const img = new Image();
    img.onload = () => {
      const c = document.createElement("canvas");
      c.width = W;
      c.height = H;
      const ctx = c.getContext("2d");
      if (!ctx) return;
      ctx.drawImage(img, 0, 0, W, H);
      const d = ctx.getImageData(0, 0, W, H);
      const [r, g, b] = hexToRgb(color);
      const a = Math.round(opacity * 255);
      const px = d.data;
      for (let i = 0; i < px.length; i += 4) {
        if (px[i] > 127) {
          px[i] = r;
          px[i + 1] = g;
          px[i + 2] = b;
          px[i + 3] = a;
        } else {
          px[i + 3] = 0;
        }
      }
      ctx.putImageData(d, 0, 0);
      if (!cancelled) setUrl(c.toDataURL());
    };
    img.src = `data:image/png;base64,${mask_b64}`;
    return () => {
      cancelled = true;
    };
  }, [mask_b64, color, W, H, opacity]);

  if (!url) return null;
  return <image href={url} x={0} y={0} width={W} height={H} />;
}
