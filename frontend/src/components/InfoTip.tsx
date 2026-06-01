import { useCallback, useRef, useState } from "react";
import { createPortal } from "react-dom";

/**
 * Small "i" info button with a styled hover/focus tooltip. The bubble is
 * rendered through a portal to <body> and positioned with fixed coordinates
 * from the icon's bounding rect, so it is never clipped by the scrollable
 * panel (and is unaffected by any transformed ancestor). Accessible: the icon
 * carries the label and is keyboard-focusable.
 */
export default function InfoTip({ text }: { text: string }) {
  const ref = useRef<HTMLSpanElement>(null);
  const [pos, setPos] = useState<{ left: number; top: number; flip: boolean } | null>(
    null
  );

  const open = useCallback(() => {
    const r = ref.current?.getBoundingClientRect();
    if (!r) return;
    const BUBBLE_W = 240;
    // Prefer placing the bubble to the right of the icon; flip to the left if
    // it would overflow the viewport's right edge.
    const flip = r.right + 8 + BUBBLE_W > window.innerWidth;
    setPos({
      left: flip ? r.left - 8 : r.right + 8,
      top: r.top + r.height / 2,
      flip,
    });
  }, []);

  const close = useCallback(() => setPos(null), []);

  return (
    <span
      ref={ref}
      className="infotip"
      tabIndex={0}
      role="img"
      aria-label={`Info: ${text}`}
      onMouseEnter={open}
      onMouseLeave={close}
      onFocus={open}
      onBlur={close}
    >
      i
      {pos &&
        createPortal(
          <span
            className={`infotip-bubble${pos.flip ? " flip" : ""}`}
            role="tooltip"
            style={{ left: pos.left, top: pos.top }}
          >
            {text}
          </span>,
          document.body
        )}
    </span>
  );
}
