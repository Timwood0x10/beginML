import { useEffect, useRef, useState } from "react";
import type { LabResult, LabParams } from "../types";
import {
  setupCanvas,
  makeScale,
  drawAxes,
  themeVar,
  type Domain,
  type Scale,
} from "../canvas";
import { useCanvasDrag } from "../useCanvasDrag";
import { useTheme } from "../../hooks/useTheme";

interface Point {
  x: number;
  y: number;
  dy: number;
}
interface ActResult extends LabResult {
  x: number[];
  y: number[];
  dy: number[];
  domain: Domain;
  point: Point;
  tangent: { x: number[]; y: number[] };
  formula: string;
  function: string;
}

function drawCurve(
  ctx: CanvasRenderingContext2D,
  s: Scale,
  xs: number[],
  ys: number[],
  color: string,
  width: number,
) {
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.beginPath();
  xs.forEach((x, i) => {
    const cx = s.px(x),
      cy = s.py(ys[i]);
    if (i === 0) ctx.moveTo(cx, cy);
    else ctx.lineTo(cx, cy);
  });
  ctx.stroke();
}

function drawActivation(
  canvas: HTMLCanvasElement,
  r: ActResult,
  hoverX: number | null,
) {
  const W = 640,
    H = 420;
  const ctx = setupCanvas(canvas, W, H);
  const s = makeScale(
    ctx,
    { x: r.domain.x as [number, number], y: r.domain.y as [number, number] },
    { l: 48, r: 20, t: 20, b: 36 },
  );
  const dark = document.documentElement.classList.contains("dark");
  ctx.fillStyle = themeVar("--ailearn-background", "#F7F0E3");
  ctx.fillRect(0, 0, W, H);
  drawAxes(
    ctx,
    s,
    { x: r.domain.x as [number, number], y: r.domain.y as [number, number] },
    {
      color: dark ? "#A99B82" : "#8A7A61",
      gridColor: dark ? "rgba(255,255,255,0.06)" : "rgba(125,118,109,0.13)",
    },
  );

  drawCurve(ctx, s, r.x, r.y, "#7A5C36", 2.5);
  drawCurve(ctx, s, r.x, r.dy, "#C8604A", 1.8);
  ctx.setLineDash([5, 4]);
  drawCurve(ctx, s, r.tangent.x, r.tangent.y, "#5B6BB0", 1.5);
  ctx.setLineDash([]);

  // hover guide line
  if (hoverX !== null) {
    ctx.strokeStyle = "rgba(99,91,79,0.4)";
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.moveTo(s.px(hoverX), s.py(r.domain.y[0]));
    ctx.lineTo(s.px(hoverX), s.py(r.domain.y[1]));
    ctx.stroke();
    ctx.setLineDash([]);
  }

  // probe point
  const px = s.px(r.point.x),
    py = s.py(r.point.y);
  ctx.fillStyle = "#2f6b3e";
  ctx.beginPath();
  ctx.arc(px, py, 7, 0, Math.PI * 2);
  ctx.fill();
  ctx.strokeStyle = themeVar("--ailearn-background", "#F7F0E3");
  ctx.lineWidth = 2.5;
  ctx.stroke();
  ctx.fillStyle = dark ? "#C9BCA6" : "#54483A";
  ctx.font = "600 12px Manrope";
  ctx.fillText(
    `(${r.point.x.toFixed(2)}, ${r.point.y.toFixed(2)})`,
    px + 10,
    py - 8,
  );

  ctx.font = "600 13px Manrope";
  ctx.fillStyle = "#7A5C36";
  ctx.fillText(
    "f(x)",
    s.px(r.x[r.x.length - 2]) + 4,
    s.py(r.y[r.y.length - 2]),
  );
  ctx.fillStyle = "#C8604A";
  ctx.fillText(
    "f'(x)",
    s.px(r.x[r.x.length - 2]) + 4,
    s.py(r.dy[r.dy.length - 2]),
  );
}

export default function ActivationLab({
  result,
  params,
  setParams,
}: {
  result: LabResult | null;
  loading: boolean;
  error: string | null;
  onAction: (k: string) => void;
  params: LabParams;
  setParams: (p: LabParams) => void;
}) {
  const ref = useRef<HTMLCanvasElement>(null);
  const scaleRef = useRef<Scale | null>(null);
  const r = result as ActResult | null;
  const [hover, setHover] = useState<number | null>(null);
  const { theme, palette } = useTheme();

  useEffect(() => {
    if (r && ref.current) {
      drawActivation(ref.current, r, hover);
      const ctx = ref.current.getContext("2d")!;
      scaleRef.current = makeScale(
        ctx,
        {
          x: r.domain.x as [number, number],
          y: r.domain.y as [number, number],
        },
        { l: 48, r: 20, t: 20, b: 36 },
      );
    }
  }, [r, hover, theme, palette]);

  const clampX = (x: number) => {
    if (!r) return x;
    const lo = r.domain.x[0],
      hi = r.domain.x[1];
    return Math.max(lo + 0.01, Math.min(hi - 0.01, x));
  };

  const { handlers } = useCanvasDrag({
    getScale: () => scaleRef.current,
    onDown: (x) => {
      setParams({ ...params, point: parseFloat(clampX(x).toFixed(2)) });
      return true;
    },
    onDrag: (x) => {
      setParams({ ...params, point: parseFloat(clampX(x).toFixed(2)) });
    },
    onHover: (x) => {
      if (r) setHover(clampX(x));
    },
    onUp: () => {},
  });

  if (!r) return null;

  return (
    <div className="flex flex-col gap-4">
      <div className="bg-surface-container-lowest dark:bg-dark-surface rounded-3xl p-4 md:p-6 shadow-ambient dark:shadow-dark-ambient border border-outline-variant/40 dark:border-white/10">
        <div className="flex flex-wrap items-center justify-between gap-3 mb-4">
          <h3 className="font-headline text-lg text-on-surface dark:text-inverse-on-surface inline-flex items-center gap-2 capitalize">
            <span
              className="material-symbols-outlined"
              style={{ fontSize: 20 }}
            >
              show_chart
            </span>
            {r.function}
            <span className="text-caption font-normal normal-case text-outline ml-2">
              (drag the green point)
            </span>
          </h3>
          <code className="font-mono text-sm text-primary dark:text-inverse-primary bg-primary-fixed/40 dark:bg-white/5 px-3 py-1.5 rounded-lg">
            {r.formula}
          </code>
        </div>
        <div className="w-full overflow-x-auto flex justify-center">
          <canvas
            ref={ref}
            className="rounded-2xl cursor-crosshair touch-none"
            {...handlers}
          />
        </div>
      </div>
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <Stat label="x" value={r.point.x.toFixed(3)} />
        <Stat label="f(x)" value={r.point.y.toFixed(3)} />
        <Stat label="f'(x)" value={r.point.dy.toFixed(3)} />
        <Stat
          label="slope angle"
          value={`${((Math.atan(r.point.dy) * 180) / Math.PI).toFixed(1)}°`}
        />
      </div>
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="bg-surface-container-lowest dark:bg-dark-surface rounded-2xl p-4 border border-outline-variant/40 dark:border-white/10 text-center">
      <div className="font-mono text-xl font-bold text-primary dark:text-inverse-primary">
        {value}
      </div>
      <div className="text-caption text-outline mt-1 uppercase tracking-wider">
        {label}
      </div>
    </div>
  );
}
