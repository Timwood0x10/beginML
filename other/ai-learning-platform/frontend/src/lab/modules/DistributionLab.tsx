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

interface Stats {
  mean: number;
  std: number;
  samples: number;
}
interface DistResult extends LabResult {
  discrete: boolean;
  distribution: string;
  name: string;
  formula: string;
  x: number[];
  y: number[];
  histEdges: number[];
  histCounts: number[];
  domain: Domain;
  stats: Stats;
}

// Nearest-sample lookup of the PDF/PMF at a given x (data comes from backend).
function sampleAt(r: DistResult, xv: number): { x: number; y: number } {
  let idx = 0;
  let best = Infinity;
  r.x.forEach((xi, i) => {
    const d = Math.abs(xi - xv);
    if (d < best) {
      best = d;
      idx = i;
    }
  });
  return { x: r.x[idx], y: r.y[idx] };
}

function drawDist(
  canvas: HTMLCanvasElement,
  r: DistResult,
  hoverX: number | null,
) {
  const W = 640,
    H = 420;
  const ctx = setupCanvas(canvas, W, H);
  const s = makeScale(
    ctx,
    { x: r.domain.x as [number, number], y: r.domain.y as [number, number] },
    { l: 52, r: 20, t: 20, b: 40 },
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

  // histogram bars
  ctx.fillStyle = dark ? "rgba(91,107,176,0.45)" : "rgba(91,107,176,0.35)";
  for (let i = 0; i < r.histCounts.length; i++) {
    const e0 = r.histEdges[i],
      e1 = r.histEdges[i + 1];
    const w = s.px(e1) - s.px(e0);
    const h = s.py(0) - s.py(r.histCounts[i]);
    if (h > 0) ctx.fillRect(s.px(e0) + 1, s.py(r.histCounts[i]), w - 2, h);
  }

  // density curve / pmf stems
  if (r.discrete) {
    ctx.strokeStyle = "#C8604A";
    ctx.lineWidth = 2;
    r.x.forEach((xi, i) => {
      ctx.beginPath();
      ctx.moveTo(s.px(xi), s.py(0));
      ctx.lineTo(s.px(xi), s.py(r.y[i]));
      ctx.stroke();
    });
    ctx.fillStyle = "#C8604A";
    r.x.forEach((xi, i) => {
      ctx.beginPath();
      ctx.arc(s.px(xi), s.py(r.y[i]), 3, 0, Math.PI * 2);
      ctx.fill();
    });
  } else {
    ctx.strokeStyle = "#7A5C36";
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    r.x.forEach((xi, i) => {
      const cx = s.px(xi),
        cy = s.py(r.y[i]);
      if (i === 0) ctx.moveTo(cx, cy);
      else ctx.lineTo(cx, cy);
    });
    ctx.stroke();
  }

  // hover guide + value marker
  if (hoverX !== null) {
    const sm = sampleAt(r, hoverX);
    const cx = s.px(sm.x),
      cy = s.py(sm.y);
    ctx.strokeStyle = "rgba(99,91,79,0.45)";
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.moveTo(cx, s.py(r.domain.y[0]));
    ctx.lineTo(cx, s.py(r.domain.y[1]));
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = "#2f6b3e";
    ctx.beginPath();
    ctx.arc(cx, cy, 5, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = themeVar("--ailearn-background", "#F7F0E3");
    ctx.lineWidth = 2;
    ctx.stroke();
    ctx.fillStyle = dark ? "#C9BCA6" : "#54483A";
    ctx.font = "600 12px Manrope";
    ctx.fillText(
      `${r.discrete ? "P" : "f"}(${sm.x.toFixed(2)}) = ${sm.y.toFixed(4)}`,
      cx + 10,
      cy - 8,
    );
  }

  ctx.fillStyle = dark ? "#A99B82" : "#8A7A61";
  ctx.font = "12px Manrope";
  ctx.fillText("x", W - 24, H - 8);
  ctx.save();
  ctx.translate(14, H / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText(r.discrete ? "probability" : "density", 0, 0);
  ctx.restore();
}

export default function DistributionLab({
  result,
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
  const r = result as DistResult | null;
  const [hoverX, setHoverX] = useState<number | null>(null);
  const { theme, palette } = useTheme();

  useEffect(() => {
    if (r && ref.current) {
      drawDist(ref.current, r, hoverX);
      const ctx = ref.current.getContext("2d")!;
      scaleRef.current = makeScale(
        ctx,
        {
          x: r.domain.x as [number, number],
          y: r.domain.y as [number, number],
        },
        { l: 52, r: 20, t: 20, b: 40 },
      );
    }
  }, [r, hoverX, theme, palette]);

  const { handlers } = useCanvasDrag({
    getScale: () => scaleRef.current,
    onDown: () => true,
    onDrag: () => {},
    onHover: (x) => {
      if (r && !Number.isNaN(x))
        setHoverX(Math.max(r.domain.x[0], Math.min(r.domain.x[1], x)));
    },
    onUp: () => {},
  });

  if (!r) return null;

  const sm = hoverX !== null ? sampleAt(r, hoverX) : null;

  return (
    <div className="flex flex-col gap-4">
      <div className="bg-surface-container-lowest dark:bg-dark-surface rounded-3xl p-4 md:p-6 shadow-ambient dark:shadow-dark-ambient border border-outline-variant/40 dark:border-white/10">
        <div className="flex flex-wrap items-center justify-between gap-3 mb-4">
          <h3 className="font-headline text-lg text-on-surface dark:text-inverse-on-surface inline-flex items-center gap-2">
            <span
              className="material-symbols-outlined"
              style={{ fontSize: 20 }}
            >
              bar_chart
            </span>
            {r.name}
            <span className="text-caption font-normal normal-case text-outline ml-2">
              (hover to inspect)
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
        {sm && (
          <p className="mt-3 text-caption text-on-surface-variant dark:text-outline">
            {r.discrete ? "P" : "f"}({sm.x.toFixed(2)}) ={" "}
            <span className="font-mono text-primary dark:text-inverse-primary">
              {sm.y.toFixed(4)}
            </span>
          </p>
        )}
      </div>
      <div className="grid grid-cols-3 gap-3">
        <Stat label="sample mean" value={r.stats.mean.toFixed(3)} />
        <Stat label="sample std" value={r.stats.std.toFixed(3)} />
        <Stat label="samples" value={String(r.stats.samples)} />
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
