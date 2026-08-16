import { useEffect, useRef, useState } from "react";
import type { LabResult } from "../types";
import {
  setupCanvas,
  makeScale,
  drawAxes,
  themeVar,
  type Domain,
} from "../canvas";
import { useTheme } from "../../hooks/useTheme";

interface Window {
  position: number;
  window: number[];
  products: number[];
  value: number;
}
interface ConvResult extends LabResult {
  input: number[];
  kernel: number[];
  output: number[];
  windows: Window[];
  kernelName: string;
  kernelSize: number;
  domain: Domain;
}

export default function ConvolutionLab({
  result,
}: {
  result: LabResult | null;
}) {
  const ref = useRef<HTMLCanvasElement>(null);
  const r = result as ConvResult | null;
  const [pos, setPos] = useState(0);
  const [playing, setPlaying] = useState(false);
  const { theme, palette } = useTheme();

  // Clamp position when data changes
  useEffect(() => {
    if (r) setPos((p) => Math.min(p, r.output.length - 1));
  }, [r]);

  // Animation
  useEffect(() => {
    if (!playing || !r) return;
    const id = setInterval(() => {
      setPos((p) => (p + 1) % r.output.length);
    }, 350);
    return () => clearInterval(id);
  }, [playing, r]);

  // Canvas drawing
  useEffect(() => {
    if (!r || !ref.current) return;
    const W = 680,
      H = 360;
    const ctx = setupCanvas(ref.current, W, H);
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

    const k = r.kernelSize;
    // input stem plot
    r.input.forEach((v, i) => {
      const inWin = i >= pos && i < pos + k;
      ctx.strokeStyle = inWin
        ? "#C8604A"
        : dark
          ? "rgba(208,197,182,0.4)"
          : "rgba(99,91,79,0.4)";
      ctx.lineWidth = inWin ? 2.5 : 1.5;
      ctx.beginPath();
      ctx.moveTo(s.px(i), s.py(0));
      ctx.lineTo(s.px(i), s.py(v));
      ctx.stroke();
      ctx.fillStyle = inWin ? "#C8604A" : dark ? "#d0c5b6" : "#7A5C36";
      ctx.beginPath();
      ctx.arc(s.px(i), s.py(v), inWin ? 3.5 : 2.5, 0, Math.PI * 2);
      ctx.fill();
    });

    // output stem plot
    r.output.forEach((v, i) => {
      const isCurrent = i === pos;
      ctx.strokeStyle = isCurrent
        ? "#2f6b3e"
        : dark
          ? "rgba(91,107,176,0.45)"
          : "rgba(91,107,176,0.5)";
      ctx.lineWidth = isCurrent ? 2.5 : 1.5;
      ctx.beginPath();
      ctx.moveTo(s.px(i), s.py(0));
      ctx.lineTo(s.px(i), s.py(v));
      ctx.stroke();
      ctx.fillStyle = isCurrent ? "#2f6b3e" : "#5B6BB0";
      ctx.beginPath();
      ctx.arc(s.px(i), s.py(v), isCurrent ? 4 : 2.5, 0, Math.PI * 2);
      ctx.fill();
    });

    // legend
    ctx.font = "600 12px Manrope";
    ctx.fillStyle = "#C8604A";
    ctx.fillText("input", W - 120, 28);
    ctx.fillStyle = "#5B6BB0";
    ctx.fillText("output", W - 60, 28);
  }, [r, pos, theme, palette]);

  if (!r) return null;
  const win = r.windows[pos];

  return (
    <div className="flex flex-col gap-4">
      <div className="bg-surface-container-lowest dark:bg-dark-surface rounded-3xl p-4 md:p-6 shadow-ambient dark:shadow-dark-ambient border border-outline-variant/40 dark:border-white/10">
        <div className="flex flex-wrap items-center justify-between gap-3 mb-4">
          <h3 className="font-headline text-lg text-on-surface dark:text-inverse-on-surface capitalize inline-flex items-center gap-2">
            <span
              className="material-symbols-outlined"
              style={{ fontSize: 20 }}
            >
              stacked_bar_chart
            </span>
            {r.kernelName} kernel
          </h3>
          <div className="flex gap-2">
            <button
              onClick={() => setPlaying((p) => !p)}
              className="inline-flex items-center gap-1.5 px-4 py-2 rounded-xl bg-primary text-on-primary dark:bg-inverse-primary dark:text-inverse-surface text-label-md font-semibold hover:opacity-90"
            >
              <span
                className="material-symbols-outlined"
                style={{ fontSize: 18 }}
              >
                {playing ? "pause" : "play_arrow"}
              </span>
              {playing ? "Pause" : "Animate"}
            </button>
          </div>
        </div>
        <div className="w-full overflow-x-auto flex justify-center">
          <canvas ref={ref} className="rounded-2xl" />
        </div>
        <input
          type="range"
          min={0}
          max={Math.max(0, r.output.length - 1)}
          value={pos}
          onChange={(e) => setPos(parseInt(e.target.value))}
          className="ailearn-range w-full mt-4"
        />
      </div>

      {/* Kernel + product detail */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <DetailCard title="Kernel" values={r.kernel} color="#7A5C36" />
        <DetailCard
          title={`Window [${pos}..${pos + r.kernelSize - 1}]`}
          values={win?.window ?? []}
          color="#C8604A"
        />
        <DetailCard
          title="Products (sum → output)"
          values={win?.products ?? []}
          color="#2f6b3e"
          highlight={win?.value}
        />
      </div>
    </div>
  );
}

function DetailCard({
  title,
  values,
  color,
  highlight,
}: {
  title: string;
  values: number[];
  color: string;
  highlight?: number;
}) {
  return (
    <div className="bg-surface-container-lowest dark:bg-dark-surface rounded-2xl p-4 border border-outline-variant/40 dark:border-white/10">
      <h4 className="text-caption uppercase tracking-wider text-on-surface-variant dark:text-outline mb-3">
        {title}
      </h4>
      <div className="flex flex-wrap gap-1.5">
        {values.map((v, i) => (
          <span
            key={i}
            className="font-mono text-xs px-2 py-1 rounded-md"
            style={{
              background: `${color}22`,
              color,
            }}
          >
            {v.toFixed(2)}
          </span>
        ))}
      </div>
      {highlight !== undefined && (
        <div className="mt-3 pt-3 border-t border-outline-variant/40 dark:border-white/10">
          <span className="text-caption text-outline">sum = </span>
          <span className="font-mono text-lg font-bold" style={{ color }}>
            {highlight.toFixed(3)}
          </span>
        </div>
      )}
    </div>
  );
}
