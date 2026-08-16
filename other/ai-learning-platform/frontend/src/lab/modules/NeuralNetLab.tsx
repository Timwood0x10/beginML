import { useEffect, useRef, useState, useCallback } from "react";
import { api } from "../../api";
import type { LabParams, LabResult } from "../types";
import {
  setupCanvas,
  makeScale,
  themeVar,
  type Domain,
  type Scale,
} from "../canvas";
import { useCanvasDrag } from "../useCanvasDrag";
import { useTheme } from "../../hooks/useTheme";

interface Point {
  x: number;
  y: number;
  cls: number;
  prob: number;
}
interface NnResult extends LabResult {
  dataset: string;
  points: Point[];
  grid: { x: number[]; y: number[]; z: number[][] };
  domain: Domain;
  losses: number[];
  finalLoss: number | null;
  accuracy: number;
  architecture: string;
  epochs: number;
  learningRate: number;
}

const COLORS = ["#C8604A", "#5B6BB0"];

function drawBoundary(
  ctx: CanvasRenderingContext2D,
  s: Scale,
  grid: NnResult["grid"],
) {
  const { x, y, z } = grid;
  for (let i = 0; i < x.length - 1; i++) {
    for (let j = 0; j < y.length - 1; j++) {
      const v = z[j][i];
      const t = Math.max(0, Math.min(1, v));
      const r = Math.round(200 + (91 - 200) * t);
      const g = Math.round(96 + (107 - 96) * t);
      const b = Math.round(74 + (176 - 74) * t);
      ctx.fillStyle = `rgba(${r},${g},${b},0.35)`;
      ctx.fillRect(
        s.px(x[i]),
        s.py(y[j + 1]),
        s.px(x[i + 1]) - s.px(x[i]) + 1,
        s.py(y[j]) - s.py(y[j + 1]) + 1,
      );
    }
  }
}

function drawContourLine(
  ctx: CanvasRenderingContext2D,
  s: Scale,
  grid: NnResult["grid"],
  level: number,
) {
  const { x, y, z } = grid;
  ctx.strokeStyle = "rgba(29,28,22,0.5)";
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  for (let i = 0; i < x.length - 1; i++) {
    for (let j = 0; j < y.length - 1; j++) {
      const v00 = z[j][i],
        v10 = z[j][i + 1],
        v11 = z[j + 1][i + 1],
        v01 = z[j + 1][i];
      const idx =
        (v00 > level ? 1 : 0) |
        (v10 > level ? 2 : 0) |
        (v11 > level ? 4 : 0) |
        (v01 > level ? 8 : 0);
      if (idx === 0 || idx === 15) continue;
      const xL = s.px(x[i]),
        xR = s.px(x[i + 1]);
      const yT = s.py(y[j]),
        yB = s.py(y[j + 1]);
      const safe = (a: number, b: number) => (a === b ? a + 1e-9 : b);
      const topX = xL + ((level - v00) / safe(v00, v10 - v00)) * (xR - xL);
      const rightY = yT + ((level - v10) / safe(v10, v11 - v10)) * (yB - yT);
      const bottomX = xR - ((level - v11) / safe(v11, v01 - v11)) * (xR - xL);
      const leftY = yB - ((level - v01) / safe(v01, v00 - v01)) * (yB - yT);
      const seg: [number, number, number, number][] = [];
      switch (idx) {
        case 1:
        case 14:
          seg.push([topX, yT, xL, leftY]);
          break;
        case 2:
        case 13:
          seg.push([topX, yT, xR, rightY]);
          break;
        case 3:
        case 12:
          seg.push([xL, leftY, xR, rightY]);
          break;
        case 4:
        case 11:
          seg.push([xR, rightY, bottomX, yB]);
          break;
        case 6:
        case 9:
          seg.push([topX, yT, bottomX, yB]);
          break;
        case 7:
        case 8:
          seg.push([xL, leftY, bottomX, yB]);
          break;
        case 5:
          seg.push([topX, yT, xL, leftY]);
          seg.push([xR, rightY, bottomX, yB]);
          break;
        case 10:
          seg.push([topX, yT, xR, rightY]);
          seg.push([xL, leftY, bottomX, yB]);
          break;
      }
      seg.forEach(([x1, y1, x2, y2]) => {
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
      });
    }
  }
  ctx.stroke();
}

function drawScene(
  canvas: HTMLCanvasElement,
  r: NnResult,
  hover: { x: number; y: number } | null,
) {
  const W = 560,
    H = 480;
  const ctx = setupCanvas(canvas, W, H);
  const s = makeScale(
    ctx,
    { x: r.domain.x as [number, number], y: r.domain.y as [number, number] },
    { l: 20, r: 20, t: 20, b: 20 },
  );
  ctx.fillStyle = themeVar("--ailearn-background", "#F7F0E3");
  ctx.fillRect(0, 0, W, H);
  drawBoundary(ctx, s, r.grid);
  drawContourLine(ctx, s, r.grid, 0.5);

  if (hover) {
    ctx.strokeStyle = "rgba(99,91,79,0.4)";
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.moveTo(s.px(hover.x), s.py(r.domain.y[0]));
    ctx.lineTo(s.px(hover.x), s.py(r.domain.y[1]));
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(s.px(r.domain.x[0]), s.py(hover.y));
    ctx.lineTo(s.px(r.domain.x[1]), s.py(hover.y));
    ctx.stroke();
    ctx.setLineDash([]);
  }

  r.points.forEach((p) => {
    ctx.fillStyle = COLORS[p.cls] ?? "#8a8376";
    ctx.beginPath();
    ctx.arc(s.px(p.x), s.py(p.y), 5, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = themeVar("--ailearn-background", "#F7F0E3");
    ctx.lineWidth = 1.5;
    ctx.stroke();
  });
}

function drawLoss(canvas: HTMLCanvasElement, r: NnResult) {
  const W = 560,
    H = 180;
  const ctx = setupCanvas(canvas, W, H);
  const pad = { l: 48, r: 16, t: 16, b: 28 };
  const maxLoss = Math.max(...r.losses, 0.01);
  const s = makeScale(
    ctx,
    { x: [0, r.losses.length - 1], y: [0, maxLoss * 1.1] },
    pad,
  );
  const dark = document.documentElement.classList.contains("dark");
  ctx.fillStyle = themeVar("--ailearn-background", "#F7F0E3");
  ctx.fillRect(0, 0, W, H);

  ctx.strokeStyle = dark ? "rgba(255,255,255,0.08)" : "rgba(125,118,109,0.12)";
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const cy = s.py((i / 4) * maxLoss * 1.1);
    ctx.beginPath();
    ctx.moveTo(pad.l, cy);
    ctx.lineTo(W - pad.r, cy);
    ctx.stroke();
  }

  ctx.strokeStyle = "#7A5C36";
  ctx.lineWidth = 2;
  ctx.beginPath();
  r.losses.forEach((v, i) => {
    const cx = s.px(i),
      cy = s.py(v);
    if (i === 0) ctx.moveTo(cx, cy);
    else ctx.lineTo(cx, cy);
  });
  ctx.stroke();

  ctx.fillStyle = dark ? "#A99B82" : "#8A7A61";
  ctx.font = "11px Manrope";
  ctx.fillText("epoch", W - 50, H - 8);
  ctx.save();
  ctx.translate(12, H / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText("loss", 0, 0);
  ctx.restore();
}

export default function NeuralNetLab({
  params,
}: {
  result: LabResult | null;
  loading: boolean;
  error: string | null;
  onAction: (k: string) => void;
  params: LabParams;
  setParams: (p: LabParams) => void;
}) {
  const { theme, palette } = useTheme();
  const boundaryRef = useRef<HTMLCanvasElement>(null);
  const lossRef = useRef<HTMLCanvasElement>(null);
  const scaleRef = useRef<Scale | null>(null);
  const [data, setData] = useState<NnResult | null>(null);
  const [customPoints, setCustomPoints] = useState<
    { x: number; y: number; cls: number }[]
  >([]);
  const [cls, setCls] = useState<0 | 1>(1);
  const [hover, setHover] = useState<{ x: number; y: number } | null>(null);
  const [computing, setComputing] = useState(false);

  const hidden = Number(params.hidden ?? 8);
  const lr = Number(params.lr ?? 0.5);
  const epochs = Number(params.epochs ?? 200);
  const dataset = String(params.dataset ?? "moons");

  const recompute = useCallback(
    async (pts: { x: number; y: number; cls: number }[]) => {
      setComputing(true);
      try {
        const res = (await api.lab.compute("neural-net", {
          dataset,
          hidden,
          lr,
          epochs,
          samples: Number(params.samples ?? 120),
          noise: Number(params.noise ?? 0),
          seed: Number(params.seed ?? 1),
          points: pts,
        })) as NnResult;
        setData(res);
      } finally {
        setComputing(false);
      }
      // eslint-disable-next-line react-hooks/exhaustive-deps
    },
    [dataset, hidden, lr, epochs, params.samples, params.noise, params.seed],
  );

  useEffect(() => {
    recompute(customPoints);
  }, [recompute, customPoints]);

  const r = data;

  useEffect(() => {
    if (r && boundaryRef.current) {
      drawScene(boundaryRef.current, r, hover);
      const ctx = boundaryRef.current.getContext("2d")!;
      scaleRef.current = makeScale(
        ctx,
        {
          x: r.domain.x as [number, number],
          y: r.domain.y as [number, number],
        },
        { l: 20, r: 20, t: 20, b: 20 },
      );
    }
  }, [r, hover, theme, palette]);

  useEffect(() => {
    if (r && lossRef.current) drawLoss(lossRef.current, r);
  }, [r, theme, palette]);

  const { handlers } = useCanvasDrag({
    getScale: () => scaleRef.current,
    onDown: (x, y) => {
      setCustomPoints((prev) => [...prev, { x, y, cls }]);
      return true;
    },
    onHover: (x, y) => {
      if (r && !Number.isNaN(x)) setHover({ x, y });
    },
    onDrag: () => {},
    onUp: () => {},
  });

  const undo = () => setCustomPoints((prev) => prev.slice(0, -1));
  const clear = () => setCustomPoints([]);
  const demo = () => setCustomPoints([]);

  return (
    <div className="flex flex-col gap-4">
      <div className="bg-surface-container-lowest dark:bg-dark-surface rounded-3xl p-4 md:p-6 shadow-ambient dark:shadow-dark-ambient border border-outline-variant/40 dark:border-white/10">
        <div className="flex flex-wrap items-center justify-between gap-3 mb-4">
          <h3 className="font-headline text-lg text-on-surface dark:text-inverse-on-surface inline-flex items-center gap-2 capitalize">
            <span
              className="material-symbols-outlined"
              style={{ fontSize: 20 }}
            >
              bubble_chart
            </span>
            {customPoints.length > 0 ? "custom" : dataset} dataset —{" "}
            {r?.architecture ?? "2-?-1"} MLP
            {computing && (
              <span className="w-3 h-3 rounded-full border-2 border-outline-variant border-t-primary animate-spin" />
            )}
          </h3>
          <div className="flex flex-wrap gap-2">
            <button
              onClick={() => setCls(1)}
              className={`px-3 py-1.5 rounded-lg text-caption font-semibold border transition ${cls === 1 ? "text-white border-transparent" : "bg-surface-container dark:bg-white/5 text-on-surface-variant dark:text-outline border-outline-variant/60 dark:border-white/10"}`}
              style={cls === 1 ? { background: COLORS[1] } : undefined}
            >
              + class
            </button>
            <button
              onClick={() => setCls(0)}
              className={`px-3 py-1.5 rounded-lg text-caption font-semibold border transition ${cls === 0 ? "text-white border-transparent" : "bg-surface-container dark:bg-white/5 text-on-surface-variant dark:text-outline border-outline-variant/60 dark:border-white/10"}`}
              style={cls === 0 ? { background: COLORS[0] } : undefined}
            >
              − class
            </button>
            <span className="w-px bg-outline-variant/60 mx-1" />
            <button
              onClick={undo}
              className="px-3 py-1.5 rounded-lg text-caption font-semibold bg-surface-container dark:bg-white/5 text-on-surface-variant dark:text-outline border border-outline-variant/60 dark:border-white/10 hover:text-primary"
            >
              undo
            </button>
            <button
              onClick={demo}
              className="px-3 py-1.5 rounded-lg text-caption font-semibold bg-surface-container dark:bg-white/5 text-on-surface-variant dark:text-outline border border-outline-variant/60 dark:border-white/10 hover:text-primary"
            >
              generated
            </button>
            <button
              onClick={clear}
              className="px-3 py-1.5 rounded-lg text-caption font-semibold bg-error-container text-on-error-container hover:opacity-90"
            >
              clear
            </button>
          </div>
        </div>
        <div className="w-full overflow-x-auto flex justify-center">
          <canvas
            ref={boundaryRef}
            className="rounded-2xl cursor-crosshair touch-none"
            {...handlers}
          />
        </div>
        <p className="mt-3 text-caption text-on-surface-variant dark:text-outline">
          Click the canvas to place a point of the selected class. Points you
          place are kept — switch to "generated" to go back to the synthetic
          dataset.
        </p>
      </div>

      <div className="bg-surface-container-lowest dark:bg-dark-surface rounded-3xl p-4 md:p-6 shadow-ambient dark:shadow-dark-ambient border border-outline-variant/40 dark:border-white/10">
        <h4 className="font-label-md text-label-md uppercase tracking-wider text-on-surface dark:text-dark-on-surface mb-3">
          Training loss
        </h4>
        <div className="w-full overflow-x-auto flex justify-center">
          <canvas ref={lossRef} className="rounded-2xl" />
        </div>
      </div>

      {r && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <Stat label="accuracy" value={`${(r.accuracy * 100).toFixed(1)}%`} />
          <Stat label="final loss" value={r.finalLoss?.toFixed(4) ?? "—"} />
          <Stat label="epochs" value={String(r.epochs)} />
          <Stat label="learning rate" value={r.learningRate.toFixed(3)} />
        </div>
      )}
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
