import { useEffect, useRef, useState, useCallback } from 'react'
import { api } from '../../api'
import type { LabParams, LabResult } from '../types'
import { setupCanvas, makeScale, type Domain } from '../canvas'

interface SvmPoint { x: number; y: number; cls: number; support: boolean }
interface SvmResult extends LabResult {
  domain: Domain
  points: SvmPoint[]
  grid: { x: number[]; y: number[]; z: number[][] }
  coef: number[] | null
  intercept: number
  nSupport: number
  accuracy: number
  kernel: string
}

const COLORS = ['#C8604A', '#5B6BB0']

export default function SvmLab({
  params, setParams,
}: {
  result: LabResult | null; loading: boolean; error: string | null
  onAction: (key: string) => void
  params: LabParams; setParams: (p: LabParams) => void
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [points, setPoints] = useState<SvmPoint[]>([])
  const [cls, setCls] = useState<0 | 1>(1)
  const [data, setData] = useState<SvmResult | null>(null)
  const [computing, setComputing] = useState(false)

  const kernel = String(params.kernel ?? 'linear')
  const C = Number(params.c ?? 1.0)
  const gamma = Number(params.gamma ?? 0.8)
  const degree = Number(params.degree ?? 3)

  const recompute = useCallback(async (pts: SvmPoint[]) => {
    setComputing(true)
    try {
      const res = await api.lab.compute('svm', {
        kernel, c: C, gamma, degree,
        points: pts.map((p) => ({ x: p.x, y: p.y, cls: p.cls })),
      }) as SvmResult
      setData(res)
    } finally {
      setComputing(false)
    }
  }, [kernel, C, gamma, degree])

  // recompute when points or control params change
  useEffect(() => {
    recompute(points)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [points, kernel, C, gamma, degree])

  // reset when reset action fires (LabPage toggles params reference)
  useEffect(() => {
    // no-op: reset is handled by the clear button below
  }, [params.reset])

  const domain: Domain = data?.domain ?? { x: [-4, 4], y: [-4, 4] }

  // Draw
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const W = 620, H = 480
    const ctx = setupCanvas(canvas, W, H)
    const s = makeScale(ctx, domain, { l: 44, r: 20, t: 20, b: 36 })
    const dark = document.documentElement.classList.contains('dark')
    const bg = dark ? '#1A1917' : '#fef9ef'
    ctx.fillStyle = bg; ctx.fillRect(0, 0, W, H)

    // decision background: color regions by sign
    if (data) {
      const { x, y, z } = data.grid
      for (let i = 0; i < x.length - 1; i++) {
        for (let j = 0; j < y.length - 1; j++) {
          const v = z[j][i]
          // map decision value to a subtle class tint
          const t = Math.max(0, Math.min(1, (v + 1.5) / 3))
          const r = Math.round(242 + (200 - 242) * (1 - t))
          const gC = Math.round(237 + (107 - 237) * (1 - t))
          const bC = Math.round(227 + (74 - 227) * (1 - t))
          const r2 = Math.round(242 + (91 - 242) * t)
          const g2 = Math.round(237 + (107 - 237) * t)
          const b2 = Math.round(227 + (176 - 227) * t)
          const cr = v >= 0 ? r : r2
          const cg = v >= 0 ? gC : g2
          const cb = v >= 0 ? bC : b2
          ctx.fillStyle = `rgba(${cr},${cg},${cb},0.35)`
          ctx.fillRect(s.px(x[i]), s.py(y[j + 1]), s.px(x[i + 1]) - s.px(x[i]) + 1, s.py(y[j]) - s.py(y[j + 1]) + 1)
        }
      }

      // decision boundary (z≈0) and margins (z≈±1) via marching-ish contour
      drawContour(ctx, s, data.grid, 0, dark ? '#f5f0e6' : '#1d1c16', 2.2)
      drawContour(ctx, s, data.grid, 1, dark ? 'rgba(245,240,230,0.4)' : 'rgba(29,28,22,0.35)', 1)
      drawContour(ctx, s, data.grid, -1, dark ? 'rgba(245,240,230,0.4)' : 'rgba(29,28,22,0.35)', 1)
    }

    // axes
    ctx.strokeStyle = dark ? 'rgba(255,255,255,0.08)' : 'rgba(125,118,109,0.15)'
    ctx.lineWidth = 1
    if (domain.x[0] <= 0 && 0 <= domain.x[1]) {
      ctx.beginPath(); ctx.moveTo(s.px(0), s.py(domain.y[0])); ctx.lineTo(s.px(0), s.py(domain.y[1])); ctx.stroke()
    }
    if (domain.y[0] <= 0 && 0 <= domain.y[1]) {
      ctx.beginPath(); ctx.moveTo(s.px(domain.x[0]), s.py(0)); ctx.lineTo(s.px(domain.x[1]), s.py(0)); ctx.stroke()
    }

    // points
    const pts = data?.points ?? points
    pts.forEach((p) => {
      ctx.fillStyle = COLORS[p.cls]
      ctx.beginPath(); ctx.arc(s.px(p.x), s.py(p.y), p.support ? 8 : 6, 0, Math.PI * 2); ctx.fill()
      if (p.support) {
        ctx.strokeStyle = bg; ctx.lineWidth = 2.5; ctx.stroke()
        ctx.strokeStyle = COLORS[p.cls]; ctx.lineWidth = 1.5
        ctx.beginPath(); ctx.arc(s.px(p.x), s.py(p.y), 11, 0, Math.PI * 2); ctx.stroke()
      }
    })
  }, [data, points, domain])

  const handleClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current!
    const rect = canvas.getBoundingClientRect()
    // makeScale works in CSS pixels (DPR transform is applied internally), so
    // we pass the CSS-relative click position directly.
    const cssX = e.clientX - rect.left
    const cssY = e.clientY - rect.top
    const tmpCtx = canvas.getContext('2d')!
    const s = makeScale(tmpCtx, domain)
    const x = s.invx(cssX), y = s.invy(cssY)
    setPoints((prev) => [...prev, { x, y, cls, support: false }])
  }

  const clear = () => { setPoints([]); setData(null) }
  const undo = () => setPoints((prev) => prev.slice(0, -1))
  const demo = () => {
    const rng = mulberry32(0)
    const pos = Array.from({ length: 8 }, () => ({ x: -1 + rng() * 2, y: 0.6 + (rng() - 0.5) * 1.4, cls: 1, support: false }))
    const neg = Array.from({ length: 8 }, () => ({ x: 1 + (rng() - 0.5) * 2, y: -0.6 + (rng() - 0.5) * 1.4, cls: 0, support: false }))
    setPoints([...pos, ...neg])
  }

  // keep parent params stable; expose nothing extra
  void setParams

  return (
    <div className="flex flex-col gap-5">
      <div className="bg-surface-container-lowest dark:bg-dark-surface rounded-3xl p-4 md:p-6 shadow-ambient dark:shadow-dark-ambient border border-outline-variant/40 dark:border-white/5">
        <div className="flex flex-wrap items-center justify-between gap-3 mb-4">
          <h3 className="font-headline text-lg text-on-surface dark:text-inverse-on-surface inline-flex items-center gap-2">
            <span className="material-symbols-outlined" style={{ fontSize: 20 }}>linear_scale</span>
            Decision boundary
            {computing && <span className="w-3 h-3 rounded-full border-2 border-outline-variant border-t-primary animate-spin ml-1" />}
          </h3>
          <div className="flex flex-wrap gap-2">
            <button
              onClick={() => setCls(1)}
              className={`px-3 py-1.5 rounded-lg text-caption font-semibold border transition ${cls === 1 ? 'text-white border-transparent' : 'bg-surface-container dark:bg-white/5 text-on-surface-variant dark:text-outline border-outline-variant/60 dark:border-white/10'}`}
              style={cls === 1 ? { background: COLORS[1] } : undefined}
            >
              + class
            </button>
            <button
              onClick={() => setCls(0)}
              className={`px-3 py-1.5 rounded-lg text-caption font-semibold border transition ${cls === 0 ? 'text-white border-transparent' : 'bg-surface-container dark:bg-white/5 text-on-surface-variant dark:text-outline border-outline-variant/60 dark:border-white/10'}`}
              style={cls === 0 ? { background: COLORS[0] } : undefined}
            >
              − class
            </button>
            <span className="w-px bg-outline-variant/60 mx-1" />
            <button onClick={undo} className="px-3 py-1.5 rounded-lg text-caption font-semibold bg-surface-container dark:bg-white/5 text-on-surface-variant dark:text-outline border border-outline-variant/60 dark:border-white/10 hover:text-primary">undo</button>
            <button onClick={demo} className="px-3 py-1.5 rounded-lg text-caption font-semibold bg-surface-container dark:bg-white/5 text-on-surface-variant dark:text-outline border border-outline-variant/60 dark:border-white/10 hover:text-primary">demo</button>
            <button onClick={clear} className="px-3 py-1.5 rounded-lg text-caption font-semibold bg-error-container text-on-error-container hover:opacity-90">clear</button>
          </div>
        </div>

        <div className="w-full overflow-x-auto flex justify-center">
          <canvas
            ref={canvasRef}
            onClick={handleClick}
            className="rounded-2xl cursor-crosshair"
          />
        </div>

        <p className="mt-3 text-caption text-on-surface-variant dark:text-outline">
          Click on the canvas to place a point of the selected class. The SVM fits
          instantly; rings mark <strong>support vectors</strong>. Use a kernel from
          the sidebar for non-linear boundaries.
        </p>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <Stat label="support vectors" value={data?.nSupport ?? 0} />
        <Stat label="training acc" value={data ? `${(data.accuracy * 100).toFixed(0)}%` : '—'} />
        <Stat label="points" value={points.length} />
        <Stat label="kernel" value={kernel} capitalize />
      </div>
    </div>
  )
}

function Stat({ label, value, capitalize }: { label: string; value: string | number; capitalize?: boolean }) {
  return (
    <div className="bg-surface-container-lowest dark:bg-dark-surface rounded-2xl p-4 border border-outline-variant/40 dark:border-white/5 text-center">
      <div className={`font-headline text-2xl font-bold text-primary dark:text-inverse-primary ${capitalize ? 'capitalize' : ''}`}>{value}</div>
      <div className="text-caption text-outline mt-1 uppercase tracking-wider">{label}</div>
    </div>
  )
}

/** Draw a single contour level on a regular grid using 2x2 marching squares. */
function drawContour(
  ctx: CanvasRenderingContext2D,
  s: ReturnType<typeof makeScale>,
  grid: { x: number[]; y: number[]; z: number[][] },
  level: number,
  color: string,
  width: number,
) {
  const { x, y, z } = grid
  ctx.strokeStyle = color
  ctx.lineWidth = width
  ctx.beginPath()
  for (let i = 0; i < x.length - 1; i++) {
    for (let j = 0; j < y.length - 1; j++) {
      const v00 = z[j][i], v10 = z[j][i + 1], v11 = z[j + 1][i + 1], v01 = z[j + 1][i]
      const idx =
        (v00 > level ? 1 : 0) | (v10 > level ? 2 : 0) | (v11 > level ? 4 : 0) | (v01 > level ? 8 : 0)
      if (idx === 0 || idx === 15) continue

      const xL = s.px(x[i]), xR = s.px(x[i + 1])
      const yT = s.py(y[j]), yB = s.py(y[j + 1])
      const safe = (a: number, b: number) => (a === b ? a + 1e-9 : b)
      // edge intersection coordinates
      const topX = xL + ((level - v00) / safe(v00, v10 - v00)) * (xR - xL)
      const rightY = yT + ((level - v10) / safe(v10, v11 - v10)) * (yB - yT)
      const bottomX = xR - ((level - v11) / safe(v11, v01 - v11)) * (xR - xL)
      const leftY = yB - ((level - v01) / safe(v01, v00 - v01)) * (yB - yT)

      // Each case draws 1 or 2 segments.
      const seg: [number, number, number, number][] = []
      switch (idx) {
        case 1: case 14: seg.push([topX, yT, xL, leftY]); break          // TL
        case 2: case 13: seg.push([topX, yT, xR, rightY]); break         // TR
        case 3: case 12: seg.push([xL, leftY, xR, rightY]); break        // TL,TR
        case 4: case 11: seg.push([xR, rightY, bottomX, yB]); break      // BR
        case 6: case 9:  seg.push([topX, yT, bottomX, yB]); break        // TR,BR / TL,BL
        case 7: case 8:  seg.push([xL, leftY, bottomX, yB]); break       // BL
        case 5: // TL,BR saddle
          seg.push([topX, yT, xL, leftY])
          seg.push([xR, rightY, bottomX, yB])
          break
        case 10: // TR,BL saddle
          seg.push([topX, yT, xR, rightY])
          seg.push([xL, leftY, bottomX, yB])
          break
      }
      seg.forEach(([x1, y1, x2, y2]) => { ctx.moveTo(x1, y1); ctx.lineTo(x2, y2) })
    }
  }
  ctx.stroke()
}

function mulberry32(seed: number) {
  let a = seed >>> 0
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0
    let t = Math.imul(a ^ (a >>> 15), 1 | a)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}
