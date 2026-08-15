import { useEffect, useRef, useState } from 'react'
import type { LabResult, LabParams } from '../types'
import { setupCanvas, makeScale, drawAxes, warmColor, type Domain, type Scale } from '../canvas'
import { useCanvasDrag } from '../useCanvasDrag'
import { useTheme } from '../../hooks/useTheme'

interface GdResult extends LabResult {
  domain: Domain
  contour: { x: number[]; y: number[]; z: number[][]; zmin: number; zmax: number }
  minimum: { x: number; y: number }
  start: { x: number; y: number }
  trajectories: { name: string; color: string; points: { x: number; y: number; loss: number }[] }[]
  finalLoss: number | null
}

// Read an active theme token so canvas drawings follow the palette.
function themeVar(name: string, fallback: string): string {
  const val = getComputedStyle(document.documentElement).getPropertyValue(name).trim()
  return val || fallback
}

function drawContour(canvas: HTMLCanvasElement, r: GdResult, hover: { x: number; y: number } | null) {
  const W = 560, H = 440
  const ctx = setupCanvas(canvas, W, H)
  const s = makeScale(ctx, r.domain)
  const bg = themeVar('--ailearn-background', '#F7F0E3')
  const ink = themeVar('--ailearn-outline', '#8A7A61')

  // filled contour heatmap
  const { x, y, z, zmin, zmax } = r.contour
  const span = zmax - zmin || 1
  for (let i = 0; i < x.length - 1; i++) {
    for (let j = 0; j < y.length - 1; j++) {
      const t = (z[j][i] - zmin) / span
      const [cr, cg, cb] = warmColor(t)
      ctx.fillStyle = `rgba(${cr},${cg},${cb},0.55)`
      ctx.fillRect(s.px(x[i]), s.py(y[j + 1]), s.px(x[i + 1]) - s.px(x[i]) + 0.5, s.py(y[j]) - s.py(y[j + 1]) + 0.5)
    }
  }
  drawAxes(ctx, s, r.domain, { color: ink, gridColor: 'rgba(125,118,109,0.15)' })

  // hover crosshair + z value
  if (hover) {
    ctx.strokeStyle = 'rgba(99,91,79,0.45)'
    ctx.setLineDash([3, 3])
    ctx.beginPath(); ctx.moveTo(s.px(hover.x), s.py(r.domain.y[0])); ctx.lineTo(s.px(hover.x), s.py(r.domain.y[1])); ctx.stroke()
    ctx.beginPath(); ctx.moveTo(s.px(r.domain.x[0]), s.py(hover.y)); ctx.lineTo(s.px(r.domain.x[1]), s.py(hover.y)); ctx.stroke()
    ctx.setLineDash([])
  }

  // global minimum
  ctx.fillStyle = '#2f6b3e'
  ctx.beginPath(); ctx.arc(s.px(r.minimum.x), s.py(r.minimum.y), 7, 0, Math.PI * 2); ctx.fill()
  ctx.strokeStyle = bg; ctx.lineWidth = 2; ctx.stroke()

  // trajectories
  for (const traj of r.trajectories) {
    ctx.strokeStyle = traj.color
    ctx.lineWidth = 2.2
    ctx.lineJoin = 'round'
    ctx.beginPath()
    traj.points.forEach((p, i) => {
      const cx = s.px(p.x), cy = s.py(p.y)
      if (i === 0) ctx.moveTo(cx, cy); else ctx.lineTo(cx, cy)
    })
    ctx.stroke()
    ctx.fillStyle = traj.color
    traj.points.forEach((p, i) => {
      if (i % Math.max(1, Math.floor(traj.points.length / 12)) === 0) {
        ctx.beginPath(); ctx.arc(s.px(p.x), s.py(p.y), 2.2, 0, Math.PI * 2); ctx.fill()
      }
    })
    const last = traj.points[traj.points.length - 1]
    if (last) {
      ctx.beginPath(); ctx.arc(s.px(last.x), s.py(last.y), 5, 0, Math.PI * 2); ctx.fill()
      ctx.strokeStyle = bg; ctx.lineWidth = 2; ctx.stroke()
    }
  }

  // start point (draggable / clickable)
  const px = s.px(r.start.x), py = s.py(r.start.y)
  ctx.fillStyle = '#C8604A'
  ctx.beginPath(); ctx.arc(px, py, 8, 0, Math.PI * 2); ctx.fill()
  ctx.strokeStyle = bg; ctx.lineWidth = 2.5; ctx.stroke()
  ctx.fillStyle = '#54483A'
  ctx.font = '600 12px Manrope'
  ctx.fillText('start', px + 10, py - 8)
}

function drawLossCurve(canvas: HTMLCanvasElement, r: GdResult) {
  const W = 560, H = 200
  const ctx = setupCanvas(canvas, W, H)
  const pad = { l: 48, r: 16, t: 16, b: 28 }
  const allLoss = r.trajectories.flatMap((t) => t.points.map((p) => p.loss))
  const maxLoss = Math.max(...allLoss, 1)
  const maxSteps = Math.max(...r.trajectories.map((t) => t.points.length))
  const domain: Domain = { x: [0, maxSteps - 1], y: [0, maxLoss * 1.05] }
  const s = makeScale(ctx, domain, pad)
  const bg = themeVar('--ailearn-background', '#F7F0E3')
  const ink = themeVar('--ailearn-outline', '#8A7A61')

  ctx.fillStyle = bg; ctx.fillRect(0, 0, W, H)
  drawAxes(ctx, s, domain, { color: ink, gridColor: 'rgba(125,118,109,0.15)' })

  for (const traj of r.trajectories) {
    ctx.strokeStyle = traj.color
    ctx.lineWidth = 2
    ctx.beginPath()
    traj.points.forEach((p, i) => {
      const cx = s.px(i), cy = s.py(Math.max(0, p.loss))
      if (i === 0) ctx.moveTo(cx, cy); else ctx.lineTo(cx, cy)
    })
    ctx.stroke()
  }
  ctx.fillStyle = ink
  ctx.font = '11px Manrope'
  ctx.fillText('step', W - 40, H - 8)
  ctx.save(); ctx.translate(14, H / 2); ctx.rotate(-Math.PI / 2)
  ctx.fillText('loss', 0, 0); ctx.restore()
}

export default function GradientDescentLab({ result, loading, params, setParams }: {
  result: LabResult | null; loading: boolean; error: string | null
  onAction: (k: string) => void; params: LabParams; setParams: (p: LabParams) => void
}) {
  const contourRef = useRef<HTMLCanvasElement>(null)
  const lossRef = useRef<HTMLCanvasElement>(null)
  const scaleRef = useRef<Scale | null>(null)
  const r = result as GdResult | null
  const [hover, setHover] = useState<{ x: number; y: number } | null>(null)
  // Redraw whenever the theme palette or light/dark mode flips — the canvas
  // otherwise keeps its old colours (drawn once when the result changed).
  const { theme, palette } = useTheme()

  useEffect(() => {
    if (r && contourRef.current) {
      drawContour(contourRef.current, r, hover)
      const ctx = contourRef.current.getContext('2d')!
      scaleRef.current = makeScale(ctx, r.domain)
    }
  }, [r, hover, theme, palette])

  useEffect(() => {
    if (r && lossRef.current) drawLossCurve(lossRef.current, r)
  }, [r, theme, palette])

  const clampPt = (x: number, y: number) => {
    if (!r) return { x, y }
    return {
      x: Math.max(r.domain.x[0] + 0.1, Math.min(r.domain.x[1] - 0.1, x)),
      y: Math.max(r.domain.y[0] + 0.1, Math.min(r.domain.y[1] - 0.1, y)),
    }
  }

  const { handlers } = useCanvasDrag({
    getScale: () => scaleRef.current,
    onDown: (x, y) => {
      const p = clampPt(x, y)
      setParams({ ...params, startX: parseFloat(p.x.toFixed(2)), startY: parseFloat(p.y.toFixed(2)) })
      return true
    },
    onDrag: (x, y) => {
      const p = clampPt(x, y)
      setParams({ ...params, startX: parseFloat(p.x.toFixed(2)), startY: parseFloat(p.y.toFixed(2)) })
    },
    onHover: (x, y) => { if (r && !Number.isNaN(x)) setHover({ x, y }) },
    onUp: () => {},
  })

  if (!r) return <div className="p-10 text-on-surface-variant dark:text-outline">{loading ? 'Computing trajectories...' : 'Adjust controls to begin.'}</div>

  return (
    <div className="flex flex-col gap-4">
      <div className="bg-surface-container-lowest dark:bg-dark-surface rounded-3xl p-4 md:p-6 shadow-ambient dark:shadow-dark-ambient border border-outline-variant/40 dark:border-white/10">
        <div className="flex flex-wrap items-center justify-between gap-3 mb-4">
          <h3 className="font-headline text-lg text-on-surface dark:text-inverse-on-surface inline-flex items-center gap-2">
            <span className="material-symbols-outlined" style={{ fontSize: 20 }}>terrain</span>
            Loss landscape
            <span className="text-caption font-normal normal-case text-outline ml-2">(click or drag to set start)</span>
          </h3>
          <div className="flex flex-wrap gap-3 text-caption">
            <Legend color="#2f6b3e" label="global minimum" />
            <Legend color="#C8604A" label="start" />
            {r.trajectories.map((t) => <Legend key={t.name} color={t.color} label={t.name} />)}
          </div>
        </div>
        <div className="w-full overflow-x-auto">
          <canvas ref={contourRef} className="rounded-2xl w-full cursor-crosshair touch-none" style={{ maxWidth: 560 }} {...handlers} />
        </div>
        {hover && (
          <p className="mt-3 text-caption text-on-surface-variant dark:text-outline">
            cursor at <span className="font-mono">({hover.x.toFixed(2)}, {hover.y.toFixed(2)})</span>
          </p>
        )}
      </div>

      <div className="bg-surface-container-lowest dark:bg-dark-surface rounded-3xl p-4 md:p-6 shadow-ambient dark:shadow-dark-ambient border border-outline-variant/40 dark:border-white/10">
        <h3 className="font-headline text-lg text-on-surface dark:text-inverse-on-surface mb-3 inline-flex items-center gap-2">
          <span className="material-symbols-outlined" style={{ fontSize: 20 }}>show_chart</span>
          Convergence
        </h3>
        <div className="w-full overflow-x-auto">
          <canvas ref={lossRef} className="rounded-2xl w-full" style={{ maxWidth: 560 }} />
        </div>
        {typeof r.finalLoss === 'number' && (
          <p className="mt-3 text-caption text-on-surface-variant dark:text-outline">
            Final loss: <span className="font-mono text-primary dark:text-inverse-primary">{r.finalLoss.toExponential(3)}</span>
          </p>
        )}
      </div>
    </div>
  )
}

function Legend({ color, label }: { color: string; label: string }) {
  return (
    <span className="inline-flex items-center gap-1.5 text-on-surface-variant dark:text-outline">
      <span className="w-3 h-3 rounded-full" style={{ background: color }} />
      {label}
    </span>
  )
}
