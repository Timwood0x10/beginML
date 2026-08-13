import { useEffect, useRef } from 'react'
import type { LabResult } from '../types'
import { setupCanvas, makeScale, drawAxes, warmColor, type Domain } from '../canvas'

interface GdResult extends LabResult {
  domain: Domain
  contour: { x: number[]; y: number[]; z: number[][]; zmin: number; zmax: number }
  minimum: { x: number; y: number }
  start: { x: number; y: number }
  trajectories: { name: string; color: string; points: { x: number; y: number; loss: number }[] }[]
  finalLoss: number | null
}

export default function GradientDescentLab({ result, loading }: { result: LabResult | null; loading: boolean }) {
  const contourRef = useRef<HTMLCanvasElement>(null)
  const lossRef = useRef<HTMLCanvasElement>(null)
  const r = result as GdResult | null

  useEffect(() => {
    if (!r || !contourRef.current) return
    const W = 560, H = 440
    const ctx = setupCanvas(contourRef.current, W, H)
    const s = makeScale(ctx, r.domain)
    const bg = document.documentElement.classList.contains('dark') ? '#1A1917' : '#fef9ef'

    // filled contour: draw each cell colored by its z value (cheap heatmap)
    const { x, y, z, zmin, zmax } = r.contour
    const span = zmax - zmin || 1
    for (let i = 0; i < x.length - 1; i++) {
      for (let j = 0; j < y.length - 1; j++) {
        const t = (z[j][i] - zmin) / span
        const [cr, cg, cb] = warmColor(t)
        ctx.fillStyle = `rgba(${cr},${cg},${cb},0.55)`
        const cx = s.px(x[i]), cy = s.py(y[j + 1])
        const w = s.px(x[i + 1]) - s.px(x[i])
        const h = s.py(y[j]) - s.py(y[j + 1])
        ctx.fillRect(cx, cy, w + 0.5, h + 0.5)
      }
    }
    drawAxes(ctx, s, r.domain, { color: '#7d766d', gridColor: 'rgba(125,118,109,0.15)' })

    // global minimum marker
    ctx.fillStyle = '#2f6b3e'
    ctx.beginPath(); ctx.arc(s.px(r.minimum.x), s.py(r.minimum.y), 7, 0, Math.PI * 2); ctx.fill()
    ctx.strokeStyle = bg; ctx.lineWidth = 2; ctx.stroke()

    // start point
    ctx.fillStyle = '#7d766d'
    ctx.beginPath(); ctx.arc(s.px(r.start.x), s.py(r.start.y), 5, 0, Math.PI * 2); ctx.fill()

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
      // point markers
      ctx.fillStyle = traj.color
      traj.points.forEach((p, i) => {
        if (i % Math.max(1, Math.floor(traj.points.length / 12)) === 0) {
          ctx.beginPath(); ctx.arc(s.px(p.x), s.py(p.y), 2.2, 0, Math.PI * 2); ctx.fill()
        }
      })
      // final position
      const last = traj.points[traj.points.length - 1]
      if (last) {
        ctx.beginPath(); ctx.arc(s.px(last.x), s.py(last.y), 5, 0, Math.PI * 2); ctx.fill()
        ctx.strokeStyle = bg; ctx.lineWidth = 2; ctx.stroke()
      }
    }
  }, [r])

  // loss curve
  useEffect(() => {
    if (!r || !lossRef.current) return
    const W = 560, H = 200
    const ctx = setupCanvas(lossRef.current, W, H)
    const pad = { l: 48, r: 16, t: 16, b: 28 }
    const allLoss = r.trajectories.flatMap((t) => t.points.map((p) => p.loss))
    const maxLoss = Math.max(...allLoss, 1)
    const maxSteps = Math.max(...r.trajectories.map((t) => t.points.length))
    const domain: Domain = { x: [0, maxSteps - 1], y: [0, maxLoss * 1.05] }
    const s = makeScale(ctx, domain, pad)
    const bg = document.documentElement.classList.contains('dark') ? '#1A1917' : '#fef9ef'

    ctx.fillStyle = bg; ctx.fillRect(0, 0, W, H)
    drawAxes(ctx, s, domain, { color: '#7d766d', gridColor: 'rgba(125,118,109,0.15)' })

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
    ctx.fillStyle = '#7d766d'
    ctx.font = '11px Manrope'
    ctx.fillText('step', W - 40, H - 8)
    ctx.save(); ctx.translate(14, H / 2); ctx.rotate(-Math.PI / 2)
    ctx.fillText('loss', 0, 0); ctx.restore()
  }, [r])

  if (!r) return <div className="p-10 text-on-surface-variant dark:text-outline">{loading ? 'Computing trajectories…' : 'Adjust controls to begin.'}</div>

  return (
    <div className="flex flex-col gap-4">
      <div className="bg-surface-container-lowest dark:bg-dark-surface rounded-3xl p-4 md:p-6 shadow-ambient dark:shadow-dark-ambient border border-outline-variant/40 dark:border-white/5">
        <div className="flex flex-wrap items-center justify-between gap-3 mb-4">
          <h3 className="font-headline text-lg text-on-surface dark:text-inverse-on-surface inline-flex items-center gap-2">
            <span className="material-symbols-outlined" style={{ fontSize: 20 }}>terrain</span>
            Loss landscape
          </h3>
          <div className="flex flex-wrap gap-3 text-caption">
            <Legend color="#2f6b3e" label="global minimum" />
            {r.trajectories.map((t) => <Legend key={t.name} color={t.color} label={t.name} />)}
          </div>
        </div>
        <div className="w-full overflow-x-auto">
          <canvas ref={contourRef} className="rounded-2xl w-full" style={{ maxWidth: 560 }} />
        </div>
      </div>

      <div className="bg-surface-container-lowest dark:bg-dark-surface rounded-3xl p-4 md:p-6 shadow-ambient dark:shadow-dark-ambient border border-outline-variant/40 dark:border-white/5">
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
