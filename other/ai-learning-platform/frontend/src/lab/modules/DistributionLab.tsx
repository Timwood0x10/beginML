import { useEffect, useRef } from 'react'
import type { LabResult } from '../types'
import { setupCanvas, makeScale, drawAxes, type Domain } from '../canvas'

interface Stats { mean: number; std: number; samples: number }
interface DistResult extends LabResult {
  discrete: boolean
  distribution: string
  name: string
  formula: string
  x: number[]
  y: number[]
  histEdges: number[]
  histCounts: number[]
  domain: Domain
  stats: Stats
}

export default function DistributionLab({ result }: { result: LabResult | null }) {
  const ref = useRef<HTMLCanvasElement>(null)
  const r = result as DistResult | null

  useEffect(() => {
    if (!r || !ref.current) return
    const W = 640, H = 420
    const ctx = setupCanvas(ref.current, W, H)
    const s = makeScale(ctx, { x: r.domain.x as [number, number], y: r.domain.y as [number, number] },
      { l: 52, r: 20, t: 20, b: 40 })
    const dark = document.documentElement.classList.contains('dark')
    ctx.fillStyle = dark ? '#1A1917' : '#fef9ef'
    ctx.fillRect(0, 0, W, H)
    drawAxes(ctx, s, { x: r.domain.x as [number, number], y: r.domain.y as [number, number] },
      { color: dark ? '#a8a19a' : '#7d766d', gridColor: dark ? 'rgba(255,255,255,0.06)' : 'rgba(125,118,109,0.13)' })

    // histogram bars
    ctx.fillStyle = dark ? 'rgba(91,107,176,0.45)' : 'rgba(91,107,176,0.35)'
    for (let i = 0; i < r.histCounts.length; i++) {
      const e0 = r.histEdges[i], e1 = r.histEdges[i + 1]
      const cx = s.px(e0), w = s.px(e1) - s.px(e0)
      const h = s.py(0) - s.py(r.histCounts[i])
      if (h > 0) ctx.fillRect(cx + 1, s.py(r.histCounts[i]), w - 2, h)
    }

    // density curve / pmf stems
    if (r.discrete) {
      ctx.strokeStyle = '#C8604A'
      ctx.lineWidth = 2
      r.x.forEach((xi, i) => {
        const cx = s.px(xi), top = s.py(r.y[i])
        ctx.beginPath()
        ctx.moveTo(cx, s.py(0))
        ctx.lineTo(cx, top)
        ctx.stroke()
      })
      ctx.fillStyle = '#C8604A'
      r.x.forEach((xi, i) => {
        ctx.beginPath(); ctx.arc(s.px(xi), s.py(r.y[i]), 3, 0, Math.PI * 2); ctx.fill()
      })
    } else {
      ctx.strokeStyle = '#635b4f'
      ctx.lineWidth = 2.5
      ctx.beginPath()
      r.x.forEach((xi, i) => {
        const cx = s.px(xi), cy = s.py(r.y[i])
        if (i === 0) ctx.moveTo(cx, cy); else ctx.lineTo(cx, cy)
      })
      ctx.stroke()
    }

    ctx.fillStyle = dark ? '#a8a19a' : '#7d766d'
    ctx.font = '12px Manrope'
    ctx.fillText('x', W - 24, H - 8)
    ctx.save(); ctx.translate(14, H / 2); ctx.rotate(-Math.PI / 2)
    ctx.fillText(r.discrete ? 'probability' : 'density', 0, 0); ctx.restore()
  }, [r])

  if (!r) return null

  return (
    <div className="flex flex-col gap-4">
      <div className="bg-surface-container-lowest dark:bg-dark-surface rounded-3xl p-4 md:p-6 shadow-ambient dark:shadow-dark-ambient border border-outline-variant/40 dark:border-white/5">
        <div className="flex flex-wrap items-center justify-between gap-3 mb-4">
          <h3 className="font-headline text-lg text-on-surface dark:text-inverse-on-surface inline-flex items-center gap-2">
            <span className="material-symbols-outlined" style={{ fontSize: 20 }}>bar_chart</span>
            {r.name}
          </h3>
          <code className="font-mono text-sm text-primary dark:text-inverse-primary bg-primary-fixed/40 dark:bg-white/5 px-3 py-1.5 rounded-lg">
            {r.formula}
          </code>
        </div>
        <div className="w-full overflow-x-auto flex justify-center">
          <canvas ref={ref} className="rounded-2xl" />
        </div>
      </div>
      <div className="grid grid-cols-3 gap-3">
        <Stat label="sample mean" value={r.stats.mean.toFixed(3)} />
        <Stat label="sample std" value={r.stats.std.toFixed(3)} />
        <Stat label="samples" value={String(r.stats.samples)} />
      </div>
    </div>
  )
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="bg-surface-container-lowest dark:bg-dark-surface rounded-2xl p-4 border border-outline-variant/40 dark:border-white/5 text-center">
      <div className="font-mono text-xl font-bold text-primary dark:text-inverse-primary">{value}</div>
      <div className="text-caption text-outline mt-1 uppercase tracking-wider">{label}</div>
    </div>
  )
}
