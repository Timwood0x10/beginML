import { useEffect, useRef } from 'react'
import type { LabResult } from '../types'
import { setupCanvas, makeScale, drawAxes, type Domain } from '../canvas'

interface Point { x: number; y: number; dy: number }
interface ActResult extends LabResult {
  x: number[]
  y: number[]
  dy: number[]
  domain: Domain
  point: Point
  tangent: { x: number[]; y: number[] }
  formula: string
  function: string
}

function drawCurve(
  ctx: CanvasRenderingContext2D,
  s: ReturnType<typeof makeScale>,
  xs: number[], ys: number[],
  color: string, width: number,
) {
  ctx.strokeStyle = color
  ctx.lineWidth = width
  ctx.beginPath()
  xs.forEach((x, i) => {
    const cx = s.px(x), cy = s.py(ys[i])
    if (i === 0) ctx.moveTo(cx, cy); else ctx.lineTo(cx, cy)
  })
  ctx.stroke()
}

export default function ActivationLab({ result }: { result: LabResult | null }) {
  const ref = useRef<HTMLCanvasElement>(null)
  const r = result as ActResult | null

  useEffect(() => {
    if (!r || !ref.current) return
    const W = 640, H = 420
    const ctx = setupCanvas(ref.current, W, H)
    const s = makeScale(ctx, { x: r.domain.x as [number, number], y: r.domain.y as [number, number] },
      { l: 48, r: 20, t: 20, b: 36 })
    const dark = document.documentElement.classList.contains('dark')
    ctx.fillStyle = dark ? '#1A1917' : '#fef9ef'
    ctx.fillRect(0, 0, W, H)
    drawAxes(ctx, s, { x: r.domain.x as [number, number], y: r.domain.y as [number, number] },
      { color: dark ? '#a8a19a' : '#7d766d', gridColor: dark ? 'rgba(255,255,255,0.06)' : 'rgba(125,118,109,0.13)' })

    // function curve
    drawCurve(ctx, s, r.x, r.y, '#635b4f', 2.5)
    // derivative
    drawCurve(ctx, s, r.x, r.dy, '#C8604A', 1.8)
    ctx.setLineDash([5, 4])
    drawCurve(ctx, s, r.tangent.x, r.tangent.y, '#5B6BB0', 1.5)
    ctx.setLineDash([])

    // evaluation point
    const px = s.px(r.point.x), py = s.py(r.point.y)
    ctx.fillStyle = '#2f6b3e'
    ctx.beginPath(); ctx.arc(px, py, 6, 0, Math.PI * 2); ctx.fill()
    ctx.strokeStyle = dark ? '#1A1917' : '#fef9ef'; ctx.lineWidth = 2; ctx.stroke()

    // labels
    ctx.font = '600 13px Manrope'
    ctx.fillStyle = '#635b4f'
    ctx.fillText('f(x)', s.px(r.x[r.x.length - 2]) + 4, s.py(r.y[r.y.length - 2]))
    ctx.fillStyle = '#C8604A'
    ctx.fillText("f'(x)", s.px(r.x[r.x.length - 2]) + 4, s.py(r.dy[r.dy.length - 2]))
  }, [r])

  if (!r) return null

  return (
    <div className="flex flex-col gap-4">
      <div className="bg-surface-container-lowest dark:bg-dark-surface rounded-3xl p-4 md:p-6 shadow-ambient dark:shadow-dark-ambient border border-outline-variant/40 dark:border-white/5">
        <div className="flex flex-wrap items-center justify-between gap-3 mb-4">
          <h3 className="font-headline text-lg text-on-surface dark:text-inverse-on-surface inline-flex items-center gap-2 capitalize">
            <span className="material-symbols-outlined" style={{ fontSize: 20 }}>show_chart</span>
            {r.function}
          </h3>
          <code className="font-mono text-sm text-primary dark:text-inverse-primary bg-primary-fixed/40 dark:bg-white/5 px-3 py-1.5 rounded-lg">
            {r.formula}
          </code>
        </div>
        <div className="w-full overflow-x-auto flex justify-center">
          <canvas ref={ref} className="rounded-2xl" />
        </div>
      </div>
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <Stat label="x" value={r.point.x.toFixed(3)} />
        <Stat label="f(x)" value={r.point.y.toFixed(3)} />
        <Stat label="f'(x)" value={r.point.dy.toFixed(3)} />
        <Stat label="slope angle" value={`${(Math.atan(r.point.dy) * 180 / Math.PI).toFixed(1)}°`} />
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
