import { useEffect, useRef, useState } from 'react'
import type { LabResult, LabParams } from '../types'
import { setupCanvas, makeScale, drawAxes, type Domain, type Scale } from '../canvas'
import { useCanvasDrag } from '../useCanvasDrag'

interface Probe { x: number; y: number; dy: number }
interface LossResult extends LabResult {
  x: number[]
  y: number[]
  domain: Domain
  loss: string
  name: string
  formula: string
  xLabel: string
  target: number
  minimum: { x: number; y: number }
  probe: Probe
  tangent: { x: number[]; y: number[] }
}

function drawLoss(canvas: HTMLCanvasElement, r: LossResult, hoverX: number | null) {
  const W = 640, H = 420
  const ctx = setupCanvas(canvas, W, H)
  const s = makeScale(ctx, { x: r.domain.x as [number, number], y: r.domain.y as [number, number] },
    { l: 52, r: 20, t: 20, b: 40 })
  const dark = document.documentElement.classList.contains('dark')
  ctx.fillStyle = dark ? '#1A1917' : '#fef9ef'
  ctx.fillRect(0, 0, W, H)
  drawAxes(ctx, s, { x: r.domain.x as [number, number], y: r.domain.y as [number, number] },
    { color: dark ? '#a8a19a' : '#7d766d', gridColor: dark ? 'rgba(255,255,255,0.06)' : 'rgba(125,118,109,0.13)' })

  if (hoverX !== null) {
    ctx.strokeStyle = 'rgba(99,91,79,0.4)'
    ctx.setLineDash([3, 3])
    ctx.beginPath(); ctx.moveTo(s.px(hoverX), s.py(r.domain.y[0])); ctx.lineTo(s.px(hoverX), s.py(r.domain.y[1])); ctx.stroke()
    ctx.setLineDash([])
  }

  // loss curve
  ctx.strokeStyle = '#635b4f'
  ctx.lineWidth = 2.5
  ctx.beginPath()
  r.x.forEach((xv, i) => {
    const cx = s.px(xv), cy = s.py(r.y[i])
    if (i === 0) ctx.moveTo(cx, cy); else ctx.lineTo(cx, cy)
  })
  ctx.stroke()

  // tangent
  ctx.strokeStyle = '#5B6BB0'
  ctx.lineWidth = 1.5
  ctx.setLineDash([5, 4])
  ctx.beginPath()
  r.tangent.x.forEach((xv, i) => {
    const cx = s.px(xv), cy = s.py(r.tangent.y[i])
    if (i === 0) ctx.moveTo(cx, cy); else ctx.lineTo(cx, cy)
  })
  ctx.stroke()
  ctx.setLineDash([])

  // minimum
  ctx.fillStyle = '#2f6b3e'
  ctx.beginPath(); ctx.arc(s.px(r.minimum.x), s.py(r.minimum.y), 6, 0, Math.PI * 2); ctx.fill()
  ctx.strokeStyle = dark ? '#1A1917' : '#fef9ef'; ctx.lineWidth = 2; ctx.stroke()

  // probe
  const px = s.px(r.probe.x), py = s.py(r.probe.y)
  ctx.fillStyle = '#C8604A'
  ctx.beginPath(); ctx.arc(px, py, 7, 0, Math.PI * 2); ctx.fill()
  ctx.strokeStyle = dark ? '#1A1917' : '#fef9ef'; ctx.lineWidth = 2.5; ctx.stroke()
  ctx.fillStyle = dark ? '#d6d0c4' : '#4b463e'
  ctx.font = '600 12px Manrope'
  ctx.fillText(`(${r.probe.x.toFixed(2)}, ${r.probe.y.toFixed(2)})`, px + 10, py - 8)

  ctx.fillStyle = dark ? '#a8a19a' : '#7d766d'
  ctx.font = '12px Manrope'
  ctx.fillText(r.xLabel, W - 80, H - 8)
  ctx.save(); ctx.translate(14, H / 2); ctx.rotate(-Math.PI / 2)
  ctx.fillText('loss', 0, 0); ctx.restore()
}

export default function LossLab({ result, params, setParams }: {
  result: LabResult | null; loading: boolean; error: string | null
  onAction: (k: string) => void; params: LabParams; setParams: (p: LabParams) => void
}) {
  const ref = useRef<HTMLCanvasElement>(null)
  const scaleRef = useRef<Scale | null>(null)
  const r = result as LossResult | null
  const [hover, setHover] = useState<number | null>(null)

  useEffect(() => {
    if (r && ref.current) {
      drawLoss(ref.current, r, hover)
      const ctx = ref.current.getContext('2d')!
      scaleRef.current = makeScale(ctx, { x: r.domain.x as [number, number], y: r.domain.y as [number, number] },
        { l: 52, r: 20, t: 20, b: 40 })
    }
  }, [r, hover])

  const clampX = (x: number) => {
    if (!r) return x
    return Math.max(r.domain.x[0] + 0.01, Math.min(r.domain.x[1] - 0.01, x))
  }

  const { handlers } = useCanvasDrag({
    getScale: () => scaleRef.current,
    onDown: (x) => { setParams({ ...params, probe: parseFloat(clampX(x).toFixed(2)) }); return true },
    onDrag: (x) => { setParams({ ...params, probe: parseFloat(clampX(x).toFixed(2)) }) },
    onHover: (x) => { if (r) setHover(clampX(x)) },
    onUp: () => {},
  })

  if (!r) return null

  return (
    <div className="flex flex-col gap-4">
      <div className="bg-surface-container-lowest dark:bg-dark-surface rounded-3xl p-4 md:p-6 shadow-ambient dark:shadow-dark-ambient border border-outline-variant/40 dark:border-white/5">
        <div className="flex flex-wrap items-center justify-between gap-3 mb-4">
          <h3 className="font-headline text-lg text-on-surface dark:text-inverse-on-surface inline-flex items-center gap-2">
            <span className="material-symbols-outlined" style={{ fontSize: 20 }}>trending_down</span>
            {r.name}
            <span className="text-caption font-normal normal-case text-outline ml-2">(drag the red point)</span>
          </h3>
          <code className="font-mono text-sm text-primary dark:text-inverse-primary bg-primary-fixed/40 dark:bg-white/5 px-3 py-1.5 rounded-lg">{r.formula}</code>
        </div>
        <div className="w-full overflow-x-auto flex justify-center">
          <canvas ref={ref} className="rounded-2xl cursor-crosshair touch-none" {...handlers} />
        </div>
      </div>
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <Stat label="target" value={String(r.target)} />
        <Stat label="min at" value={r.minimum.x.toFixed(3)} />
        <Stat label="probe loss" value={r.probe.y.toFixed(3)} />
        <Stat label="gradient" value={r.probe.dy.toFixed(3)} />
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
