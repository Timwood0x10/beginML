import { useEffect, useRef, useState } from 'react'
import type { LabResult, LabParams } from '../types'
import { setupCanvas, makeScale, drawAxes, warmColor, type Domain, type Scale } from '../canvas'
import { useCanvasDrag } from '../useCanvasDrag'

interface RegResult extends LabResult {
  domain: Domain
  contour: { x: number[]; y: number[]; z: number[][]; zmax: number }
  optimum: number[]
  constraint: number
  shapes: { type: string; points: number[][] }[]
  contacts: { penalty: string; point: number[]; sparse: boolean }[]
}

function drawReg(canvas: HTMLCanvasElement, r: RegResult, hover: { x: number; y: number } | null) {
  const W = 620, H = 480
  const ctx = setupCanvas(canvas, W, H)
  const s = makeScale(ctx, r.domain, { l: 44, r: 20, t: 20, b: 36 })
  const dark = document.documentElement.classList.contains('dark')
  const bg = dark ? '#1A1917' : '#fef9ef'

  // contour heatmap
  const { x, y, z, zmax } = r.contour
  for (let i = 0; i < x.length - 1; i++) {
    for (let j = 0; j < y.length - 1; j++) {
      const t = z[j][i] / zmax
      const [cr, cg, cb] = warmColor(t)
      ctx.fillStyle = `rgba(${cr},${cg},${cb},0.5)`
      ctx.fillRect(s.px(x[i]), s.py(y[j + 1]), s.px(x[i + 1]) - s.px(x[i]) + 0.5, s.py(y[j]) - s.py(y[j + 1]) + 0.5)
    }
  }
  drawAxes(ctx, s, r.domain, { color: dark ? '#a8a19a' : '#7d766d', gridColor: dark ? 'rgba(255,255,255,0.06)' : 'rgba(125,118,109,0.13)' })

  // hover guide lines
  if (hover) {
    ctx.strokeStyle = 'rgba(99,91,79,0.4)'
    ctx.setLineDash([3, 3])
    ctx.beginPath(); ctx.moveTo(s.px(hover.x), s.py(r.domain.y[0])); ctx.lineTo(s.px(hover.x), s.py(r.domain.y[1])); ctx.stroke()
    ctx.beginPath(); ctx.moveTo(s.px(r.domain.x[0]), s.py(hover.y)); ctx.lineTo(s.px(r.domain.x[1]), s.py(hover.y)); ctx.stroke()
    ctx.setLineDash([])
  }

  // constraint shapes
  for (const shape of r.shapes) {
    ctx.beginPath()
    shape.points.forEach((p, i) => {
      const cx = s.px(p[0]), cy = s.py(p[1])
      if (i === 0) ctx.moveTo(cx, cy); else ctx.lineTo(cx, cy)
    })
    ctx.closePath()
    ctx.strokeStyle = shape.type === 'l1' ? '#C8604A' : '#5B6BB0'
    ctx.setLineDash(shape.type === 'l2' ? [6, 4] : [])
    ctx.lineWidth = 2.5
    ctx.stroke()
    ctx.setLineDash([])
  }

  // contact points
  for (const c of r.contacts) {
    const [px, py] = c.point
    ctx.fillStyle = c.penalty === 'l1' ? '#C8604A' : '#5B6BB0'
    ctx.beginPath(); ctx.arc(s.px(px), s.py(py), 6, 0, Math.PI * 2); ctx.fill()
    ctx.strokeStyle = bg; ctx.lineWidth = 2; ctx.stroke()
  }

  // optimum marker (draggable)
  const [ox, oy] = r.optimum
  const px = s.px(ox), py = s.py(oy)
  ctx.fillStyle = '#2f6b3e'
  ctx.beginPath(); ctx.arc(px, py, 8, 0, Math.PI * 2); ctx.fill()
  ctx.strokeStyle = bg; ctx.lineWidth = 2.5; ctx.stroke()
  ctx.fillStyle = dark ? '#d6d0c4' : '#4b463e'
  ctx.font = '600 12px Manrope'
  ctx.fillText(`(${ox.toFixed(2)}, ${oy.toFixed(2)})`, px + 10, py - 10)
}

export default function RegularizationLab({ result, params, setParams }: {
  result: LabResult | null; loading: boolean; error: string | null
  onAction: (k: string) => void; params: LabParams; setParams: (p: LabParams) => void
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const scaleRef = useRef<Scale | null>(null)
  const r = result as RegResult | null
  const [hover, setHover] = useState<{ x: number; y: number } | null>(null)

  useEffect(() => {
    if (r && canvasRef.current) {
      drawReg(canvasRef.current, r, hover)
      const ctx = canvasRef.current.getContext('2d')!
      scaleRef.current = makeScale(ctx, r.domain, { l: 44, r: 20, t: 20, b: 36 })
    }
  }, [r, hover])

  const clamp = (v: number, lo: number, hi: number) => Math.max(lo + 0.1, Math.min(hi - 0.1, v))

  const { handlers } = useCanvasDrag({
    getScale: () => scaleRef.current,
    onDown: (x, y) => {
      if (!r) return false
      setParams({ ...params, optX: parseFloat(clamp(x, 0.2, 2.8).toFixed(2)), optY: parseFloat(clamp(y, 0.2, 2.8).toFixed(2)) })
      return true
    },
    onDrag: (x, y) => {
      if (!r) return
      setParams({ ...params, optX: parseFloat(clamp(x, 0.2, 2.8).toFixed(2)), optY: parseFloat(clamp(y, 0.2, 2.8).toFixed(2)) })
    },
    onHover: (x, y) => { if (r && !Number.isNaN(x)) setHover({ x, y }) },
    onUp: () => {},
  })

  if (!r) return null

  return (
    <div className="flex flex-col gap-5">
      <div className="bg-surface-container-lowest dark:bg-dark-surface rounded-3xl p-4 md:p-6 shadow-ambient dark:shadow-dark-ambient border border-outline-variant/40 dark:border-white/5">
        <div className="flex flex-wrap items-center justify-between gap-3 mb-4">
          <h3 className="font-headline text-lg text-on-surface dark:text-inverse-on-surface inline-flex items-center gap-2">
            <span className="material-symbols-outlined" style={{ fontSize: 20 }}>circle</span>
            Constrained minimum
            <span className="text-caption font-normal normal-case text-outline ml-2">(drag the green point)</span>
          </h3>
          <div className="flex flex-wrap gap-3 text-caption">
            <Legend color="#2f6b3e" label="unconstrained optimum" />
            {r.shapes.map((sh) => (
              <Legend key={sh.type} color={sh.type === 'l1' ? '#C8604A' : '#5B6BB0'} label={`${sh.type.toUpperCase()} ball`} dashed={sh.type === 'l2'} />
            ))}
          </div>
        </div>
        <div className="w-full overflow-x-auto flex justify-center">
          <canvas ref={canvasRef} className="rounded-2xl cursor-crosshair touch-none" {...handlers} />
        </div>
        <p className="mt-3 text-caption text-on-surface-variant dark:text-outline">
          Drag the green optimum to move the loss contours, or use C / loss tilt
          sliders. The contact point is the regularized solution: L1 lands on an
          axis (sparsity), L2 shrinks smoothly.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {r.contacts.map((c) => (
          <div key={c.penalty} className="bg-surface-container-lowest dark:bg-dark-surface rounded-2xl p-5 border border-outline-variant/40 dark:border-white/5">
            <div className="flex items-center justify-between mb-3">
              <h4 className="font-label-md text-label-md uppercase tracking-wider inline-flex items-center gap-2"
                  style={{ color: c.penalty === 'l1' ? '#C8604A' : '#5B6BB0' }}>
                {c.penalty === 'l1' ? 'Lasso (L1)' : 'Ridge (L2)'}
              </h4>
              {c.sparse ? (
                <span className="text-caption font-semibold px-2 py-1 rounded-full bg-[#f3dfdc] text-[#8a3a35] dark:bg-[#3d2a28] dark:text-[#e9b8b2]">sparse solution</span>
              ) : (
                <span className="text-caption font-semibold px-2 py-1 rounded-full bg-surface-variant text-on-surface-variant dark:bg-white/10 dark:text-outline">dense weights</span>
              )}
            </div>
            <div className="font-mono text-body-md text-on-surface dark:text-dark-on-surface mb-2">
              w* = ({c.point[0].toFixed(3)}, {c.point[1].toFixed(3)})
            </div>
            <p className="text-caption text-on-surface-variant dark:text-outline leading-relaxed">
              {c.penalty === 'l1'
                ? 'The diamond corner lies on an axis, driving one weight to exactly zero. This is why Lasso performs feature selection.'
                : 'The circle touches a contour in the interior of a quadrant — both weights shrink but neither is forced to zero.'}
            </p>
          </div>
        ))}
      </div>
    </div>
  )
}

function Legend({ color, label, dashed }: { color: string; label: string; dashed?: boolean }) {
  return (
    <span className="inline-flex items-center gap-1.5 text-on-surface-variant dark:text-outline">
      {dashed ? <span className="w-4 border-t-2 border-dashed" style={{ borderColor: color }} /> : <span className="w-3 h-3 rounded-full" style={{ background: color }} />}
      {label}
    </span>
  )
}
