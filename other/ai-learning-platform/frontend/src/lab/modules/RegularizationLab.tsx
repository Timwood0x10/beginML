import { useEffect, useRef } from 'react'
import type { LabResult } from '../types'
import { setupCanvas, makeScale, drawAxes, warmColor, type Domain } from '../canvas'

interface RegResult extends LabResult {
  domain: Domain
  contour: { x: number[]; y: number[]; z: number[][]; zmax: number }
  optimum: number[]
  constraint: number
  shapes: { type: string; points: number[][] }[]
  contacts: { penalty: string; point: number[]; sparse: boolean }[]
}

export default function RegularizationLab({ result }: { result: LabResult | null }) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const r = result as RegResult | null

  useEffect(() => {
    if (!r || !canvasRef.current) return
    const W = 620, H = 480
    const ctx = setupCanvas(canvasRef.current, W, H)
    const s = makeScale(ctx, r.domain, { l: 44, r: 20, t: 20, b: 36 })
    const dark = document.documentElement.classList.contains('dark')
    const bg = dark ? '#1A1917' : '#fef9ef'

    // contour heatmap of the quadratic loss
    const { x, y, z, zmax } = r.contour
    for (let i = 0; i < x.length - 1; i++) {
      for (let j = 0; j < y.length - 1; j++) {
        const t = z[j][i] / zmax
        const [cr, cg, cb] = warmColor(t)
        ctx.fillStyle = `rgba(${cr},${cg},${cb},0.5)`
        const cx = s.px(x[i]), cy = s.py(y[j + 1])
        const w = s.px(x[i + 1]) - s.px(x[i])
        const h = s.py(y[j]) - s.py(y[j + 1])
        ctx.fillRect(cx, cy, w + 0.5, h + 0.5)
      }
    }
    drawAxes(ctx, s, r.domain, { color: dark ? '#a8a19a' : '#7d766d', gridColor: dark ? 'rgba(255,255,255,0.06)' : 'rgba(125,118,109,0.13)' })

    // unconstrained optimum (where the loss is centered)
    const [ox, oy] = r.optimum
    ctx.fillStyle = '#2f6b3e'
    ctx.beginPath(); ctx.arc(s.px(ox), s.py(oy), 6, 0, Math.PI * 2); ctx.fill()
    ctx.strokeStyle = bg; ctx.lineWidth = 2; ctx.stroke()

    // constraint shapes
    for (const shape of r.shapes) {
      ctx.beginPath()
      shape.points.forEach((p, i) => {
        const cx = s.px(p[0]), cy = s.py(p[1])
        if (i === 0) ctx.moveTo(cx, cy); else ctx.lineTo(cx, cy)
      })
      ctx.closePath()
      if (shape.type === 'l1') {
        ctx.strokeStyle = '#C8604A'
        ctx.setLineDash([])
      } else {
        ctx.strokeStyle = '#5B6BB0'
        ctx.setLineDash([6, 4])
      }
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
  }, [r])

  if (!r) return null

  return (
    <div className="flex flex-col gap-5">
      <div className="bg-surface-container-lowest dark:bg-dark-surface rounded-3xl p-4 md:p-6 shadow-ambient dark:shadow-dark-ambient border border-outline-variant/40 dark:border-white/5">
        <div className="flex flex-wrap items-center justify-between gap-3 mb-4">
          <h3 className="font-headline text-lg text-on-surface dark:text-inverse-on-surface inline-flex items-center gap-2">
            <span className="material-symbols-outlined" style={{ fontSize: 20 }}>circle</span>
            Constrained minimum
          </h3>
          <div className="flex flex-wrap gap-3 text-caption">
            <Legend color="#2f6b3e" label="unconstrained optimum" />
            {r.shapes.map((sh) => (
              <Legend key={sh.type} color={sh.type === 'l1' ? '#C8604A' : '#5B6BB0'} label={`${sh.type.toUpperCase()} ball`} dashed={sh.type === 'l2'} />
            ))}
          </div>
        </div>
        <div className="w-full overflow-x-auto flex justify-center">
          <canvas ref={canvasRef} className="rounded-2xl" />
        </div>
        <p className="mt-3 text-caption text-on-surface-variant dark:text-outline">
          The colored ball is the constraint set ‖w‖ ≤ C. The contact point is the
          regularized solution. For L1 it tends to land on an axis → a weight is
          exactly zero (sparsity); for L2 it lies smoothly between the axes.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {r.contacts.map((c) => (
          <div
            key={c.penalty}
            className="bg-surface-container-lowest dark:bg-dark-surface rounded-2xl p-5 border border-outline-variant/40 dark:border-white/5"
          >
            <div className="flex items-center justify-between mb-3">
              <h4 className="font-label-md text-label-md uppercase tracking-wider inline-flex items-center gap-2"
                  style={{ color: c.penalty === 'l1' ? '#C8604A' : '#5B6BB0' }}>
                {c.penalty === 'l1' ? 'Lasso (L1)' : 'Ridge (L2)'}
              </h4>
              {c.sparse ? (
                <span className="text-caption font-semibold px-2 py-1 rounded-full bg-[#f3dfdc] text-[#8a3a35] dark:bg-[#3d2a28] dark:text-[#e9b8b2]">
                  sparse solution
                </span>
              ) : (
                <span className="text-caption font-semibold px-2 py-1 rounded-full bg-surface-variant text-on-surface-variant dark:bg-white/10 dark:text-outline">
                  dense weights
                </span>
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
      {dashed ? (
        <span className="w-4 border-t-2 border-dashed" style={{ borderColor: color }} />
      ) : (
        <span className="w-3 h-3 rounded-full" style={{ background: color }} />
      )}
      {label}
    </span>
  )
}
