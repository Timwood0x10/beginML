import { useEffect, useRef } from 'react'
import type { LabResult } from '../types'
import { setupCanvas, makeScale, drawAxes, type Domain } from '../canvas'

interface PcaResult extends LabResult {
  domain: Domain
  points: number[][]
  reconstructed: number[][]
  residuals: number[][][]
  mean: number[]
  eigenvalues: number[]
  eigenvectors: { name: string; segment: number[][]; variance: number }[]
  components: number
  explainedVariance: number[]
  keptVariance: number
}

export default function PcaLab({ result }: { result: LabResult | null }) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const r = result as PcaResult | null

  useEffect(() => {
    if (!r || !canvasRef.current) return
    const W = 620, H = 480
    const ctx = setupCanvas(canvasRef.current, W, H)
    const s = makeScale(ctx, r.domain, { l: 44, r: 20, t: 20, b: 36 })
    const dark = document.documentElement.classList.contains('dark')
    const bg = dark ? '#1E1913' : '#F7F0E3'
    const axisColor = dark ? '#A99B82' : '#8A7A61'

    ctx.fillStyle = bg; ctx.fillRect(0, 0, W, H)
    drawAxes(ctx, s, r.domain, { color: axisColor, gridColor: dark ? 'rgba(255,255,255,0.06)' : 'rgba(125,118,109,0.13)' })

    // residual lines (original -> projection)
    ctx.strokeStyle = dark ? 'rgba(208,197,182,0.25)' : 'rgba(99,91,79,0.25)'
    ctx.lineWidth = 1
    for (const [a, b] of r.residuals) {
      ctx.beginPath()
      ctx.moveTo(s.px(a[0]), s.py(a[1]))
      ctx.lineTo(s.px(b[0]), s.py(b[1]))
      ctx.stroke()
    }

    // reconstructed (projected) points
    ctx.fillStyle = '#C8604A'
    for (const p of r.reconstructed) {
      ctx.beginPath(); ctx.arc(s.px(p[0]), s.py(p[1]), 3, 0, Math.PI * 2); ctx.fill()
    }

    // original points
    ctx.fillStyle = dark ? 'rgba(208,197,182,0.9)' : 'rgba(99,91,79,0.85)'
    for (const p of r.points) {
      ctx.beginPath(); ctx.arc(s.px(p[0]), s.py(p[1]), 3.5, 0, Math.PI * 2); ctx.fill()
    }

    // eigenvectors (principal axes) through the mean
    const [mx, my] = r.mean
    const colors = ['#7A5C36', '#C8604A']
    r.eigenvectors.forEach((ev, i) => {
      const [, tip] = ev.segment
      ctx.strokeStyle = colors[i]
      ctx.lineWidth = 2.5
      ctx.beginPath()
      ctx.moveTo(s.px(mx), s.py(my))
      ctx.lineTo(s.px(mx + tip[0]), s.py(my + tip[1]))
      ctx.stroke()
      // arrowhead
      const ang = Math.atan2(tip[1], tip[0])
      const ex = s.px(mx + tip[0]), ey = s.py(my + tip[1])
      ctx.beginPath()
      ctx.moveTo(ex, ey)
      ctx.lineTo(ex - 8 * Math.cos(ang - 0.4), ey + 8 * Math.sin(ang - 0.4))
      ctx.lineTo(ex - 8 * Math.cos(ang + 0.4), ey + 8 * Math.sin(ang + 0.4))
      ctx.closePath(); ctx.fill()
      ctx.fillStyle = colors[i]
      ctx.font = '600 12px Manrope'
      ctx.fillText(ev.name, ex + 6, ey - 6)
    })

    // mean marker
    ctx.fillStyle = '#2f6b3e'
    ctx.beginPath(); ctx.arc(s.px(mx), s.py(my), 5, 0, Math.PI * 2); ctx.fill()
    ctx.strokeStyle = bg; ctx.lineWidth = 2; ctx.stroke()
  }, [r])

  if (!r) return null

  return (
    <div className="flex flex-col gap-5">
      <div className="bg-surface-container-lowest dark:bg-dark-surface rounded-3xl p-4 md:p-6 shadow-ambient dark:shadow-dark-ambient border border-outline-variant/40 dark:border-white/10">
        <div className="flex flex-wrap items-center justify-between gap-3 mb-4">
          <h3 className="font-headline text-lg text-on-surface dark:text-inverse-on-surface inline-flex items-center gap-2">
            <span className="material-symbols-outlined" style={{ fontSize: 20 }}>center_focus_strong</span>
            Data &amp; principal axes
          </h3>
          <div className="flex flex-wrap gap-3 text-caption">
            <Legend color="#7A5C36" label="original data" />
            <Legend color="#C8604A" label="reconstruction" />
            <Legend color="#2f6b3e" label="mean" />
          </div>
        </div>
        <div className="w-full overflow-x-auto flex justify-center">
          <canvas ref={canvasRef} className="rounded-2xl" />
        </div>
        <p className="mt-3 text-caption text-on-surface-variant dark:text-outline">
          Arrows are the eigenvectors of the covariance matrix, scaled by their
          standard deviation. The pale lines are projection residuals.
        </p>
      </div>

      {/* Variance explained */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="bg-surface-container-lowest dark:bg-dark-surface rounded-2xl p-5 border border-outline-variant/40 dark:border-white/10">
          <h4 className="font-label-md text-label-md uppercase tracking-wider text-on-surface dark:text-dark-on-surface mb-4">Explained variance</h4>
          <div className="flex flex-col gap-3">
            {r.eigenvectors.map((ev, i) => (
              <div key={ev.name}>
                <div className="flex justify-between text-caption mb-1">
                  <span className="text-on-surface dark:text-dark-on-surface font-semibold">{ev.name}</span>
                  <span className="font-mono text-primary dark:text-inverse-primary">{(ev.variance * 100).toFixed(1)}%</span>
                </div>
                <div className="h-2.5 bg-surface-variant dark:bg-white/10 rounded-full overflow-hidden">
                  <div className="h-full rounded-full" style={{ width: `${ev.variance * 100}%`, background: i === 0 ? '#7A5C36' : '#C8604A' }} />
                </div>
              </div>
            ))}
          </div>
          <div className="mt-4 pt-4 border-t border-outline-variant/50 dark:border-white/10 flex justify-between text-body-md">
            <span className="text-on-surface-variant dark:text-outline">Kept with {r.components} component{r.components === 1 ? '' : 's'}</span>
            <span className="font-headline font-bold text-primary dark:text-inverse-primary">{(r.keptVariance * 100).toFixed(1)}%</span>
          </div>
        </div>

        <div className="bg-surface-container-lowest dark:bg-dark-surface rounded-2xl p-5 border border-outline-variant/40 dark:border-white/10">
          <h4 className="font-label-md text-label-md uppercase tracking-wider text-on-surface dark:text-dark-on-surface mb-3">Eigenvalues</h4>
          <p className="text-body-md text-on-surface-variant dark:text-outline mb-4 leading-relaxed">
            The eigenvalues measure how much variance lives along each principal
            direction — and therefore how much information each component preserves.
          </p>
          <div className="flex gap-4">
            {r.eigenvalues.map((v, i) => (
              <div key={i} className="flex-1 bg-surface-container dark:bg-white/5 rounded-xl p-4 text-center">
                <div className="font-mono text-2xl font-bold text-primary dark:text-inverse-primary">{v.toFixed(2)}</div>
                <div className="text-caption text-outline mt-1">λ{i + 1}</div>
              </div>
            ))}
          </div>
          <p className="text-caption text-outline mt-4">
            Slide <strong>Components kept</strong> to 0 to project onto the mean, 1 to
            drop the low-variance axis, or 2 to reconstruct perfectly.
          </p>
        </div>
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
