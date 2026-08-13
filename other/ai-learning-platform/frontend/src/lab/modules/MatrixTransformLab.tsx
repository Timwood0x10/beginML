import { useEffect, useRef, useState } from 'react'
import type { LabResult, LabParams } from '../types'
import { setupCanvas, makeScale, drawAxes, type Domain, type Scale } from '../canvas'
import { useCanvasDrag } from '../useCanvasDrag'

interface Eigen { value: number; vector: number[] }
interface MatrixResult extends LabResult {
  matrix: number[][]
  det: number
  trace: number
  rank: number
  eigen: Eigen[]
  basis: { ex: number[]; ey: number[] }
  gridOriginal: number[][][]
  gridTransformed: number[][][]
  squareOriginal: number[][]
  squareTransformed: number[][]
  circleTransformed: number[][]
  domain: Domain
}

function drawGrid(ctx: CanvasRenderingContext2D, s: Scale, lines: number[][][], color: string, width: number) {
  ctx.strokeStyle = color
  ctx.lineWidth = width
  lines.forEach(([a, b]) => {
    ctx.beginPath()
    ctx.moveTo(s.px(a[0]), s.py(a[1]))
    ctx.lineTo(s.px(b[0]), s.py(b[1]))
    ctx.stroke()
  })
}

function drawPolygon(ctx: CanvasRenderingContext2D, s: Scale, pts: number[][], stroke: string, fill?: string) {
  if (fill) {
    ctx.fillStyle = fill
    ctx.beginPath()
    pts.forEach((p, i) => i === 0 ? ctx.moveTo(s.px(p[0]), s.py(p[1])) : ctx.lineTo(s.px(p[0]), s.py(p[1])))
    ctx.closePath(); ctx.fill()
  }
  ctx.strokeStyle = stroke
  ctx.lineWidth = 2
  ctx.beginPath()
  pts.forEach((p, i) => i === 0 ? ctx.moveTo(s.px(p[0]), s.py(p[1])) : ctx.lineTo(s.px(p[0]), s.py(p[1])))
  ctx.closePath(); ctx.stroke()
}

function drawArrow(ctx: CanvasRenderingContext2D, s: Scale, to: number[], color: string) {
  const ex = s.px(to[0]), ey = s.py(to[1])
  ctx.strokeStyle = color
  ctx.fillStyle = color
  ctx.lineWidth = 2.5
  ctx.beginPath()
  ctx.moveTo(s.px(0), s.py(0))
  ctx.lineTo(ex, ey)
  ctx.stroke()
  const angle = Math.atan2(s.py(0) - ey, ex - s.px(0))
  ctx.beginPath()
  ctx.moveTo(ex, ey)
  ctx.lineTo(ex - 9 * Math.cos(angle - 0.4), ey + 9 * Math.sin(angle - 0.4))
  ctx.lineTo(ex - 9 * Math.cos(angle + 0.4), ey + 9 * Math.sin(angle + 0.4))
  ctx.closePath(); ctx.fill()
}

function drawScene(canvas: HTMLCanvasElement, r: MatrixResult, pt: { x: number; y: number } | null, hover: { x: number; y: number } | null) {
  const W = 560, H = 560
  const ctx = setupCanvas(canvas, W, H)
  const s = makeScale(ctx, { x: r.domain.x as [number, number], y: r.domain.y as [number, number] }, { l: 20, r: 20, t: 20, b: 20 })
  const dark = document.documentElement.classList.contains('dark')
  ctx.fillStyle = dark ? '#1A1917' : '#fef9ef'
  ctx.fillRect(0, 0, W, H)

  drawGrid(ctx, s, r.gridOriginal, dark ? 'rgba(255,255,255,0.06)' : 'rgba(125,118,109,0.1)', 1)
  drawGrid(ctx, s, r.gridTransformed, dark ? 'rgba(208,197,182,0.3)' : 'rgba(99,91,79,0.35)', 1.2)

  drawPolygon(ctx, s, r.squareOriginal, dark ? 'rgba(255,255,255,0.25)' : 'rgba(125,118,109,0.4)', 'rgba(125,118,109,0.05)')
  drawPolygon(ctx, s, r.squareTransformed, '#C8604A', 'rgba(200,96,74,0.1)')

  ctx.strokeStyle = '#5B6BB0'
  ctx.lineWidth = 2
  ctx.beginPath()
  r.circleTransformed.forEach((p, i) => i === 0 ? ctx.moveTo(s.px(p[0]), s.py(p[1])) : ctx.lineTo(s.px(p[0]), s.py(p[1])))
  ctx.closePath(); ctx.stroke()

  drawAxes(ctx, s, { x: r.domain.x as [number, number], y: r.domain.y as [number, number] }, { color: dark ? '#a8a19a' : '#7d766d', gridColor: 'transparent' })

  drawArrow(ctx, s, r.basis.ex, '#2f6b3e')
  drawArrow(ctx, s, r.basis.ey, '#8B5CF6')

  // user-placed point + its transform
  if (pt) {
    const M = r.matrix
    const tx = M[0][0] * pt.x + M[0][1] * pt.y
    const ty = M[1][0] * pt.x + M[1][1] * pt.y
    // connecting line original -> transformed
    ctx.strokeStyle = dark ? 'rgba(208,197,182,0.5)' : 'rgba(99,91,79,0.5)'
    ctx.setLineDash([4, 4])
    ctx.lineWidth = 1.5
    ctx.beginPath()
    ctx.moveTo(s.px(pt.x), s.py(pt.y))
    ctx.lineTo(s.px(tx), s.py(ty))
    ctx.stroke()
    ctx.setLineDash([])

    // original point
    ctx.fillStyle = '#2f6b3e'
    ctx.beginPath(); ctx.arc(s.px(pt.x), s.py(pt.y), 7, 0, Math.PI * 2); ctx.fill()
    ctx.strokeStyle = dark ? '#1A1917' : '#fef9ef'; ctx.lineWidth = 2; ctx.stroke()
    ctx.fillStyle = dark ? '#d6d0c4' : '#4b463e'
    ctx.font = '600 12px Manrope'
    ctx.fillText(`p (${pt.x.toFixed(2)}, ${pt.y.toFixed(2)})`, s.px(pt.x) + 10, s.py(pt.y) - 10)

    // transformed point
    ctx.fillStyle = '#C8604A'
    ctx.beginPath(); ctx.arc(s.px(tx), s.py(ty), 7, 0, Math.PI * 2); ctx.fill()
    ctx.strokeStyle = dark ? '#1A1917' : '#fef9ef'; ctx.lineWidth = 2; ctx.stroke()
    ctx.fillStyle = '#C8604A'
    ctx.fillText(`M·p (${tx.toFixed(2)}, ${ty.toFixed(2)})`, s.px(tx) + 10, s.py(ty) - 10)
  }

  if (hover && !pt) {
    ctx.strokeStyle = 'rgba(99,91,79,0.35)'
    ctx.setLineDash([3, 3])
    ctx.beginPath(); ctx.moveTo(s.px(hover.x), s.py(r.domain.y[0])); ctx.lineTo(s.px(hover.x), s.py(r.domain.y[1])); ctx.stroke()
    ctx.beginPath(); ctx.moveTo(s.px(r.domain.x[0]), s.py(hover.y)); ctx.lineTo(s.px(r.domain.x[1]), s.py(hover.y)); ctx.stroke()
    ctx.setLineDash([])
  }
}

export default function MatrixTransformLab({ result }: {
  result: LabResult | null; loading: boolean; error: string | null
  onAction: (k: string) => void; params: LabParams; setParams: (p: LabParams) => void
}) {
  const ref = useRef<HTMLCanvasElement>(null)
  const scaleRef = useRef<Scale | null>(null)
  const r = result as MatrixResult | null
  const [pt, setPt] = useState<{ x: number; y: number } | null>(null)
  const [hover, setHover] = useState<{ x: number; y: number } | null>(null)

  useEffect(() => {
    if (r && ref.current) {
      drawScene(ref.current, r, pt, hover)
      const ctx = ref.current.getContext('2d')!
      scaleRef.current = makeScale(ctx, { x: r.domain.x as [number, number], y: r.domain.y as [number, number] }, { l: 20, r: 20, t: 20, b: 20 })
    }
  }, [r, pt, hover])

  // clear the placed point when the matrix changes
  useEffect(() => {
    setPt(null)
  }, [r?.matrix])

  const { handlers } = useCanvasDrag({
    getScale: () => scaleRef.current,
    onDown: (x, y) => { setPt({ x, y }); return true },
    onDrag: (x, y) => { setPt({ x, y }) },
    onHover: (x, y) => { if (r && !Number.isNaN(x)) setHover({ x, y }) },
    onUp: () => {},
  })

  if (!r) return null

  return (
    <div className="flex flex-col lg:flex-row gap-4">
      <div className="bg-surface-container-lowest dark:bg-dark-surface rounded-3xl p-4 md:p-6 shadow-ambient dark:shadow-dark-ambient border border-outline-variant/40 dark:border-white/5">
        <h3 className="font-headline text-lg text-on-surface dark:text-inverse-on-surface mb-3 inline-flex items-center gap-2">
          <span className="material-symbols-outlined" style={{ fontSize: 20 }}>grid_view</span>
          Linear transform
          <span className="text-caption font-normal normal-case text-outline ml-2">(click to place a point)</span>
        </h3>
        <div className="w-full overflow-x-auto flex justify-center">
          <canvas ref={ref} className="rounded-2xl cursor-crosshair touch-none" {...handlers} />
        </div>
        {pt && (
          <p className="mt-3 text-caption text-on-surface-variant dark:text-outline">
            p = ({pt.x.toFixed(2)}, {pt.y.toFixed(2)}) → M·p = ({(
              r.matrix[0][0] * pt.x + r.matrix[0][1] * pt.y
            ).toFixed(2)}, {(
              r.matrix[1][0] * pt.x + r.matrix[1][1] * pt.y
            ).toFixed(2)})
          </p>
        )}
      </div>

      <div className="lg:w-72 flex flex-col gap-3">
        <div className="bg-surface-container-lowest dark:bg-dark-surface rounded-2xl p-4 border border-outline-variant/40 dark:border-white/5">
          <h4 className="text-caption uppercase tracking-wider text-outline mb-3">Matrix</h4>
          <div className="grid grid-cols-2 gap-2 font-mono text-lg text-center">
            {r.matrix.flat().map((v, i) => (
              <div key={i} className="bg-surface-container dark:bg-white/5 rounded-lg py-2 text-primary dark:text-inverse-primary font-bold">{v.toFixed(2)}</div>
            ))}
          </div>
        </div>

        <div className="grid grid-cols-3 gap-2">
          <Stat label="det" value={r.det.toFixed(2)} />
          <Stat label="trace" value={r.trace.toFixed(2)} />
          <Stat label="rank" value={String(r.rank)} />
        </div>

        <div className="bg-surface-container-lowest dark:bg-dark-surface rounded-2xl p-4 border border-outline-variant/40 dark:border-white/5">
          <h4 className="text-caption uppercase tracking-wider text-outline mb-2">Eigenvalues</h4>
          {r.eigen.length > 0 ? r.eigen.map((e, i) => (
            <div key={i} className="flex items-center gap-2 mb-2 last:mb-0">
              <span className="font-mono text-sm font-bold text-primary dark:text-inverse-primary w-12">λ={e.value.toFixed(2)}</span>
              <span className="font-mono text-xs text-on-surface-variant dark:text-outline">v=[{e.vector[0].toFixed(2)}, {e.vector[1].toFixed(2)}]</span>
            </div>
          )) : <p className="text-caption text-outline italic">Complex (e.g. rotation)</p>}
        </div>

        <div className="flex gap-3 text-caption">
          <span className="inline-flex items-center gap-1.5 text-on-surface-variant dark:text-outline"><span className="w-3 h-0.5 bg-[#2f6b3e]" /> e₁</span>
          <span className="inline-flex items-center gap-1.5 text-on-surface-variant dark:text-outline"><span className="w-3 h-0.5 bg-[#8B5CF6]" /> e₂</span>
          <span className="inline-flex items-center gap-1.5 text-on-surface-variant dark:text-outline"><span className="w-3 h-0.5 bg-[#C8604A]" /> square</span>
        </div>
      </div>
    </div>
  )
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="bg-surface-container-lowest dark:bg-dark-surface rounded-2xl p-3 border border-outline-variant/40 dark:border-white/5 text-center">
      <div className="font-mono text-base font-bold text-primary dark:text-inverse-primary">{value}</div>
      <div className="text-caption text-outline mt-0.5 uppercase tracking-wider">{label}</div>
    </div>
  )
}
