import { useEffect, useRef } from 'react'
import type { LabResult } from '../types'
import { setupCanvas, makeScale, type Domain } from '../canvas'

interface Point { x: number; y: number; cls: number; prob: number }
interface NnResult extends LabResult {
  dataset: string
  points: Point[]
  grid: { x: number[]; y: number[]; z: number[][] }
  domain: Domain
  losses: number[]
  finalLoss: number | null
  accuracy: number
  architecture: string
  epochs: number
  learningRate: number
}

function drawBoundary(
  ctx: CanvasRenderingContext2D, s: ReturnType<typeof makeScale>,
  grid: { x: number[]; y: number[]; z: number[][] },
) {
  const { x, y, z } = grid
  for (let i = 0; i < x.length - 1; i++) {
    for (let j = 0; j < y.length - 1; j++) {
      const v = z[j][i]
      const t = Math.max(0, Math.min(1, v))
      const r = Math.round(200 + (91 - 200) * t)
      const g = Math.round(96 + (107 - 96) * t)
      const b = Math.round(74 + (176 - 74) * t)
      ctx.fillStyle = `rgba(${r},${g},${b},0.35)`
      const cx = s.px(x[i]), cy = s.py(y[j + 1])
      const w = s.px(x[i + 1]) - s.px(x[i])
      const h = s.py(y[j]) - s.py(y[j + 1])
      ctx.fillRect(cx, cy, w + 1, h + 1)
    }
  }
}

function drawContourLine(
  ctx: CanvasRenderingContext2D, s: ReturnType<typeof makeScale>,
  grid: { x: number[]; y: number[]; z: number[][] }, level: number,
) {
  const { x, y, z } = grid
  ctx.strokeStyle = 'rgba(29,28,22,0.5)'
  ctx.lineWidth = 1.5
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
      const topX = xL + ((level - v00) / safe(v00, v10 - v00)) * (xR - xL)
      const rightY = yT + ((level - v10) / safe(v10, v11 - v10)) * (yB - yT)
      const bottomX = xR - ((level - v11) / safe(v11, v01 - v11)) * (xR - xL)
      const leftY = yB - ((level - v01) / safe(v01, v00 - v01)) * (yB - yT)
      const seg: [number, number, number, number][] = []
      switch (idx) {
        case 1: case 14: seg.push([topX, yT, xL, leftY]); break
        case 2: case 13: seg.push([topX, yT, xR, rightY]); break
        case 3: case 12: seg.push([xL, leftY, xR, rightY]); break
        case 4: case 11: seg.push([xR, rightY, bottomX, yB]); break
        case 6: case 9: seg.push([topX, yT, bottomX, yB]); break
        case 7: case 8: seg.push([xL, leftY, bottomX, yB]); break
        case 5: seg.push([topX, yT, xL, leftY]); seg.push([xR, rightY, bottomX, yB]); break
        case 10: seg.push([topX, yT, xR, rightY]); seg.push([xL, leftY, bottomX, yB]); break
      }
      seg.forEach(([x1, y1, x2, y2]) => { ctx.moveTo(x1, y1); ctx.lineTo(x2, y2) })
    }
  }
  ctx.stroke()
}

export default function NeuralNetLab({ result }: { result: LabResult | null }) {
  const boundaryRef = useRef<HTMLCanvasElement>(null)
  const lossRef = useRef<HTMLCanvasElement>(null)
  const r = result as NnResult | null

  useEffect(() => {
    if (!r || !boundaryRef.current) return
    const W = 560, H = 480
    const ctx = setupCanvas(boundaryRef.current, W, H)
    const s = makeScale(ctx, { x: r.domain.x as [number, number], y: r.domain.y as [number, number] },
      { l: 20, r: 20, t: 20, b: 20 })
    const dark = document.documentElement.classList.contains('dark')
    ctx.fillStyle = dark ? '#1A1917' : '#fef9ef'
    ctx.fillRect(0, 0, W, H)
    drawBoundary(ctx, s, r.grid)
    drawContourLine(ctx, s, r.grid, 0.5)

    r.points.forEach((p) => {
      ctx.fillStyle = p.cls === 1 ? '#5B6BB0' : '#C8604A'
      ctx.beginPath()
      ctx.arc(s.px(p.x), s.py(p.y), 5, 0, Math.PI * 2)
      ctx.fill()
      ctx.strokeStyle = dark ? '#1A1917' : '#fef9ef'
      ctx.lineWidth = 1.5
      ctx.stroke()
    })
  }, [r])

  useEffect(() => {
    if (!r || !lossRef.current) return
    const W = 560, H = 180
    const ctx = setupCanvas(lossRef.current, W, H)
    const pad = { l: 48, r: 16, t: 16, b: 28 }
    const maxLoss = Math.max(...r.losses, 0.01)
    const s = makeScale(ctx, { x: [0, r.losses.length - 1], y: [0, maxLoss * 1.1] }, pad)
    const dark = document.documentElement.classList.contains('dark')
    ctx.fillStyle = dark ? '#1A1917' : '#fef9ef'
    ctx.fillRect(0, 0, W, H)

    ctx.strokeStyle = dark ? 'rgba(255,255,255,0.08)' : 'rgba(125,118,109,0.12)'
    ctx.lineWidth = 1
    for (let i = 0; i <= 4; i++) {
      const yv = (i / 4) * maxLoss * 1.1
      const cy = s.py(yv)
      ctx.beginPath(); ctx.moveTo(pad.l, cy); ctx.lineTo(W - pad.r, cy); ctx.stroke()
    }

    ctx.strokeStyle = '#635b4f'
    ctx.lineWidth = 2
    ctx.beginPath()
    r.losses.forEach((v, i) => {
      const cx = s.px(i), cy = s.py(v)
      if (i === 0) ctx.moveTo(cx, cy); else ctx.lineTo(cx, cy)
    })
    ctx.stroke()

    ctx.fillStyle = dark ? '#a8a19a' : '#7d766d'
    ctx.font = '11px Manrope'
    ctx.fillText('epoch', W - 50, H - 8)
    ctx.save(); ctx.translate(12, H / 2); ctx.rotate(-Math.PI / 2)
    ctx.fillText('loss', 0, 0); ctx.restore()
  }, [r])

  if (!r) return null

  return (
    <div className="flex flex-col gap-4">
      <div className="bg-surface-container-lowest dark:bg-dark-surface rounded-3xl p-4 md:p-6 shadow-ambient dark:shadow-dark-ambient border border-outline-variant/40 dark:border-white/5">
        <h3 className="font-headline text-lg text-on-surface dark:text-inverse-on-surface mb-3 inline-flex items-center gap-2 capitalize">
          <span className="material-symbols-outlined" style={{ fontSize: 20 }}>bubble_chart</span>
          {r.dataset} dataset — {r.architecture} MLP
        </h3>
        <div className="w-full overflow-x-auto flex justify-center">
          <canvas ref={boundaryRef} className="rounded-2xl" />
        </div>
      </div>

      <div className="bg-surface-container-lowest dark:bg-dark-surface rounded-3xl p-4 md:p-6 shadow-ambient dark:shadow-dark-ambient border border-outline-variant/40 dark:border-white/5">
        <h4 className="font-label-md text-label-md uppercase tracking-wider text-on-surface dark:text-dark-on-surface mb-3">Training loss</h4>
        <div className="w-full overflow-x-auto flex justify-center">
          <canvas ref={lossRef} className="rounded-2xl" />
        </div>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <Stat label="accuracy" value={`${(r.accuracy * 100).toFixed(1)}%`} />
        <Stat label="final loss" value={r.finalLoss?.toFixed(4) ?? '—'} />
        <Stat label="epochs" value={String(r.epochs)} />
        <Stat label="learning rate" value={r.learningRate.toFixed(3)} />
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
