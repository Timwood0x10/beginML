import { useEffect, useRef, useState } from 'react'
import type { LabResult, LabParams } from '../types'
import { setupCanvas, makeScale, drawAxes, type Domain, type Scale } from '../canvas'
import { useCanvasDrag } from '../useCanvasDrag'

interface BernoulliResult extends LabResult {
  mode: 'bernoulli'
  x: number[]
  entropy: number[]
  kl: number[]
  crossEntropy: number[]
  domain: Domain
  p: number
  entropyP: number
  klAtP: number
  ceAtP: number
  formula: string
}
interface CategoricalResult extends LabResult {
  mode: 'categorical'
  categories: string[]
  p: number[]
  q: number[]
  domain: Domain
  entropyP: number
  entropyQ: number
  crossEntropy: number
  kl: number
  temperature: number
  formula: string
}

export default function EntropyLab({ result, params, setParams }: {
  result: LabResult | null; loading: boolean; error: string | null
  onAction: (k: string) => void; params: LabParams; setParams: (p: LabParams) => void
}) {
  if (!result) return null
  return result.mode === 'categorical'
    ? <CategoricalView r={result as CategoricalResult} />
    : <BernoulliView r={result as BernoulliResult} params={params} setParams={setParams} />
}

interface Inspect { x: number; h: number; kl: number; ce: number }

function BernoulliView({ r, params, setParams }: {
  r: BernoulliResult; params: LabParams; setParams: (p: LabParams) => void
}) {
  const ref = useRef<HTMLCanvasElement>(null)
  const scaleRef = useRef<Scale | null>(null)
  const [inspect, setInspect] = useState<Inspect | null>(null)
  const [hover, setHover] = useState<number | null>(null)

  // Look up curve values at a given q by nearest sample (no formula here).
  const sampleAt = (q: number): Inspect => {
    let idx = 0
    let best = Infinity
    r.x.forEach((xv, i) => {
      const d = Math.abs(xv - q)
      if (d < best) { best = d; idx = i }
    })
    return { x: r.x[idx], h: r.entropy[idx], kl: r.kl[idx], ce: r.crossEntropy[idx] }
  }

  useEffect(() => {
    if (!ref.current) return
    const W = 640, H = 420
    const ctx = setupCanvas(ref.current, W, H)
    const s = makeScale(ctx, { x: r.domain.x as [number, number], y: r.domain.y as [number, number] }, { l: 52, r: 20, t: 20, b: 40 })
    scaleRef.current = s
    const dark = document.documentElement.classList.contains('dark')
    ctx.fillStyle = dark ? '#1E1913' : '#F7F0E3'
    ctx.fillRect(0, 0, W, H)
    drawAxes(ctx, s, { x: r.domain.x as [number, number], y: r.domain.y as [number, number] },
      { color: dark ? '#A99B82' : '#8A7A61', gridColor: dark ? 'rgba(255,255,255,0.06)' : 'rgba(125,118,109,0.13)' })

    const curves: [number[], string][] = [
      [r.entropy, '#7A5C36'], [r.kl, '#C8604A'], [r.crossEntropy, '#5B6BB0'],
    ]
    curves.forEach(([data, color]) => {
      ctx.strokeStyle = color
      ctx.lineWidth = 2.2
      ctx.beginPath()
      data.forEach((v, i) => {
        const cx = s.px(r.x[i]), cy = s.py(v)
        if (i === 0) ctx.moveTo(cx, cy); else ctx.lineTo(cx, cy)
      })
      ctx.stroke()
    })

    // hover guide
    if (hover !== null && !inspect) {
      ctx.strokeStyle = 'rgba(99,91,79,0.4)'
      ctx.setLineDash([3, 3])
      ctx.beginPath(); ctx.moveTo(s.px(hover), s.py(r.domain.y[0])); ctx.lineTo(s.px(hover), s.py(r.domain.y[1])); ctx.stroke()
      ctx.setLineDash([])
    }

    // P line
    ctx.strokeStyle = '#2f6b3e'
    ctx.lineWidth = 1.5
    ctx.setLineDash([4, 4])
    ctx.beginPath()
    ctx.moveTo(s.px(r.p), s.py(r.domain.y[0]))
    ctx.lineTo(s.px(r.p), s.py(r.domain.y[1]))
    ctx.stroke()
    ctx.setLineDash([])

    // inspect marker
    if (inspect) {
      const ix = s.px(inspect.x)
      ctx.strokeStyle = '#2f6b3e'
      ctx.lineWidth = 2
      ctx.beginPath(); ctx.moveTo(ix, s.py(r.domain.y[0])); ctx.lineTo(ix, s.py(r.domain.y[1])); ctx.stroke()
      const pts: [number, string][] = [[inspect.h, '#7A5C36'], [inspect.kl, '#C8604A'], [inspect.ce, '#5B6BB0']]
      pts.forEach(([v, color]) => {
        ctx.fillStyle = color
        ctx.beginPath(); ctx.arc(ix, s.py(v), 5, 0, Math.PI * 2); ctx.fill()
      })
    }

    // legend
    const items: [string, string, string][] = [
      ['#7A5C36', 'H(Q)', 'entropy'],
      ['#C8604A', 'KL(P||Q)', 'divergence'],
      ['#5B6BB0', 'H(P,Q)', 'cross-entropy'],
    ]
    ctx.font = '600 12px Manrope'
    items.forEach(([color, sym, label], i) => {
      ctx.fillStyle = color
      ctx.fillRect(W - 140, 18 + i * 18, 14, 3)
      ctx.fillStyle = dark ? '#C9BCA6' : '#54483A'
      ctx.fillText(`${sym} ${label}`, W - 120, 24 + i * 18)
    })
    ctx.fillStyle = '#2f6b3e'
    ctx.fillText(`P = ${r.p.toFixed(2)}`, s.px(r.p) + 8, 36)
  }, [r, inspect, hover])

  const { handlers } = useCanvasDrag({
    getScale: () => scaleRef.current,
    onDown: (x) => { setInspect(sampleAt(x)); return true },
    onDrag: (x) => { setInspect(sampleAt(x)) },
    onHover: (x) => { if (!Number.isNaN(x)) setHover(x) },
    onUp: () => {},
  })

  return (
    <div className="flex flex-col gap-4">
      <FormulaCard formula={r.formula} title="Bernoulli P vs Q" />
      <div className="bg-surface-container-lowest dark:bg-dark-surface rounded-3xl p-4 md:p-6 shadow-ambient dark:shadow-dark-ambient border border-outline-variant/40 dark:border-white/10">
        <div className="w-full overflow-x-auto flex justify-center">
          <canvas ref={ref} className="rounded-2xl cursor-crosshair touch-none" {...handlers} />
        </div>
        <p className="mt-3 text-caption text-on-surface-variant dark:text-outline">Click the chart to inspect H, KL and cross-entropy at any Q.</p>
      </div>
      <div className="grid grid-cols-3 gap-3">
        <Stat label="H(P)" value={r.entropyP.toFixed(3)} />
        <Stat label="KL(P||P)" value={r.klAtP.toFixed(3)} />
        <Stat label="H(P,P)" value={r.ceAtP.toFixed(3)} />
      </div>
      {inspect && (
        <div className="bg-surface-container-low dark:bg-dark-surface rounded-2xl p-4 border border-outline-variant/40 dark:border-white/10 grid grid-cols-3 gap-3">
          <Stat label={`H(Q=${inspect.x.toFixed(2)})`} value={inspect.h.toFixed(3)} />
          <Stat label={`KL(P||Q)`} value={inspect.kl.toFixed(3)} />
          <Stat label={`H(P,Q)`} value={inspect.ce.toFixed(3)} />
        </div>
      )}
    </div>
  )
}

function CategoricalView({ r }: { r: CategoricalResult }) {
  const maxVal = Math.max(...r.p, ...r.q, 0.01)
  return (
    <div className="flex flex-col gap-4">
      <FormulaCard formula={r.formula} title={`Categorical (k=${r.categories.length}, T=${r.temperature})`} />
      <div className="bg-surface-container-lowest dark:bg-dark-surface rounded-3xl p-6 border border-outline-variant/40 dark:border-white/10">
        <div className="flex items-end gap-3 h-48">
          {r.categories.map((cat, i) => {
            const hp = (r.p[i] / maxVal) * 100
            const hq = (r.q[i] / maxVal) * 100
            return (
              <div key={cat} className="flex-1 flex flex-col items-center gap-1">
                <div className="flex items-end gap-1 h-full w-full">
                  <div className="flex-1 rounded-t-md transition-all" style={{ height: `${hp}%`, background: '#7A5C36' }} title={`P=${r.p[i].toFixed(3)}`} />
                  <div className="flex-1 rounded-t-md transition-all" style={{ height: `${hq}%`, background: '#C8604A' }} title={`Q=${r.q[i].toFixed(3)}`} />
                </div>
                <span className="text-caption text-outline font-mono">{cat}</span>
              </div>
            )
          })}
        </div>
        <div className="flex gap-4 mt-4 text-caption">
          <span className="inline-flex items-center gap-1.5 text-on-surface-variant dark:text-outline"><span className="w-3 h-3 rounded-sm bg-[#7A5C36]" /> P (true)</span>
          <span className="inline-flex items-center gap-1.5 text-on-surface-variant dark:text-outline"><span className="w-3 h-3 rounded-sm bg-[#C8604A]" /> Q (model, T={r.temperature})</span>
        </div>
      </div>
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <Stat label="H(P)" value={r.entropyP.toFixed(3)} />
        <Stat label="H(Q)" value={r.entropyQ.toFixed(3)} />
        <Stat label="H(P,Q)" value={r.crossEntropy.toFixed(3)} />
        <Stat label="KL(P||Q)" value={r.kl.toFixed(3)} highlight />
      </div>
    </div>
  )
}

function FormulaCard({ formula, title }: { formula: string; title: string }) {
  return (
    <div className="bg-surface-container-lowest dark:bg-dark-surface rounded-3xl p-5 border border-outline-variant/40 dark:border-white/10 flex items-center justify-between gap-4 flex-wrap">
      <h3 className="font-headline text-lg text-on-surface dark:text-inverse-on-surface inline-flex items-center gap-2">
        <span className="material-symbols-outlined" style={{ fontSize: 20 }}>water_drop</span>
        {title}
      </h3>
      <code className="font-mono text-sm text-primary dark:text-inverse-primary bg-primary-fixed/40 dark:bg-white/5 px-3 py-1.5 rounded-lg">{formula}</code>
    </div>
  )
}

function Stat({ label, value, highlight }: { label: string; value: string; highlight?: boolean }) {
  return (
    <div className={`rounded-2xl p-4 border text-center ${
      highlight
        ? 'bg-primary/10 dark:bg-inverse-primary/15 border-primary/30 dark:border-inverse-primary/30'
        : 'bg-surface-container-lowest dark:bg-dark-surface border-outline-variant/40 dark:border-white/10'
    }`}>
      <div className={`font-mono text-xl font-bold ${highlight ? 'text-primary dark:text-inverse-primary' : 'text-primary dark:text-inverse-primary'}`}>{value}</div>
      <div className="text-caption text-outline mt-1 uppercase tracking-wider">{label}</div>
    </div>
  )
}
