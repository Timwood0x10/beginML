import { useMemo, useState } from 'react'
import type { LabResult } from '../types'

// Interactive visualization for the transformer-training lab module.
// Shows the training loss curve plus the per-head attention heatmaps that
// emerge after training on the predict-the-previous-token task.

interface TransformerResult extends LabResult {
  tokens: string[]
  n: number
  heads: number
  layers: number
  lr: number
  epochs: number
  causal: boolean
  losses: number[]
  finalLoss: number
  attn: number[][][][] // layers -> heads -> n x n
  task: string
}

export default function TransformerLab({ result }: { result: LabResult | null }) {
  const r = result as TransformerResult | null
  const [layer, setLayer] = useState(0)
  const [head, setHead] = useState(0)
  const [hover, setHover] = useState<{ row: number; col: number } | null>(null)

  if (!r) return null

  const safeLayer = Math.min(layer, Math.max(0, r.attn.length - 1))
  const layerAttn = r.attn[safeLayer] ?? []
  const safeHead = Math.min(head, Math.max(0, layerAttn.length - 1))
  const matrix: number[][] = layerAttn[safeHead] ?? []

  return (
    <div className="flex flex-col gap-5">
      <LossCurve losses={r.losses} epochs={r.epochs} finalLoss={r.finalLoss} />
      <div className="bg-surface-container-lowest dark:bg-dark-surface rounded-3xl p-6 shadow-ambient dark:shadow-dark-ambient border border-outline-variant/40 dark:border-white/10">
        <div className="flex flex-wrap items-center justify-between gap-3 mb-4">
          <h3 className="font-headline text-lg text-on-surface dark:text-inverse-on-surface inline-flex items-center gap-2">
            <span className="material-symbols-outlined" style={{ fontSize: 20 }}>grid_on</span>
            Learned attention (layer {safeLayer + 1}, head {safeHead + 1})
          </h3>
          <div className="flex flex-col items-end gap-2">
            <div className="inline-flex rounded-xl bg-surface-container dark:bg-white/5 p-1">
              {Array.from({ length: r.layers }).map((_, li) => (
                <button
                  key={li}
                  onClick={() => { setLayer(li); setHead(0) }}
                  className={`px-3 py-1.5 rounded-lg text-caption font-semibold transition ${
                    safeLayer === li ? 'bg-primary text-on-primary dark:bg-inverse-primary dark:text-inverse-surface' : 'text-on-surface-variant dark:text-outline'
                  }`}
                >
                  L{li + 1}
                </button>
              ))}
            </div>
            <div className="inline-flex rounded-xl bg-surface-container dark:bg-white/5 p-1">
              {Array.from({ length: r.heads }).map((_, hi) => (
                <button
                  key={hi}
                  onClick={() => setHead(hi)}
                  className={`px-3 py-1.5 rounded-lg text-caption font-semibold transition ${
                    safeHead === hi ? 'bg-primary text-on-primary dark:bg-inverse-primary dark:text-inverse-surface' : 'text-on-surface-variant dark:text-outline'
                  }`}
                >
                  H{hi + 1}
                </button>
              ))}
            </div>
          </div>
        </div>

        <div className="w-full overflow-x-auto">
          <div className="inline-block min-w-full">
            <div className="flex ml-16">
              {r.tokens.map((t) => (
                <div key={t} className="w-11 md:w-13 text-center text-caption font-semibold text-on-surface-variant dark:text-outline pb-1">
                  {t}
                </div>
              ))}
            </div>
            {matrix.map((row, i) => (
              <div key={i} className="flex items-center">
                <div className="w-16 text-right pr-2 text-caption font-semibold text-on-surface-variant dark:text-outline shrink-0">
                  {r.tokens[i]}
                </div>
                {row.map((v, j) => {
                  const masked = r.causal && j > i
                  const isHover = hover?.row === i && hover?.col === j
                  return (
                    <div
                      key={j}
                      onMouseEnter={() => setHover({ row: i, col: j })}
                      onMouseLeave={() => setHover(null)}
                      className={`w-11 h-11 md:w-13 md:h-13 m-0.5 rounded-lg flex items-center justify-center font-mono text-caption transition-transform ${isHover ? 'ring-2 ring-primary dark:ring-inverse-primary scale-105 z-10' : ''}`}
                      style={{
                        background: masked
                          ? 'repeating-linear-gradient(45deg, rgba(125,118,109,0.12), rgba(125,118,109,0.12) 4px, transparent 4px, transparent 8px)'
                          : heatColor(v),
                        color: v > 0.55 ? '#fff' : '#3B3023',
                      }}
                      title={`${r.tokens[i]} → ${r.tokens[j]}: ${v.toFixed(3)}${masked ? ' (masked)' : ''}`}
                    >
                      {masked ? '' : v.toFixed(2)}
                    </div>
                  )
                })}
              </div>
            ))}
          </div>
        </div>

        {hover && !(r.causal && hover.col > hover.row) && (
          <p className="mt-3 text-caption text-on-surface-variant dark:text-outline">
            <span className="font-mono text-primary dark:text-inverse-primary">{r.tokens[hover.row]}</span>
            {' attends to '}
            <span className="font-mono text-primary dark:text-inverse-primary">{r.tokens[hover.col]}</span>
            {' with weight '}
            <span className="font-mono font-semibold">{matrix[hover.row][hover.col].toFixed(3)}</span>
          </p>
        )}
        <p className="mt-2 text-caption text-outline">
          Task: {r.task}. The model learns to attend one step back — watch the
          diagonal pattern sharpen as it trains.
        </p>
      </div>
    </div>
  )
}

// Cream -> clay heat color for attention weights in [0, 1].
function heatColor(v: number): string {
  const t = Math.max(0, Math.min(1, v))
  const r1 = Math.round(242 + (184 - 242) * t)
  const g1 = Math.round(237 + (96 - 237) * t)
  const b1 = Math.round(227 + (74 - 227) * t)
  return `rgb(${r1},${g1},${b1})`
}

// Hand-drawn SVG loss curve with gridlines and a final-loss callout.
function LossCurve({ losses, epochs, finalLoss }: { losses: number[]; epochs: number; finalLoss: number }) {
  const W = 640
  const H = 220
  const PAD = { l: 46, r: 14, t: 16, b: 30 }
  const maxLoss = useMemo(() => Math.max(...losses, 0.01), [losses])

  const points = losses.map((v, i) => {
    const x = PAD.l + (i / Math.max(1, losses.length - 1)) * (W - PAD.l - PAD.r)
    const y = PAD.t + (1 - v / maxLoss) * (H - PAD.t - PAD.b)
    return { x, y }
  })
  const path = points.map((p, i) => `${i === 0 ? 'M' : 'L'}${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(' ')

  return (
    <div className="bg-surface-container-lowest dark:bg-dark-surface rounded-3xl p-6 shadow-ambient dark:shadow-dark-ambient border border-outline-variant/40 dark:border-white/10">
      <div className="flex flex-wrap items-center justify-between gap-3 mb-3">
        <h3 className="font-headline text-lg text-on-surface dark:text-inverse-on-surface inline-flex items-center gap-2">
          <span className="material-symbols-outlined" style={{ fontSize: 20 }}>trending_down</span>
          Training loss
        </h3>
        <span className="inline-flex items-center gap-2 text-caption font-mono text-primary dark:text-inverse-primary bg-primary-fixed/40 dark:bg-white/5 px-3 py-1 rounded-xl">
          final loss: {finalLoss.toFixed(4)}
        </span>
      </div>

      <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img" aria-label="Training loss curve">
        {[0, 0.25, 0.5, 0.75, 1].map((t) => {
          const y = PAD.t + t * (H - PAD.t - PAD.b)
          const val = (maxLoss * (1 - t)).toFixed(2)
          return (
            <g key={t}>
              <line x1={PAD.l} y1={y} x2={W - PAD.r} y2={y} stroke="rgba(140,122,107,0.18)" strokeDasharray="3 4" />
              <text x={PAD.l - 8} y={y + 4} textAnchor="end" className="fill-on-surface-variant dark:fill-outline" fontSize="11">
                {val}
              </text>
            </g>
          )
        })}
        <line x1={PAD.l} y1={PAD.t} x2={PAD.l} y2={H - PAD.b} stroke="rgba(140,122,107,0.35)" />
        <line x1={PAD.l} y1={H - PAD.b} x2={W - PAD.r} y2={H - PAD.b} stroke="rgba(140,122,107,0.35)" />
        <text x={(PAD.l + W - PAD.r) / 2} y={H - 8} textAnchor="middle" className="fill-on-surface-variant dark:fill-outline" fontSize="11">
          training step ({epochs} total)
        </text>
        {path && <path d={path} fill="none" stroke="#A8382A" strokeWidth="2.2" strokeLinejoin="round" strokeLinecap="round" />}
        {points.length > 0 && (
          <circle cx={points[points.length - 1].x} cy={points[points.length - 1].y} r="4" fill="#A8382A" />
        )}
      </svg>
    </div>
  )
}
