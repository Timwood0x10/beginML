import { useEffect, useRef, useState } from 'react'
import type { LabResult, LabParams } from '../types'
import { setupCanvas, makeScale, themeVar, type Scale } from '../canvas'
import { useTheme } from '../../hooks/useTheme'
import { useI18n } from '../../i18n/context'
import { labTextsZh, labTextsEn, fmt } from '../../i18n/lab'

interface RopeResult extends LabResult {
  position: number
  position_b: number
  frequency: number
  theta0: number
  dims: number
  pair: boolean
  distance_mode: boolean
  angle_m: number
  angle_n: number
  relative_phase: number
  point_m: [number, number]
  point_n: [number, number]
  similarity: number
  sim_at_4: number
  sim_at_8: number
  distance_curve: { distances: number[]; similarities: number[] }
  frequency_lens: { dim: number; theta: number; angle_at_m: number; angle_at_n: number }[]
  provenance: string
}

function drawObservatory(canvas: HTMLCanvasElement, r: RopeResult) {
  const W = 640, H = 560
  const ctx = setupCanvas(canvas, W, H)
  const s = makeScale(ctx, { x: [-1.5, 1.5], y: [-1.5, 1.5] }, { l: 20, r: 20, t: 20, b: 20 })
  const dark = document.documentElement.classList.contains('dark')
  const bg = themeVar('--ailearn-background', '#F7F0E3')
  const ink = dark ? '#A99B82' : '#8A7A61'
  ctx.fillStyle = bg
  ctx.fillRect(0, 0, W, H)

  const cx = s.px(0)
  const cy = s.py(0)
  const radius = s.px(1) - s.px(0)

  // axes
  ctx.strokeStyle = dark ? 'rgba(255,255,255,0.08)' : 'rgba(125,118,109,0.15)'
  ctx.lineWidth = 1
  ctx.beginPath(); ctx.moveTo(s.px(-1.4), cy); ctx.lineTo(s.px(1.4), cy); ctx.stroke()
  ctx.beginPath(); ctx.moveTo(cx, s.py(-1.4)); ctx.lineTo(cx, s.py(1.4)); ctx.stroke()
  // unit circle
  ctx.strokeStyle = dark ? 'rgba(255,255,255,0.25)' : 'rgba(125,118,109,0.35)'
  ctx.setLineDash([4, 4])
  ctx.beginPath(); ctx.arc(cx, cy, radius, 0, Math.PI * 2); ctx.stroke()
  ctx.setLineDash([])

  // rotation sweep for token A (from angle 0 to angle_m)
  if (r.angle_m !== 0) {
    ctx.fillStyle = 'rgba(200,96,74,0.16)'
    ctx.beginPath()
    ctx.moveTo(cx, cy)
    ctx.arc(cx, cy, radius, 0, -r.angle_m, true)
    ctx.closePath()
    ctx.fill()
    ctx.strokeStyle = 'rgba(200,96,74,0.5)'
    ctx.setLineDash([6, 4])
    ctx.beginPath(); ctx.arc(cx, cy, radius, 0, -r.angle_m, true); ctx.stroke()
    ctx.setLineDash([])
  }

  // token A star
  const [ax, ay] = r.point_m
  drawStar(ctx, s.px(ax), s.py(ay), 9, '#C8604A')
  ctx.fillStyle = dark ? '#C9BCA6' : '#54483A'
  ctx.font = '600 12px Manrope'
  ctx.fillText(`A @ pos ${r.position}`, s.px(ax) + 12, s.py(ay) - 10)

  // token B star
  if (r.pair) {
    const [bx, by] = r.point_n
    drawStar(ctx, s.px(bx), s.py(by), 9, '#5B6BB0')
    ctx.fillStyle = dark ? '#C9BCA6' : '#54483A'
    ctx.fillText(`B @ pos ${r.position_b}`, s.px(bx) + 12, s.py(by) - 10)

    // relative phase arc between the two stars (drawn from B's angle to A's)
    const a0 = -r.angle_n
    const a1 = -r.angle_m
    ctx.strokeStyle = dark ? 'rgba(91,107,176,0.7)' : 'rgba(91,107,176,0.6)'
    ctx.lineWidth = 2
    ctx.beginPath(); ctx.arc(cx, cy, radius * 0.55, Math.min(a0, a1), Math.max(a0, a1)); ctx.stroke()

    // phase label at mid-angle
    const mid = (a0 + a1) / 2
    const lr = radius * 0.68
    ctx.fillStyle = '#5B6BB0'
    ctx.font = '600 12px Manrope'
    ctx.fillText(`Δθ = ${r.relative_phase.toFixed(2)} rad`, cx + Math.cos(mid) * lr - 40, cy + Math.sin(mid) * lr + 4)
  }

  // "start" marker at angle 0
  ctx.fillStyle = dark ? 'rgba(255,255,255,0.4)' : 'rgba(125,118,109,0.4)'
  ctx.beginPath(); ctx.arc(cx + radius, cy, 3, 0, Math.PI * 2); ctx.fill()
  ctx.font = '11px Manrope'
  ctx.fillText('0 (no rotation)', cx + radius + 8, cy + 4)

  // angle annotation
  ctx.fillStyle = ink
  ctx.font = '12px Manrope'
  ctx.fillText(`angle A = ${r.angle_m.toFixed(2)} rad`, 24, 28)
  if (r.pair) ctx.fillText(`angle B = ${r.angle_n.toFixed(2)} rad`, 24, 46)
  ctx.fillText(`θ₀ = ${r.theta0.toFixed(3)} rad / pos`, 24, 64)
}

function drawStar(ctx: CanvasRenderingContext2D, x: number, y: number, r: number, color: string) {
  ctx.save()
  ctx.fillStyle = color
  ctx.strokeStyle = '#fff'
  ctx.lineWidth = 2
  ctx.beginPath()
  for (let i = 0; i < 10; i++) {
    const rad = i % 2 === 0 ? r : r * 0.45
    const a = (Math.PI / 5) * i - Math.PI / 2
    const px = x + Math.cos(a) * rad
    const py = y + Math.sin(a) * rad
    if (i === 0) ctx.moveTo(px, py)
    else ctx.lineTo(px, py)
  }
  ctx.closePath()
  ctx.fill()
  ctx.stroke()
  ctx.restore()
}

export default function RotaryObservatoryLab({ result, loading }: {
  result: LabResult | null; loading: boolean; error: string | null
  onAction: (k: string) => void; params: LabParams; setParams: (p: LabParams) => void
}) {
  const { lang } = useI18n()
  const texts = (lang === 'zh' ? labTextsZh : labTextsEn)['rotary-observatory']
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const scaleRef = useRef<Scale | null>(null)
  const r = result as RopeResult | null
  const { theme, palette } = useTheme()
  const [answer, setAnswer] = useState<string | null>(null)
  const [submitted, setSubmitted] = useState(false)

  useEffect(() => {
    if (r && canvasRef.current) {
      drawObservatory(canvasRef.current, r)
      const ctx = canvasRef.current.getContext('2d')!
      scaleRef.current = makeScale(ctx, { x: [-1.5, 1.5], y: [-1.5, 1.5] }, { l: 20, r: 20, t: 20, b: 20 })
    }
  }, [r, theme, palette])

  if (!r) {
    return <div className="p-10 text-on-surface-variant dark:text-outline">{loading ? texts.ui.computing : texts.ui.adjustControls}</div>
  }

  // Challenge verdict from actual computed evidence (deterministic, no LLM).
  const correct = answer === (r.sim_at_8 > r.sim_at_4 ? 'up' : r.sim_at_8 < r.sim_at_4 ? 'down' : 'same')
  const curve = r.distance_curve
  const curDist = Math.abs(r.position - r.position_b)
  const curveMax = Math.max(...curve.similarities.map(Math.abs), 0.01)

  return (
    <div className="flex flex-col gap-5">
      {/* Question layer */}
      <div className="bg-surface-container-lowest dark:bg-dark-surface rounded-3xl p-5 border border-outline-variant/40 dark:border-white/10">
        <div className="flex items-start gap-3">
          <span className="text-xl mt-0.5">🔍</span>
          <div>
            <div className="text-caption uppercase tracking-wider font-semibold text-outline mb-1">{texts.ui.question}</div>
            <p className="font-headline text-lg text-on-surface dark:text-inverse-on-surface leading-snug">
              {texts.question}
            </p>
          </div>
        </div>
      </div>

      {/* Observatory canvas */}
      <div className="bg-surface-container-lowest dark:bg-dark-surface rounded-3xl p-4 md:p-6 shadow-ambient dark:shadow-dark-ambient border border-outline-variant/40 dark:border-white/10">
        <div className="flex flex-wrap items-center justify-between gap-3 mb-3">
          <h3 className="font-headline text-lg text-on-surface dark:text-inverse-on-surface inline-flex items-center gap-2">
            <span className="material-symbols-outlined" style={{ fontSize: 20 }}>rotate_right</span>
            {texts.ui.title}
          </h3>
          <div className="inline-flex items-center gap-2 text-caption text-on-surface-variant dark:text-outline">
            <span className="material-symbols-outlined" style={{ fontSize: 16 }}>schedule</span>
            {texts.ui.similarityLabel} <span className="font-mono text-primary dark:text-inverse-primary">{r.similarity.toFixed(3)}</span>
          </div>
        </div>
        <div className="w-full overflow-x-auto flex justify-center">
          <canvas ref={canvasRef} className="rounded-2xl" style={{ maxWidth: 640 }} />
        </div>
        {r.distance_mode && (
          <div className="mt-3 text-center text-label-md font-semibold text-primary dark:text-inverse-primary">
            {texts.aha}
          </div>
        )}
      </div>

      {/* FIND THE DISTANCE: similarity vs distance */}
      {r.distance_mode && (
        <div className="bg-surface-container-lowest dark:bg-dark-surface rounded-3xl p-4 md:p-6 shadow-ambient dark:shadow-dark-ambient border border-outline-variant/40 dark:border-white/10">
          <h3 className="font-headline text-lg text-on-surface dark:text-inverse-on-surface mb-4 inline-flex items-center gap-2">
            <span className="material-symbols-outlined" style={{ fontSize: 20 }}>straighten</span>
            {texts.ui.findDistance}
          </h3>
          <div className="flex items-end gap-1 h-40">
            {curve.distances.map((d, i) => {
              const v = curve.similarities[i]
              const isCurrent = d === curDist
              const h = (Math.abs(v) / curveMax) * 100
              return (
                <div key={d} className="flex-1 flex flex-col justify-end items-center min-w-0" title={`d=${d} → sim=${v.toFixed(3)}`}>
                  <div
                    className="w-full rounded-t"
                    style={{
                      height: `${Math.max(h, 1)}%`,
                      background: v >= 0 ? '#2f6b3e' : '#C8604A',
                      opacity: isCurrent ? 1 : 0.45,
                      boxShadow: isCurrent ? '0 0 0 2px rgba(91,107,176,0.7)' : 'none',
                    }}
                  />
                </div>
              )
            })}
          </div>
          <div className="flex mt-1">
            {curve.distances.map((d) => (
              <span key={d} className="flex-1 text-center text-[9px] text-outline">{d % 4 === 0 ? d : ''}</span>
            ))}
          </div>
          <p className="mt-3 text-caption text-on-surface-variant dark:text-outline">
            {fmt(texts.ui.distanceCaption, {
              pa: r.position,
              pb: r.position_b,
              d: curDist,
              s: r.similarity.toFixed(3),
            })}
          </p>
        </div>
      )}

      {/* Frequency lens */}
      <div className="bg-surface-container-lowest dark:bg-dark-surface rounded-3xl p-4 md:p-6 shadow-ambient dark:shadow-dark-ambient border border-outline-variant/40 dark:border-white/10">
        <h3 className="font-headline text-lg text-on-surface dark:text-inverse-on-surface mb-1 inline-flex items-center gap-2">
          <span className="material-symbols-outlined" style={{ fontSize: 20 }}>lens_blur</span>
          {texts.ui.frequencyLens}
        </h3>
        <p className="text-caption text-on-surface-variant dark:text-outline mb-4">
          {texts.ui.frequencyLensHint}
        </p>
        <div className="flex flex-col gap-1.5">
          {r.frequency_lens.map((d) => (
            <div key={d.dim} className="flex items-center gap-3">
              <span className="w-8 text-right text-caption font-mono text-outline">d{d.dim}</span>
              <div className="flex-1 h-4 bg-surface-container dark:bg-white/5 rounded-md overflow-hidden">
                <div
                  className="h-full rounded-md"
                  style={{
                    width: `${(d.theta / r.frequency_lens[0].theta) * 100}%`,
                    background: d.dim % 2 === 0 ? '#5B6BB0' : '#7A5C36',
                    opacity: 0.6 + 0.4 * (d.theta / r.frequency_lens[0].theta),
                  }}
                />
              </div>
              <span className="w-24 text-caption font-mono text-on-surface-variant dark:text-outline tabular-nums text-right">
                {d.theta.toFixed(4)} {texts.ui.radPerPos}
              </span>
            </div>
          ))}
        </div>
      </div>

      {/* Challenge layer — independent of Controls */}
      <div className="bg-surface-container-lowest dark:bg-dark-surface rounded-3xl p-5 border border-outline-variant/40 dark:border-white/10">
        <div className="flex items-center gap-2 mb-1">
          <span className="text-lg">🧪</span>
          <h3 className="font-label-md text-label-md uppercase tracking-wider text-on-surface dark:text-dark-on-surface">{texts.ui.makePrediction}</h3>
        </div>
        <p className="text-body-md text-on-surface dark:text-inverse-on-surface mb-3">{texts.challengeQuestion}</p>
        <div className="flex flex-wrap gap-2 mb-3">
          {texts.challengeOptions.map((opt) => (
            <button
              key={opt.value}
              onClick={() => { setAnswer(opt.value); setSubmitted(false) }}
              className={`px-4 py-2 rounded-xl text-label-md font-semibold border transition-all ${
                answer === opt.value
                  ? 'bg-primary text-on-primary border-primary dark:bg-inverse-primary dark:text-inverse-surface'
                  : 'bg-surface-container-lowest dark:bg-dark-surface text-on-surface-variant dark:text-outline border-outline-variant/60 dark:border-white/10 hover:border-primary/50'
              }`}
            >
              {opt.label}
            </button>
          ))}
        </div>
        <button
          disabled={answer === null}
          onClick={() => setSubmitted(true)}
          className="inline-flex items-center gap-2 px-5 py-2 rounded-xl bg-on-surface text-on-primary dark:bg-inverse-surface dark:text-inverse-surface font-label-md text-label-md hover:opacity-90 transition disabled:opacity-40"
        >
          {texts.ui.runExperiment}
        </button>

        {submitted && answer !== null && (
          <div className={`mt-4 rounded-2xl p-4 border ${correct ? 'border-[#2f6b3e]/40' : 'border-[#C8604A]/40'} bg-surface-container dark:bg-white/5`}>
            <div className="flex items-center gap-2 mb-2">
              <span className="text-lg">{correct ? '✅' : '❌'}</span>
              <span className={`font-label-md font-bold ${correct ? 'text-[#2f6b3e] dark:text-[#9ed0a8]' : 'text-[#C8604A] dark:text-[#f0b3a4]'}`}>
                {correct ? texts.ui.correct : texts.ui.notQuite}
              </span>
              <span className="text-caption text-outline ml-auto font-mono">
                {fmt(texts.ui.verdictEvidence, { s4: r.sim_at_4.toFixed(3), s8: r.sim_at_8.toFixed(3) })}
              </span>
            </div>
            <ul className="text-caption text-on-surface-variant dark:text-outline space-y-1">
              {texts.explanation.map((line) => (
                <li key={line}>• {fmt(line, { s4: r.sim_at_4.toFixed(3), s8: r.sim_at_8.toFixed(3) })}</li>
              ))}
            </ul>
          </div>
        )}
      </div>

      {/* Related Notes */}
      <div className="bg-surface-container-lowest dark:bg-dark-surface rounded-3xl p-5 border border-outline-variant/40 dark:border-white/10">
        <h3 className="font-label-md text-label-md uppercase tracking-wider text-on-surface dark:text-dark-on-surface mb-3">📚 {texts.ui.relatedNotes}</h3>
        <div className="flex flex-col gap-2">
          {texts.notes.map((n) => (
            <div key={n.src} className="flex items-center gap-3 px-4 py-2.5 rounded-xl bg-surface-container dark:bg-white/5 border border-outline-variant/40 dark:border-white/10">
              <span className="material-symbols-outlined text-outline" style={{ fontSize: 18 }}>menu_book</span>
              <div className="min-w-0">
                <div className="text-body-md text-on-surface dark:text-dark-on-surface truncate">{n.title}</div>
                <div className="text-caption text-outline font-mono truncate">{n.src}</div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
