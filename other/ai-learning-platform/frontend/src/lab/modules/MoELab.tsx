import { useMemo, useState } from 'react'
import type { LabResult, LabParams } from '../types'
import { useI18n } from '../../i18n/context'
import { labTextsZh, labTextsEn, fmt } from '../../i18n/lab'

interface ExpertInfo {
  id: number
  name: string
  load: number
  load_pct: number
  tokens: number[]
}

interface MoEResult extends LabResult {
  tokens: string[]
  n_tokens: number
  n_experts: number
  top_k: number
  temperature: number
  routing: number[][]
  experts: ExpertInfo[]
  loads: number[]
  load_total: number
  seed: number
  simulation_mode: boolean
  provenance: string
}

const EXPERT_COLORS = ['#C8604A', '#5B6BB0', '#2f6b3e', '#7A5C36', '#8a3a35', '#4a6b8a', '#8a6b3a', '#6b5b8a']

export default function MoELab({ result, loading, onAction }: {
  result: LabResult | null; loading: boolean; error: string | null
  onAction: (k: string) => void; params: LabParams; setParams: (p: LabParams) => void
}) {
  const { lang } = useI18n()
  const texts = (lang === 'zh' ? labTextsZh : labTextsEn)['moe-expert-routing']
  const r = result as MoEResult | null
  const [answer, setAnswer] = useState<string | null>(null)
  const [submitted, setSubmitted] = useState(false)
  const [selToken, setSelToken] = useState<number | null>(null)

  const correct = useMemo(() => {
    if (!r) return false
    // Evidence-based verdict: busy experts when the load is concentrated.
    const maxL = Math.max(...r.loads)
    const minL = Math.min(...r.loads)
    return answer === (maxL > minL * 1.5 ? 'busiest' : 'even')
  }, [r, answer])

  if (!r) {
    return <div className="p-10 text-on-surface-variant dark:text-outline">{loading ? texts.ui.computing : texts.ui.adjustControls}</div>
  }

  const sel = selToken === null ? 0 : selToken
  const row = r.routing[sel] ?? []
  const maxLoad = Math.max(...r.loads, 1e-9)
  const maxRow = Math.max(...row, 1e-9)
  const maxLoadPct = Math.max(...r.experts.map((e) => e.load_pct), 1e-9)

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

      {/* Routing room */}
      <div className="bg-surface-container-lowest dark:bg-dark-surface rounded-3xl p-4 md:p-6 shadow-ambient dark:shadow-dark-ambient border border-outline-variant/40 dark:border-white/10">
        <div className="flex flex-wrap items-center justify-between gap-3 mb-3">
          <h3 className="font-headline text-lg text-on-surface dark:text-inverse-on-surface inline-flex items-center gap-2">
            <span className="material-symbols-outlined" style={{ fontSize: 20 }}>call_split</span>
            {texts.ui.title}
          </h3>
          <span className="text-caption text-on-surface-variant dark:text-outline">
            {texts.ui.topK} = <span className="font-mono text-primary dark:text-inverse-primary">{r.top_k}</span>
            {' · '}{texts.ui.gate} T = <span className="font-mono">{r.temperature.toFixed(2)}</span>
          </span>
        </div>

        {/* tokens left → experts right, edges = routing weight */}
        <div className="w-full overflow-x-auto">
          <svg viewBox="0 0 720 260" className="min-w-[620px] w-full" role="img" aria-label="expert routing">
            {/* token nodes (left) */}
            {r.tokens.map((t, i) => {
              const x = 30
              const y = 26 + i * (208 / Math.max(1, r.n_tokens - 1))
              return (
                <g key={i} onClick={() => setSelToken(i)} className="cursor-pointer">
                  <circle cx={x} cy={y} r={i === sel ? 14 : 11}
                    fill={i === sel ? '#C8604A' : 'rgba(125,118,109,0.18)'}
                    stroke={i === sel ? '#fff' : 'rgba(125,118,109,0.4)'} strokeWidth={i === sel ? 2 : 1} />
                  <text x={x} y={y + 4} textAnchor="middle" fontSize={i === sel ? 11 : 9}
                    fontWeight={i === sel ? 700 : 500} fill={i === sel ? '#fff' : 'var(--ailearn-on-surface, #3B3023)'}>
                    {t}
                  </text>
                </g>
              )
            })}

            {/* edges from selected token to experts */}
            {row.map((w, e) => {
              if (w <= 0.01) return null
              const ex = r.experts[e]
              const exX = 560
              const exY = 26 + e * (208 / Math.max(1, r.n_experts - 1))
              return (
                <line key={e} x1={52} y1={26 + sel * (208 / Math.max(1, r.n_tokens - 1))}
                  x2={exX - 16} y2={exY}
                  stroke={EXPERT_COLORS[e % EXPERT_COLORS.length]}
                  strokeWidth={Math.max(1, w * 9)} opacity={0.25 + w * 0.75} />
              )
            })}

            {/* expert nodes (right) */}
            {r.experts.map((ex, e) => {
              const x = 560
              const y = 26 + e * (208 / Math.max(1, r.n_experts - 1))
              const busy = ex.load_pct > 30
              return (
                <g key={e}>
                  <rect x={x} y={y - 15} width={140} height={30} rx={12}
                    fill={EXPERT_COLORS[e % EXPERT_COLORS.length]}
                    opacity={busy ? 0.85 : 0.55} />
                  <text x={x + 70} y={y + 4} textAnchor="middle" fontSize="11" fontWeight="600" fill="#fff">
                    {ex.name}
                  </text>
                  <text x={x + 70} y={y + 15} textAnchor="middle" fontSize="9" fill="#fff" opacity="0.85">
                    {ex.load_pct.toFixed(0)}%
                  </text>
                </g>
              )
            })}
          </svg>
        </div>
        <p className="mt-2 text-caption text-on-surface-variant dark:text-outline">
          {texts.ui.clicked} · <span className="font-mono">{r.tokens[sel]}</span>
        </p>
      </div>

      {/* Selected token routing bars + expert loads */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="bg-surface-container-lowest dark:bg-dark-surface rounded-3xl p-4 md:p-6 shadow-ambient dark:shadow-dark-ambient border border-outline-variant/40 dark:border-white/10">
          <h3 className="font-headline text-lg text-on-surface dark:text-inverse-on-surface mb-3 inline-flex items-center gap-2">
            <span className="material-symbols-outlined" style={{ fontSize: 20 }}>route</span>
            {texts.ui.routing} · {r.tokens[sel]}
          </h3>
          <div className="flex flex-col gap-1.5">
            {r.experts.map((ex, e) => {
              const w = row[e] ?? 0
              const kept = w > 0.01
              return (
                <div key={e} className="flex items-center gap-3">
                  <span className="w-24 text-right text-caption font-semibold truncate" style={{ color: EXPERT_COLORS[e % EXPERT_COLORS.length] }}>
                    {ex.name}
                  </span>
                  <div className="flex-1 h-4 bg-surface-container dark:bg-white/5 rounded-md overflow-hidden">
                    <div className="h-full rounded-md transition-all duration-300"
                      style={{
                        width: `${Math.max((w / maxRow) * 100, kept ? 1 : 0)}%`,
                        background: kept ? EXPERT_COLORS[e % EXPERT_COLORS.length] : 'repeating-linear-gradient(45deg, rgba(125,118,109,0.15), rgba(125,118,109,0.15) 3px, transparent 3px, transparent 6px)',
                        opacity: kept ? 0.35 + w * 0.65 : 0.4,
                      }} />
                  </div>
                  <span className="w-10 text-right font-mono text-caption text-on-surface-variant dark:text-outline tabular-nums">{w.toFixed(2)}</span>
                </div>
              )
            })}
          </div>
        </div>

        <div className="bg-surface-container-lowest dark:bg-dark-surface rounded-3xl p-4 md:p-6 shadow-ambient dark:shadow-dark-ambient border border-outline-variant/40 dark:border-white/10">
          <h3 className="font-headline text-lg text-on-surface dark:text-inverse-on-surface mb-3 inline-flex items-center gap-2">
            <span className="material-symbols-outlined" style={{ fontSize: 20 }}>equalizer</span>
            {texts.ui.load}
          </h3>
          <div className="flex flex-col gap-1.5">
            {r.experts.map((ex, e) => (
              <div key={e} className="flex items-center gap-3">
                <span className="w-24 text-right text-caption font-semibold truncate" style={{ color: EXPERT_COLORS[e % EXPERT_COLORS.length] }}>
                  {ex.name}
                </span>
                <div className="flex-1 h-4 bg-surface-container dark:bg-white/5 rounded-md overflow-hidden">
                  <div className="h-full rounded-md transition-all duration-300"
                    style={{ width: `${(ex.load_pct / maxLoadPct) * 100}%`, background: EXPERT_COLORS[e % EXPERT_COLORS.length], opacity: 0.65 }} />
                </div>
                <span className="w-12 text-right font-mono text-caption text-on-surface-variant dark:text-outline tabular-nums">
                  {ex.load_pct.toFixed(0)}%
                </span>
              </div>
            ))}
          </div>
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
                {fmt(texts.ui.verdictEvidence, { max: Math.max(...r.loads).toFixed(2), min: Math.min(...r.loads).toFixed(2) })}
              </span>
            </div>
            <ul className="text-caption text-on-surface-variant dark:text-outline space-y-1">
              {texts.explanation.map((line) => (
                <li key={line}>• {fmt(line, { max: Math.max(...r.loads).toFixed(2), min: Math.min(...r.loads).toFixed(2) })}</li>
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
