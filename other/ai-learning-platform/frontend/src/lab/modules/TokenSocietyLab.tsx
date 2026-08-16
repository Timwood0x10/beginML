import { useEffect, useMemo, useRef, useState } from 'react'
import type { LabResult, LabParams } from '../types'
import { useI18n } from '../../i18n/context'
import { labTextsZh, labTextsEn, fmt } from '../../i18n/lab'
import { ExplainBox } from '../Journal'
import { QuestList, type Quest } from '../QuestList'

interface HeadInfo {
  head: number
  name: string
  avg_dist: number
  local_ratio: number
  diag_ratio: number
  weights: number[][]
}

interface SocietyResult extends LabResult {
  tokens: string[]
  n: number
  sentence: string
  heads: HeadInfo[]
  n_heads: number
  seed: number
  simulation_mode: boolean
  provenance: string
}

const HEAD_COLORS = ['#C8604A', '#5B6BB0', '#2f6b3e', '#7A5C36', '#8a3a35', '#4a6b8a', '#8a6b3a', '#6b5b8a']

export default function TokenSocietyLab({ result, loading, onAction, params, onRecord }: {
  result: LabResult | null; loading: boolean; error: string | null
  onAction: (k: string) => void; params: LabParams; setParams: (p: LabParams) => void
  onRecord?: (entry: { question: string; prediction: string; correct: boolean; evidence: string; params: LabParams }) => void
}) {
  const { lang } = useI18n()
  const texts = (lang === 'zh' ? labTextsZh : labTextsEn)['token-society']
  const r = result as SocietyResult | null
  const [headIdx, setHeadIdx] = useState(0)
  const [selToken, setSelToken] = useState<number | null>(null)
  const [answer, setAnswer] = useState<string | null>(null)
  const [submitted, setSubmitted] = useState(false)

  const correct = useMemo(() => {
    if (!r) return false
    // Evidence-based verdict: the head with the largest avg_dist is the
    // long-distance expert.
    const scout = r.heads.reduce((a, b) => (b.avg_dist > a.avg_dist ? b : a))
    const ans = scout.name === 'The Long-Distance Scout' ? 'scout'
      : scout.name === 'The Repeater' ? 'repeater' : 'nearby'
    return answer === ans
  }, [r, answer])

  // --- L2 Manipulate Challenge ------------------------------------------
  // Success is DERIVED from the selected token's attention row under the
  // current head (the max weight). The learner switches heads/sentences to
  // concentrate or spread the attention until it passes.
  const [l2Goal, setL2Goal] = useState<'focused' | 'spread' | null>(null)
  const recordedL2 = useRef<string | null>(null)

  const selTokenIdx = selToken === null ? Math.floor((r?.n ?? 2) / 2) : selToken
  const currentMaxW = r && r.heads[Math.min(headIdx, r.heads.length - 1)]
    ? Math.max(...r.heads[Math.min(headIdx, r.heads.length - 1)].weights[selTokenIdx])
    : 0

  const l2Achieved = l2Goal === 'focused'
    ? currentMaxW >= 0.5
    : l2Goal === 'spread'
      ? currentMaxW <= 0.35
      : false

  // Record the achievement into the Journal exactly once per goal.
  useEffect(() => {
    if (r && l2Goal && l2Achieved && recordedL2.current !== l2Goal) {
      recordedL2.current = l2Goal
      const head = r.heads[Math.min(headIdx, r.heads.length - 1)]
      onRecord?.({
        question: l2Goal === 'focused' ? texts.ui.l2Focused : texts.ui.l2Spread,
        prediction: 'manual tuning',
        correct: true,
        evidence: `head=${head.name}, maxW=${currentMaxW.toFixed(3)}`,
        params: { ...params },
      })
    }
  }, [l2Goal, l2Achieved, r, headIdx, selTokenIdx, currentMaxW, onRecord, texts, params])

  if (!r) {
    return <div className="p-10 text-on-surface-variant dark:text-outline">{loading ? texts.ui.computing : texts.ui.adjustControls}</div>
  }

  const head = r.heads[Math.min(headIdx, r.heads.length - 1)]
  const sel = selToken === null ? Math.floor(r.n / 2) : selToken

  // --- Exploration quests (derived from computed result) -----------------
  const quests: Quest[] = [
    { id: 'scout', label: texts.quests![0], done: head.name === 'The Long-Distance Scout' },
    { id: 'focused', label: texts.quests![1], done: currentMaxW >= 0.5 },
    { id: 'spread', label: texts.quests![2], done: currentMaxW <= 0.35 },
  ]

  // society graph geometry
  const N = r.n
  const W = Math.max(560, N * 74)
  const H = 240
  const cy = H / 2
  const xOf = (i: number) => 36 + (i / Math.max(1, N - 1)) * (W - 72)

  // edges from the selected token (weights[sel][j])
  const edges = r.tokens.map((_, j) => {
    const w = head.weights[sel][j]
    return { j, w, x: xOf(j) }
  })

  return (
    <div className="flex flex-col gap-5">
      {/* Exploration quests */}
      <QuestList
        quests={quests}
        onRecord={onRecord}
        params={params}
        evidence={`scout avg_dist=${r.heads.reduce((a, b) => (b.avg_dist > a.avg_dist ? b : a)).avg_dist.toFixed(3)}`}
      />

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

      {/* Society graph */}
      <div className="bg-surface-container-lowest dark:bg-dark-surface rounded-3xl p-4 md:p-6 shadow-ambient dark:shadow-dark-ambient border border-outline-variant/40 dark:border-white/10">
        <div className="flex flex-wrap items-center justify-between gap-3 mb-3">
          <h3 className="font-headline text-lg text-on-surface dark:text-inverse-on-surface inline-flex items-center gap-2">
            <span className="material-symbols-outlined" style={{ fontSize: 20 }}>diversity_3</span>
            {texts.ui.title}
          </h3>
          <div className="inline-flex items-center gap-3 text-caption text-on-surface-variant dark:text-outline">
            <span className="font-mono text-primary dark:text-inverse-primary">{head.name}</span>
            <button
              onClick={() => onAction('reshuffle')}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-primary/10 dark:bg-inverse-primary/10 border border-primary/40 dark:border-inverse-primary/40 text-label-md font-semibold hover:opacity-90 transition"
            >
              <span className="material-symbols-outlined" style={{ fontSize: 16 }}>refresh</span>
              NEW SOCIETY
            </button>
          </div>
        </div>

        <div className="w-full overflow-x-auto">
          <svg viewBox={`0 0 ${W} ${H}`} className="min-w-[560px] w-full" role="img" aria-label="token society">
            {/* edges from the selected token */}
            {edges.map(({ j, w }) => (
              <line
                key={j}
                x1={xOf(sel)} y1={cy} x2={xOf(j)} y2={cy}
                stroke={HEAD_COLORS[head.head % HEAD_COLORS.length]}
                strokeWidth={Math.max(0.6, w * 7)}
                opacity={0.22 + w * 0.78}
              />
            ))}
            {/* tokens */}
            {r.tokens.map((t, i) => {
              const isSel = i === sel
              return (
                <g key={i} onClick={() => setSelToken(i)} className="cursor-pointer">
                  <circle
                    cx={xOf(i)} cy={cy} r={isSel ? 17 : 13}
                    fill={isSel ? HEAD_COLORS[head.head % HEAD_COLORS.length] : 'rgba(125,118,109,0.18)'}
                    stroke={isSel ? '#fff' : 'rgba(125,118,109,0.4)'}
                    strokeWidth={isSel ? 2 : 1}
                  />
                  <text
                    x={xOf(i)} y={cy + 4} textAnchor="middle"
                    fontSize={isSel ? 11 : 10} fontWeight={isSel ? 700 : 500}
                    fill={isSel ? '#fff' : 'var(--ailearn-on-surface, #3B3023)'}
                  >
                    {t.length > 9 ? t.slice(0, 8) + '…' : t}
                  </text>
                </g>
              )
            })}
          </svg>
        </div>
        <p className="mt-2 text-caption text-on-surface-variant dark:text-outline">
          {texts.ui.clickToken} · <span className="font-mono">{r.tokens[sel]}</span>
        </p>
      </div>

      {/* WHO DOES X LISTEN TO */}
      <div className="bg-surface-container-lowest dark:bg-dark-surface rounded-3xl p-4 md:p-6 shadow-ambient dark:shadow-dark-ambient border border-outline-variant/40 dark:border-white/10">
        <h3 className="font-headline text-lg text-on-surface dark:text-inverse-on-surface mb-1 inline-flex items-center gap-2">
          <span className="material-symbols-outlined" style={{ fontSize: 20 }}>hearing</span>
          {texts.ui.whoListens} <span className="font-mono text-primary dark:text-inverse-primary">{r.tokens[sel]}</span> {texts.ui.listensTo}
        </h3>
        <div className="flex flex-col gap-1.5 mt-3">
          {r.tokens.map((t, j) => {
            const w = head.weights[sel][j]
            const isSelf = j === sel
            return (
              <div key={j} className="flex items-center gap-3">
                <span className="w-16 text-right text-caption font-semibold truncate text-on-surface dark:text-dark-on-surface">{t}</span>
                <div className="flex-1 h-4 bg-surface-container dark:bg-white/5 rounded-md overflow-hidden">
                  <div
                    className="h-full rounded-md transition-all duration-300"
                    style={{
                      width: `${Math.max(w * 100, 1)}%`,
                      background: isSelf ? '#7d766d' : HEAD_COLORS[head.head % HEAD_COLORS.length],
                      opacity: 0.35 + w * 0.65,
                    }}
                  />
                </div>
                <span className="w-12 text-right font-mono text-caption text-on-surface-variant dark:text-outline tabular-nums">{w.toFixed(2)}</span>
              </div>
            )
          })}
        </div>
      </div>

      {/* Observers panel */}
      <div className="bg-surface-container-lowest dark:bg-dark-surface rounded-3xl p-4 md:p-6 shadow-ambient dark:shadow-dark-ambient border border-outline-variant/40 dark:border-white/10">
        <h3 className="font-headline text-lg text-on-surface dark:text-inverse-on-surface mb-3 inline-flex items-center gap-2">
          <span className="material-symbols-outlined" style={{ fontSize: 20 }}>visibility</span>
          {texts.ui.observers}
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
          {r.heads.map((h, i) => {
            const active = i === head.head
            return (
              <button
                key={i}
                onClick={() => setHeadIdx(i)}
                className={`text-left rounded-2xl px-4 py-3 border transition-all ${
                  active
                    ? 'border-primary/50 dark:border-inverse-primary/50 bg-primary/10 dark:bg-inverse-primary/10'
                    : 'border-outline-variant/40 dark:border-white/10 bg-surface-container dark:bg-white/5 hover:border-primary/40'
                }`}
              >
                <div className="flex items-center gap-2 mb-1">
                  <span className="w-3 h-3 rounded-full" style={{ background: HEAD_COLORS[i % HEAD_COLORS.length] }} />
                  <span className="font-label-md font-bold text-on-surface dark:text-dark-on-surface">{h.name}</span>
                  <span className="ml-auto font-mono text-caption text-outline">H{h.head}</span>
                </div>
                <div className="flex flex-wrap gap-3 text-caption text-on-surface-variant dark:text-outline">
                  <span>{texts.ui.avgDist} <span className="font-mono text-primary dark:text-inverse-primary">{h.avg_dist.toFixed(2)}</span></span>
                  <span>{texts.ui.localRatio} <span className="font-mono">{h.local_ratio.toFixed(2)}</span></span>
                  <span>{texts.ui.diagRatio} <span className="font-mono">{h.diag_ratio.toFixed(2)}</span></span>
                </div>
              </button>
            )
          })}
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
          onClick={() => {
            setSubmitted(true)
            if (onRecord && r) {
              const label = texts.challengeOptions.find((o) => o.value === answer)?.label ?? answer ?? ''
              const scout = r.heads.reduce((a, b) => (b.avg_dist > a.avg_dist ? b : a))
              onRecord({
                question: texts.challengeQuestion,
                prediction: label,
                correct,
                evidence: `scout avg_dist=${scout.avg_dist.toFixed(3)}`,
                params: { ...params },
              })
            }
          }}
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
                {fmt(texts.ui.verdictEvidence, { d: r.heads.reduce((a, b) => (b.avg_dist > a.avg_dist ? b : a)).avg_dist.toFixed(2) })}
              </span>
            </div>
            <ul className="text-caption text-on-surface-variant dark:text-outline space-y-1">
              {texts.explanation.map((line) => (
                <li key={line}>• {line}</li>
              ))}
            </ul>
          </div>
        )}
      </div>

      {/* L2 Manipulate Challenge — independent of Controls, independent of Predict */}
      <div className="bg-surface-container-lowest dark:bg-dark-surface rounded-3xl p-5 border border-outline-variant/40 dark:border-white/10">
        <div className="flex items-center gap-2 mb-1">
          <span className="text-lg">🎛️</span>
          <h3 className="font-label-md text-label-md uppercase tracking-wider text-on-surface dark:text-dark-on-surface">{texts.ui.l2Title}</h3>
        </div>
        <p className="text-body-md text-on-surface dark:text-inverse-on-surface mb-3">{texts.ui.l2Tagline}</p>

        <div className="flex flex-wrap gap-2 mb-3">
          <button
            onClick={() => { setL2Goal('focused'); recordedL2.current = null }}
            className={`px-4 py-2 rounded-xl text-label-md font-semibold border transition-all ${
              l2Goal === 'focused'
                ? 'bg-primary text-on-primary border-primary dark:bg-inverse-primary dark:text-inverse-surface'
                : 'bg-surface-container-lowest dark:bg-dark-surface text-on-surface-variant dark:text-outline border-outline-variant/60 dark:border-white/10 hover:border-primary/50'
            }`}
          >
            {texts.ui.l2Focused}
          </button>
          <button
            onClick={() => { setL2Goal('spread'); recordedL2.current = null }}
            className={`px-4 py-2 rounded-xl text-label-md font-semibold border transition-all ${
              l2Goal === 'spread'
                ? 'bg-primary text-on-primary border-primary dark:bg-inverse-primary dark:text-inverse-surface'
                : 'bg-surface-container-lowest dark:bg-dark-surface text-on-surface-variant dark:text-outline border-outline-variant/60 dark:border-white/10 hover:border-primary/50'
            }`}
          >
            {texts.ui.l2Spread}
          </button>
        </div>

        {l2Goal && (
          <div>
            <div className="mb-2 text-caption text-on-surface-variant dark:text-outline">
              {texts.ui.l2MaxW}: <span className="font-mono text-primary dark:text-inverse-primary">{currentMaxW.toFixed(3)}</span>
            </div>
            <div className={`rounded-2xl p-3 border ${
              l2Achieved
                ? 'border-[#2f6b3e]/40 bg-[#2f6b3e]/10 text-[#2f6b3e] dark:text-[#9ed0a8]'
                : 'border-outline-variant/40 dark:border-white/10 bg-surface-container dark:bg-white/5 text-on-surface-variant dark:text-outline'
            }`}>
              <div className="text-label-md font-semibold mb-1">
                {l2Achieved ? texts.ui.l2Success : texts.ui.l2NotYet}
              </div>
              <div className="text-caption">{texts.ui.l2Hint}</div>
            </div>
            <button
              onClick={() => { setL2Goal(null); recordedL2.current = null }}
              className="mt-3 inline-flex items-center gap-1.5 text-caption text-on-surface-variant dark:text-outline hover:text-primary dark:hover:text-inverse-primary transition"
            >
              <span className="material-symbols-outlined" style={{ fontSize: 15 }}>restart_alt</span>
              {texts.ui.l2Reset}
            </button>
          </div>
        )}
      </div>

      {/* L3 Explain — learner-owned, never machine-graded */}
      <ExplainBox
        onRecord={onRecord}
        evidence={`scout avg_dist=${r.heads.reduce((a, b) => (b.avg_dist > a.avg_dist ? b : a)).avg_dist.toFixed(3)}`}
        params={params}
      />

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
