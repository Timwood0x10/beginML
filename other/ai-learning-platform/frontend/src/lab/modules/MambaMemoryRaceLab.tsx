import { useEffect, useMemo, useRef, useState } from 'react'
import type { LabResult, LabParams } from '../types'
import { useI18n } from '../../i18n/context'
import { labTextsZh, labTextsEn, fmt } from '../../i18n/lab'
import { ExplainBox } from '../Journal'
import { QuestList, type Quest } from '../QuestList'

interface RaceResult extends LabResult {
  length: number
  d: number
  layers: number
  lengths: number[]
  transformer_flops: number[]
  mamba_flops: number[]
  transformer_mem: number[]
  mamba_mem: number[]
  transformer_flops_now: number
  mamba_flops_now: number
  flops_ratio: number
  transformer_mem_now: number
  mamba_mem_now: number
  mem_ratio: number
  cross_point: number | null
  walk_trans: number[] // per-token lookback cost for Transformer
  walk_mamba: number[] // per-token lookback cost for Mamba (always 1)
  cum_trans: number[] // cumulative cost as the learner walks the sequence
  cum_mamba: number[]
  provenance: string
}

export default function MambaMemoryRaceLab({ result, loading, params, onRecord }: {
  result: LabResult | null; loading: boolean; error: string | null
  onAction: (k: string) => void; params: LabParams; setParams: (p: LabParams) => void
  onRecord?: (entry: { question: string; prediction: string; correct: boolean; evidence: string; params: LabParams }) => void
}) {
  const { lang } = useI18n()
  const texts = (lang === 'zh' ? labTextsZh : labTextsEn)['mamba-memory-race']
  const r = result as RaceResult | null
  const [answer, setAnswer] = useState<string | null>(null)
  const [submitted, setSubmitted] = useState(false)
  // Sequence walk: the learner steps token by token, feeling the lookback.
  const [walkIdx, setWalkIdx] = useState(0)

  useEffect(() => {
    setWalkIdx(0)
  }, [r?.length])

  const correct = useMemo(() => {
    if (!r) return false
    // Evidence-based verdict: the gap must widen (quadratic vs linear).
    return answer === 'wider'
  }, [r, answer])

  // --- L2 Manipulate Challenge ------------------------------------------
  // Success is DERIVED from the computed FLOPs gap (a result, not a
  // control). The learner drags the sequence length until the quadratic
  // curve pulls ahead by the target multiple.
  // Hooks must live ABOVE the `if (!r) return` early exit.
  const [l2Goal, setL2Goal] = useState<'gap8' | 'gap16' | null>(null)
  const recordedL2 = useRef<string | null>(null)

  const l2Achieved = l2Goal === 'gap8'
    ? (r?.flops_ratio ?? 0) >= 8
    : l2Goal === 'gap16'
      ? (r?.flops_ratio ?? 0) >= 16
      : false

  // Record the achievement into the Journal exactly once per goal.
  useEffect(() => {
    if (r && l2Goal && l2Achieved && recordedL2.current !== l2Goal) {
      recordedL2.current = l2Goal
      onRecord?.({
        question: l2Goal === 'gap8' ? texts.ui.l2Gap8 : texts.ui.l2Gap16,
        prediction: 'manual tuning',
        correct: true,
        evidence: `FLOPs ${r.transformer_flops_now.toFixed(0)} vs ${r.mamba_flops_now.toFixed(0)} (${r.flops_ratio.toFixed(1)}×)`,
        params: { ...params },
      })
    }
  }, [l2Goal, l2Achieved, r, onRecord, texts, params])

  if (!r) {
    return <div className="p-10 text-on-surface-variant dark:text-outline">{loading ? texts.ui.computing : texts.ui.adjustControls}</div>
  }

  // --- FLOPs curve (log-y to keep both visible) --------------------------
  const W = 640, H = 300
  const pad = { l: 56, r: 18, t: 14, b: 40 }
  const plotW = W - pad.l - pad.r
  const plotH = H - pad.t - pad.b
  const lMin = Math.min(...r.lengths)
  const lMax = Math.max(...r.lengths)
  const allVals = [...r.transformer_flops, ...r.mamba_flops]
  const vMin = Math.min(...allVals)
  const vMax = Math.max(...allVals)
  const lpx = (L: number) => pad.l + ((L - lMin) / (lMax - lMin)) * plotW
  const lpy = (v: number) => pad.t + (1 - Math.log(v / vMin) / Math.log(vMax / vMin)) * plotH
  const poly = (vals: number[]) => r.lengths.map((L, i) => `${lpx(L).toFixed(1)},${lpy(vals[i]).toFixed(1)}`).join(' ')

  // --- memory bars -------------------------------------------------------
  const memMax = Math.max(r.transformer_mem_now, r.mamba_mem_now, 1e-9)

  // --- Exploration quests (derived from computed result) -----------------
  const quests: Quest[] = [
    { id: 'gap8', label: texts.quests![0], done: r.flops_ratio >= 8 },
    { id: 'gap16', label: texts.quests![1], done: r.flops_ratio >= 16 },
    { id: 'mem', label: texts.quests![2], done: r.mem_ratio >= 8 },
  ]

  return (
    <div className="flex flex-col gap-5">
      {/* Exploration quests */}
      <QuestList
        quests={quests}
        onRecord={onRecord}
        params={params}
        evidence={`FLOPs ${r.transformer_flops_now.toFixed(0)} vs ${r.mamba_flops_now.toFixed(0)} (${r.flops_ratio.toFixed(1)}×)`}
      />

      {/* Sequence walk — step through the tokens and FEEL the lookback */}
      <div className="bg-surface-container-lowest dark:bg-dark-surface rounded-3xl p-4 md:p-6 shadow-ambient dark:shadow-dark-ambient border border-outline-variant/40 dark:border-white/10">
        <div className="flex flex-wrap items-center justify-between gap-3 mb-3">
          <h3 className="font-headline text-lg text-on-surface dark:text-inverse-on-surface inline-flex items-center gap-2">
            <span className="material-symbols-outlined" style={{ fontSize: 20 }}>footprint</span>
            {texts.ui.walkTitle}
          </h3>
          <div className="flex items-center gap-2">
            <button
              onClick={() => setWalkIdx((i) => Math.max(0, i - 1))}
              disabled={walkIdx === 0}
              className="px-3 py-1.5 rounded-xl text-label-md font-semibold bg-surface-variant dark:bg-white/10 text-on-surface dark:text-dark-on-surface hover:opacity-80 transition disabled:opacity-40"
            >
              ←
            </button>
            <span className="font-mono text-caption text-on-surface-variant dark:text-outline tabular-nums">{walkIdx + 1}/{r.length}</span>
            <button
              onClick={() => setWalkIdx((i) => Math.min(r.length - 1, i + 1))}
              disabled={walkIdx >= r.length - 1}
              className="px-3 py-1.5 rounded-xl text-label-md font-semibold bg-primary text-on-primary dark:bg-inverse-primary dark:text-inverse-surface hover:opacity-80 transition disabled:opacity-40"
            >
              →
            </button>
          </div>
        </div>
        <p className="text-caption text-on-surface-variant dark:text-outline mb-3">{texts.ui.walkHint}</p>

        {/* token dots */}
        <div className="flex items-center gap-1 overflow-x-auto pb-2 mb-3">
          {r.walk_trans.map((_, i) => (
            <button
              key={i}
              onClick={() => setWalkIdx(i)}
              title={`token ${i}`}
              className={`shrink-0 w-5 h-5 rounded-full flex items-center justify-center text-[9px] font-bold border transition-all ${
                i < walkIdx
                  ? 'bg-[#C8604A]/30 text-[#C8604A] dark:text-[#f0b3a4] border-[#C8604A]/40'
                  : i === walkIdx
                    ? 'bg-primary text-on-primary border-primary dark:bg-inverse-primary dark:text-inverse-surface scale-110'
                    : 'bg-surface-container dark:bg-white/5 text-outline border-outline-variant/40 dark:border-white/10'
              }`}
            >
              {i + 1}
            </button>
          ))}
        </div>

        {/* lookback cost at the current token */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3 mb-3">
          <div className="rounded-2xl px-4 py-3 border border-[#C8604A]/40 bg-[#C8604A]/10">
            <div className="text-caption text-[#C8604A] dark:text-[#f0b3a4] font-semibold mb-1">{texts.ui.transformer}</div>
            <div className="font-mono text-body-md text-on-surface dark:text-dark-on-surface">
              {texts.ui.lookback}: {r.walk_trans[walkIdx].toFixed(0)} <span className="text-caption text-outline">{texts.ui.tokens}</span>
            </div>
            <div className="text-caption text-on-surface-variant dark:text-outline mt-0.5">
              {texts.ui.cumulative}: <span className="font-mono">{r.cum_trans[walkIdx].toFixed(0)}</span>
            </div>
          </div>
          <div className="rounded-2xl px-4 py-3 border border-[#2f6b3e]/40 bg-[#2f6b3e]/10">
            <div className="text-caption text-[#2f6b3e] dark:text-[#9ed0a8] font-semibold mb-1">{texts.ui.mamba}</div>
            <div className="font-mono text-body-md text-on-surface dark:text-dark-on-surface">
              {texts.ui.lookback}: {r.walk_mamba[walkIdx].toFixed(0)} <span className="text-caption text-outline">{texts.ui.fixedState}</span>
            </div>
            <div className="text-caption text-on-surface-variant dark:text-outline mt-0.5">
              {texts.ui.cumulative}: <span className="font-mono">{r.cum_mamba[walkIdx].toFixed(0)}</span>
            </div>
          </div>
        </div>

        {/* cumulative cost race bars */}
        <div className="flex flex-col gap-1.5">
          <div className="flex items-center gap-3">
            <span className="w-24 text-right text-caption font-semibold text-[#C8604A]">{texts.ui.transformer}</span>
            <div className="flex-1 h-4 bg-surface-container dark:bg-white/5 rounded-md overflow-hidden">
              <div className="h-full rounded-md bg-[#C8604A] transition-all duration-300"
                style={{ width: `${(r.cum_trans[walkIdx] / Math.max(r.cum_trans[r.length - 1], 1)) * 100}%`, opacity: 0.75 }} />
            </div>
          </div>
          <div className="flex items-center gap-3">
            <span className="w-24 text-right text-caption font-semibold text-[#2f6b3e]">{texts.ui.mamba}</span>
            <div className="flex-1 h-4 bg-surface-container dark:bg-white/5 rounded-md overflow-hidden">
              <div className="h-full rounded-md bg-[#2f6b3e] transition-all duration-300"
                style={{ width: `${(r.cum_mamba[walkIdx] / Math.max(r.cum_trans[r.length - 1], 1)) * 100}%`, opacity: 0.75 }} />
            </div>
          </div>
        </div>
      </div>

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

      {/* FLOPs race */}
      <div className="bg-surface-container-lowest dark:bg-dark-surface rounded-3xl p-4 md:p-6 shadow-ambient dark:shadow-dark-ambient border border-outline-variant/40 dark:border-white/10">
        <div className="flex flex-wrap items-center justify-between gap-3 mb-3">
          <h3 className="font-headline text-lg text-on-surface dark:text-inverse-on-surface inline-flex items-center gap-2">
            <span className="material-symbols-outlined" style={{ fontSize: 20 }}>speed</span>
            {texts.ui.title}
          </h3>
          <span className="text-caption text-on-surface-variant dark:text-outline">
            {fmt(texts.ui.now, { l: r.length })}
            {' · '}{fmt(texts.ui.flopsRatio, { r: r.flops_ratio.toFixed(1) })}
          </span>
        </div>

        <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img" aria-label="flops race">
          {/* grid */}
          {[0, 0.25, 0.5, 0.75, 1].map((t) => {
            const y = pad.t + t * plotH
            const v = vMin * Math.pow(vMax / vMin, 1 - t)
            return (
              <g key={t}>
                <line x1={pad.l} y1={y} x2={W - pad.r} y2={y} stroke="rgba(125,118,109,0.14)" strokeDasharray="2 4" />
                <text x={pad.l - 8} y={y + 4} textAnchor="end" fontSize="9" className="fill-outline">
                  {v >= 1e6 ? `${(v / 1e6).toFixed(1)}M` : v >= 1e3 ? `${(v / 1e3).toFixed(1)}k` : v.toFixed(0)}
                </text>
              </g>
            )
          })}
          {/* x axis */}
          <line x1={pad.l} y1={pad.t + plotH} x2={W - pad.r} y2={pad.t + plotH} stroke="rgba(125,118,109,0.4)" />
          {[32, 64, 128, 256].map((L) => (
            <g key={L}>
              <line x1={lpx(L)} y1={pad.t + plotH} x2={lpx(L)} y2={pad.t + plotH + 4} stroke="rgba(125,118,109,0.4)" />
              <text x={lpx(L)} y={H - 14} textAnchor="middle" fontSize="9" className="fill-outline">{L}</text>
            </g>
          ))}
          <text x={(pad.l + W - pad.r) / 2} y={H - 2} textAnchor="middle" fontSize="10" className="fill-outline">
            {texts.ui.length} →
          </text>

          {/* curves */}
          <polyline points={poly(r.transformer_flops)} fill="none" stroke="#C8604A" strokeWidth="2.4" />
          <polyline points={poly(r.mamba_flops)} fill="none" stroke="#2f6b3e" strokeWidth="2.4" />

          {/* current-length cursor */}
          <line x1={lpx(r.length)} y1={pad.t} x2={lpx(r.length)} y2={pad.t + plotH} stroke="rgba(125,118,109,0.5)" strokeWidth="1.2" strokeDasharray="3 3" />
          <circle cx={lpx(r.length)} cy={lpy(r.transformer_flops_now)} r="4.5" fill="#C8604A" stroke="#fff" strokeWidth="1.5" />
          <circle cx={lpx(r.length)} cy={lpy(r.mamba_flops_now)} r="4.5" fill="#2f6b3e" stroke="#fff" strokeWidth="1.5" />
        </svg>

        <div className="mt-2 flex flex-wrap gap-4 text-caption text-on-surface-variant dark:text-outline">
          <span className="inline-flex items-center gap-1.5"><span className="w-4 h-0.5 rounded bg-[#C8604A]" /> {texts.ui.transformer}</span>
          <span className="inline-flex items-center gap-1.5"><span className="w-4 h-0.5 rounded bg-[#2f6b3e]" /> {texts.ui.mamba}</span>
          {r.cross_point && (
            <span className="ml-auto font-mono text-primary dark:text-inverse-primary">
              {fmt(texts.ui.cross, { c: r.cross_point })}
            </span>
          )}
        </div>
      </div>

      {/* Memory bars */}
      <div className="bg-surface-container-lowest dark:bg-dark-surface rounded-3xl p-4 md:p-6 shadow-ambient dark:shadow-dark-ambient border border-outline-variant/40 dark:border-white/10">
        <div className="flex items-center justify-between gap-3 mb-3">
          <h3 className="font-headline text-lg text-on-surface dark:text-inverse-on-surface inline-flex items-center gap-2">
            <span className="material-symbols-outlined" style={{ fontSize: 20 }}>memory</span>
            {texts.ui.memory}
          </h3>
          <span className="text-caption text-on-surface-variant dark:text-outline">
            {fmt(texts.ui.memRatio, { r: r.mem_ratio.toFixed(1) })}
          </span>
        </div>
        <div className="flex flex-col gap-2">
          <div className="flex items-center gap-3">
            <span className="w-28 text-right text-caption font-semibold text-[#C8604A]">{texts.ui.transformer}</span>
            <div className="flex-1 h-5 bg-surface-container dark:bg-white/5 rounded-md overflow-hidden">
              <div className="h-full rounded-md bg-[#C8604A] transition-all duration-300"
                style={{ width: `${(r.transformer_mem_now / memMax) * 100}%`, opacity: 0.75 }} />
            </div>
            <span className="w-16 text-right font-mono text-caption text-on-surface-variant dark:text-outline tabular-nums">
              {r.transformer_mem_now.toFixed(0)}
            </span>
          </div>
          <div className="flex items-center gap-3">
            <span className="w-28 text-right text-caption font-semibold text-[#2f6b3e]">{texts.ui.mamba}</span>
            <div className="flex-1 h-5 bg-surface-container dark:bg-white/5 rounded-md overflow-hidden">
              <div className="h-full rounded-md bg-[#2f6b3e] transition-all duration-300"
                style={{ width: `${(r.mamba_mem_now / memMax) * 100}%`, opacity: 0.75 }} />
            </div>
            <span className="w-16 text-right font-mono text-caption text-on-surface-variant dark:text-outline tabular-nums">
              {r.mamba_mem_now.toFixed(0)}
            </span>
          </div>
        </div>
        <p className="mt-3 text-caption text-on-surface-variant dark:text-outline">
          {texts.ui.flops} @ L={r.length}：{texts.ui.verdictEvidence.replace('{t}', String(r.transformer_flops_now)).replace('{m}', String(r.mamba_flops_now)).replace('{r}', r.flops_ratio.toFixed(1))}
        </p>
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
              onRecord({
                question: texts.challengeQuestion,
                prediction: label,
                correct,
                evidence: `FLOPs ${r.transformer_flops_now.toFixed(0)} vs ${r.mamba_flops_now.toFixed(0)} (${r.flops_ratio.toFixed(1)}×)`,
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
                {fmt(texts.ui.verdictEvidence, {
                  t: r.transformer_flops_now.toFixed(0),
                  m: r.mamba_flops_now.toFixed(0),
                  r: r.flops_ratio.toFixed(1),
                })}
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
            onClick={() => { setL2Goal('gap8'); recordedL2.current = null }}
            className={`px-4 py-2 rounded-xl text-label-md font-semibold border transition-all ${
              l2Goal === 'gap8'
                ? 'bg-primary text-on-primary border-primary dark:bg-inverse-primary dark:text-inverse-surface'
                : 'bg-surface-container-lowest dark:bg-dark-surface text-on-surface-variant dark:text-outline border-outline-variant/60 dark:border-white/10 hover:border-primary/50'
            }`}
          >
            {texts.ui.l2Gap8}
          </button>
          <button
            onClick={() => { setL2Goal('gap16'); recordedL2.current = null }}
            className={`px-4 py-2 rounded-xl text-label-md font-semibold border transition-all ${
              l2Goal === 'gap16'
                ? 'bg-primary text-on-primary border-primary dark:bg-inverse-primary dark:text-inverse-surface'
                : 'bg-surface-container-lowest dark:bg-dark-surface text-on-surface-variant dark:text-outline border-outline-variant/60 dark:border-white/10 hover:border-primary/50'
            }`}
          >
            {texts.ui.l2Gap16}
          </button>
        </div>

        {l2Goal && (
          <div>
            <div className="mb-2 text-caption text-on-surface-variant dark:text-outline">
              {texts.ui.l2CurGap}: <span className="font-mono text-primary dark:text-inverse-primary">{r.flops_ratio.toFixed(1)}×</span>
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
        evidence={`FLOPs ${r.transformer_flops_now.toFixed(0)} vs ${r.mamba_flops_now.toFixed(0)} (${r.flops_ratio.toFixed(1)}×)`}
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
