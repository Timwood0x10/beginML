import { useMemo, useState } from 'react'
import type { LabResult, LabParams } from '../types'
import { useI18n } from '../../i18n/context'
import { labTextsZh, labTextsEn, fmt } from '../../i18n/lab'
import { ExplainBox } from '../Journal'
import { VerificationPanel } from '../VerificationPanel'
import { verificationData } from '../verification'
import { NotesPanel } from '../NotesPanel'
import { QuestList, type Quest } from '../QuestList'

interface FeatureInfo {
  idx: number
  name: string
  max: number
  mean: number
  sparsity: number
  active_tokens: number[]
  is_true: boolean
  true_semantic: string | null
}

interface HuntResult extends LabResult {
  tokens: string[]
  n_tokens: number
  n_features: number
  n_true: number
  true_idx: number[]
  activation: number[][] // [token][feature]
  features: FeatureInfo[]
  reveal: boolean
  seed: number
  simulation_mode: boolean
  provenance: string
}

const REAL_COLOR = '#2f6b3e'
const NOISE_COLOR = '#C8604A'

export default function FeatureHuntLab({ result, loading, onAction, params, setParams, onRecord }: {
  result: LabResult | null; loading: boolean; error: string | null
  onAction: (k: string) => void; params: LabParams; setParams: (p: LabParams) => void
  onRecord?: (entry: { question: string; prediction: string; correct: boolean; evidence: string; params: LabParams; insight?: string }) => void
}) {
  const { lang } = useI18n()
  const texts = (lang === 'zh' ? labTextsZh : labTextsEn)['feature-hunt']
  const r = result as HuntResult | null
  const [answer, setAnswer] = useState<string | null>(null)
  const [submitted, setSubmitted] = useState(false)
  const [selIdx, setSelIdx] = useState<number | null>(null)
  // hunt: set of feature indices the learner bets are real
  const [marked, setMarked] = useState<Set<number>>(new Set())
  const [verdict, setVerdict] = useState<{ precision: number; recall: number } | null>(null)

  const correct = useMemo(() => {
    if (!r) return false
    // Evidence-based verdict: noise neurons stay below 0.2.
    return answer === 'low'
  }, [r, answer])

  if (!r) {
    return <div className="p-10 text-on-surface-variant dark:text-outline">{loading ? texts.ui.computing : texts.ui.adjustControls}</div>
  }

  const sel = selIdx === null ? 0 : selIdx
  const selFeat = r.features[sel]
  const actMax = Math.max(...r.activation.flat(), 0.01)
  const trueSet = new Set(r.true_idx)

  // --- Exploration quests (derived from hunt verdict) --------------------
  const quests: Quest[] = [
    { id: 'all', label: texts.quests![0], done: verdict !== null && verdict.precision === 1 && verdict.recall === 1 },
    { id: 'nofp', label: texts.quests![1], done: verdict !== null && verdict.precision === 1 },
    { id: 'recall', label: texts.quests![2], done: verdict !== null && verdict.recall === 1 },
  ]

  // heatmap geometry
  const W = 620, H = 240
  const pad = { l: 34, r: 12, t: 14, b: 26 }
  const cellW = (W - pad.l - pad.r) / r.n_features
  const cellH = (H - pad.t - pad.b) / r.n_tokens
  const heat = (v: number) => {
    const t = Math.min(1, Math.max(0, v / actMax))
    return `color-mix(in srgb, ${REAL_COLOR} ${Math.round(t * 88)}%, #F3EBD9)`
  }

  const toggleMark = (i: number) => {
    setMarked((prev) => {
      const next = new Set(prev)
      if (next.has(i)) next.delete(i)
      else next.add(i)
      return next
    })
    setVerdict(null)
  }

  const submitVerdict = () => {
    if (marked.size === 0) return
    const tp = [...marked].filter((i) => trueSet.has(i)).length
    const fp = marked.size - tp
    const precision = tp / (tp + fp || 1)
    const recall = tp / (r.n_true || 1)
    setVerdict({ precision, recall })
    if (onRecord) {
      onRecord({
        question: texts.ui.hunt,
        prediction: `${tp}/${r.n_true} 真特征命中`,
        correct: precision >= 0.75,
        evidence: `precision=${precision.toFixed(2)}, recall=${recall.toFixed(2)}`,
        params: { ...params },
      })
    }
  }

  return (
    <div className="flex flex-col gap-5">
      {/* Exploration quests */}
      <QuestList
        quests={quests}
        labId="feature-hunt"
        onRecord={onRecord}
        params={params}
        evidence={`true=${r.true_idx.join(',')}`}
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

      {/* Activation heatmap */}
      <div className="bg-surface-container-lowest dark:bg-dark-surface rounded-3xl p-4 md:p-6 shadow-ambient dark:shadow-dark-ambient border border-outline-variant/40 dark:border-white/10">
        <div className="flex flex-wrap items-center justify-between gap-3 mb-3">
          <h3 className="font-headline text-lg text-on-surface dark:text-inverse-on-surface inline-flex items-center gap-2">
            <span className="material-symbols-outlined" style={{ fontSize: 20 }}>track_changes</span>
            {texts.ui.title}
          </h3>
          <span className="text-caption text-on-surface-variant dark:text-outline">
            {r.n_features} {texts.ui.neuron} · {r.n_true} {texts.ui.isTrue} · seed {r.seed}
          </span>
        </div>

        <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img" aria-label="activation heatmap">
          {r.activation.map((row, ti) =>
            row.map((v, fi) => (
              <rect
                key={`${ti}-${fi}`}
                x={pad.l + fi * cellW}
                y={pad.t + ti * cellH}
                width={Math.max(cellW, 0.8)}
                height={Math.max(cellH, 0.8)}
                fill={heat(v)}
                opacity={0.9}
              />
            )),
          )}
          {/* y labels: tokens */}
          {r.tokens.map((t, ti) => (
            <text key={ti} x={pad.l - 6} y={pad.t + ti * cellH + 3} textAnchor="end" fontSize="9" className="fill-outline">{t}</text>
          ))}
          {/* x labels: feature idx */}
          {r.features.filter((f) => f.idx % 4 === 0).map((f) => (
            <text key={f.idx} x={pad.l + f.idx * cellW + cellW / 2} y={H - 8} textAnchor="middle" fontSize="8" className="fill-outline">{f.idx}</text>
          ))}
          {/* selected feature column highlight */}
          <rect x={pad.l + sel * cellW} y={pad.t} width={cellW} height={H - pad.t - pad.b}
            fill="none" stroke="rgba(255,255,255,0.85)" strokeWidth="1.5" strokeDasharray="3 3" />
        </svg>

        <div className="mt-2 flex flex-wrap items-center gap-4 text-caption text-on-surface-variant dark:text-outline">
          <span className="inline-flex items-center gap-1.5"><span className="w-3 h-3 rounded-sm" style={{ background: '#F3EBD9' }} /> 0</span>
          <span className="inline-flex items-center gap-1.5"><span className="w-3 h-3 rounded-sm" style={{ background: REAL_COLOR }} /> {actMax.toFixed(2)}</span>
          <span className="ml-auto">{texts.ui.heatmap}（{r.provenance}）</span>
        </div>
      </div>

      {/* Selected neuron detail + hunt */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="bg-surface-container-lowest dark:bg-dark-surface rounded-3xl p-4 md:p-6 shadow-ambient dark:shadow-dark-ambient border border-outline-variant/40 dark:border-white/10">
          <div className="flex items-center justify-between gap-2 mb-3">
            <h3 className="font-headline text-lg text-on-surface dark:text-inverse-on-surface inline-flex items-center gap-2">
              <span className="material-symbols-outlined" style={{ fontSize: 20 }}>neurology</span>
              {texts.ui.neuron} {selFeat.idx}
            </h3>
            <span className={`px-2.5 py-1 rounded-full text-caption font-bold ${
              selFeat.is_true ? 'bg-[#2f6b3e]/15 text-[#2f6b3e] dark:text-[#9ed0a8]' : 'bg-[#C8604A]/15 text-[#C8604A] dark:text-[#f0b3a4]'
            }`}>
              {selFeat.is_true ? texts.ui.isTrue : texts.ui.isNoise}
            </span>
          </div>

          <div className="grid grid-cols-3 gap-2 mb-3 text-caption text-on-surface-variant dark:text-outline">
            <div className="rounded-xl bg-surface-container dark:bg-white/5 px-3 py-2">
              <div className="text-outline">{texts.ui.max}</div>
              <div className="font-mono text-primary dark:text-inverse-primary">{selFeat.max.toFixed(3)}</div>
            </div>
            <div className="rounded-xl bg-surface-container dark:bg-white/5 px-3 py-2">
              <div className="text-outline">{texts.ui.mean}</div>
              <div className="font-mono">{selFeat.mean.toFixed(3)}</div>
            </div>
            <div className="rounded-xl bg-surface-container dark:bg-white/5 px-3 py-2">
              <div className="text-outline">{texts.ui.sparsity}</div>
              <div className="font-mono">{(selFeat.sparsity * 100).toFixed(0)}%</div>
            </div>
          </div>

          <div className="text-caption text-on-surface-variant dark:text-outline mb-2">{texts.ui.activeOn}:</div>
          <div className="flex flex-wrap gap-1.5">
            {r.tokens.map((t, ti) => {
              const v = r.activation[ti][sel]
              const isActive = v > 0.5
              return (
                <span key={ti} className={`px-2 py-1 rounded-lg text-caption font-semibold border ${
                  isActive
                    ? 'bg-[#2f6b3e]/15 text-[#2f6b3e] dark:text-[#9ed0a8] border-[#2f6b3e]/30'
                    : 'bg-surface-container dark:bg-white/5 text-outline border-outline-variant/40 dark:border-white/10'
                }`}>
                  {t} <span className="font-mono opacity-70">{v.toFixed(2)}</span>
                </span>
              )
            })}
          </div>
          {selFeat.true_semantic && (
            <div className="mt-3 text-caption text-on-surface-variant dark:text-outline">
              {texts.ui.revealedSemantic}: <span className="font-mono text-primary dark:text-inverse-primary">{selFeat.true_semantic}</span>
            </div>
          )}
        </div>

        {/* Hunt: mark neurons you believe are real */}
        <div className="bg-surface-container-lowest dark:bg-dark-surface rounded-3xl p-4 md:p-6 shadow-ambient dark:shadow-dark-ambient border border-outline-variant/40 dark:border-white/10">
          <h3 className="font-headline text-lg text-on-surface dark:text-inverse-on-surface mb-1 inline-flex items-center gap-2">
            <span className="material-symbols-outlined" style={{ fontSize: 20 }}>hiking</span>
            {texts.ui.hunt}
          </h3>
          <p className="text-caption text-on-surface-variant dark:text-outline mb-3">
            {texts.ui.mark}（{r.n_true} 个是真）
          </p>
          <div className="flex flex-wrap gap-1.5 mb-3">
            {r.features.map((f) => (
              <button
                key={f.idx}
                onClick={() => { setSelIdx(f.idx); toggleMark(f.idx) }}
                className={`px-3 py-1.5 rounded-xl text-caption font-semibold border transition-all ${
                  marked.has(f.idx)
                    ? 'bg-primary text-on-primary border-primary dark:bg-inverse-primary dark:text-inverse-surface'
                    : 'bg-surface-container dark:bg-white/5 text-on-surface-variant dark:text-outline border-outline-variant/40 dark:border-white/10 hover:border-primary/50'
                }`}
              >
                {f.idx}
              </button>
            ))}
          </div>
          <button
            disabled={marked.size === 0}
            onClick={submitVerdict}
            className="inline-flex items-center gap-2 px-5 py-2 rounded-xl bg-on-surface text-on-primary dark:bg-inverse-surface dark:text-inverse-surface font-label-md text-label-md hover:opacity-90 transition disabled:opacity-40"
          >
            {texts.ui.submit}
          </button>

          {verdict && (
            <div className={`mt-4 rounded-2xl p-4 border ${
              verdict.precision >= 0.75 ? 'border-[#2f6b3e]/40' : 'border-[#C8604A]/40'
            } bg-surface-container dark:bg-white/5`}>
              <div className="flex items-center gap-2 mb-2">
                <span className="text-lg">{verdict.precision >= 0.75 ? '🏆' : '🎯'}</span>
                <span className={`font-label-md font-bold ${
                  verdict.precision >= 0.75 ? 'text-[#2f6b3e] dark:text-[#9ed0a8]' : 'text-[#C8604A] dark:text-[#f0b3a4]'
                }`}>
                  {texts.ui.precision}: {(verdict.precision * 100).toFixed(0)}% · {texts.ui.recall}: {(verdict.recall * 100).toFixed(0)}%
                </span>
              </div>
              <div className="flex flex-wrap gap-1.5">
                {r.features.filter((f) => f.is_true).map((f) => (
                  <span key={f.idx} className="px-2 py-1 rounded-lg bg-[#2f6b3e]/15 text-[#2f6b3e] dark:text-[#9ed0a8] text-caption font-semibold">
                    {texts.ui.revealed} {f.idx} ({f.true_semantic})
                  </span>
                ))}
              </div>
            </div>
          )}
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
              onRecord({
                question: texts.challengeQuestion,
                prediction: label,
                correct,
                evidence: `noise max=${r.features.filter((f) => !f.is_true).reduce((a, b) => Math.max(a, b.max), 0).toFixed(2)}`,
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
              <span className="text-caption text-outline ml-auto font-mono">{texts.ui.verdictEvidence}</span>
            </div>
            <ul className="text-caption text-on-surface-variant dark:text-outline space-y-1">
              {texts.explanation.map((line) => (
                <li key={line}>• {line}</li>
              ))}
            </ul>
          </div>
        )}
      </div>

      {/* Knowledge verification — multi-source */}
      <VerificationPanel entry={verificationData['feature-hunt']} />

      {/* Related notes */}
      <NotesPanel notes={texts.notes} />

      {/* L3 Explain — learner-owned, never machine-graded */}
      <ExplainBox
        onRecord={onRecord}
        evidence={`true=${r.true_idx.join(',')}, noise max=${r.features.filter((f) => !f.is_true).reduce((a, b) => Math.max(a, b.max), 0).toFixed(2)}`}
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
