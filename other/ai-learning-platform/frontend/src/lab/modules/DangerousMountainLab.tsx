import { useEffect, useMemo, useRef, useState } from 'react'
import type { LabResult, LabParams } from '../types'
import { useI18n } from '../../i18n/context'
import { labTextsZh, labTextsEn, fmt } from '../../i18n/lab'
import { ExplainBox } from '../Journal'
import { VerificationPanel } from '../VerificationPanel'
import { verificationData } from '../verification'
import { NotesPanel } from '../NotesPanel'
import { QuestList, type Quest } from '../QuestList'
import { ABPanel, type ABShot } from '../ABPanel'

interface MountainResult extends LabResult {
  caps: number[]
  train: number[]
  test: number[]
  classic: number[]
  n: number
  noise: number
  capacity: number
  index: number
  train_error: number
  test_error: number
  classic_error: number
  state: 'underfit' | 'danger' | 'overparam'
  peak_cap: number
  peak_error: number
  interpolation_threshold: number
  danger_zone: [number, number]
  provenance: string
}

export default function DangerousMountainLab({ result, loading, params, onRecord }: {
  result: LabResult | null; loading: boolean; error: string | null
  onAction: (k: string) => void; params: LabParams; setParams: (p: LabParams) => void
  onRecord?: (entry: { question: string; prediction: string; correct: boolean; evidence: string; params: LabParams }) => void
}) {
  const { lang } = useI18n()
  const texts = (lang === 'zh' ? labTextsZh : labTextsEn)['dangerous-mountain']
  const r = result as MountainResult | null
  const [answer, setAnswer] = useState<string | null>(null)
  const [submitted, setSubmitted] = useState(false)
  // THEORY → REALITY blend: 0 = classical U-curve, 1 = observed double descent
  const [blend, setBlend] = useState(1)

  const correct = useMemo(() => {
    if (!r) return false
    // Evidence-based verdict (deterministic, no LLM): does test error fall
    // again in the overparameterized regime compared with underfit?
    const underfit = r.test.filter((_, i) => r.caps[i] < 0.75 * r.n)
    const overparam = r.test.filter((_, i) => r.caps[i] > 1.5 * r.n)
    const underMin = Math.min(...underfit)
    const overMin = Math.min(...overparam)
    return answer === (overMin < underMin ? 'down' : 'up')
  }, [r, answer])

  // --- L2 Manipulate Challenge ------------------------------------------
  // Success is DERIVED from the computed curve (a result, not a control).
  // The learner tunes capacity / samples / noise until the model lands in
  // the overparameterized valley or right on the double-descent peak.
  // Hooks must live ABOVE the `if (!r) return` early exit.
  const [l2Goal, setL2Goal] = useState<'valley' | 'peak' | null>(null)
  const recordedL2 = useRef<string | null>(null)

  const l2UnderfitMin = r ? Math.min(...r.test.filter((_, i) => r.caps[i] < 0.75 * r.n)) : Infinity
  const l2Achieved = l2Goal === 'valley'
    ? (r?.state ?? '') === 'overparam' && (r?.test_error ?? Infinity) < l2UnderfitMin
    : l2Goal === 'peak'
      ? r?.capacity === r?.peak_cap
      : false

  // Record the achievement into the Journal exactly once per goal.
  useEffect(() => {
    if (r && l2Goal && l2Achieved && recordedL2.current !== l2Goal) {
      recordedL2.current = l2Goal
      onRecord?.({
        question: l2Goal === 'valley' ? texts.ui.l2Valley : texts.ui.l2Peak,
        prediction: 'manual tuning',
        correct: true,
        evidence: `state=${r.state}, test=${r.test_error.toFixed(4)}`,
        params: { ...params },
      })
    }
  }, [l2Goal, l2Achieved, r, onRecord, texts, params])

  // --- A/B snapshots (hooks above the early exit too) --------------------
  const [shotA, setShotA] = useState<ABShot | null>(null)
  const [shotB, setShotB] = useState<ABShot | null>(null)

  if (!r) {
    return <div className="p-10 text-on-surface-variant dark:text-outline">{loading ? texts.ui.computing : texts.ui.adjustControls}</div>
  }

  // --- Exploration quests (derived from computed result) -----------------
  const underfitMin = Math.min(...r.test.filter((_, i) => r.caps[i] < 0.75 * r.n))
  const quests: Quest[] = [
    { id: 'valley', label: texts.quests![0], done: r.state === 'overparam' && r.test_error < underfitMin },
    { id: 'peak', label: texts.quests![1], done: r.capacity === r.peak_cap },
    { id: 'min', label: texts.quests![2], done: r.test_error === Math.min(...r.test) },
  ]

  const makeShot = (name: string): ABShot => ({
    name,
    metrics: [
      { key: 'capacity', label: 'capacity', value: r.capacity },
      { key: 'test', label: 'test err', value: r.test_error, higherBetter: false },
      { key: 'train', label: 'train err', value: r.train_error, higherBetter: false },
    ],
  })

  const { caps, train, test, classic } = r
  const underfit = test.filter((_, i) => caps[i] < 0.75 * r.n)
  const overparam = test.filter((_, i) => caps[i] > 1.5 * r.n)
  const underMin = Math.min(...underfit)
  const overMin = Math.min(...overparam)

  // chart geometry
  const W = 660, H = 300
  const pad = { l: 50, r: 16, t: 16, b: 34 }
  const plotW = W - pad.l - pad.r
  const plotH = H - pad.t - pad.b
  const capMin = Math.min(...caps)
  const capMax = Math.max(...caps)
  const yMax = Math.max(...test, ...train) * 1.15

  const px = (cap: number) => pad.l + ((cap - capMin) / (capMax - capMin)) * plotW
  const py = (err: number) => pad.t + (1 - Math.min(err, yMax) / yMax) * plotH
  const poly = (vals: number[]) => caps.map((c, i) => `${px(c).toFixed(1)},${py(vals[i]).toFixed(1)}`).join(' ')
  // theory→reality fused curve, clamped to the chart top
  const fused = test.map((t, i) => classic[i] * (1 - blend) + t * blend)

  const [dzLo, dzHi] = r.danger_zone
  const cur = r.capacity
  const curErr = r.test_error

  const stateCard = {
    underfit: { icon: '🌱', title: texts.ui.stateUnderfit, desc: texts.ui.stateUnderfitDesc, cls: 'border-[#5B6BB0]/40 text-[#5B6BB0] dark:text-[#aab4dd]' },
    danger: { icon: '⚠️', title: texts.ui.stateDanger, desc: texts.ui.stateDangerDesc, cls: 'border-[#C8604A]/40 text-[#C8604A] dark:text-[#f0b3a4]' },
    overparam: { icon: '🏔️', title: texts.ui.stateOverparam, desc: texts.ui.stateOverparamDesc, cls: 'border-[#2f6b3e]/40 text-[#2f6b3e] dark:text-[#9ed0a8]' },
  }[r.state]

  return (
    <div className="flex flex-col gap-5">
      {/* Exploration quests */}
      <QuestList
        quests={quests}
        labId="dangerous-mountain"
        onRecord={onRecord}
        params={params}
        evidence={`state=${r.state}, test=${r.test_error.toFixed(4)}`}
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

      {/* Mountain chart */}
      <div className="bg-surface-container-lowest dark:bg-dark-surface rounded-3xl p-4 md:p-6 shadow-ambient dark:shadow-dark-ambient border border-outline-variant/40 dark:border-white/10">
        <div className="flex flex-wrap items-center justify-between gap-3 mb-2">
          <h3 className="font-headline text-lg text-on-surface dark:text-inverse-on-surface inline-flex items-center gap-2">
            <span className="material-symbols-outlined" style={{ fontSize: 20 }}>terrain</span>
            {texts.ui.title}
          </h3>
          <span className="text-caption text-on-surface-variant dark:text-outline">
            {texts.ui.capacityLabel} <span className="font-mono text-primary dark:text-inverse-primary">{cur}</span>
            {' · '}n = <span className="font-mono">{r.n}</span>
            {' · '}noise = <span className="font-mono">{r.noise.toFixed(2)}</span>
          </span>
        </div>

        <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img" aria-label="double descent">
          {/* danger zone */}
          <rect
            x={px(dzLo)} y={pad.t} width={px(dzHi) - px(dzLo)} height={plotH}
            fill="rgba(200,96,74,0.10)" stroke="rgba(200,96,74,0.25)" strokeDasharray="3 3"
          />
          <text x={(px(dzLo) + px(dzHi)) / 2} y={pad.t + 14} textAnchor="middle" fontSize="10" fill="rgba(200,96,74,0.8)">
            ⚠ DANGER ZONE
          </text>

          {/* y grid */}
          {[0, 0.25, 0.5, 0.75, 1].map((t) => {
            const y = pad.t + t * plotH
            const val = (yMax * (1 - t)).toFixed(1)
            return (
              <g key={t}>
                <line x1={pad.l} y1={y} x2={W - pad.r} y2={y} stroke="rgba(125,118,109,0.14)" strokeDasharray="2 4" />
                <text x={pad.l - 8} y={y + 4} textAnchor="end" fontSize="10" className="fill-outline">{val}</text>
              </g>
            )
          })}
          {/* x axis */}
          <line x1={pad.l} y1={pad.t + plotH} x2={W - pad.r} y2={pad.t + plotH} stroke="rgba(125,118,109,0.4)" />
          {[0, 100, 200, 300].map((c) => (
            <g key={c}>
              <line x1={px(c)} y1={pad.t + plotH} x2={px(c)} y2={pad.t + plotH + 4} stroke="rgba(125,118,109,0.4)" />
              <text x={px(c)} y={H - 12} textAnchor="middle" fontSize="10" className="fill-outline">{c}</text>
            </g>
          ))}
          <text x={(pad.l + W - pad.r) / 2} y={H - 4} textAnchor="middle" fontSize="10" className="fill-outline">
            {texts.ui.capacityLabel} → (model capacity)
          </text>

          {/* classical U (dashed, grey) */}
          <polyline points={poly(classic)} fill="none" stroke="rgba(125,118,109,0.55)" strokeWidth="1.6" strokeDasharray="5 4" />
          {/* fused theory→reality */}
          <polyline points={poly(fused)} fill="none" stroke="rgba(91,107,176,0.55)" strokeWidth="2" strokeDasharray={blend < 0.98 ? '6 4' : 'none'} />
          {/* train (green) */}
          <polyline points={poly(train)} fill="none" stroke="#2f6b3e" strokeWidth="2" />
          {/* test (clay) */}
          <polyline points={poly(test)} fill="none" stroke="#C8604A" strokeWidth="2.4" />

          {/* peak marker */}
          <circle cx={px(r.peak_cap)} cy={py(r.peak_error)} r="4.5" fill="#C8604A" stroke="#fff" strokeWidth="1.5" />
          <text x={px(r.peak_cap) + 8} y={py(r.peak_error) - 8} fontSize="10" fill="#C8604A" fontWeight="600">
            peak {r.peak_error.toFixed(1)} @ cap {r.peak_cap}
          </text>

          {/* current capacity cursor */}
          <line x1={px(cur)} y1={pad.t} x2={px(cur)} y2={pad.t + plotH} stroke="rgba(200,96,74,0.7)" strokeWidth="1.4" />
          <circle cx={px(cur)} cy={py(curErr)} r="5" fill="#C8604A" stroke="#fff" strokeWidth="2" />
        </svg>

        {/* THEORY → REALITY blend */}
        <div className="mt-3">
          <div className="flex items-center justify-between text-caption text-on-surface-variant dark:text-outline mb-1">
            <span>{texts.ui.classicView}</span>
            <span>{texts.ui.theoryToReality}</span>
            <span>{texts.ui.modernView}</span>
          </div>
          <input
            type="range" min={0} max={1} step={0.02} value={blend}
            onChange={(e) => setBlend(parseFloat(e.target.value))}
            className="ailearn-range w-full"
          />
        </div>

        {/* legend */}
        <div className="flex flex-wrap gap-4 mt-3 text-caption text-on-surface-variant dark:text-outline">
          <span className="inline-flex items-center gap-1.5"><span className="w-4 h-0.5 rounded bg-[#C8604A]" /> {texts.ui.testError}</span>
          <span className="inline-flex items-center gap-1.5"><span className="w-4 h-0.5 rounded bg-[#2f6b3e]" /> {texts.ui.trainError}</span>
          <span className="inline-flex items-center gap-1.5"><span className="w-4 border-t-2 border-dashed border-[#7d766d]" /> {texts.ui.classicView}</span>
        </div>
      </div>

      {/* State card */}
      <div className={`rounded-3xl p-5 border ${stateCard.cls} bg-surface-container-lowest dark:bg-dark-surface`}>
        <div className="flex items-center gap-2 mb-1">
          <span className="text-lg">{stateCard.icon}</span>
          <h3 className="font-label-md text-label-md font-bold">{stateCard.title}</h3>
          <span className="text-caption text-outline ml-auto font-mono">
            {fmt(texts.ui.verdictEvidence, { u: underMin.toFixed(2), p: r.peak_error.toFixed(2), o: overMin.toFixed(2) })}
          </span>
        </div>
        <p className="text-body-md text-on-surface dark:text-inverse-on-surface">{stateCard.desc}</p>
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
                evidence: `state=${r.state}, test=${r.test_error.toFixed(4)}`,
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
                {fmt(texts.ui.verdictEvidence, { u: underMin.toFixed(2), p: r.peak_error.toFixed(2), o: overMin.toFixed(2) })}
              </span>
            </div>
            <ul className="text-caption text-on-surface-variant dark:text-outline space-y-1">
              {texts.explanation.map((line) => (
                <li key={line}>• {fmt(line, { u: underMin.toFixed(2), p: r.peak_error.toFixed(2), o: overMin.toFixed(2) })}</li>
              ))}
            </ul>
            <div className="mt-2 text-label-md font-semibold text-primary dark:text-inverse-primary">{texts.aha}</div>
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
            onClick={() => { setL2Goal('valley'); recordedL2.current = null }}
            className={`px-4 py-2 rounded-xl text-label-md font-semibold border transition-all ${
              l2Goal === 'valley'
                ? 'bg-primary text-on-primary border-primary dark:bg-inverse-primary dark:text-inverse-surface'
                : 'bg-surface-container-lowest dark:bg-dark-surface text-on-surface-variant dark:text-outline border-outline-variant/60 dark:border-white/10 hover:border-primary/50'
            }`}
          >
            {texts.ui.l2Valley}
          </button>
          <button
            onClick={() => { setL2Goal('peak'); recordedL2.current = null }}
            className={`px-4 py-2 rounded-xl text-label-md font-semibold border transition-all ${
              l2Goal === 'peak'
                ? 'bg-primary text-on-primary border-primary dark:bg-inverse-primary dark:text-inverse-surface'
                : 'bg-surface-container-lowest dark:bg-dark-surface text-on-surface-variant dark:text-outline border-outline-variant/60 dark:border-white/10 hover:border-primary/50'
            }`}
          >
            {texts.ui.l2Peak}
          </button>
        </div>

        {l2Goal && (
          <div>
            <div className="mb-2 text-caption text-on-surface-variant dark:text-outline">
              {texts.ui.l2CurState}: <span className="font-mono text-primary dark:text-inverse-primary">{r.state}</span>
              {' · '}test <span className="font-mono">{r.test_error.toFixed(4)}</span>
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

      {/* A/B compare */}
      <ABPanel
        a={shotA}
        b={shotB}
        onSnapA={() => setShotA(makeShot(`A: cap=${r.capacity}`))}
        onSnapB={() => setShotB(makeShot(`B: cap=${r.capacity}`))}
        onSwap={() => { setShotA(shotB); setShotB(shotA) }}
        onClear={() => { setShotA(null); setShotB(null) }}
      />

      {/* Knowledge verification — multi-source */}
      <VerificationPanel entry={verificationData['dangerous-mountain']} />

      {/* Related notes */}
      <NotesPanel notes={texts.notes} />

      {/* L3 Explain — learner-owned, never machine-graded */}
      <ExplainBox
        onRecord={onRecord}
        evidence={`state=${r.state}, test=${r.test_error.toFixed(4)}`}
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
