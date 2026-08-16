import { useEffect, useMemo, useRef, useState } from 'react'
import type { LabResult, LabParams } from '../types'
import { useI18n } from '../../i18n/context'
import { labTextsZh, labTextsEn, fmt } from '../../i18n/lab'
import { ExplainBox } from '../Journal'
import { QuestList, type Quest } from '../QuestList'
import { ABPanel, type ABShot } from '../ABPanel'

interface HealthInfo {
  label: string
  ok: boolean
  detail: string
}

interface MRIResult extends LabResult {
  scan: string
  steps: number[]
  layers: number[]
  heatmap: number[][] // [layer][step]
  current_layer: number
  current_step: number
  layer_curve: number[]
  step_profile: number[]
  health: HealthInfo
  channels: string[]
  repair: number
  repair_applied: boolean
  seed: number
  simulation_mode: boolean
  provenance: string
}

const CHANNEL_COLOR = {
  loss: { hi: '#C8604A', lo: '#F3EBD9' },   // clay → pale
  entropy: { hi: '#5B6BB0', lo: '#EDE9F4' }, // indigo → pale
  grad_norm: { hi: '#2f6b3e', lo: '#E8F0E9' }, // green → pale
}

export default function TransformerMRILab({ result, loading, params, setParams, onRecord }: {
  result: LabResult | null; loading: boolean; error: string | null
  onAction: (k: string) => void; params: LabParams; setParams: (p: LabParams) => void
  onRecord?: (entry: { question: string; prediction: string; correct: boolean; evidence: string; params: LabParams; insight?: string }) => void
}) {
  const { lang } = useI18n()
  const texts = (lang === 'zh' ? labTextsZh : labTextsEn)['transformer-mri']
  const r = result as MRIResult | null
  const [answer, setAnswer] = useState<string | null>(null)
  const [submitted, setSubmitted] = useState(false)

  const correct = useMemo(() => {
    if (!r) return false
    // Evidence-based verdict: gradient-norm channel exposes the pathology.
    return answer === 'grad'
  }, [r, answer])

  // --- L2 Manipulate Challenge ------------------------------------------
  // Success is DERIVED from the scan channel + health verdict (results,
  // not controls). The learner switches channels/slices until the
  // pathology is exposed or the healthy zone is found.
  // Hooks must live ABOVE the `if (!r) return` early exit.
  const [l2Goal, setL2Goal] = useState<'pathology' | 'healthy' | null>(null)
  const recordedL2 = useRef<string | null>(null)

  const l2Achieved = l2Goal === 'pathology'
    ? r?.scan === 'grad_norm' && !(r?.health.ok ?? true)
    : l2Goal === 'healthy'
      ? r?.scan === 'loss' && (r?.health.ok ?? false)
      : false

  // Record the achievement into the Journal exactly once per goal.
  useEffect(() => {
    if (r && l2Goal && l2Achieved && recordedL2.current !== l2Goal) {
      recordedL2.current = l2Goal
      onRecord?.({
        question: l2Goal === 'pathology' ? texts.ui.l2Pathology : texts.ui.l2Healthy,
        prediction: 'manual tuning',
        correct: true,
        evidence: `${r.health.label}: ${r.health.detail}`,
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

  // --- heatmap geometry --------------------------------------------------
  const W = 640, H = 300
  const pad = { l: 44, r: 14, t: 14, b: 30 }
  const plotW = W - pad.l - pad.r
  const plotH = H - pad.t - pad.b
  const nSteps = r.steps.length
  const nLayers = r.layers.length
  const cellW = plotW / nSteps
  const cellH = plotH / nLayers

  const vMin = 0
  const vMax = Math.max(...r.heatmap.flat(), 0.01)
  const color = CHANNEL_COLOR[r.scan as keyof typeof CHANNEL_COLOR] ?? CHANNEL_COLOR.loss

  // --- Exploration quests (derived from computed result) -----------------
  const quests: Quest[] = [
    { id: 'pathology', label: texts.quests![0], done: r.scan === 'grad_norm' && !r.health.ok },
    { id: 'healthy', label: texts.quests![1], done: r.scan === 'loss' && r.health.ok },
    { id: 'boundary', label: texts.quests![2], done: r.scan === 'grad_norm' && r.current_layer >= 8 },
  ]

  const makeShot = (name: string): ABShot => ({
    name,
    metrics: [
      { key: 'layer', label: 'slice layer', value: r.current_layer },
      { key: 'step', label: 'slice step', value: r.current_step },
      { key: 'deep', label: 'deep L11', value: r.heatmap[11][r.current_step] },
    ],
  })
  const tint = (v: number) => {
    const t = Math.min(1, Math.max(0, (v - vMin) / (vMax - vMin)))
    return `color-mix(in srgb, ${color.hi} ${Math.round(t * 88)}%, ${color.lo})`
  }

  // curve geometry (same x axis as heatmap)
  const curveY = (v: number) => pad.t + plotH - (v / vMax) * plotH
  const layerCurvePath = r.layer_curve.map((v, i) => {
    const x = pad.l + (i / Math.max(1, nSteps - 1)) * plotW
    return `${i === 0 ? 'M' : 'L'}${x.toFixed(1)},${curveY(v).toFixed(1)}`
  }).join(' ')

  // step profile geometry
  const profMax = Math.max(...r.step_profile, 0.01)
  const profX = (v: number) => pad.l + (v / profMax) * plotW

  return (
    <div className="flex flex-col gap-5">
      {/* Exploration quests */}
      <QuestList
        quests={quests}
        onRecord={onRecord}
        params={params}
        evidence={`${r.health.label}: ${r.health.detail}`}
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

      {/* Heatmap + curves */}
      <div className="bg-surface-container-lowest dark:bg-dark-surface rounded-3xl p-4 md:p-6 shadow-ambient dark:shadow-dark-ambient border border-outline-variant/40 dark:border-white/10">
        <div className="flex flex-wrap items-center justify-between gap-3 mb-3">
          <h3 className="font-headline text-lg text-on-surface dark:text-inverse-on-surface inline-flex items-center gap-2">
            <span className="material-symbols-outlined" style={{ fontSize: 20 }}>monitor_heart</span>
            {texts.ui.title}
          </h3>
          <span className="text-caption text-on-surface-variant dark:text-outline">
            {texts.ui.scan}: <span className="font-mono text-primary dark:text-inverse-primary">{r.scan}</span>
            {' · '}L{r.current_layer} · step {r.current_step}
          </span>
        </div>

        <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img" aria-label="mri heatmap">
          {/* heatmap cells */}
          {r.heatmap.map((row, li) =>
            row.map((v, si) => (
              <rect
                key={`${li}-${si}`}
                x={pad.l + si * cellW}
                y={pad.t + li * cellH}
                width={Math.max(cellW, 0.6)}
                height={Math.max(cellH, 0.6)}
                fill={tint(v)}
                opacity={0.9}
              />
            )),
          )}
          {/* y labels: layers */}
          {[0, 3, 6, 9, 11].map((li) => (
            <text key={li} x={pad.l - 6} y={pad.t + li * cellH + 3} textAnchor="end" fontSize="9" className="fill-outline">L{li}</text>
          ))}
          {/* x labels: steps */}
          {[0, 25, 50, 75, 99].map((si) => (
            <text key={si} x={pad.l + (si / 99) * plotW} y={H - 10} textAnchor="middle" fontSize="9" className="fill-outline">{si}</text>
          ))}
          {/* slice cursor: current step vertical */}
          <line
            x1={pad.l + (r.current_step / Math.max(1, nSteps - 1)) * plotW}
            y1={pad.t}
            x2={pad.l + (r.current_step / Math.max(1, nSteps - 1)) * plotW}
            y2={pad.t + plotH}
            stroke="rgba(255,255,255,0.85)"
            strokeWidth="1.2"
            strokeDasharray="3 3"
          />
        </svg>

        <div className="mt-2 flex flex-wrap items-center gap-4 text-caption text-on-surface-variant dark:text-outline">
          <span className="inline-flex items-center gap-1.5">
            <span className="w-3 h-3 rounded-sm" style={{ background: color.lo }} /> 0
          </span>
          <span className="inline-flex items-center gap-1.5">
            <span className="w-3 h-3 rounded-sm" style={{ background: color.hi }} /> {vMax.toFixed(2)}
          </span>
          <span className="ml-auto">← step → · {texts.ui.heatmap} ({r.provenance})</span>
        </div>
      </div>

      {/* Layer curve + step profile */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="bg-surface-container-lowest dark:bg-dark-surface rounded-3xl p-4 md:p-6 shadow-ambient dark:shadow-dark-ambient border border-outline-variant/40 dark:border-white/10">
          <h3 className="font-headline text-lg text-on-surface dark:text-inverse-on-surface mb-3 inline-flex items-center gap-2">
            <span className="material-symbols-outlined" style={{ fontSize: 20 }}>show_chart</span>
            {texts.ui.layerCurve} · L{r.current_layer}
          </h3>
          <svg viewBox={`0 0 320 120`} className="w-full" role="img" aria-label="layer curve">
            <polyline points={r.layer_curve.map((v, i) => {
              const x = (i / Math.max(1, nSteps - 1)) * 300 + 10
              const y = 10 + 100 - (v / vMax) * 100
              return `${x.toFixed(1)},${y.toFixed(1)}`
            }).join(' ')} fill="none" stroke={color.hi} strokeWidth="2" />
            <circle cx={10 + (r.current_step / Math.max(1, nSteps - 1)) * 300} cy={10 + 100 - (r.layer_curve[r.current_step] / vMax) * 100} r="3.5" fill="#C8604A" stroke="#fff" strokeWidth="1" />
          </svg>
        </div>

        <div className="bg-surface-container-lowest dark:bg-dark-surface rounded-3xl p-4 md:p-6 shadow-ambient dark:shadow-dark-ambient border border-outline-variant/40 dark:border-white/10">
          <h3 className="font-headline text-lg text-on-surface dark:text-inverse-on-surface mb-3 inline-flex items-center gap-2">
            <span className="material-symbols-outlined" style={{ fontSize: 20 }}>view_column</span>
            {texts.ui.stepProfile} · step {r.current_step}
          </h3>
          <svg viewBox={`0 0 320 120`} className="w-full" role="img" aria-label="step profile">
            {r.step_profile.map((v, li) => (
              <g key={li}>
                <line x1={12} y1={10 + li * 9} x2={12 + profX(v)} y2={10 + li * 9} stroke={color.hi} strokeWidth="3" opacity={0.5 + 0.5 * (v / profMax)} />
                <text x={20 + profX(v)} y={10 + li * 9 + 3} fontSize="8" className="fill-outline">L{li}</text>
              </g>
            ))}
          </svg>
        </div>
      </div>

      {/* Health verdict + Fix-the-pathology mechanic */}
      <div className={`rounded-3xl p-4 md:p-6 border ${
        r.health.ok ? 'border-[#2f6b3e]/40 dark:border-white/10' : 'border-[#C8604A]/40 dark:border-white/10'
      } bg-surface-container-lowest dark:bg-dark-surface`}>
        <div className="flex items-center gap-2 mb-1">
          <span className="text-lg">{r.health.ok ? '🩺' : '⚠️'}</span>
          <h3 className="font-headline text-lg text-on-surface dark:text-inverse-on-surface inline-flex items-center gap-2">
            {r.health.ok ? texts.ui.healthy : texts.ui.pathology}
          </h3>
          {r.repair_applied && (
            <span className="px-2.5 py-1 rounded-full bg-[#2f6b3e]/15 text-[#2f6b3e] dark:text-[#9ed0a8] text-caption font-bold">
              ✚ repair ×{r.repair.toFixed(1)}
            </span>
          )}
        </div>
        <div className="font-mono text-caption text-on-surface-variant dark:text-outline">
          {fmt(texts.ui.verdictEvidence, { label: r.health.label, detail: r.health.detail })}
        </div>

        {/* Fix: residual scaling heals the vanishing-gradient pathology */}
        {r.scan === 'grad_norm' && !r.health.ok && (
          <div className="mt-3 rounded-2xl p-3 border border-outline-variant/40 dark:border-white/10 bg-surface-container dark:bg-white/5">
            <div className="text-caption text-on-surface-variant dark:text-outline mb-2">{texts.ui.repairHint}</div>
            <button
              onClick={() => setParams({ ...params, repair: '4' })}
              className="inline-flex items-center gap-2 px-4 py-2 rounded-xl bg-on-surface text-on-primary dark:bg-inverse-surface dark:text-inverse-surface font-label-md text-label-md hover:opacity-90 transition"
            >
              <span className="material-symbols-outlined" style={{ fontSize: 16 }}>healing</span>
              {texts.ui.repairBtn}
            </button>
          </div>
        )}
        {r.repair_applied && (
          <button
            onClick={() => setParams({ ...params, repair: '1' })}
            className="mt-3 inline-flex items-center gap-1.5 text-caption text-on-surface-variant dark:text-outline hover:text-[#C8604A] transition"
          >
            <span className="material-symbols-outlined" style={{ fontSize: 15 }}>undo</span>
            {texts.ui.repairUndo}
          </button>
        )}
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
                evidence: `${r.health.label}: ${r.health.detail}`,
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
                {fmt(texts.ui.verdictEvidence, { label: r.health.label, detail: r.health.detail })}
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
            onClick={() => { setL2Goal('pathology'); recordedL2.current = null }}
            className={`px-4 py-2 rounded-xl text-label-md font-semibold border transition-all ${
              l2Goal === 'pathology'
                ? 'bg-primary text-on-primary border-primary dark:bg-inverse-primary dark:text-inverse-surface'
                : 'bg-surface-container-lowest dark:bg-dark-surface text-on-surface-variant dark:text-outline border-outline-variant/60 dark:border-white/10 hover:border-primary/50'
            }`}
          >
            {texts.ui.l2Pathology}
          </button>
          <button
            onClick={() => { setL2Goal('healthy'); recordedL2.current = null }}
            className={`px-4 py-2 rounded-xl text-label-md font-semibold border transition-all ${
              l2Goal === 'healthy'
                ? 'bg-primary text-on-primary border-primary dark:bg-inverse-primary dark:text-inverse-surface'
                : 'bg-surface-container-lowest dark:bg-dark-surface text-on-surface-variant dark:text-outline border-outline-variant/60 dark:border-white/10 hover:border-primary/50'
            }`}
          >
            {texts.ui.l2Healthy}
          </button>
        </div>

        {l2Goal && (
          <div>
            <div className="mb-2 text-caption text-on-surface-variant dark:text-outline">
              {texts.ui.l2CurScan}: <span className="font-mono text-primary dark:text-inverse-primary">{r.scan}</span>
              {' · '}{r.health.label} {r.health.ok ? '✓' : '✗'}
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
        onSnapA={() => setShotA(makeShot(`A: L${r.current_layer}@${r.current_step}`))}
        onSnapB={() => setShotB(makeShot(`B: L${r.current_layer}@${r.current_step}`))}
        onSwap={() => { setShotA(shotB); setShotB(shotA) }}
        onClear={() => { setShotA(null); setShotB(null) }}
      />

      {/* L3 Explain — learner-owned, never machine-graded */}
      <ExplainBox
        onRecord={onRecord}
        evidence={`${r.health.label}: ${r.health.detail}`}
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
