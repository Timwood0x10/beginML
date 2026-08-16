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

interface Injection {
  layer: number
  attention: number
  ffn: number
}

interface RiverResult extends LabResult {
  tokens: string[]
  n_tokens: number
  layers: number[]
  trajectories: number[][][] // [layer][token][x, y]
  injections: Injection[]
  show: string
  seed: number
  simulation_mode: boolean
  provenance: string
}

const TOKEN_COLORS = ['#C8604A', '#5B6BB0', '#2f6b3e', '#7A5C36', '#8a3a35', '#4a6b8a', '#8a6b3a', '#6b5b8a', '#3a8a6b', '#8a4a6b']

export default function RepresentationRiverLab({ result, loading, params, onRecord }: {
  result: LabResult | null; loading: boolean; error: string | null
  onAction: (k: string) => void; params: LabParams; setParams: (p: LabParams) => void
  onRecord?: (entry: { question: string; prediction: string; correct: boolean; evidence: string; params: LabParams }) => void
}) {
  const { lang } = useI18n()
  const texts = (lang === 'zh' ? labTextsZh : labTextsEn)['representation-river']
  const r = result as RiverResult | null
  const [answer, setAnswer] = useState<string | null>(null)
  const [submitted, setSubmitted] = useState(false)

  const correct = useMemo(() => {
    if (!r) return false
    // Attention injection mixes tokens; FFN transforms each token alone.
    return answer === 'mix'
  }, [r, answer])

  // --- L2 Manipulate Challenge ------------------------------------------
  // Success is DERIVED from the last injection (a result, not a control).
  // The learner tunes injection strength until the trajectory stays stable
  // or the injection is strong enough to yank it off course.
  // Hooks must live ABOVE the `if (!r) return` early exit.
  const [l2Goal, setL2Goal] = useState<'stable' | 'break' | null>(null)
  const recordedL2 = useRef<string | null>(null)

  const l2LastInj = r ? r.injections[r.injections.length - 1] : null
  const l2Achieved = l2Goal === 'stable'
    ? (l2LastInj?.attention ?? 1) < 0.3 && (l2LastInj?.ffn ?? 1) < 0.3
    : l2Goal === 'break'
      ? (l2LastInj?.attention ?? 0) >= 0.8 || (l2LastInj?.ffn ?? 0) >= 0.8
      : false

  // Record the achievement into the Journal exactly once per goal.
  useEffect(() => {
    if (r && l2Goal && l2Achieved && recordedL2.current !== l2Goal) {
      recordedL2.current = l2Goal
      const inj = r.injections[r.injections.length - 1]
      onRecord?.({
        question: l2Goal === 'stable' ? texts.ui.l2Stable : texts.ui.l2Break,
        prediction: 'manual tuning',
        correct: true,
        evidence: `attn=${inj.attention.toFixed(3)}, ffn=${inj.ffn.toFixed(3)}`,
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
  const lastInj = r.injections[r.injections.length - 1]
  const quests: Quest[] = [
    { id: 'stable', label: texts.quests![0], done: lastInj.attention < 0.3 && lastInj.ffn < 0.3 },
    { id: 'maxattn', label: texts.quests![1], done: lastInj.attention >= 0.8 || lastInj.ffn >= 0.8 },
    { id: 'both', label: texts.quests![2], done: lastInj.attention >= 0.8 && lastInj.ffn >= 0.8 },
  ]

  const makeShot = (name: string): ABShot => ({
    name,
    metrics: [
      { key: 'attn', label: 'attn inj', value: lastInj.attention },
      { key: 'ffn', label: 'ffn inj', value: lastInj.ffn },
      { key: 'layers', label: 'layers', value: r.layers.length },
    ],
  })

  // --- trajectory plot geometry ----------------------------------------
  const W = 640, H = 380
  const pad = { l: 40, r: 20, t: 16, b: 44 }
  const plotW = W - pad.l - pad.r
  const plotH = H - pad.t - pad.b
  const all = r.trajectories.flat(2)
  const xMin = Math.min(...all.map((v, i) => (i % 2 === 0 ? v : Infinity)))
  const yMin = Math.min(...all.map((v, i) => (i % 2 === 1 ? v : Infinity)))
  const xMax = Math.max(...all.map((v, i) => (i % 2 === 0 ? v : -Infinity)))
  const yMax = Math.max(...all.map((v, i) => (i % 2 === 1 ? v : -Infinity)))
  const xSpan = xMax - xMin || 1
  const ySpan = yMax - yMin || 1
  const px = (v: number) => pad.l + ((v - xMin) / xSpan) * plotW
  const py = (v: number) => pad.t + (1 - (v - yMin) / ySpan) * plotH

  const layerLabels = r.layers.filter((l) => l % 2 === 0)
  const maxLayer = r.layers[r.layers.length - 1]

  return (
    <div className="flex flex-col gap-5">
      {/* SIMULATION MODE banner */}
      <div className="rounded-3xl px-5 py-3 border border-[#8a3a35]/40 bg-[#f3dfdc]/40 dark:bg-[#3d2a28]/40 flex items-center gap-3">
        <span className="material-symbols-outlined text-[#8a3a35]" style={{ fontSize: 20 }}>science</span>
        <div>
          <div className="text-label-md font-bold text-[#8a3a35]">{texts.ui.simulationMode}</div>
          <div className="text-caption text-on-surface-variant dark:text-outline">{texts.ui.simulationHint}</div>
        </div>
      </div>

      {/* Exploration quests */}
      <QuestList
        quests={quests}
        labId="representation-river"
        onRecord={onRecord}
        params={params}
        evidence={`attn=${lastInj.attention.toFixed(3)}, ffn=${lastInj.ffn.toFixed(3)}`}
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

      {/* River: token trajectories */}
      <div className="bg-surface-container-lowest dark:bg-dark-surface rounded-3xl p-4 md:p-6 shadow-ambient dark:shadow-dark-ambient border border-outline-variant/40 dark:border-white/10">
        <div className="flex flex-wrap items-center justify-between gap-3 mb-3">
          <h3 className="font-headline text-lg text-on-surface dark:text-inverse-on-surface inline-flex items-center gap-2">
            <span className="material-symbols-outlined" style={{ fontSize: 20 }}>water</span>
            {texts.ui.river}
          </h3>
          <span className="text-caption text-on-surface-variant dark:text-outline">
            {texts.ui.trajectory} · seed <span className="font-mono">{r.seed}</span>
          </span>
        </div>

        <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img" aria-label="representation river">
          {/* layer gridlines */}
          {layerLabels.map((l) => {
            const x = pad.l + (l / maxLayer) * plotW
            return (
              <g key={l}>
                <line x1={x} y1={pad.t} x2={x} y2={pad.t + plotH} stroke="rgba(125,118,109,0.12)" strokeDasharray="2 4" />
                <text x={x} y={H - 12} textAnchor="middle" fontSize="10" className="fill-outline">{l}</text>
              </g>
            )
          })}
          {/* x axis label */}
          <text x={(pad.l + W - pad.r) / 2} y={H - 4} textAnchor="middle" fontSize="10" className="fill-outline">
            {texts.ui.layer} 0 → {maxLayer}
          </text>

          {/* trajectories */}
          {r.tokens.map((_, t) => {
            const pts = r.trajectories.map((layer) => `${px(layer[t][0]).toFixed(1)},${py(layer[t][1]).toFixed(1)}`).join(' ')
            return (
              <g key={t}>
                <polyline points={pts} fill="none" stroke={TOKEN_COLORS[t % TOKEN_COLORS.length]}
                  strokeWidth="2" opacity="0.85" strokeLinejoin="round" />
                {/* layer dots */}
                {r.trajectories.map((layer, l) => (
                  <circle key={l} cx={px(layer[t][0])} cy={py(layer[t][1])} r={l === 0 || l === r.trajectories.length - 1 ? 4 : 2.4}
                    fill={TOKEN_COLORS[t % TOKEN_COLORS.length]} stroke={l === 0 || l === r.trajectories.length - 1 ? '#fff' : 'none'}
                    strokeWidth="1" />
                ))}
                {/* token label at the end */}
                <text x={px(r.trajectories[maxLayer][t][0]) + 8} y={py(r.trajectories[maxLayer][t][1]) + 3}
                  fontSize="10" fontWeight="600" fill={TOKEN_COLORS[t % TOKEN_COLORS.length]}>
                  {r.tokens[t]}
                </text>
              </g>
            )
          })}
        </svg>

        {/* legend */}
        <div className="mt-2 flex flex-wrap gap-3 text-caption text-on-surface-variant dark:text-outline">
          {r.tokens.map((t, i) => (
            <span key={t} className="inline-flex items-center gap-1.5">
              <span className="w-3 h-3 rounded-full" style={{ background: TOKEN_COLORS[i % TOKEN_COLORS.length] }} />
              {t}
            </span>
          ))}
        </div>
      </div>

      {/* Injections */}
      <div className="bg-surface-container-lowest dark:bg-dark-surface rounded-3xl p-4 md:p-6 shadow-ambient dark:shadow-dark-ambient border border-outline-variant/40 dark:border-white/10">
        <div className="flex flex-wrap items-center justify-between gap-3 mb-3">
          <h3 className="font-headline text-lg text-on-surface dark:text-inverse-on-surface inline-flex items-center gap-2">
            <span className="material-symbols-outlined" style={{ fontSize: 20 }}>tributary</span>
            {texts.ui.river} · {texts.ui.attention} / {texts.ui.ffn}
          </h3>
          <span className="text-caption text-on-surface-variant dark:text-outline">
            show = <span className="font-mono text-primary dark:text-inverse-primary">{r.show}</span>
          </span>
        </div>

        <div className="flex items-end gap-1 h-40">
          {r.injections.map((inj) => {
            const maxI = Math.max(...r.injections.map((x) => Math.max(x.attention, x.ffn)), 0.01)
            const aH = (inj.attention / maxI) * 100
            const fH = (inj.ffn / maxI) * 100
            return (
              <div key={inj.layer} className="flex-1 flex flex-col justify-end items-center min-w-0 gap-0.5" title={`L${inj.layer}: attn ${inj.attention.toFixed(3)} ffn ${inj.ffn.toFixed(3)}`}>
                <div className="w-full rounded-t" style={{ height: `${Math.max(aH, 1)}%`, background: '#5B6BB0', opacity: 0.8 }} />
                <div className="w-full rounded-t" style={{ height: `${Math.max(fH, 1)}%`, background: '#C8604A', opacity: 0.8 }} />
              </div>
            )
          })}
        </div>
        <div className="flex mt-1">
          {r.injections.map((inj) => (
            <span key={inj.layer} className="flex-1 text-center text-[9px] text-outline">{inj.layer % 3 === 0 ? inj.layer : ''}</span>
          ))}
        </div>
        <div className="mt-2 flex flex-wrap gap-4 text-caption text-on-surface-variant dark:text-outline">
          <span className="inline-flex items-center gap-1.5"><span className="w-3 h-2 rounded-sm bg-[#5B6BB0]" /> {texts.ui.attention}</span>
          <span className="inline-flex items-center gap-1.5"><span className="w-3 h-2 rounded-sm bg-[#C8604A]" /> {texts.ui.ffn}</span>
          <span className="ml-auto font-mono">
            {fmt(texts.ui.verdictEvidence, { a: lastInj.attention.toFixed(2), f: lastInj.ffn.toFixed(2) })} @ L{lastInj.layer}
          </span>
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
              const inj = r.injections[r.injections.length - 1]
              onRecord({
                question: texts.challengeQuestion,
                prediction: label,
                correct,
                evidence: `attn=${inj.attention.toFixed(3)}, ffn=${inj.ffn.toFixed(3)}`,
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
                {fmt(texts.ui.verdictEvidence, { a: lastInj.attention.toFixed(2), f: lastInj.ffn.toFixed(2) })}
              </span>
            </div>
            <ul className="text-caption text-on-surface-variant dark:text-outline space-y-1">
              {texts.explanation.map((line) => (
                <li key={line}>• {fmt(line, { a: lastInj.attention.toFixed(2), f: lastInj.ffn.toFixed(2) })}</li>
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
            onClick={() => { setL2Goal('stable'); recordedL2.current = null }}
            className={`px-4 py-2 rounded-xl text-label-md font-semibold border transition-all ${
              l2Goal === 'stable'
                ? 'bg-primary text-on-primary border-primary dark:bg-inverse-primary dark:text-inverse-surface'
                : 'bg-surface-container-lowest dark:bg-dark-surface text-on-surface-variant dark:text-outline border-outline-variant/60 dark:border-white/10 hover:border-primary/50'
            }`}
          >
            {texts.ui.l2Stable}
          </button>
          <button
            onClick={() => { setL2Goal('break'); recordedL2.current = null }}
            className={`px-4 py-2 rounded-xl text-label-md font-semibold border transition-all ${
              l2Goal === 'break'
                ? 'bg-primary text-on-primary border-primary dark:bg-inverse-primary dark:text-inverse-surface'
                : 'bg-surface-container-lowest dark:bg-dark-surface text-on-surface-variant dark:text-outline border-outline-variant/60 dark:border-white/10 hover:border-primary/50'
            }`}
          >
            {texts.ui.l2Break}
          </button>
        </div>

        {l2Goal && (
          <div>
            <div className="mb-2 text-caption text-on-surface-variant dark:text-outline">
              {texts.ui.l2CurInj}: <span className="font-mono text-primary dark:text-inverse-primary">attn={lastInj.attention.toFixed(3)}, ffn={lastInj.ffn.toFixed(3)}</span>
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
        onSnapA={() => setShotA(makeShot(`A: attn=${lastInj.attention.toFixed(2)}`))}
        onSnapB={() => setShotB(makeShot(`B: attn=${lastInj.attention.toFixed(2)}`))}
        onSwap={() => { setShotA(shotB); setShotB(shotA) }}
        onClear={() => { setShotA(null); setShotB(null) }}
      />

      {/* Knowledge verification — multi-source */}
      <VerificationPanel entry={verificationData['representation-river']} />

      {/* Related notes */}
      <NotesPanel notes={texts.notes} />

      {/* L3 Explain — learner-owned, never machine-graded */}
      <ExplainBox
        onRecord={onRecord}
        evidence={`attn=${r.injections[r.injections.length - 1].attention.toFixed(3)}, ffn=${r.injections[r.injections.length - 1].ffn.toFixed(3)}`}
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
