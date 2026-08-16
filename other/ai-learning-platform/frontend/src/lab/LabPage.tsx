import { useEffect, useMemo, useState, useCallback, useRef } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { api } from '../api'
import type { LabModule, LabParams, LabResult } from './types'
import { ControlRow, defaultParams } from './Controls'
import { useI18n } from '../i18n/context'
import { labModulesZh, labModulesEn, labGroupsZh, labGroupsEn, controlLabelsZh, controlLabelsEn } from '../i18n/lab'
import { useDiscoveries, JournalPanel } from './Journal'
import { QUEST_CHAIN_KEY } from './QuestList'
import GradientDescentLab from './modules/GradientDescentLab'
import AttentionLab from './modules/AttentionLab'
import TransformerLab from './modules/TransformerLab'
import AgentForgeLab from './modules/AgentForgeLab'
import PcaLab from './modules/PcaLab'
import RegularizationLab from './modules/RegularizationLab'
import SvmLab from './modules/SvmLab'
import ActivationLab from './modules/ActivationLab'
import ConvolutionLab from './modules/ConvolutionLab'
import LossLab from './modules/LossLab'
import MatrixTransformLab from './modules/MatrixTransformLab'
import DistributionLab from './modules/DistributionLab'
import EntropyLab from './modules/EntropyLab'
import NeuralNetLab from './modules/NeuralNetLab'
import SamplingMachineLab from './modules/SamplingMachineLab'
import RotaryObservatoryLab from './modules/RotaryObservatoryLab'
import DangerousMountainLab from './modules/DangerousMountainLab'
import ShootingRangeLab from './modules/ShootingRangeLab'
import WeightFreezerLab from './modules/WeightFreezerLab'
import RepresentationRiverLab from './modules/RepresentationRiverLab'
import TokenSocietyLab from './modules/TokenSocietyLab'
import DetectiveLab from './modules/DetectiveLab'
import MoELab from './modules/MoELab'
import MambaMemoryRaceLab from './modules/MambaMemoryRaceLab'
import TransformerMRILab from './modules/TransformerMRILab'
import FeatureHuntLab from './modules/FeatureHuntLab'

interface LabComponentProps {
  result: LabResult | null
  loading: boolean
  error: string | null
  onAction: (key: string) => void
  params: LabParams
  setParams: (p: LabParams) => void
  /** Record a Challenge outcome into the Experiment Journal. */
  onRecord?: (entry: {
    question: string
    prediction: string
    correct: boolean
    evidence: string
    params: LabParams
    insight?: string
  }) => void
}

const LAB_COMPONENTS: Record<string, React.ComponentType<LabComponentProps>> = {
  'gradient-descent': GradientDescentLab,
  attention: AttentionLab,
  'transformer-training': TransformerLab,
  'agent-forge': AgentForgeLab,
  pca: PcaLab,
  regularization: RegularizationLab,
  svm: SvmLab,
  activations: ActivationLab,
  convolution: ConvolutionLab,
  losses: LossLab,
  'matrix-transform': MatrixTransformLab,
  distributions: DistributionLab,
  entropy: EntropyLab,
  'neural-net': NeuralNetLab,
  'sampling-machine': SamplingMachineLab,
  'rotary-observatory': RotaryObservatoryLab,
  'dangerous-mountain': DangerousMountainLab,
  'shooting-range': ShootingRangeLab,
  'weight-freezer': WeightFreezerLab,
  'representation-river': RepresentationRiverLab,
  'token-society': TokenSocietyLab,
  'transformer-detective': DetectiveLab,
  'moe-expert-routing': MoELab,
  'mamba-memory-race': MambaMemoryRaceLab,
  'transformer-mri': TransformerMRILab,
  'feature-hunt': FeatureHuntLab,
}

export default function LabPage() {
  const { lang, t } = useI18n()
  const { moduleId } = useParams<{ moduleId: string }>()
  const navigate = useNavigate()
  const [modules, setModules] = useState<LabModule[]>([])
  const [result, setResult] = useState<LabResult | null>(null)
  const [resultFor, setResultFor] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [params, setParams] = useState<LabParams>({})
  const reqToken = useRef(0)

  const labMeta = lang === 'zh' ? labModulesZh : labModulesEn
  const controlLabels = lang === 'zh' ? controlLabelsZh : controlLabelsEn
  const { entries, addEntry, updateInsight, clear } = useDiscoveries()

  // Cross-experiment challenge chain: which labs have ALL quests done.
  const [questComplete, setQuestComplete] = useState<Set<string>>(() => {
    try {
      const raw = window.localStorage.getItem(QUEST_CHAIN_KEY)
      return raw ? new Set<string>(JSON.parse(raw)) : new Set<string>()
    } catch {
      return new Set<string>()
    }
  })

  useEffect(() => {
    const reload = () => {
      try {
        const raw = window.localStorage.getItem(QUEST_CHAIN_KEY)
        setQuestComplete(raw ? new Set<string>(JSON.parse(raw)) : new Set<string>())
      } catch {
        setQuestComplete(new Set<string>())
      }
    }
    window.addEventListener('ailearn-quest-complete', reload)
    return () => window.removeEventListener('ailearn-quest-complete', reload)
  }, [])

  useEffect(() => {
    let alive = true
    api.lab.modules().then((r) => alive && setModules(r.modules)).catch(() => {})
    return () => { alive = false }
  }, [])

  const active = useMemo(
    () => modules.find((m) => m.id === moduleId) ?? modules[0],
    [modules, moduleId],
  )

  const record = useCallback((entry: {
    question: string
    prediction: string
    correct: boolean
    evidence: string
    params: LabParams
    insight?: string
  }) => {
    if (!active) return
    addEntry({
      experimentId: active.id,
      experimentTitle: labMeta[active.id]?.title ?? active.title,
      ...entry,
    })
  }, [active, labMeta, addEntry])

  // Reset state when switching modules — prevents stale-result crashes.
  useEffect(() => {
    setResult(null)
    setResultFor(null)
    setError(null)
    if (active) setParams(defaultParams(active.controls))
  }, [active])

  const compute = useCallback(async (nextParams: LabParams) => {
    if (!active) return
    const token = ++reqToken.current
    const moduleId = active.id
    setLoading(true)
    setError(null)
    try {
      const r = await api.lab.compute(moduleId, nextParams)
      if (token === reqToken.current) {
        setResult(r)
        setResultFor(moduleId)
      }
    } catch (e) {
      if (token === reqToken.current) {
        setError(e instanceof Error ? e.message : String(e))
      }
    } finally {
      if (token === reqToken.current) setLoading(false)
    }
  }, [active])

  useEffect(() => {
    if (active && Object.keys(params).length > 0) compute(params)
  }, [active, params, compute])

  const onControlChange = (key: string, value: string | number | boolean) => {
    setParams((p) => ({ ...p, [key]: value }))
  }
  const onAction = (key: string) => {
    if (key === 'reset') setParams((p) => ({ ...p }))
    // Sampling Machine: each SAMPLE run advances the seed so the seeded
    // sampler draws a fresh stochastic batch (Simulation Contract).
    else if (key === 'sample') setParams((p) => ({ ...p, seed: (Number(p.seed) || 1) + 1 }))
    // Weight Freezer: SHAKE WEIGHTS regenerates the seeded weight cloud.
    else if (key === 'reshuffle') setParams((p) => ({ ...p, seed: (Number(p.seed) || 1) + 1 }))
  }

  if (modules.length === 0) {
    return <div className="p-8 text-on-surface-variant dark:text-outline">{t.common.loading}</div>
  }

  const Lab = active ? LAB_COMPONENTS[active.id] : null
  // Only pass result to the component if it belongs to the active module.
  // The reset useEffect runs AFTER render, so this guard prevents a newly
  // mounted module from briefly receiving the previous module's data shape.
  const safeResult = active && resultFor === active.id ? result : null

  // Challenge chain helpers: a lab is "completed" when all its quests are
  // done; its `next_experiment` then shows as unlocked.
  const completed = (id: string) => questComplete.has(id)
  const unlocked = (id: string) => modules.some((m) => m.next_experiment === id && completed(m.id))

  // Sidebar sections — the Experiment Lab is organized by "what I want to
  // explore", not by a flat folder list. Modules without a `group` fall into
  // the `classic` bucket (the original math lab).
  const groupOrder = ['now-experimenting', 'rotary-observatory', 'model-behavior', 'learning-dynamics', 'model-efficiency', 'classic']
  const grouped = new Map<string, LabModule[]>()
  for (const m of modules) {
    const g = m.group ?? 'classic'
    if (!grouped.has(g)) grouped.set(g, [])
    grouped.get(g)!.push(m)
  }
  const orderedGroups = groupOrder.filter((g) => grouped.has(g))
  for (const g of grouped.keys()) if (!orderedGroups.includes(g)) orderedGroups.push(g)
  const labGroups = lang === 'zh' ? labGroupsZh : labGroupsEn

  return (
    <div data-scope="tlab" className="flex flex-col md:flex-row gap-6 pt-2 -mx-margin-mobile md:mx-0">
      <aside
        className="hidden md:flex flex-col w-72 shrink-0 rounded-3xl p-5 shadow-ambient dark:shadow-dark-ambient border sticky top-24 self-start max-h-[calc(100vh-7rem)] overflow-y-auto"
        style={{ background: 'var(--tlab-panel)', borderColor: 'var(--tlab-border)' }}
      >
        <div className="px-2 mb-5">
          <div className="flex items-center gap-3 mb-2">
            <div className="w-10 h-10 rounded-full bg-primary dark:bg-inverse-primary text-on-primary dark:text-inverse-surface flex items-center justify-center">
              <span className="material-symbols-outlined fill" style={{ fontSize: 22 }}>science</span>
            </div>
            <div>
              <h2 className="font-headline text-headline-lg-mobile text-primary dark:text-inverse-primary leading-tight">{t.lab.title}</h2>
              <p className="text-caption text-on-surface-variant dark:text-outline">{t.lab.subtitle}</p>
            </div>
          </div>
        </div>

        <nav className="flex flex-col gap-4 mb-5">
          {orderedGroups.map((g) => (
            <div key={g}>
              <h4 className="px-2 mb-1 text-caption uppercase tracking-wider font-semibold text-outline">{labGroups[g] ?? g}</h4>
              <div className="flex flex-col gap-1">
                {grouped.get(g)!.map((m) => {
                  const isActive = active?.id === m.id
                  return (
                    <button
                      key={m.id}
                      onClick={() => navigate(`/lab/${m.id}`)}
                      className={`flex items-center gap-3 px-4 py-2.5 rounded-xl text-left transition-all ${
                        isActive
                          ? 'bg-primary dark:bg-inverse-primary text-on-primary dark:text-inverse-surface font-semibold shadow-sm'
                          : 'text-on-surface-variant dark:text-outline hover:bg-surface-variant dark:hover:bg-white/5'
                      }`}
                    >
                      <span className="material-symbols-outlined" style={{ fontSize: 20 }}>{m.icon}</span>
                      {completed(m.id) ? (
                        <span className="material-symbols-outlined shrink-0" style={{ fontSize: 16, color: isActive ? 'inherit' : '#2f6b3e' }}>verified</span>
                      ) : unlocked(m.id) ? (
                        <span className="material-symbols-outlined shrink-0" style={{ fontSize: 16 }}>lock_open</span>
                      ) : null}
                      <span className="flex-1 min-w-0">
                        <span className="block text-body-md truncate">{labMeta[m.id]?.title ?? m.title}</span>
                        <span className={`block text-caption truncate ${isActive ? 'opacity-80' : ''}`}>{labMeta[m.id]?.subtitle ?? m.subtitle}</span>
                      </span>
                    </button>
                  )
                })}
              </div>
            </div>
          ))}
        </nav>

        {active && active.controls.length > 0 && (
          <div className="border-t border-outline-variant/50 dark:border-white/10 pt-4 flex flex-col gap-4">
            <div className="px-2">
              <h4 className="font-label-md text-label-md uppercase tracking-wider text-on-surface dark:text-dark-on-surface mb-3">{t.lab.controls}</h4>
              <div className="flex flex-col gap-4">
                {active.controls.map((c) => (
                  <ControlRow key={c.key} control={c} label={controlLabels[c.key] ?? c.label} value={params[c.key]} onChange={onControlChange} onAction={onAction} />
                ))}
              </div>
            </div>
            {loading && (
              <div className="px-2 text-caption text-outline inline-flex items-center gap-2">
                <span className="w-3 h-3 rounded-full border-2 border-outline-variant border-t-primary animate-spin" />
                {t.lab.computing}
              </div>
            )}
          </div>
        )}
      </aside>

      <div className="md:hidden flex gap-2 overflow-x-auto pb-1 -mx-margin-mobile px-margin-mobile">
        {modules.map((m) => {
          const isActive = active?.id === m.id
          return (
            <button
              key={m.id}
              onClick={() => navigate(`/lab/${m.id}`)}
              className={`shrink-0 inline-flex items-center gap-2 px-4 py-2 rounded-full text-label-md font-semibold border ${
                isActive
                  ? 'bg-primary text-on-primary border-primary dark:bg-inverse-primary dark:text-inverse-surface'
                  : 'bg-surface-container dark:bg-dark-surface-elevated text-on-surface-variant dark:text-outline border-outline-variant/60 dark:border-white/10'
              }`}
            >
              <span className="material-symbols-outlined" style={{ fontSize: 17 }}>{m.icon}</span>
              {labMeta[m.id]?.title ?? m.title}
            </button>
          )
        })}
      </div>

      <main className="flex-1 min-w-0 flex flex-col gap-5">
        {active && (
          <header
            className="rounded-3xl p-6 shadow-ambient dark:shadow-dark-ambient border"
            style={{ background: 'var(--tlab-panel)', borderColor: 'var(--tlab-border)' }}
          >
            {/* Scoped --tlab-* accent: the experiment area's own material line */}
            <div className="tlab-header-line mb-4" />
            <span className="text-caption uppercase tracking-wider font-semibold text-primary dark:text-inverse-primary">{t.home.categoryNames[active.category] ?? active.category}</span>
            <h1 className="font-headline text-headline-lg-mobile md:text-headline-xl text-on-surface dark:text-inverse-on-surface mt-1">{labMeta[active.id]?.title ?? active.title}</h1>
            <p className="text-body-md text-on-surface-variant dark:text-outline mt-2 max-w-3xl leading-relaxed">{labMeta[active.id]?.blurb ?? active.blurb}</p>

            {/* Cross-experiment link (Discovery Graph edge): the next
                experiment this one points to, shown as a jump chip */}
            {active.next_experiment && (() => {
              const nxt = modules.find((m) => m.id === active.next_experiment)
              return nxt ? (
                <button
                  onClick={() => navigate(`/lab/${nxt.id}`)}
                  className="mt-4 inline-flex items-center gap-2 px-4 py-2 rounded-full border border-primary/40 dark:border-inverse-primary/40 bg-primary/10 dark:bg-inverse-primary/10 text-on-surface dark:text-inverse-on-surface font-label-md text-label-md hover:opacity-90 transition"
                  title={nxt.subtitle}
                >
                  <span className="material-symbols-outlined" style={{ fontSize: 17 }}>{nxt.icon}</span>
                  {labMeta[nxt.id]?.title ?? nxt.title}
                  {unlocked(nxt.id) ? (
                    <span className="material-symbols-outlined" style={{ fontSize: 15 }}>lock_open</span>
                  ) : (
                    <span className="material-symbols-outlined text-outline" style={{ fontSize: 15 }}>lock</span>
                  )}
                  <span className="material-symbols-outlined" style={{ fontSize: 16 }}>arrow_forward</span>
                </button>
              ) : null
            })()}
          </header>
        )}

        {active && active.controls.length > 0 && (
          <details className="md:hidden bg-surface-container dark:bg-dark-surface-elevated rounded-2xl p-4 border border-outline-variant/40 dark:border-white/10">
            <summary className="font-label-md text-label-md font-semibold cursor-pointer inline-flex items-center gap-2">
              <span className="material-symbols-outlined" style={{ fontSize: 18 }}>tune</span>
              {t.lab.controls}
            </summary>
            <div className="flex flex-col gap-4 mt-4">
              {active.controls.map((c) => (
                <ControlRow key={c.key} control={c} label={controlLabels[c.key] ?? c.label} value={params[c.key]} onChange={onControlChange} onAction={onAction} />
              ))}
            </div>
          </details>
        )}

        {error ? (
          <div className="bg-error-container text-on-error-container rounded-2xl p-6 text-body-md">{error}</div>
        ) : Lab ? (
          <Lab result={safeResult} loading={loading} error={error} onAction={onAction} params={params} setParams={setParams} onRecord={record} />
        ) : null}

        {/* Experiment Journal — the learner's own discovery trail */}
        <JournalPanel
          entries={entries}
          onClear={clear}
          onUpdateInsight={updateInsight}
          nextQuestion={active?.next_question}
          onExplore={active?.next_experiment ? () => navigate(`/lab/${active.next_experiment!}`) : undefined}
        />
      </main>
    </div>
  )
}
