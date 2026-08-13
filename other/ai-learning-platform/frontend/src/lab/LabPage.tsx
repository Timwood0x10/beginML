import { useEffect, useMemo, useState, useCallback, useRef } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { api } from '../api'
import type { LabModule, LabParams, LabResult } from './types'
import { ControlRow, defaultParams } from './Controls'
import { useI18n } from '../i18n/context'
import { labModulesZh, labModulesEn, controlLabelsZh, controlLabelsEn } from '../i18n/lab'
import GradientDescentLab from './modules/GradientDescentLab'
import AttentionLab from './modules/AttentionLab'
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

interface LabComponentProps {
  result: LabResult | null
  loading: boolean
  error: string | null
  onAction: (key: string) => void
  params: LabParams
  setParams: (p: LabParams) => void
}

const LAB_COMPONENTS: Record<string, React.ComponentType<LabComponentProps>> = {
  'gradient-descent': GradientDescentLab,
  attention: AttentionLab,
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

  useEffect(() => {
    let alive = true
    api.lab.modules().then((r) => alive && setModules(r.modules)).catch(() => {})
    return () => { alive = false }
  }, [])

  const active = useMemo(
    () => modules.find((m) => m.id === moduleId) ?? modules[0],
    [modules, moduleId],
  )

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
  }

  if (modules.length === 0) {
    return <div className="p-8 text-on-surface-variant dark:text-outline">{t.common.loading}</div>
  }

  const Lab = active ? LAB_COMPONENTS[active.id] : null
  // Only pass result to the component if it belongs to the active module.
  // The reset useEffect runs AFTER render, so this guard prevents a newly
  // mounted module from briefly receiving the previous module's data shape.
  const safeResult = active && resultFor === active.id ? result : null

  return (
    <div className="flex flex-col md:flex-row gap-6 pt-2 -mx-margin-mobile md:mx-0">
      <aside className="hidden md:flex flex-col w-72 shrink-0 bg-surface-container dark:bg-dark-surface-elevated rounded-3xl p-5 shadow-ambient dark:shadow-dark-ambient border border-outline-variant/40 dark:border-white/10 sticky top-24 self-start max-h-[calc(100vh-7rem)] overflow-y-auto">
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

        <nav className="flex flex-col gap-1 mb-5">
          {modules.map((m) => {
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
                <span className="flex-1 min-w-0">
                  <span className="block text-body-md truncate">{labMeta[m.id]?.title ?? m.title}</span>
                  <span className={`block text-caption truncate ${isActive ? 'opacity-80' : ''}`}>{labMeta[m.id]?.subtitle ?? m.subtitle}</span>
                </span>
              </button>
            )
          })}
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
          <header className="bg-surface-container-low dark:bg-dark-surface rounded-3xl p-6 shadow-ambient dark:shadow-dark-ambient border border-outline-variant/40 dark:border-white/10">
            <span className="text-caption uppercase tracking-wider font-semibold text-primary dark:text-inverse-primary">{t.home.categoryNames[active.category] ?? active.category}</span>
            <h1 className="font-headline text-headline-lg-mobile md:text-headline-xl text-on-surface dark:text-inverse-on-surface mt-1">{labMeta[active.id]?.title ?? active.title}</h1>
            <p className="text-body-md text-on-surface-variant dark:text-outline mt-2 max-w-3xl leading-relaxed">{labMeta[active.id]?.blurb ?? active.blurb}</p>
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
          <Lab result={safeResult} loading={loading} error={error} onAction={onAction} params={params} setParams={setParams} />
        ) : null}
      </main>
    </div>
  )
}
