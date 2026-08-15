// Experiment Journal — a learner's own discovery trail (plan §12).
//
// Every completed Challenge can be saved as a Discovery entry: the question,
// the prediction, whether it was right, the evidence and the params that
// produced it. Entries persist in localStorage under "ailearn-discoveries"
// and are shown as MY DISCOVERIES cards. The Insight belongs to the learner
// (No-LLM principle) — we only record what happened in the experiment.
import { useCallback, useEffect, useState } from 'react'
import { useI18n } from '../i18n/context'
import { fmt } from '../i18n/lab'
import type { LabParams } from './types'

export interface DiscoveryEntry {
  id: string
  experimentId: string
  experimentTitle: string
  question: string
  prediction: string
  correct: boolean
  evidence: string
  params: LabParams
  createdAt: number
}

const STORAGE_KEY = 'ailearn-discoveries'
const MAX_ENTRIES = 24

const UI: Record<'zh' | 'en', Record<string, string>> = {
  zh: {
    title: '我的发现',
    subtitle: 'Predict → Experiment → Discover',
    empty: '还没有发现。做一次实验、提交一次预测，把它存进来。',
    emptyHint: '在实验里提交预测后，这里会生成一张 Experiment Card。',
    prediction: '预测',
    correct: '✓',
    wrong: '✗',
    nextQuestion: '下一个问题',
    explore: '去探索 →',
    deleteAll: '清空',
    evidence: '证据',
  },
  en: {
    title: 'My Discoveries',
    subtitle: 'Predict → Experiment → Discover',
    empty: 'No discoveries yet. Run an experiment, submit a prediction, and save it here.',
    emptyHint: 'Submit a prediction in any experiment to generate an Experiment Card.',
    prediction: 'Prediction',
    correct: '✓',
    wrong: '✗',
    nextQuestion: 'Next question',
    explore: 'Explore →',
    deleteAll: 'Clear',
    evidence: 'Evidence',
  },
}

function load(): DiscoveryEntry[] {
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY)
    if (!raw) return []
    const parsed = JSON.parse(raw)
    return Array.isArray(parsed) ? parsed.slice(0, MAX_ENTRIES) : []
  } catch {
    return []
  }
}

export function useDiscoveries() {
  const [entries, setEntries] = useState<DiscoveryEntry[]>(load)

  useEffect(() => {
    try {
      window.localStorage.setItem(STORAGE_KEY, JSON.stringify(entries.slice(0, MAX_ENTRIES)))
    } catch {
      /* storage full / unavailable — ignore, memory-only */
    }
  }, [entries])

  const addEntry = useCallback((e: Omit<DiscoveryEntry, 'id' | 'createdAt'>) => {
    setEntries((prev) => [
      { ...e, id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`, createdAt: Date.now() },
      ...prev,
    ].slice(0, MAX_ENTRIES))
  }, [])

  const clear = useCallback(() => setEntries([]), [])
  return { entries, addEntry, clear }
}

/**
 * JournalPanel — MY DISCOVERIES cards + the experiment's next-question exit.
 * Rendered once per lab page, below the active experiment.
 */
export function JournalPanel({
  entries,
  onClear,
  nextQuestion,
  onExplore,
}: {
  entries: DiscoveryEntry[]
  onClear: () => void
  nextQuestion?: string
  onExplore?: () => void
}) {
  const { lang } = useI18n()
  const ui = UI[lang]

  return (
    <div className="flex flex-col gap-4">
      {/* Next-question exit (Discovery Graph edge) */}
      {nextQuestion && (
        <div className="bg-surface-container-lowest dark:bg-dark-surface rounded-3xl p-5 border border-outline-variant/40 dark:border-white/10">
          <div className="flex flex-wrap items-center gap-3">
            <span className="text-lg">🧭</span>
            <div className="min-w-0 flex-1">
              <div className="text-caption uppercase tracking-wider font-semibold text-outline mb-0.5">{ui.nextQuestion}</div>
              <div className="text-body-md text-on-surface dark:text-inverse-on-surface">{nextQuestion}</div>
            </div>
            {onExplore && (
              <button
                onClick={onExplore}
                className="inline-flex items-center gap-1.5 px-4 py-2 rounded-xl bg-primary text-on-primary dark:bg-inverse-primary dark:text-inverse-surface font-label-md text-label-md hover:opacity-90 transition"
              >
                {ui.explore}
              </button>
            )}
          </div>
        </div>
      )}

      {/* MY DISCOVERIES */}
      <div className="bg-surface-container-lowest dark:bg-dark-surface rounded-3xl p-4 md:p-6 shadow-ambient dark:shadow-dark-ambient border border-outline-variant/40 dark:border-white/10">
        <div className="flex items-center justify-between gap-3 mb-3">
          <h3 className="font-headline text-lg text-on-surface dark:text-inverse-on-surface inline-flex items-center gap-2">
            <span className="material-symbols-outlined" style={{ fontSize: 20 }}>bookmark</span>
            {ui.title}
          </h3>
          {entries.length > 0 && (
            <button
              onClick={onClear}
              className="inline-flex items-center gap-1 text-caption text-on-surface-variant dark:text-outline hover:text-[#C8604A] transition"
            >
              <span className="material-symbols-outlined" style={{ fontSize: 15 }}>delete_sweep</span>
              {ui.deleteAll}
            </button>
          )}
        </div>
        <p className="text-caption text-on-surface-variant dark:text-outline mb-4">{ui.subtitle}</p>

        {entries.length === 0 ? (
          <div className="rounded-2xl border border-dashed border-outline-variant/50 dark:border-white/15 p-6 text-center">
            <div className="text-2xl mb-2">🔬</div>
            <div className="text-body-md text-on-surface dark:text-dark-on-surface">{ui.empty}</div>
            <div className="text-caption text-outline mt-1">{ui.emptyHint}</div>
          </div>
        ) : (
          <div className="flex flex-col gap-2">
            {entries.map((e) => (
              <div key={e.id} className="rounded-2xl px-4 py-3 border border-outline-variant/40 dark:border-white/10 bg-surface-container dark:bg-white/5">
                <div className="flex items-center gap-2 mb-1">
                  <span className={`text-sm ${e.correct ? 'text-[#2f6b3e] dark:text-[#9ed0a8]' : 'text-[#C8604A] dark:text-[#f0b3a4]'}`}>
                    {e.correct ? ui.correct : ui.wrong}
                  </span>
                  <span className="font-label-md font-semibold text-on-surface dark:text-dark-on-surface truncate">{e.experimentTitle}</span>
                  <span className="ml-auto text-caption text-outline font-mono shrink-0">
                    {new Date(e.createdAt).toLocaleDateString(lang === 'zh' ? 'zh-CN' : 'en-US', { month: 'short', day: 'numeric' })}
                  </span>
                </div>
                <div className="text-caption text-on-surface-variant dark:text-outline mb-1">{e.question}</div>
                <div className="flex flex-wrap items-center gap-2 text-caption">
                  <span className="inline-flex items-center gap-1 text-outline">
                    {ui.prediction}: <span className="font-mono text-on-surface dark:text-dark-on-surface">{e.prediction}</span>
                  </span>
                  <span className="text-outline">·</span>
                  <span className="font-mono text-outline truncate">{e.evidence}</span>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
