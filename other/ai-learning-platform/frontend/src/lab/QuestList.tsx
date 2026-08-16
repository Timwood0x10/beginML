// QuestList — the exploration-quest tracker (interactivity layer).
//
// Every experiment can define 3 progressive quests ("find the peak", "make
// the error minimal", "break it"). QuestList renders them as checkable goals
// and auto-records the FIRST completion of each quest into the Experiment
// Journal as a Discovery (correct: true, prediction: quest). Goal completion
// is derived by each experiment from its computed result — never guessed.
// Copy lives here (shared UI) so enabling quests in a new experiment is one
// <QuestList quests={...} /> line.
import { useEffect, useRef } from 'react'
import { useI18n } from '../i18n/context'
import { fmt } from '../i18n/lab'
import type { LabParams } from './types'

export interface Quest {
  id: string
  label: string
  done: boolean
}

const UI: Record<'zh' | 'en', Record<string, string>> = {
  zh: {
    title: '探索任务',
    hint: '拖动控件，亲手达成每个目标。',
    done: '✓',
    progress: '{n}/{total} 达成',
  },
  en: {
    title: 'Exploration Quests',
    hint: 'Adjust the controls and hit each goal by hand.',
    done: '✓',
    progress: '{n}/{total} done',
  },
}

export function QuestList({
  quests,
  onRecord,
  params,
  evidence,
}: {
  quests: Quest[]
  onRecord?: (entry: {
    question: string
    prediction: string
    correct: boolean
    evidence: string
    params: LabParams
    insight?: string
  }) => void
  params: LabParams
  evidence: string
}) {
  const { lang } = useI18n()
  const ui = UI[lang]
  const doneCount = quests.filter((q) => q.done).length

  // Record a quest ONLY at the moment it flips from false → true. `prevDone`
  // holds the done-set of the previous render; the first render (null) just
  // establishes the baseline without recording, so remounting the component
  // (e.g. navigating away and back) never re-writes the same quest to the
  // Journal.
  const prevDone = useRef<Set<string> | null>(null)

  useEffect(() => {
    if (!onRecord) return
    const nowDone = new Set(quests.filter((q) => q.done).map((q) => q.id))
    if (prevDone.current) {
      for (const id of nowDone) {
        if (!prevDone.current.has(id)) {
          const q = quests.find((x) => x.id === id)
          if (q) {
            onRecord({
              question: q.label,
              prediction: 'quest completed',
              correct: true,
              evidence,
              params: { ...params },
            })
          }
        }
      }
    }
    prevDone.current = nowDone
  }, [quests, onRecord, evidence, params])

  return (
    <div className="bg-surface-container-lowest dark:bg-dark-surface rounded-3xl p-5 border border-outline-variant/40 dark:border-white/10">
      <div className="flex items-center gap-2 mb-1">
        <span className="text-lg">🧭</span>
        <h3 className="font-label-md text-label-md uppercase tracking-wider text-on-surface dark:text-dark-on-surface">{ui.title}</h3>
        <span className="ml-auto text-caption text-outline font-mono">{fmt(ui.progress, { n: doneCount, total: quests.length })}</span>
      </div>
      <p className="text-caption text-on-surface-variant dark:text-outline mb-3">{ui.hint}</p>
      <div className="flex flex-col gap-2">
        {quests.map((q) => (
          <div
            key={q.id}
            className={`flex items-center gap-3 px-4 py-2.5 rounded-xl border transition-all ${
              q.done
                ? 'border-[#2f6b3e]/40 bg-[#2f6b3e]/10'
                : 'border-outline-variant/40 dark:border-white/10 bg-surface-container dark:bg-white/5'
            }`}
          >
            <span
              className={`w-5 h-5 rounded-full flex items-center justify-center text-[11px] font-bold shrink-0 ${
                q.done ? 'bg-[#2f6b3e] text-white' : 'bg-surface-variant dark:bg-white/10 text-outline'
              }`}
            >
              {q.done ? ui.done : ''}
            </span>
            <span
              className={`text-body-md ${
                q.done ? 'text-[#2f6b3e] dark:text-[#9ed0a8]' : 'text-on-surface dark:text-dark-on-surface'
              }`}
            >
              {q.label}
            </span>
          </div>
        ))}
      </div>
    </div>
  )
}
