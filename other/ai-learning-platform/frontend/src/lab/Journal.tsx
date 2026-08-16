// Experiment Journal — a learner's own discovery trail (plan §12).
//
// Every completed Challenge can be saved as a Discovery entry: the question,
// the prediction, whether it was right, the evidence and the params that
// produced it. Entries persist in localStorage under "ailearn-discoveries"
// and are shown as MY DISCOVERIES cards. The Insight belongs to the learner
// (No-LLM principle) — we only record what happened in the experiment.
import { useCallback, useEffect, useMemo, useState } from 'react'
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
  /** Learner's own words — L3 Explain. Never machine-written (No-LLM). */
  insight?: string
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
    insight: '我的发现',
    editInsight: '写下我的发现',
    save: '保存',
    cancel: '取消',
    insightPlaceholder: '用你自己的话写下：这次实验揭示了什么？',
    viewAll: '全部',
    viewByExp: '按实验',
    viewTimeline: '时间线',
    export: '导出 Markdown',
    exported: '已导出',
    groupCount: '{n} 条发现',
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
    insight: 'My insight',
    editInsight: 'Write my insight',
    save: 'Save',
    cancel: 'Cancel',
    insightPlaceholder: 'In your own words: what did this experiment reveal?',
    viewAll: 'All',
    viewByExp: 'By experiment',
    viewTimeline: 'Timeline',
    export: 'Export Markdown',
    exported: 'Exported',
    groupCount: '{n} discoveries',
  },
}

function load(): DiscoveryEntry[] {
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY)
    if (!raw) return []
    const parsed = JSON.parse(raw)
    if (!Array.isArray(parsed)) return []
    // Dedupe: entries are stored newest-first, so keep the FIRST occurrence
    // of each identical (experiment, question, prediction, evidence) key.
    // This cleans up duplicates written by older QuestList auto-recording.
    const seen = new Map<string, DiscoveryEntry>()
    for (const e of parsed) {
      const key = [e.experimentId, e.question, e.prediction, e.evidence].join('|')
      if (!seen.has(key)) seen.set(key, e)
    }
    return [...seen.values()].slice(0, MAX_ENTRIES)
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

  const updateInsight = useCallback((id: string, insight: string) => {
    setEntries((prev) => prev.map((e) => (e.id === id ? { ...e, insight } : e)))
  }, [])

  const clear = useCallback(() => setEntries([]), [])
  return { entries, addEntry, updateInsight, clear }
}

const L3_UI: Record<'zh' | 'en', { title: string; placeholder: string; save: string; saved: string }> = {
  zh: {
    title: '用自己的话解释',
    placeholder: '这次实验揭示了什么？写下你的发现（不会自动判定，这是你的）。',
    save: '存入我的发现',
    saved: '✓ 已存入「我的发现」，可在页面底部补充或修改。',
  },
  en: {
    title: 'Explain it in your own words',
    placeholder: 'What did this experiment reveal? Write your insight (no auto-grading — it is yours).',
    save: 'Save to My Discoveries',
    saved: '✓ Saved to My Discoveries — refine it at the bottom of the page.',
  },
}

/**
 * ExplainBox — L3 Explain, the learner-owned layer.
 *
 * A textarea + save button that records the learner's own words as a
 * Discovery entry with an `insight`. Never machine-graded (No-LLM): the
 * insight belongs to the user. Reused by every experiment so the L3 layer
 * is one shared component, not copy-paste per lab. Copy lives here (shared
 * UI), so adding L3 to a new experiment is one <ExplainBox /> line.
 */
export function ExplainBox({
  onRecord,
  evidence,
  params,
}: {
  onRecord?: (entry: {
    question: string
    prediction: string
    correct: boolean
    evidence: string
    params: LabParams
    insight?: string
  }) => void
  evidence: string
  params: LabParams
}) {
  const { lang } = useI18n()
  const ui = L3_UI[lang]
  const [draft, setDraft] = useState('')
  const [saved, setSaved] = useState(false)

  return (
    <div className="bg-surface-container-lowest dark:bg-dark-surface rounded-3xl p-5 border border-outline-variant/40 dark:border-white/10">
      <div className="flex items-center gap-2 mb-1">
        <span className="text-lg">✍️</span>
        <h3 className="font-label-md text-label-md uppercase tracking-wider text-on-surface dark:text-dark-on-surface">{ui.title}</h3>
      </div>
      <p className="text-body-md text-on-surface dark:text-inverse-on-surface mb-3">{ui.placeholder}</p>
      <textarea
        value={draft}
        onChange={(ev) => { setDraft(ev.target.value); setSaved(false) }}
        rows={3}
        placeholder={ui.placeholder}
        className="w-full rounded-2xl px-4 py-3 bg-surface-container dark:bg-white/5 border border-outline-variant/50 dark:border-white/15 text-body-md text-on-surface dark:text-dark-on-surface placeholder:text-outline focus:outline-none focus:border-primary/50 resize-none"
      />
      <div className="mt-3 flex flex-wrap items-center gap-3">
        <button
          disabled={draft.trim() === ''}
          onClick={() => {
            if (onRecord) {
              onRecord({
                question: ui.title,
                prediction: '',
                correct: true,
                evidence,
                params: { ...params },
                insight: draft.trim(),
              })
              setSaved(true)
            }
          }}
          className="inline-flex items-center gap-2 px-5 py-2 rounded-xl bg-on-surface text-on-primary dark:bg-inverse-surface dark:text-inverse-surface font-label-md text-label-md hover:opacity-90 transition disabled:opacity-40"
        >
          <span className="material-symbols-outlined" style={{ fontSize: 17 }}>save</span>
          {ui.save}
        </button>
        {saved && <span className="text-caption text-[#2f6b3e] dark:text-[#9ed0a8]">{ui.saved}</span>}
      </div>
    </div>
  )
}

/**
 * JournalPanel — MY DISCOVERIES cards + the experiment's next-question exit.
 * Rendered once per lab page, below the active experiment.
 */
export function JournalPanel({
  entries,
  onClear,
  onUpdateInsight,
  nextQuestion,
  onExplore,
}: {
  entries: DiscoveryEntry[]
  onClear: () => void
  onUpdateInsight?: (id: string, insight: string) => void
  nextQuestion?: string
  onExplore?: () => void
}) {
  const { lang } = useI18n()
  const ui = UI[lang]
  const [editingId, setEditingId] = useState<string | null>(null)
  const [draft, setDraft] = useState('')
  const [view, setView] = useState<'all' | 'byexp' | 'timeline'>('all')
  const [exported, setExported] = useState(false)

  /** Group entries by experiment title (stable insertion order). */
  const groups = useMemo(() => {
    const m = new Map<string, DiscoveryEntry[]>()
    for (const e of entries) {
      if (!m.has(e.experimentTitle)) m.set(e.experimentTitle, [])
      m.get(e.experimentTitle)!.push(e)
    }
    return [...m.entries()]
  }, [entries])

  /** Timeline: oldest → newest (the exploration chain). */
  const timeline = useMemo(
    () => [...entries].sort((a, b) => a.createdAt - b.createdAt),
    [entries],
  )

  /** Export all discoveries as a Markdown file. */
  const exportMd = useCallback(() => {
    const body = entries.map((e) => [
      `## ${e.experimentTitle} — ${e.correct ? '✓' : '✗'}`,
      `- **Question:** ${e.question}`,
      `- **Prediction:** ${e.prediction}`,
      `- **Evidence:** ${e.evidence}`,
      e.insight ? `- **Insight:** ${e.insight}` : '',
      `- _${new Date(e.createdAt).toISOString()}_`,
      '',
    ].join('\n')).join('\n')
    const md = `# My Discoveries\n\n${body || '_empty_'}\n`
    const blob = new Blob([md], { type: 'text/markdown;charset=utf-8' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'my-discoveries.md'
    a.click()
    URL.revokeObjectURL(url)
    setExported(true)
    window.setTimeout(() => setExported(false), 2000)
  }, [entries])

  /** One discovery card (shared by all views). */
  const card = (e: DiscoveryEntry) => (
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

      {/* L3 Explain: the learner's own insight, editable inline */}
      <div className="mt-2">
        {editingId === e.id ? (
          <div className="rounded-xl border border-outline-variant/50 dark:border-white/15 p-2">
            <textarea
              value={draft}
              onChange={(ev) => setDraft(ev.target.value)}
              placeholder={ui.insightPlaceholder}
              rows={2}
              className="w-full bg-transparent text-caption text-on-surface dark:text-dark-on-surface placeholder:text-outline focus:outline-none resize-none"
            />
            <div className="flex justify-end gap-2 mt-1">
              <button
                onClick={() => setEditingId(null)}
                className="px-3 py-1 rounded-lg text-caption text-on-surface-variant dark:text-outline hover:bg-surface-variant dark:hover:bg-white/5 transition"
              >
                {ui.cancel}
              </button>
              <button
                onClick={() => {
                  if (onUpdateInsight) onUpdateInsight(e.id, draft.trim())
                  setEditingId(null)
                }}
                disabled={draft.trim() === ''}
                className="px-3 py-1 rounded-lg text-caption font-semibold bg-primary text-on-primary dark:bg-inverse-primary dark:text-inverse-surface disabled:opacity-40 transition"
              >
                {ui.save}
              </button>
            </div>
          </div>
        ) : e.insight ? (
          <div className="rounded-xl border-l-2 border-primary/50 dark:border-inverse-primary/50 bg-surface-container dark:bg-white/5 px-3 py-2 text-caption text-on-surface dark:text-dark-on-surface">
            <span className="text-outline mr-1.5">💡</span>{e.insight}
          </div>
        ) : null}

        {onUpdateInsight && editingId !== e.id && (
          <button
            onClick={() => { setEditingId(e.id); setDraft(e.insight ?? '') }}
            className="mt-1 inline-flex items-center gap-1 text-caption text-outline hover:text-primary dark:hover:text-inverse-primary transition"
          >
            <span className="material-symbols-outlined" style={{ fontSize: 14 }}>edit</span>
            {ui.editInsight}
          </button>
        )}
      </div>
    </div>
  )

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

        {entries.length > 0 && (
          <div className="flex flex-wrap items-center gap-2 mb-4">
            {/* view switcher */}
            <div className="inline-flex gap-1 p-1 rounded-xl bg-surface-container dark:bg-white/5 border border-outline-variant/40 dark:border-white/10">
              {([['all', ui.viewAll], ['byexp', ui.viewByExp], ['timeline', ui.viewTimeline]] as const).map(([v, label]) => (
                <button
                  key={v}
                  onClick={() => setView(v)}
                  className={`px-3 py-1.5 rounded-lg text-label-md font-semibold transition-all ${
                    view === v
                      ? 'bg-primary text-on-primary dark:bg-inverse-primary dark:text-inverse-surface'
                      : 'text-on-surface-variant dark:text-outline hover:bg-surface-variant dark:hover:bg-white/10'
                  }`}
                >
                  {label}
                </button>
              ))}
            </div>
            {/* export markdown */}
            <button
              onClick={exportMd}
              className="inline-flex items-center gap-1.5 px-3 py-2 rounded-xl text-label-md font-semibold border border-outline-variant/50 dark:border-white/15 text-on-surface-variant dark:text-outline hover:border-primary/50 hover:text-primary dark:hover:text-inverse-primary transition"
            >
              <span className="material-symbols-outlined" style={{ fontSize: 15 }}>download</span>
              {exported ? ui.exported : ui.export}
            </button>
            <span className="ml-auto text-caption text-outline font-mono">{fmt(ui.groupCount, { n: entries.length })}</span>
          </div>
        )}

        {entries.length === 0 ? (
          <div className="rounded-2xl border border-dashed border-outline-variant/50 dark:border-white/15 p-6 text-center">
            <div className="text-2xl mb-2">🔬</div>
            <div className="text-body-md text-on-surface dark:text-dark-on-surface">{ui.empty}</div>
            <div className="text-caption text-outline mt-1">{ui.emptyHint}</div>
          </div>
        ) : view === 'byexp' ? (
          <div className="flex flex-col gap-4">
            {groups.map(([title, list]) => (
              <div key={title}>
                <h4 className="text-caption uppercase tracking-wider font-semibold text-outline mb-2">
                  {title} <span className="font-mono text-outline/70">({list.length})</span>
                </h4>
                <div className="flex flex-col gap-2">{list.map(card)}</div>
              </div>
            ))}
          </div>
        ) : view === 'timeline' ? (
          <div className="flex flex-col gap-2">
            {timeline.map((e, i) => (
              <div key={e.id} className="flex items-stretch gap-3">
                {/* chain index */}
                <div className="flex flex-col items-center shrink-0">
                  <span className="w-7 h-7 rounded-full bg-primary/15 dark:bg-inverse-primary/15 text-primary dark:text-inverse-primary flex items-center justify-center text-caption font-bold">
                    {i + 1}
                  </span>
                  {i < timeline.length - 1 && <span className="w-px flex-1 bg-outline-variant/40 dark:bg-white/10" />}
                </div>
                <div className="flex-1 min-w-0 pb-3">{card(e)}</div>
              </div>
            ))}
          </div>
        ) : (
          <div className="flex flex-col gap-2">{entries.map(card)}</div>
        )}
      </div>
    </div>
  )
}
