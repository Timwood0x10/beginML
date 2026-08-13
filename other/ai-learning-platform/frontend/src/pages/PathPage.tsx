import { useEffect, useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import { api } from '../api'
import type { NotesResponse, Note } from '../types'
import { Spinner, ErrorState } from '../components/States'

const CATEGORY_ORDER = ['math', 'attention', 'hybrid', 'paper', 'general']
const CATEGORY_META: Record<
  string,
  { phase: string; title: string; blurb: string; icon: string }
> = {
  math: {
    phase: 'Phase I',
    title: 'Mathematical foundations',
    blurb: 'Calculus, linear algebra, probability and optimization — the bedrock beneath every model.',
    icon: 'functions',
  },
  attention: {
    phase: 'Phase II',
    title: 'Self-attention & Transformers',
    blurb: 'From scaled dot-product attention to multi-head, encoders, decoders and RoPE.',
    icon: 'psychology',
  },
  hybrid: {
    phase: 'Phase III',
    title: 'Hybrid & frontier architectures',
    blurb: 'MoE, Mamba/SSMs, MLA, MTP, quantization and post-Transformer reasoning.',
    icon: 'bolt',
  },
  paper: {
    phase: 'Phase IV',
    title: 'Research close-readings',
    blurb: 'Annotated papers and empirical studies, read alongside the foundations.',
    icon: 'description',
  },
  general: {
    phase: 'Reference',
    title: 'General notes',
    blurb: 'Foundational overviews and reference material.',
    icon: 'article',
  },
}

function pathKey(note: Note): string {
  // numeric prefix in filename like "0.1." / "11." → use for ordering
  const m = note.filename.match(/^(\d+(?:\.\d+)*)/)
  return m ? m[1] : note.filename
}

export default function PathPage() {
  const [data, setData] = useState<NotesResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let alive = true
    api
      .notes()
      .then((res) => alive && setData(res))
      .catch((e) => alive && setError(e instanceof Error ? e.message : String(e)))
      .finally(() => alive && setLoading(false))
    return () => {
      alive = false
    }
  }, [])

  const phases = useMemo(() => {
    if (!data) return []
    const groups = new Map<string, Note[]>()
    for (const note of data.notes) {
      const id = note.category.id
      if (!groups.has(id)) groups.set(id, [])
      groups.get(id)!.push(note)
    }
    return CATEGORY_ORDER.filter((id) => groups.has(id)).map((id) => {
      const notes = [...groups.get(id)!].sort((a, b) =>
        pathKey(a).localeCompare(pathKey(b), undefined, { numeric: true }),
      )
      const minutes = notes.reduce((s, n) => s + n.readingTime, 0)
      return { id, meta: CATEGORY_META[id] ?? CATEGORY_META.general, notes, minutes }
    })
  }, [data])

  if (loading) return <Spinner label="Laying out your learning path…" />
  if (error || !data) return <ErrorState message={error ?? 'Failed to load path.'} />

  const totalMinutes = phases.reduce((s, p) => s + p.minutes, 0)

  return (
    <div className="flex flex-col gap-10 pt-2">
      <header className="flex flex-col gap-3">
        <span className="inline-flex items-center gap-2 self-start text-caption font-semibold uppercase tracking-[0.15em] text-primary dark:text-inverse-primary">
          <span className="material-symbols-outlined" style={{ fontSize: 16 }}>route</span>
          Guided curriculum
        </span>
        <h1 className="font-headline text-headline-lg-mobile md:text-headline-xl text-on-surface dark:text-inverse-on-surface max-w-3xl">
          A deliberate path through the notes
        </h1>
        <p className="font-body-lg text-on-surface-variant dark:text-outline max-w-2xl leading-relaxed">
          {data.total} notes arranged in four phases — foundations first, then architectures, then
          the frontier. Roughly {Math.round(totalMinutes / 60)} hours of focused reading.
        </p>
      </header>

      {/* Desktop: two-column phase rail + timeline */}
      <div className="relative max-w-4xl mx-auto w-full">
        {/* vertical spine */}
        <div className="absolute left-5 md:left-8 top-2 bottom-2 w-px bg-outline-variant dark:bg-white/10" />

        {phases.map((phase, phaseIdx) => (
          <section key={phase.id} className="relative mb-12 last:mb-0">
            {/* Phase node */}
            <div className="relative z-10 flex gap-6 md:gap-8 items-start mb-6 pl-0">
              <div
                className={`w-10 h-10 md:w-16 md:h-16 rounded-full flex items-center justify-center shrink-0 shadow-ambient dark:shadow-dark-ambient ${
                  phaseIdx === 0
                    ? 'bg-primary dark:bg-inverse-primary text-on-primary dark:text-inverse-surface ring-4 ring-primary-fixed/60 dark:ring-inverse-primary/20'
                    : 'bg-surface-container dark:bg-dark-surface-elevated text-primary dark:text-inverse-primary border border-outline-variant/60 dark:border-white/10'
                }`}
              >
                <span
                  className="material-symbols-outlined"
                  style={{ fontSize: phaseIdx === 0 ? 30 : 26 }}
                >
                  {phase.meta.icon}
                </span>
              </div>

              <div className="flex-1 pt-1 md:pt-3">
                <span className="text-caption uppercase tracking-wider text-outline font-semibold">
                  {phase.meta.phase}
                </span>
                <h2 className="font-headline text-headline-lg-mobile text-on-surface dark:text-inverse-on-surface mt-0.5">
                  {phase.meta.title}
                </h2>
                <p className="text-body-md text-on-surface-variant dark:text-outline mt-1 max-w-2xl leading-relaxed">
                  {phase.meta.blurb}
                </p>
                <div className="inline-flex items-center gap-3 mt-3 text-caption text-on-surface-variant dark:text-outline">
                  <span className="inline-flex items-center gap-1">
                    <span className="material-symbols-outlined" style={{ fontSize: 14 }}>description</span>
                    {phase.notes.length} notes
                  </span>
                  <span className="w-1 h-1 rounded-full bg-outline-variant" />
                  <span className="inline-flex items-center gap-1">
                    <span className="material-symbols-outlined" style={{ fontSize: 14 }}>schedule</span>
                    {phase.minutes} min
                  </span>
                </div>
              </div>
            </div>

            {/* Lessons in this phase */}
            <div className="pl-14 md:pl-24 flex flex-col gap-3">
              {phase.notes.map((note, i) => {
                const isCurrent = phaseIdx === 0 && i === 0
                return (
                  <Link
                    key={note.id}
                    to={`/note/${note.id}`}
                    className={`group relative flex items-start gap-4 rounded-2xl p-4 md:p-5 border transition-all duration-200 ${
                      isCurrent
                        ? 'bg-surface-container-lowest dark:bg-dark-surface-elevated border-primary/30 dark:border-inverse-primary/30 shadow-ambient dark:shadow-dark-ambient'
                        : 'bg-surface-container-lowest/60 dark:bg-dark-surface-elevated/60 border-outline-variant/40 dark:border-white/10 hover:bg-surface-container-lowest dark:hover:bg-dark-surface-elevated hover:border-outline-variant dark:hover:border-white/15 hover:-translate-y-0.5'
                    }`}
                  >
                    <div
                      className={`w-8 h-8 rounded-full flex items-center justify-center shrink-0 mt-0.5 ${
                        isCurrent
                          ? 'bg-primary text-on-primary dark:bg-inverse-primary dark:text-inverse-surface'
                          : 'bg-surface-variant dark:bg-white/5 text-on-surface-variant dark:text-outline'
                      }`}
                    >
                      <span className="text-caption font-bold">{i + 1}</span>
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className="flex flex-wrap items-center gap-2 mb-1">
                        <h3 className="font-headline text-lg text-on-surface dark:text-inverse-on-surface leading-snug group-hover:text-primary dark:group-hover:text-inverse-primary transition-colors">
                          {note.title}
                        </h3>
                        {isCurrent && (
                          <span className="px-2 py-0.5 rounded-full text-caption font-semibold bg-primary/10 text-primary dark:bg-inverse-primary/15 dark:text-inverse-primary">
                            Start here
                          </span>
                        )}
                      </div>
                      {note.description && (
                        <p className="text-body-md text-on-surface-variant dark:text-outline line-clamp-2 leading-relaxed">
                          {note.description}
                        </p>
                      )}
                      <div className="flex items-center gap-3 mt-2 text-caption text-outline">
                        <span className="inline-flex items-center gap-1">
                          <span className="material-symbols-outlined" style={{ fontSize: 13 }}>schedule</span>
                          {note.readingTime} min
                        </span>
                        <span className="inline-flex items-center gap-1">
                          <span className="material-symbols-outlined" style={{ fontSize: 13 }}>subject</span>
                          {note.wordCount.toLocaleString()} words
                        </span>
                      </div>
                    </div>
                    <span className="material-symbols-outlined self-center text-outline group-hover:text-primary dark:group-hover:text-inverse-primary transition-all group-hover:translate-x-0.5" style={{ fontSize: 20 }}>
                      arrow_forward
                    </span>
                  </Link>
                )
              })}
            </div>
          </section>
        ))}
      </div>
    </div>
  )
}
