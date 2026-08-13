import { useEffect, useMemo, useState, useCallback } from 'react'
import { useSearchParams } from 'react-router-dom'
import { api } from '../api'
import type { Note, NotesResponse } from '../types'
import NoteCard from '../components/NoteCard'
import { Spinner, ErrorState, EmptyState } from '../components/States'

function useDebounced<T>(value: T, delay: number): T {
  const [debounced, setDebounced] = useState(value)
  useEffect(() => {
    const t = setTimeout(() => setDebounced(value), delay)
    return () => clearTimeout(t)
  }, [value, delay])
  return debounced
}

export default function BrowsePage() {
  const [params, setParams] = useSearchParams()
  const categoryParam = params.get('category')
  const [search, setSearch] = useState(params.get('q') ?? '')
  const debounced = useDebounced(search.trim(), 250)

  const [data, setData] = useState<NotesResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const load = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const res = await api.notes({
        category: categoryParam,
        search: debounced || null,
      })
      setData(res)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setLoading(false)
    }
  }, [categoryParam, debounced])

  useEffect(() => {
    load()
  }, [load])

  const setCategory = useCallback(
    (id: string | null) => {
      const next = new URLSearchParams(params)
      if (id) next.set('category', id)
      else next.delete('category')
      setParams(next, { replace: true })
    },
    [params, setParams],
  )

  const categories = data?.categories ?? []

  const sortedNotes = useMemo(() => {
    if (!data) return [] as Note[]
    // When searching, backend already ranks by relevance. Otherwise sort by
    // path to give a deterministic, curriculum-like order.
    if (debounced) return data.notes
    return [...data.notes].sort((a, b) => a.path.localeCompare(b.path, undefined, { numeric: true }))
  }, [data, debounced])

  return (
    <div className="flex flex-col gap-8 pt-2">
      {/* Header */}
      <header className="flex flex-col gap-3">
        <h1 className="font-headline text-headline-lg-mobile md:text-headline-xl text-on-surface dark:text-inverse-on-surface">
          Library
        </h1>
        <p className="text-body-lg text-on-surface-variant dark:text-outline max-w-2xl">
          {data
            ? `${data.total} notes across ${categories.length} subjects — semantic search powered by scikit-learn.`
            : 'Loading your notes…'}
        </p>
      </header>

      {/* Search + filters */}
      <div className="flex flex-col gap-4">
        <div className="relative">
          <span className="material-symbols-outlined absolute left-4 top-1/2 -translate-y-1/2 text-outline" style={{ fontSize: 22 }}>
            search
          </span>
          <input
            type="text"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search by concept, e.g. gradient descent, RoPE, MoE…"
            className="w-full bg-surface-container dark:bg-dark-surface-elevated text-on-surface dark:text-dark-on-surface placeholder:text-outline rounded-2xl py-3.5 pl-12 pr-12 border border-outline-variant/50 dark:border-white/10 focus:border-primary dark:focus:border-inverse-primary focus:ring-2 focus:ring-primary/20 outline-none text-body-md transition"
          />
          {search && (
            <button
              onClick={() => setSearch('')}
              aria-label="Clear search"
              className="absolute right-3 top-1/2 -translate-y-1/2 w-8 h-8 rounded-full flex items-center justify-center text-outline hover:bg-surface-variant dark:hover:bg-white/10 transition"
            >
              <span className="material-symbols-outlined" style={{ fontSize: 18 }}>close</span>
            </button>
          )}
        </div>

        <div className="flex flex-wrap gap-2">
          <FilterChip
            active={!categoryParam}
            onClick={() => setCategory(null)}
            icon="all_inclusive"
            label="All"
            count={data ? data.total : undefined}
          />
          {categories.map((c) => (
            <FilterChip
              key={c.id}
              active={categoryParam === c.id}
              onClick={() => setCategory(c.id)}
              icon={c.icon}
              label={c.en}
              count={c.count}
            />
          ))}
        </div>
      </div>

      {/* Content */}
      {loading ? (
        <Spinner label={debounced ? `Searching for “${debounced}”…` : 'Loading notes…'} />
      ) : error ? (
        <ErrorState message={error} onRetry={load} />
      ) : sortedNotes.length === 0 ? (
        <EmptyState
          title="No notes match your filters"
          subtitle={debounced ? `Nothing matched “${debounced}”. Try another concept, or clear the search.` : 'Try selecting a different subject.'}
        />
      ) : (
        <>
          <div className="flex items-center justify-between text-caption text-on-surface-variant dark:text-outline">
            <span>
              Showing <strong className="text-on-surface dark:text-dark-on-surface">{sortedNotes.length}</strong> of {data?.total}
              {debounced && <> · ranked by relevance</>}
            </span>
            {(debounced || categoryParam) && (
              <button
                onClick={() => {
                  setSearch('')
                  setCategory(null)
                }}
                className="text-primary dark:text-inverse-primary font-semibold hover:underline inline-flex items-center gap-1"
              >
                <span className="material-symbols-outlined" style={{ fontSize: 14 }}>filter_alt_off</span>
                Clear filters
              </button>
            )}
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
            {sortedNotes.map((note) => (
              <NoteCard key={note.id} note={note} />
            ))}
          </div>
        </>
      )}
    </div>
  )
}

function FilterChip({
  active,
  onClick,
  icon,
  label,
  count,
}: {
  active: boolean
  onClick: () => void
  icon: string
  label: string
  count?: number
}) {
  return (
    <button
      onClick={onClick}
      className={`inline-flex items-center gap-2 px-4 py-2 rounded-full text-label-md font-semibold border transition-all duration-200 ${
        active
          ? 'bg-primary dark:bg-inverse-primary text-on-primary dark:text-inverse-surface border-primary dark:border-inverse-primary shadow-sm'
          : 'bg-surface-container-lowest dark:bg-dark-surface-elevated text-on-surface-variant dark:text-outline border-outline-variant/60 dark:border-white/10 hover:border-primary/40 dark:hover:border-inverse-primary/40 hover:text-on-surface dark:hover:text-dark-on-surface'
      }`}
    >
      <span className="material-symbols-outlined" style={{ fontSize: 17 }}>{icon}</span>
      {label}
      {typeof count === 'number' && (
        <span className={`px-1.5 py-0.5 rounded-full text-caption ${active ? 'bg-white/20' : 'bg-surface-variant dark:bg-white/10'}`}>
          {count}
        </span>
      )}
    </button>
  )
}
