import { useEffect, useRef, useState, useCallback } from 'react'
import { Link, useParams, useNavigate } from 'react-router-dom'
import katex from 'katex'
import 'katex/dist/katex.min.css'
import { api } from '../api'
import type { NoteDetail, Heading } from '../types'
import { Spinner, ErrorState } from '../components/States'
import CategoryBadge from '../components/CategoryBadge'

/**
 * Render pymdownx-arithmatex output. Generic mode wraps math in
 * <span class="arithmatex">\(...\)</span> (inline) and
 * <div class="arithmatex">\[...\]</div> (block). We extract the TeX and
 * typeset with KaTeX directly.
 */
function typesetMath(root: HTMLElement) {
  const nodes = root.querySelectorAll<HTMLElement>('.arithmatex')
  nodes.forEach((node) => {
    const raw = node.textContent ?? ''
    let tex = raw.trim()
    // strip \( \) / \[ \] / $ delimiters
    tex = tex.replace(/\\\(([\s\S]*)\\\)/, '$1')
    tex = tex.replace(/\\\[([\s\S]*)\\\]/, '$1')
    tex = tex.replace(/^\$+|\$+$/g, '').trim()
    const display = node.tagName.toLowerCase() === 'div'
    try {
      katex.render(tex, node, {
        displayMode: display,
        throwOnError: false,
        output: 'html',
      })
    } catch {
      node.textContent = raw
    }
  })
}

function slugify(text: string): string {
  return text
    .toLowerCase()
    .replace(/[^\w\- ]/g, '')
    .trim()
    .replace(/\s+/g, '-')
}

export default function NotePage() {
  const { id } = useParams<{ id: string }>()
  const navigate = useNavigate()
  const [note, setNote] = useState<NoteDetail | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [activeSlug, setActiveSlug] = useState<string>('')
  const contentRef = useRef<HTMLDivElement | null>(null)

  const load = useCallback(async () => {
    if (!id) return
    setLoading(true)
    setError(null)
    try {
      const detail = await api.note(id)
      setNote(detail)
      window.scrollTo({ top: 0 })
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setLoading(false)
    }
  }, [id])

  useEffect(() => {
    load()
  }, [load])

  // Typeset math + add heading ids after content renders
  useEffect(() => {
    if (!note || !contentRef.current) return
    const root = contentRef.current
    root.querySelectorAll('h1, h2, h3').forEach((h) => {
      const el = h as HTMLElement
      if (!el.id) el.id = slugify(el.textContent ?? '')
    })
    // Open external links in new tab
    root.querySelectorAll('a[href^="http"]').forEach((a) => {
      a.setAttribute('target', '_blank')
      a.setAttribute('rel', 'noopener noreferrer')
    })
    typesetMath(root)
  }, [note])

  // Scroll-spy for the TOC
  useEffect(() => {
    if (!note || !contentRef.current) return
    const headings = contentRef.current.querySelectorAll<HTMLElement>('h2, h3')
    if (headings.length === 0) return
    const observer = new IntersectionObserver(
      (entries) => {
        const visible = entries
          .filter((e) => e.isIntersecting)
          .sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top)
        if (visible[0]) setActiveSlug(visible[0].target.id)
      },
      { rootMargin: '-96px 0px -70% 0px', threshold: 0 },
    )
    headings.forEach((h) => observer.observe(h))
    return () => observer.disconnect()
  }, [note])

  // Keyboard: ← back
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') navigate(-1)
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [navigate])

  if (loading) return <Spinner label="Loading note…" />
  if (error || !note) return <ErrorState message={error ?? 'Note not found.'} onRetry={load} />

  const toc: Heading[] = note.headings.filter((h) => h.level >= 2 && h.level <= 3)

  return (
    <div className="flex flex-col gap-8 pt-2">
      {/* Breadcrumb / back */}
      <div className="flex items-center gap-2 text-caption text-on-surface-variant dark:text-outline">
        <button
          onClick={() => navigate(-1)}
          className="inline-flex items-center gap-1 hover:text-primary dark:hover:text-inverse-primary transition-colors font-semibold"
        >
          <span className="material-symbols-outlined" style={{ fontSize: 16 }}>arrow_back</span>
          Back
        </button>
        <span className="text-outline-variant dark:text-white/20">/</span>
        <Link to="/browse" className="hover:text-primary dark:hover:text-inverse-primary transition-colors">
          Library
        </Link>
        <span className="text-outline-variant dark:text-white/20">/</span>
        <span className="text-on-surface dark:text-dark-on-surface truncate max-w-[50vw]">{note.title}</span>
      </div>

      <div className="flex flex-col lg:flex-row gap-10">
        {/* Article */}
        <article className="flex-1 min-w-0 bg-surface-container-lowest dark:bg-dark-surface-elevated rounded-3xl shadow-ambient dark:shadow-dark-ambient border border-outline-variant/50 dark:border-white/5 p-6 md:p-12">
          <header className="mb-8 pb-6 border-b border-outline-variant/50 dark:border-white/5">
            <div className="flex flex-wrap items-center gap-3 mb-4">
              <CategoryBadge category={note.category} size="md" />
              <span className="inline-flex items-center gap-1.5 text-caption text-on-surface-variant dark:text-outline">
                <span className="material-symbols-outlined" style={{ fontSize: 15 }}>schedule</span>
                {note.readingTime} min read
              </span>
              <span className="inline-flex items-center gap-1.5 text-caption text-on-surface-variant dark:text-outline">
                <span className="material-symbols-outlined" style={{ fontSize: 15 }}>subject</span>
                {note.wordCount.toLocaleString()} words
              </span>
            </div>
            <h1 className="font-headline text-3xl md:text-4xl font-bold text-on-surface dark:text-inverse-on-surface leading-tight">
              {note.title}
            </h1>
            {note.description && (
              <p className="mt-4 text-body-lg text-on-surface-variant dark:text-outline leading-relaxed">
                {note.description}
              </p>
            )}
          </header>

          <div
            ref={contentRef}
            className="prose-ailearn"
            dangerouslySetInnerHTML={{ __html: note.html }}
          />
        </article>

        {/* Right rail: TOC + related */}
        <aside className="lg:w-72 shrink-0 flex flex-col gap-6">
          {toc.length > 0 && (
            <div className="lg:sticky lg:top-28 bg-surface-container-low dark:bg-dark-surface rounded-2xl p-5 border border-outline-variant/40 dark:border-white/5">
              <div className="flex items-center gap-2 mb-3 text-on-surface dark:text-dark-on-surface">
                <span className="material-symbols-outlined" style={{ fontSize: 18 }}>list</span>
                <h4 className="font-label-md text-label-md uppercase tracking-wider">Contents</h4>
              </div>
              <nav className="flex flex-col">
                {toc.map((h) => (
                  <a
                    key={h.slug + h.text}
                    href={`#${h.slug || slugify(h.text)}`}
                    className={`toc-link ${h.level === 3 ? 'ml-3' : ''} ${
                      activeSlug === (h.slug || slugify(h.text)) ? 'active' : ''
                    }`}
                  >
                    {h.text}
                  </a>
                ))}
              </nav>
            </div>
          )}

          {note.related.length > 0 && (
            <div className="bg-surface-container-low dark:bg-dark-surface rounded-2xl p-5 border border-outline-variant/40 dark:border-white/5">
              <div className="flex items-center gap-2 mb-4 text-on-surface dark:text-dark-on-surface">
                <span className="material-symbols-outlined" style={{ fontSize: 18 }}>hub</span>
                <h4 className="font-label-md text-label-md uppercase tracking-wider">Related notes</h4>
              </div>
              <div className="flex flex-col gap-1">
                {note.related.map((r) => (
                  <Link
                    key={r.id}
                    to={`/note/${r.id}`}
                    className="group block rounded-xl px-3 py-2.5 -mx-1 hover:bg-surface-variant dark:hover:bg-white/5 transition-colors"
                  >
                    <div className="text-body-md text-on-surface dark:text-dark-on-surface font-semibold leading-snug group-hover:text-primary dark:group-hover:text-inverse-primary line-clamp-2">
                      {r.title}
                    </div>
                    <div className="flex items-center gap-2 mt-1">
                      <span className="text-caption text-outline">{r.category.en}</span>
                      {typeof r.score === 'number' && (
                        <span className="text-caption text-outline">
                          · {Math.round(r.score * 100)}% similar
                        </span>
                      )}
                    </div>
                  </Link>
                ))}
              </div>
            </div>
          )}

          <button
            onClick={() => navigate(-1)}
            className="lg:hidden w-full inline-flex items-center justify-center gap-2 px-5 py-3 rounded-2xl bg-surface-container dark:bg-dark-surface-elevated border border-outline-variant/60 dark:border-white/10 font-label-md text-label-md text-on-surface dark:text-dark-on-surface"
          >
            <span className="material-symbols-outlined" style={{ fontSize: 18 }}>arrow_back</span>
            Back to library
          </button>
        </aside>
      </div>
    </div>
  )
}
