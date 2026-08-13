import { useEffect, useRef, useState, useCallback } from 'react'
import { Link, useParams, useNavigate } from 'react-router-dom'
import { api } from '../api'
import type { NoteDetail, Heading } from '../types'
import { Spinner, ErrorState } from '../components/States'
import CategoryBadge from '../components/CategoryBadge'
import { useI18n } from '../i18n/context'
import { useBookPagination } from '../hooks/useBookPagination'

// Math is rendered server-side as MathML by the backend (latex2mathml) and
// rendered natively by the browser — no frontend math library needed.

// Book page geometry — a comfortable codex leaf with a book-ish ratio.
// Width adapts to the container via ResizeObserver; height follows ratio.
const MAX_PAGE_WIDTH = 720
const PAGE_RATIO = 1.12 // height = width * ratio (a slightly tall page)
// .book-base/.sheet-face padding — the measure container must mirror it so
// measured heights match the rendered content area exactly.
const BOOK_PAD = { top: 48, right: 56, bottom: 40, left: 56 }
const BOOK_PAD_Y = BOOK_PAD.top + BOOK_PAD.bottom

function slugify(text: string): string {
  return text
    .toLowerCase()
    .replace(/[^\w\- ]/g, '')
    .trim()
    .replace(/\s+/g, '-')
}

function toRoman(n: number): string {
  if (n <= 0 || n >= 4000) return String(n)
  const table: [number, string][] = [
    [1000, 'M'], [900, 'CM'], [500, 'D'], [400, 'CD'],
    [100, 'C'], [90, 'XC'], [50, 'L'], [40, 'XL'],
    [10, 'X'], [9, 'IX'], [5, 'V'], [4, 'IV'], [1, 'I'],
  ]
  let out = ''
  let v = n
  for (const [num, sym] of table) {
    while (v >= num) {
      out += sym
      v -= num
    }
  }
  return out
}

export default function NotePage() {
  const { lang, t } = useI18n()
  const { id } = useParams<{ id: string }>()
  const navigate = useNavigate()
  const [note, setNote] = useState<NoteDetail | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [activeSlug, setActiveSlug] = useState<string>('')
  const [pageIndex, setPageIndex] = useState(0)
  const [flip, setFlip] = useState<{ dir: 'next' | 'prev'; from: number; to: number } | null>(null)
  const [pendingScroll, setPendingScroll] = useState<string | null>(null)
  const [pageWidth, setPageWidth] = useState(660)
  const contentRef = useRef<HTMLDivElement | null>(null)
  const articleRef = useRef<HTMLElement | null>(null)

  const pageHeight = Math.round(pageWidth * PAGE_RATIO)

  // Adapt the book leaf to the available container width. Runs once the note
  // has loaded (the article is only rendered then), and on every resize.
  useEffect(() => {
    const el = articleRef.current
    if (!el) return
    const measure = () => {
      const avail = el.clientWidth - 48 // book sits inside the padded article
      setPageWidth(Math.max(320, Math.min(MAX_PAGE_WIDTH, avail)))
    }
    measure()
    const ro = new ResizeObserver(measure)
    ro.observe(el)
    return () => ro.disconnect()
  }, [note])

  const load = useCallback(async () => {
    if (!id) return
    setLoading(true)
    setError(null)
    try {
      const detail = await api.note(id, lang)
      setNote(detail)
      setPageIndex(0)
      window.scrollTo({ top: 0 })
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setLoading(false)
    }
  }, [id, lang])

  useEffect(() => {
    load()
  }, [load])

  const { measureRef, pages } = useBookPagination(note?.html ?? '', pageHeight - BOOK_PAD_Y)

  // Find which paginated leaf contains a heading slug (id="<slug>").
  const findPageForSlug = useCallback(
    (slug: string): number => {
      for (let i = 0; i < pages.length; i++) {
        if (pages[i].includes(`id="${slug}"`)) return i
      }
      return -1
    },
    [pages],
  )

  const handleTocClick = useCallback(
    (e: React.MouseEvent, h: Heading) => {
      e.preventDefault()
      // Use the frontend slugify (same function that generates in-page ids),
      // NOT the backend slug — the two differ for CJK text.
      const slug = slugify(h.text)
      const target = findPageForSlug(slug)
      if (target < 0) return
      if (target === pageIndex) {
        contentRef.current?.querySelector(`#${CSS.escape(slug)}`)?.scrollIntoView({ behavior: 'smooth', block: 'start' })
        setActiveSlug(slug)
        return
      }
      setPendingScroll(slug)
      setFlip({ dir: target > pageIndex ? 'next' : 'prev', from: pageIndex, to: target })
    },
    [findPageForSlug, pageIndex],
  )

  // Scroll-spy for the TOC (only observes the currently visible page)
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
  }, [note, pageIndex])

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
          {t.common.back}
        </button>
        <span className="text-outline-variant dark:text-white/20">/</span>
        <Link to="/browse" className="hover:text-primary dark:hover:text-inverse-primary transition-colors">
          {t.nav.library}
        </Link>
        <span className="text-outline-variant dark:text-white/20">/</span>
        <span className="text-on-surface dark:text-dark-on-surface truncate max-w-[50vw]">{note.title}</span>
      </div>

      <div className="flex flex-col lg:flex-row gap-10">
        {/* Article */}
        <article
          ref={articleRef}
          className="flex-1 min-w-0 bg-surface-container-lowest dark:bg-dark-surface-elevated rounded-3xl shadow-ambient dark:shadow-dark-ambient border border-outline-variant/50 dark:border-white/10 p-6 md:p-12"
        >
          <header className="mb-8 pb-6 border-b border-outline-variant/50 dark:border-white/10">
            <div className="flex flex-wrap items-center gap-3 mb-4">
              <CategoryBadge category={note.category} size="md" />
              <span className="inline-flex items-center gap-1.5 text-caption text-on-surface-variant dark:text-outline">
                <span className="material-symbols-outlined" style={{ fontSize: 15 }}>schedule</span>
                {note.readingTime} {t.note.minRead}
              </span>
              <span className="inline-flex items-center gap-1.5 text-caption text-on-surface-variant dark:text-outline">
                <span className="material-symbols-outlined" style={{ fontSize: 15 }}>subject</span>
                {note.wordCount.toLocaleString()} {t.home.words}
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

          {/* Hidden measure container — drives pagination layout. React owns
              its content via dangerouslySetInnerHTML; it sits off-screen but
              keeps layout so offsetHeight stays meaningful. It mirrors the
              book page padding so measured heights match the rendered page. */}
          <div
            ref={measureRef}
            className="prose-ailearn"
            style={{
              width: pageWidth,
              padding: `${BOOK_PAD.top}px ${BOOK_PAD.right}px ${BOOK_PAD.bottom}px ${BOOK_PAD.left}px`,
              boxSizing: 'border-box',
              position: 'absolute',
              left: '-9999px',
              top: 0,
              visibility: 'hidden',
              pointerEvents: 'none',
            }}
            aria-hidden="true"
            dangerouslySetInnerHTML={{ __html: note.html }}
          />

          {/* Book body: base page + turning sheet */}
          <div className="book mt-4" style={{ width: pageWidth, height: pageHeight }}>
            {/* Base page — the leaf revealed underneath the turning sheet */}
            <div
              ref={contentRef}
              className="book-base prose-ailearn"
              style={{ width: pageWidth, height: pageHeight }}
            >
              <div dangerouslySetInnerHTML={{ __html: pages[flip ? flip.to : pageIndex] ?? '' }} />
            </div>

            {/* Turning sheet — front holds the outgoing page, back is paper */}
            {flip && (
              <div
                className={`book-sheet is-turning ${flip.dir === 'next' ? 'flip-next' : 'flip-prev'}`}
                onAnimationEnd={() => {
                  setPageIndex(flip.to)
                  setFlip(null)
                  if (pendingScroll) {
                    // After the leaf settles, reveal the requested heading.
                    requestAnimationFrame(() => {
                      contentRef.current?.querySelector(`#${CSS.escape(pendingScroll)}`)?.scrollIntoView({ behavior: 'smooth', block: 'start' })
                      setActiveSlug(pendingScroll)
                      setPendingScroll(null)
                    })
                  }
                }}
              >
                <div className="sheet-face front prose-ailearn">
                  <div dangerouslySetInnerHTML={{ __html: pages[flip.from] ?? '' }} />
                </div>
                <div className="sheet-face back" />
              </div>
            )}
          </div>

          {/* Page navigation */}
          {pages.length > 1 && (
            <div className="flex flex-col items-center gap-3 mt-6">
              <div className="flex items-center gap-3">
                <button
                  onClick={() => {
                    if (flip || pageIndex === 0) return
                    setFlip({ dir: 'prev', from: pageIndex, to: pageIndex - 1 })
                  }}
                  disabled={flip !== null || pageIndex === 0}
                  aria-label="Previous page"
                  className="w-10 h-10 rounded-full flex items-center justify-center text-on-surface-variant dark:text-outline hover:bg-surface-variant dark:hover:bg-white/10 disabled:opacity-30 transition"
                >
                  <span className="material-symbols-outlined" style={{ fontSize: 20 }}>chevron_left</span>
                </button>
                <span className="font-codex text-caption text-on-surface-variant dark:text-outline">
                  {toRoman(pageIndex + 1)} / {toRoman(pages.length)}
                </span>
                <button
                  onClick={() => {
                    if (flip || pageIndex >= pages.length - 1) return
                    setFlip({ dir: 'next', from: pageIndex, to: pageIndex + 1 })
                  }}
                  disabled={flip !== null || pageIndex >= pages.length - 1}
                  aria-label="Next page"
                  className="w-10 h-10 rounded-full flex items-center justify-center text-on-surface-variant dark:text-outline hover:bg-surface-variant dark:hover:bg-white/10 disabled:opacity-30 transition"
                >
                  <span className="material-symbols-outlined" style={{ fontSize: 20 }}>chevron_right</span>
                </button>
              </div>
              <div className="flex items-center gap-2">
                {pages.map((_, i) => (
                  <button
                    key={i}
                    onClick={() => {
                      if (flip || i === pageIndex) return
                      setFlip({ dir: i > pageIndex ? 'next' : 'prev', from: pageIndex, to: i })
                    }}
                    aria-label={`Page ${i + 1}`}
                    className={`page-dot ${i === pageIndex ? 'active' : ''}`}
                  />
                ))}
              </div>
            </div>
          )}
        </article>

        {/* Right rail: TOC + related */}
        <aside className="lg:w-72 shrink-0 flex flex-col gap-6">
          {toc.length > 0 && (
            <div className="lg:sticky lg:top-28 bg-surface-container-low dark:bg-dark-surface rounded-2xl p-5 border border-outline-variant/40 dark:border-white/10">
              <div className="flex items-center gap-2 mb-3 text-on-surface dark:text-dark-on-surface">
                <span className="material-symbols-outlined" style={{ fontSize: 18 }}>list</span>
                <h4 className="font-label-md text-label-md uppercase tracking-wider">{t.note.contents}</h4>
              </div>
              <nav className="flex flex-col">
                {toc.map((h) => {
                  const slug = slugify(h.text)
                  return (
                    <a
                      key={h.slug + h.text}
                      href={`#${slug}`}
                      onClick={(e) => handleTocClick(e, h)}
                      className={`toc-link ${h.level === 3 ? 'ml-3' : ''} ${
                        activeSlug === slug ? 'active' : ''
                      }`}
                    >
                      {h.text}
                    </a>
                  )
                })}
              </nav>
            </div>
          )}

          {note.related.length > 0 && (
            <div className="bg-surface-container-low dark:bg-dark-surface rounded-2xl p-5 border border-outline-variant/40 dark:border-white/10">
              <div className="flex items-center gap-2 mb-4 text-on-surface dark:text-dark-on-surface">
                <span className="material-symbols-outlined" style={{ fontSize: 18 }}>hub</span>
                <h4 className="font-label-md text-label-md uppercase tracking-wider">{t.note.related}</h4>
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
                      <span className="text-caption text-outline">{t.home.categoryNames[r.category.id] ?? r.category.en}</span>
                      {typeof r.score === 'number' && (
                        <span className="text-caption text-outline">
                          · {Math.round(r.score * 100)}% {t.note.similar}
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
            {t.note.backToLibrary}
          </button>
        </aside>
      </div>
    </div>
  )
}
