import { useEffect, useRef, useState, useCallback } from 'react'
import { Link, useParams, useNavigate } from 'react-router-dom'
import { PageFlip } from 'page-flip'
import { api } from '../api'
import type { NoteDetail, Heading } from '../types'
import { Spinner, ErrorState } from '../components/States'
import CategoryBadge from '../components/CategoryBadge'
import { useI18n } from '../i18n/context'
import { useBookPagination } from '../hooks/useBookPagination'

// Math is rendered server-side as MathML by the backend (latex2mathml) and
// rendered natively by the browser — no frontend math library needed.

// Book page geometry — a comfortable codex leaf.
// Width adapts to the container via ResizeObserver; height follows ratio.
// Use a wider page (up to 860px) with a more natural aspect ratio to give
// code blocks and tables room to breathe without horizontal scrolling.
const MAX_PAGE_WIDTH = 860
const PAGE_RATIO = 1.0 // height = width * ratio (closer to golden ratio for screen reading)
// .page padding — the measure container must mirror it so measured heights
// match the rendered content area exactly.
const BOOK_PAD = { top: 40, right: 48, bottom: 36, left: 48 }
const BOOK_PAD_Y = BOOK_PAD.top + BOOK_PAD.bottom

function slugify(text: string): string {
  return text
    .toLowerCase()
    .replace(/[^\w\- ]/g, '')
    .trim()
    .replace(/\s+/g, '-')
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
  const [pageWidth, setPageWidth] = useState(720)
  const articleRef = useRef<HTMLElement | null>(null)
  // The flipbook lives inside this div. StPageFlip owns the subtree; React
  // only provides the empty host so the instance can be rebuilt on demand.
  const bookElRef = useRef<HTMLDivElement | null>(null)
  const pageFlipRef = useRef<PageFlip | null>(null)

  const pageHeight = Math.round(pageWidth * PAGE_RATIO)

  // Adapt the book leaf to the available container width. Runs once the note
  // has loaded (the article is only rendered then), and on every resize.
  useEffect(() => {
    const el = articleRef.current
    if (!el) return
    const measure = () => {
      // Use the content area width (clientWidth minus horizontal padding)
      const cs = getComputedStyle(el)
      const padLeft = parseFloat(cs.paddingLeft) || 0
      const padRight = parseFloat(cs.paddingRight) || 0
      const avail = el.clientWidth - padLeft - padRight - 8 // small buffer
      setPageWidth(Math.max(360, Math.min(MAX_PAGE_WIDTH, avail)))
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

  const { measureRef, pages } = useBookPagination(note?.html ?? '', pageHeight, pageWidth)

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

  // --- StPageFlip lifecycle ----------------------------------------------
  // Rebuilds the book whenever the note (pages) or the leaf geometry changes.
  useEffect(() => {
    const el = bookElRef.current
    if (!el || pages.length === 0) return

    // Keep a reference to the parent BEFORE destroy() — StPageFlip removes
    // the host element from the DOM when destroyed, so parentElement would
    // be null afterwards.
    const parent = el.parentElement
    if (pageFlipRef.current) {
      pageFlipRef.current.destroy()
      pageFlipRef.current = null
    }
    if (parent && !el.isConnected) parent.appendChild(el)
    el.replaceChildren()

    const pageEls = pages.map((html, idx) => {
      const div = document.createElement('div')
      div.className = 'page prose-ailearn'
      div.style.position = 'relative'
      div.innerHTML = html
      // Add a small cinnabar seal page number in the bottom-right corner
      const seal = document.createElement('div')
      seal.className = 'page-corner-seal'
      seal.textContent = String(idx + 1)
      div.appendChild(seal)
      return div
    })
    el.append(...pageEls)

    const flip = new PageFlip(el, {
      width: pageWidth,
      height: pageHeight,
      size: 'fixed',
      showCover: false,
      usePortrait: true,
      flippingTime: 750,
      drawShadow: true,
      maxShadowOpacity: 0.55,
      startPage: 0,
      useMouseEvents: true,
      mobileScrollSupport: true,
      swipeDistance: 30,
      clickEventForward: true,
    })
    flip.loadFromHTML(pageEls)
    // StPageFlip auto-marks the last page as "hard" (sturdy cover) which uses
    // a 3D rotateY path; earlier pages use a 2D polygon clip-path. The two
    // paths render slightly different fold animations — force every page to
    // "soft" so forward and backward flips look identical.
    for (let i = 0; i < pageEls.length; i++) {
      const p = flip.getPage(i) as { setDensity?: (d: string) => void } | undefined
      p?.setDensity?.('soft')
    }
    flip.on('flip', (e) => setPageIndex(e.data))
    pageFlipRef.current = flip
    // Start on the current page after a rebuild (e.g. resize mid-read).
    if (pageIndex > 0) flip.turnToPage(pageIndex)
  }, [pages, pageWidth, pageHeight]) // eslint-disable-line react-hooks/exhaustive-deps

  const handleTocClick = useCallback(
    (e: React.MouseEvent, h: Heading) => {
      e.preventDefault()
      // Use the frontend slugify (same function that generates in-page ids),
      // NOT the backend slug — the two differ for CJK text.
      const slug = slugify(h.text)
      const target = findPageForSlug(slug)
      if (target < 0) return
      setActiveSlug(slug)
      if (target === pageIndex) return
      pageFlipRef.current?.flip(target)
    },
    [findPageForSlug, pageIndex],
  )

  // Scroll-spy for the TOC (observes only the currently visible leaf). On
  // pages whose content scrolls inside the leaf (oversized blocks) the spy
  // follows the internal scroll; regular leaves simply highlight the first
  // heading.
  useEffect(() => {
    const host = bookElRef.current
    if (!host) return
    const pageEl = host.querySelectorAll<HTMLElement>('.page')[pageIndex]
    if (!pageEl) return
    const headings = pageEl.querySelectorAll<HTMLElement>('h2, h3')
    if (headings.length === 0) return
    const observer = new IntersectionObserver(
      (entries) => {
        const visible = entries
          .filter((e) => e.isIntersecting)
          .sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top)
        if (visible[0]) setActiveSlug(visible[0].target.id)
      },
      { rootMargin: '-16px 0px -70% 0px', threshold: 0 },
    )
    headings.forEach((h) => observer.observe(h))
    return () => observer.disconnect()
  }, [note, pageIndex, pages])

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
    <div className="flex flex-col gap-6 pt-2">
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
          className="flex-1 min-w-0 relative bg-surface-container-lowest dark:bg-dark-surface-elevated rounded-3xl shadow-ambient dark:shadow-dark-ambient border border-outline-variant/50 dark:border-white/10 p-5 md:p-10"
        >
          <header className="mb-6 pb-5 border-b border-outline-variant/50 dark:border-white/10">
            <div className="flex flex-wrap items-center gap-3 mb-3">
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
            <h1 className="font-headline text-2xl md:text-3xl font-bold text-on-surface dark:text-inverse-on-surface leading-tight">
              {note.title}
            </h1>
            {note.description && (
              <p className="mt-3 text-body-lg text-on-surface-variant dark:text-outline leading-relaxed">
                {note.description}
              </p>
            )}
          </header>

          {/* Hidden measure container — drives pagination layout. React owns
              its content via dangerouslySetInnerHTML; it sits WAY off-screen
              (position:fixed so it cannot inflate body scrollHeight) but keeps
              layout so offsetHeight stays meaningful. It mirrors the
              book page padding/border/box-sizing exactly so measured heights
              match the rendered page pixel-for-pixel. */}
          <div
            ref={measureRef}
            className="prose-ailearn page-measure"
            style={{
              position: 'fixed',
              left: '-99999px',
              top: '-99999px',
              width: pageWidth,
              padding: `${BOOK_PAD.top}px ${BOOK_PAD.right}px ${BOOK_PAD.bottom}px ${BOOK_PAD.left}px`,
              boxSizing: 'border-box',
              border: '1px solid transparent',
              overflow: 'visible',
              visibility: 'hidden',
              pointerEvents: 'none',
              height: 'auto',
              minHeight: 'unset',
              maxHeight: 'none',
              zIndex: -1,
            }}
            aria-hidden="true"
            dangerouslySetInnerHTML={{ __html: note.html }}
          />

          {/* Book — StPageFlip turns the pages with a real paper fold. The
              host div is a stable React-owned shell; the library manages the
              page DOM and canvas layers inside it. */}
          <div className="book-3d mt-2" style={{ width: pageWidth, height: pageHeight }}>
            <div ref={bookElRef} className="book-flip-host" style={{ width: pageWidth, height: pageHeight }} />
            {/* Peel hint — invites dragging when idle (hidden while turning) */}
            <div className="book-peel-hint" aria-hidden="true" />
          </div>

          {/* Page navigation — Cinnabar Seal (朱砂印) style */}
          {pages.length > 1 && (
            <div className="flex flex-col items-center gap-3 mt-5">
              <div className="flex items-center gap-3">
                <button
                  onClick={() => pageFlipRef.current?.flipPrev()}
                  disabled={pageIndex === 0}
                  aria-label="Previous page"
                  className="page-nav-btn"
                >
                  <span className="material-symbols-outlined" style={{ fontSize: 18 }}>chevron_left</span>
                </button>
                <span className="page-number-seal">
                  {pageIndex + 1} / {pages.length}
                </span>
                <button
                  onClick={() => pageFlipRef.current?.flipNext()}
                  disabled={pageIndex >= pages.length - 1}
                  aria-label="Next page"
                  className="page-nav-btn"
                >
                  <span className="material-symbols-outlined" style={{ fontSize: 18 }}>chevron_right</span>
                </button>
              </div>
              <div className="flex items-center gap-2">
                {pages.map((_, i) => (
                  <button
                    key={i}
                    onClick={() => {
                      if (i === pageIndex) return
                      pageFlipRef.current?.flip(i)
                    }}
                    aria-label={`Page ${i + 1}`}
                    className={`page-dot ${i === pageIndex ? 'active' : ''}`}
                  />
                ))}
              </div>
            </div>
          )}
        </article>

        {/* Right rail: TOC + related. The whole rail sticks as one unit so
            the related-notes card stays on screen while reading — sticking
            only the TOC let the related card scroll out of view. */}
        <aside className="lg:w-72 shrink-0 flex flex-col gap-6">
          <div className="flex flex-col gap-6 lg:sticky lg:top-8">
            {toc.length > 0 && (
              <div className="max-h-[38vh] overflow-y-auto bg-surface-container-low dark:bg-dark-surface rounded-2xl p-5 border border-outline-variant/40 dark:border-white/10">
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
              <div className="max-h-[42vh] overflow-y-auto bg-surface-container-low dark:bg-dark-surface rounded-2xl p-5 border border-outline-variant/40 dark:border-white/10">
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
          </div>

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
