// NotesPanel — related-notes browser that LINKS INTO the project's own
// note system (improved mapping).
//
// The platform serves every note at /note/:id (NotePage — flipbook, math,
// mermaid, TF-IDF related notes). This panel lists the experiment's related
// notes (title + source path from the i18n `notes` table) and, on click,
// resolves the note id from the backend index and navigates to its detail
// page. Resolution is robust: it indexes BOTH language trees (zh/ + en/),
// tries the exact `{lang}/{src}` path first, then the bare relative path,
// then a suffix match — so a note that only exists in the other language
// (or whose prefix differs) still jumps correctly. Copy lives here (shared).
import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { api } from '../api'
import { useI18n } from '../i18n/context'

export interface LabNote {
  title: string
  src: string
}

const UI: Record<'zh' | 'en', Record<string, string>> = {
  zh: {
    title: '相关笔记',
    hint: '点击跳转到项目笔记页（翻书阅读 + 更多相关笔记）。',
    open: '打开笔记',
    loading: '笔记索引加载中…',
    missing: '笔记未收录',
    missingHint: '笔记库中找不到「{src}」——把该笔记加入笔记库后即可跳转。',
  },
  en: {
    title: 'Related Notes',
    hint: 'Click to jump to the note page (flipbook reading + related notes).',
    open: 'Open note',
    loading: 'Loading note index…',
    missing: 'Note not found',
    missingHint: 'Could not find "{src}" in the note index — add it to the notes library to enable the jump.',
  },
}

export function NotesPanel({ notes }: { notes: LabNote[] }) {
  const { lang } = useI18n()
  const ui = UI[lang]
  const navigate = useNavigate()
  const [pathToId, setPathToId] = useState<Map<string, string> | null>(null)
  const [notFound, setNotFound] = useState<string | null>(null)

  // Index BOTH language trees so a note living only under zh/ or only under
  // en/ still resolves. Keys: the full path (e.g. zh/Self-Attention/11.RoPE.md)
  // AND the bare relative path (Self-Attention/11.RoPE.md).
  useEffect(() => {
    let alive = true
    Promise.all([
      api.notes({ lang: 'zh' }).catch(() => null),
      api.notes({ lang: 'en' }).catch(() => null),
    ]).then(([zh, en]) => {
      if (!alive) return
      const m = new Map<string, string>()
      for (const res of [zh, en]) {
        if (!res) continue
        for (const n of res.notes) {
          if (!m.has(n.path)) m.set(n.path, n.id)
          const bare = n.path.replace(/^(zh|en)\//, '')
          if (!m.has(bare)) m.set(bare, n.id)
        }
      }
      setPathToId(m)
    })
    return () => {
      alive = false
    }
  }, [])

  if (notes.length === 0) return null

  const open = (src: string) => {
    // 1) exact path in the current language → 2) bare relative path →
    // 3) suffix match (in case the directory tree differs slightly).
    const exact = pathToId?.get(`${lang}/${src}`) ?? pathToId?.get(src)
    if (exact) {
      navigate(`/note/${exact}`)
      return
    }
    const suffix = pathToId ? [...pathToId.entries()].find(([p]) => p.endsWith(src))?.[1] : undefined
    if (suffix) {
      navigate(`/note/${suffix}`)
      return
    }
    setNotFound(src)
  }

  return (
    <div className="bg-surface-container-lowest dark:bg-dark-surface rounded-3xl p-5 border border-outline-variant/40 dark:border-white/10">
      <div className="flex items-center gap-2 mb-1">
        <span className="text-lg">📓</span>
        <h3 className="font-label-md text-label-md uppercase tracking-wider text-on-surface dark:text-dark-on-surface">{ui.title}</h3>
      </div>
      <p className="text-caption text-on-surface-variant dark:text-outline mb-3">{ui.hint}</p>

      <div className="flex flex-col gap-2">
        {notes.map((n) => {
          const isMissing = notFound === n.src
          return (
            <div
              key={n.src}
              className="rounded-xl border border-outline-variant/40 dark:border-white/10 bg-surface-container dark:bg-white/5 overflow-hidden"
            >
              <button
                onClick={() => open(n.src)}
                disabled={pathToId === null}
                className="w-full flex items-center gap-3 px-4 py-2.5 text-left hover:bg-surface-variant dark:hover:bg-white/10 transition disabled:opacity-60"
              >
                <span className="material-symbols-outlined text-primary dark:text-inverse-primary" style={{ fontSize: 18 }}>description</span>
                <span className="flex-1 min-w-0">
                  <span className="block text-body-md text-on-surface dark:text-dark-on-surface truncate">{n.title}</span>
                  <span className="block text-caption text-outline font-mono truncate">{n.src}</span>
                </span>
                <span className="text-caption text-on-surface-variant dark:text-outline shrink-0 inline-flex items-center gap-1">
                  {pathToId === null ? ui.loading : ui.open}
                  {pathToId !== null && <span className="material-symbols-outlined" style={{ fontSize: 14 }}>arrow_forward</span>}
                </span>
              </button>

              {isMissing && (
                <div className="border-t border-outline-variant/40 dark:border-white/10 px-4 py-3">
                  <div className="rounded-xl p-3 border border-dashed border-outline-variant/60 dark:border-white/15">
                    <div className="text-label-md font-semibold text-on-surface-variant dark:text-outline mb-1">📄 {ui.missing}</div>
                    <div className="text-caption text-on-surface-variant dark:text-outline">{ui.missingHint.replace('{src}', n.src)}</div>
                  </div>
                </div>
              )}
            </div>
          )
        })}
      </div>
    </div>
  )
}
