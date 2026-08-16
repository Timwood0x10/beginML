// VerificationPanel — multi-source knowledge verification ("查证挑战").
//
// Renders an experiment's core knowledge claims, each backed by 2-3
// authoritative sources (papers / official docs) from verification.ts.
// The learner reads the sources, confirms the claims against the experiment
// they just ran, and marks the knowledge as verified (persisted to
// localStorage). Copy lives here (shared UI) so wiring a new experiment is
// one <VerificationPanel entry={verificationData['xxx']} /> line.
import { useEffect, useState } from 'react'
import { useI18n } from '../i18n/context'
import type { VerificationEntry, VerificationSource } from './verification'

const STORAGE_KEY = 'ailearn-verified'

function loadVerified(): Set<string> {
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY)
    if (!raw) return new Set()
    const parsed = JSON.parse(raw)
    return Array.isArray(parsed) ? new Set(parsed) : new Set()
  } catch {
    return new Set()
  }
}

const UI: Record<'zh' | 'en', Record<string, string>> = {
  zh: {
    title: '多方查证',
    hint: '每条核心断言都配了权威来源。读一读，确认它与实验现象一致，再标记已查证。',
    claims: '核心断言',
    sources: '权威来源',
    paper: '论文',
    docs: '官方文档',
    verify: '我已查证',
    verified: '已查证',
    open: '打开',
    sourceTag: '来源',
  },
  en: {
    title: 'Multi-source Verification',
    hint: 'Each core claim is backed by authoritative sources. Read them, confirm they match the experiment, then mark it verified.',
    claims: 'Core claims',
    sources: 'Authoritative sources',
    paper: 'Paper',
    docs: 'Official docs',
    verify: 'Mark verified',
    verified: 'Verified',
    open: 'Open',
    sourceTag: 'Source',
  },
}

function SourceRow({ s, ui }: { s: VerificationSource; ui: Record<string, string> }) {
  const { lang } = useI18n()
  const label = lang === 'zh' ? s.title : s.titleEn
  const tag = s.kind === 'paper' ? ui.paper : ui.docs
  return (
    <a
      href={s.url}
      target="_blank"
      rel="noopener noreferrer"
      className="flex items-start gap-2 group"
    >
      <span
        className={`shrink-0 mt-0.5 px-2 py-0.5 rounded-md text-[10px] font-bold uppercase tracking-wide ${
          s.kind === 'paper'
            ? 'bg-[#5B6BB0]/15 text-[#5B6BB0] dark:text-[#aab3e8]'
            : 'bg-[#2f6b3e]/15 text-[#2f6b3e] dark:text-[#9ed0a8]'
        }`}
      >
        {tag}
      </span>
      <span className="text-caption text-on-surface-variant dark:text-outline group-hover:text-primary dark:group-hover:text-inverse-primary transition truncate">
        {label}
        <span className="material-symbols-outlined align-middle ml-1" style={{ fontSize: 13 }}>open_in_new</span>
      </span>
    </a>
  )
}

export function VerificationPanel({ entry }: { entry: VerificationEntry }) {
  const { lang } = useI18n()
  const ui = UI[lang]
  const [verified, setVerified] = useState<Set<string>>(() => loadVerified())
  const done = verified.has(entry.id)

  useEffect(() => {
    try {
      window.localStorage.setItem(STORAGE_KEY, JSON.stringify([...verified]))
    } catch {
      /* storage may be unavailable — ignore */
    }
  }, [verified])

  const mark = () => {
    const next = new Set(verified)
    next.add(entry.id)
    setVerified(next)
  }

  return (
    <div className={`rounded-3xl p-5 border transition-all ${
      done
        ? 'border-[#2f6b3e]/40 dark:border-white/10 bg-[#2f6b3e]/5'
        : 'border-outline-variant/40 dark:border-white/10 bg-surface-container-lowest dark:bg-dark-surface'
    }`}>
      <div className="flex items-center gap-2 mb-1">
        <span className="text-lg">🛡️</span>
        <h3 className="font-label-md text-label-md uppercase tracking-wider text-on-surface dark:text-dark-on-surface">{ui.title}</h3>
        {done && (
          <span className="px-2.5 py-1 rounded-full bg-[#2f6b3e]/15 text-[#2f6b3e] dark:text-[#9ed0a8] text-caption font-bold ml-auto">
            {ui.verified}
          </span>
        )}
      </div>
      <p className="text-caption text-on-surface-variant dark:text-outline mb-3">{ui.hint}</p>

      {/* Core claims — each claim carries its own authoritative sources */}
      <div className="mb-3">
        <div className="text-caption uppercase tracking-wider text-outline font-semibold mb-2">{ui.claims}</div>
        <div className="flex flex-col gap-2">
          {entry.claims.map((c, i) => (
            <div key={i} className="rounded-xl border border-outline-variant/40 dark:border-white/10 bg-surface-container dark:bg-white/5 p-4">
              <div className="flex items-start gap-2">
                <span className="font-mono text-caption text-outline mt-0.5 shrink-0">#{i + 1}</span>
                <span className="text-body-md text-on-surface dark:text-dark-on-surface">
                  {lang === 'zh' ? c.zh : c.en}
                </span>
              </div>
              <div className="mt-2.5 pl-6 flex flex-col gap-1.5">
                {c.sources.map((s, j) => (
                  <SourceRow key={j} s={s} ui={ui} />
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Verify action */}
      {!done ? (
        <button
          onClick={mark}
          className="inline-flex items-center gap-2 px-4 py-2 rounded-xl bg-on-surface text-on-primary dark:bg-inverse-surface dark:text-inverse-surface font-label-md text-label-md hover:opacity-90 transition"
        >
          <span className="material-symbols-outlined" style={{ fontSize: 16 }}>verified</span>
          {ui.verify}
        </button>
      ) : (
        <div className="flex items-center gap-2 text-caption text-[#2f6b3e] dark:text-[#9ed0a8]">
          <span className="material-symbols-outlined" style={{ fontSize: 16 }}>verified</span>
          {ui.verified}
        </div>
      )}
    </div>
  )
}
