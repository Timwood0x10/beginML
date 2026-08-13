// Language context — zh/en with persistence and a small transition.
import { createContext, useContext, useEffect, useState, useCallback, type ReactNode } from 'react'
import { zh, en, type Lang, type UIDict } from './translations'

const STORAGE_KEY = 'aiscope-lang'

const DICTS: Record<Lang, UIDict> = { zh, en }

interface I18nValue {
  lang: Lang
  setLang: (l: Lang) => void
  toggleLang: () => void
  t: UIDict
}

const I18nContext = createContext<I18nValue | null>(null)

function getInitialLang(): Lang {
  if (typeof window === 'undefined') return 'zh'
  const stored = window.localStorage.getItem(STORAGE_KEY) as Lang | null
  if (stored === 'zh' || stored === 'en') return stored
  return 'zh'
}

export function I18nProvider({ children }: { children: ReactNode }) {
  const [lang, setLangState] = useState<Lang>(getInitialLang)

  useEffect(() => {
    window.localStorage.setItem(STORAGE_KEY, lang)
    document.documentElement.lang = lang
  }, [lang])

  const setLang = useCallback((l: Lang) => setLangState(l), [])
  const toggleLang = useCallback(() => {
    setLangState((prev) => (prev === 'zh' ? 'en' : 'zh'))
  }, [])

  return (
    <I18nContext.Provider value={{ lang, setLang, toggleLang, t: DICTS[lang] }}>
      {children}
    </I18nContext.Provider>
  )
}

export function useI18n(): I18nValue {
  const ctx = useContext(I18nContext)
  if (!ctx) throw new Error('useI18n must be used inside I18nProvider')
  return ctx
}
