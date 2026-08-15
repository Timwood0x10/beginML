import { useEffect, useRef, useState } from 'react'
import { NavLink, Outlet, useLocation } from 'react-router-dom'
import { useTheme } from '../hooks/useTheme'
import { useI18n } from '../i18n/context'

// All color palettes — shown as a text dropdown, not just dots.
const THEME_OPTIONS = [
  { id: 'parchment', bg: '#F7F0E3', name: 'Parchment', zh: '羊皮纸', desc: 'warm aged paper' },
  { id: 'matcha', bg: '#EAE8DC', name: 'Matcha', zh: '抹茶', desc: 'sage green' },
  { id: 'qingdai', bg: 'linear-gradient(135deg,#F3F6F4,#56679F 70%,#8179A8)', name: 'QingDai', zh: '青黛', desc: 'bamboo-indigo lab' },
  { id: 'pine', bg: 'linear-gradient(135deg,#F2F5F1,#5F8276 70%,#72849A)', name: 'Pine Mist', zh: '松烟', desc: 'grey-green forest' },
  { id: 'studio', bg: 'linear-gradient(135deg,#F3F4F2,#738895 70%,#B7836C)', name: 'Studio', zh: '工作室', desc: 'fog-blue & clay' },
  { id: 'moss', bg: 'linear-gradient(135deg,#F1F3EF,#71826C 70%,#899096)', name: 'Moss & Stone', zh: '苔石', desc: 'moss-green field lab' },
] as const

type ThemeId = (typeof THEME_OPTIONS)[number]['id']

function ThemeDropdown({ compact = false }: { compact?: boolean }) {
  const { palette, setPalette } = useTheme()
  const { lang } = useI18n()
  const [open, setOpen] = useState(false)
  const ref = useRef<HTMLDivElement | null>(null)

  const current = THEME_OPTIONS.find((o) => o.id === palette) ?? THEME_OPTIONS[0]

  // Close when clicking outside the dropdown.
  useEffect(() => {
    if (!open) return
    const onClick = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false)
    }
    window.addEventListener('mousedown', onClick)
    return () => window.removeEventListener('mousedown', onClick)
  }, [open])

  return (
    <div ref={ref} className="relative">
      <button
        onClick={() => setOpen((o) => !o)}
        aria-haspopup="listbox"
        aria-expanded={open}
        className={`inline-flex items-center gap-1.5 rounded-full text-label-md font-semibold text-on-surface-variant dark:text-outline hover:bg-surface-variant dark:hover:bg-white/5 transition-colors ${
          compact ? 'h-9 px-2.5' : 'h-10 px-3'
        }`}
      >
        <span
          className="w-3.5 h-3.5 rounded-full border border-black/10 dark:border-white/20"
          style={{ background: current.bg }}
        />
        <span>{lang === 'zh' ? current.zh : current.name}</span>
        <span className="material-symbols-outlined text-[15px]" style={{ fontSize: 15 }}>arrow_drop_down</span>
      </button>

      {open && (
        <div
          role="listbox"
          className="absolute right-0 top-full mt-2 w-64 rounded-2xl bg-surface-container-lowest dark:bg-dark-surface-elevated border border-outline-variant/50 dark:border-white/10 shadow-ambient-lg dark:shadow-dark-ambient p-1.5 z-50"
        >
          {THEME_OPTIONS.map((o) => (
            <button
              key={o.id}
              role="option"
              aria-selected={palette === o.id}
              onClick={() => {
                setPalette(o.id as ThemeId)
                setOpen(false)
              }}
              className={`w-full flex items-center gap-2.5 px-2.5 py-2 rounded-xl text-left transition-colors ${
                palette === o.id
                  ? 'bg-primary text-on-primary dark:bg-inverse-primary dark:text-inverse-surface'
                  : 'text-on-surface dark:text-dark-on-surface hover:bg-surface-variant dark:hover:bg-white/5'
              }`}
            >
              <span
                className="w-5 h-5 rounded-full shrink-0 border border-black/10 dark:border-white/20"
                style={{ background: o.bg }}
              />
              <span className="min-w-0">
                <span className="block text-caption font-semibold">{lang === 'zh' ? `${o.zh} · ${o.name}` : `${o.name} · ${o.zh}`}</span>
                <span className={`block text-[10px] truncate ${palette === o.id ? 'opacity-80' : 'text-outline'}`}>{o.desc}</span>
              </span>
            </button>
          ))}
        </div>
      )}
    </div>
  )
}

export default function Layout() {
  const { theme, toggle } = useTheme()
  const { lang, toggleLang, t } = useI18n()
  const location = useLocation()

  const navItems = [
    { to: '/', label: t.nav.roadmap, icon: 'space_dashboard', end: true },
    { to: '/browse', label: t.nav.library, icon: 'menu_book' },
    { to: '/path', label: t.nav.path, icon: 'route' },
    { to: '/map', label: t.nav.map, icon: 'hub' },
    { to: '/agent', label: t.nav.agent, icon: 'smart_toy' },
    { to: '/lab', label: t.nav.lab, icon: 'science' },
  ]

  return (
    <div className="min-h-screen flex flex-col bg-surface dark:bg-dark-surface text-on-surface dark:text-dark-on-surface relative overflow-x-clip">
      {/* subtle paper texture */}
      <div
        className="fixed inset-0 pointer-events-none z-0 opacity-[0.025] dark:opacity-[0.04]"
        style={{
          backgroundImage:
            'radial-gradient(circle at 1px 1px, currentColor 1px, transparent 0)',
          backgroundSize: '24px 24px',
        }}
      />

      {/* Top navigation bar — floating island per template. Not sticky:
          it scrolls away with the page instead of following the reader. */}
      <nav className="hidden md:flex justify-between items-center w-[calc(100%-48px)] max-w-container-max mx-auto bg-surface-container dark:bg-dark-surface-elevated rounded-2xl px-6 py-3 mt-4 shadow-ambient dark:shadow-dark-ambient z-40 border border-outline-variant/40 dark:border-white/10">
        <NavLink to="/" className="flex items-center gap-3 group cursor-pointer">
          <div className="w-9 h-9 rounded-xl bg-primary text-on-primary flex items-center justify-center group-hover:scale-105 transition-transform">
            <span className="material-symbols-outlined fill" style={{ fontSize: 20 }}>auto_stories</span>
          </div>
          <span className="font-headline font-bold text-headline-lg-mobile text-on-surface dark:text-inverse-on-surface tracking-tight">
            AIScope
          </span>
        </NavLink>

        <div className="flex items-center gap-1">
          {navItems.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.end}
              className={({ isActive }) =>
                `flex items-center gap-2 px-4 py-2 rounded-full font-label-md text-label-md transition-all duration-200 ${
                  isActive
                    ? 'bg-primary text-on-primary shadow-sm'
                    : 'text-on-surface-variant dark:text-outline hover:bg-surface-variant dark:hover:bg-white/5'
                }`
              }
            >
              <span className="material-symbols-outlined" style={{ fontSize: 18 }}>{item.icon}</span>
              {item.label}
            </NavLink>
          ))}
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={toggleLang}
            aria-label="Switch language"
            className="h-10 px-3 rounded-full flex items-center justify-center text-label-md font-semibold text-on-surface-variant dark:text-outline hover:bg-surface-variant dark:hover:bg-white/5 transition-colors"
          >
            {lang === 'zh' ? 'EN' : '中文'}
          </button>
          <ThemeDropdown />
          <button
            onClick={toggle}
            aria-label="Toggle theme"
            className="w-10 h-10 rounded-full flex items-center justify-center text-on-surface-variant dark:text-outline hover:bg-surface-variant dark:hover:bg-white/5 transition-colors"
          >
            <span className="material-symbols-outlined" style={{ fontSize: 20 }}>
              {theme === 'dark' ? 'light_mode' : 'dark_mode'}
            </span>
          </button>
        </div>
      </nav>

      {/* Mobile top bar */}
      <div className="md:hidden flex items-center justify-between px-5 py-4 bg-surface-container dark:bg-dark-surface-elevated z-40 border-b border-outline-variant/40 dark:border-white/10">
        <NavLink to="/" className="flex items-center gap-2 group cursor-pointer">
          <div className="w-8 h-8 rounded-lg bg-primary text-on-primary flex items-center justify-center group-hover:scale-105 transition-transform">
            <span className="material-symbols-outlined fill" style={{ fontSize: 18 }}>auto_stories</span>
          </div>
          <span className="font-headline font-bold text-lg">AIScope</span>
        </NavLink>
        <div className="flex items-center gap-1">
          <button
            onClick={toggleLang}
            aria-label="Switch language"
            className="h-9 px-2.5 rounded-full flex items-center justify-center text-label-md font-semibold text-on-surface-variant dark:text-outline"
          >
            {lang === 'zh' ? 'EN' : '中文'}
          </button>
          <ThemeDropdown compact />
          <button
            onClick={toggle}
            className="w-9 h-9 rounded-full flex items-center justify-center text-on-surface-variant dark:text-outline"
          >
            <span className="material-symbols-outlined" style={{ fontSize: 20 }}>
              {theme === 'dark' ? 'light_mode' : 'dark_mode'}
            </span>
          </button>
        </div>
      </div>

      {/* Mobile bottom nav */}
      <nav className="md:hidden fixed bottom-0 left-0 right-0 z-40 bg-surface-container dark:bg-dark-surface-elevated border-t border-outline-variant/40 dark:border-white/10 px-2 py-2 flex justify-around">
        {navItems.map((item) => {
          const active = item.end
            ? location.pathname === item.to
            : location.pathname.startsWith(item.to)
          return (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.end}
              className={`flex flex-col items-center gap-0.5 px-3 py-1.5 rounded-xl transition-colors ${
                active ? 'text-primary dark:text-inverse-primary' : 'text-on-surface-variant dark:text-outline'
              }`}
            >
              <span className="material-symbols-outlined" style={{ fontSize: 22 }}>{item.icon}</span>
              <span className="text-[10px] font-semibold">{item.label}</span>
            </NavLink>
          )
        })}
      </nav>

      <main className="flex-1 w-full max-w-container-max mx-auto px-margin-mobile md:px-margin-desktop py-6 md:py-8 relative z-10 pb-24 md:pb-8 animate-fade-in">
        <Outlet />
      </main>

      <footer className="w-full max-w-container-max mx-auto px-gutter py-5 border-t border-outline-variant/50 dark:border-white/10 flex flex-col md:flex-row items-center justify-between gap-2 text-caption text-on-surface-variant dark:text-outline relative z-10">
        <div className="flex items-center gap-2 font-label-md text-primary dark:text-inverse-primary">
          <span className="material-symbols-outlined" style={{ fontSize: 15 }}>auto_stories</span>
          AIScope · Scholarly notes on AI &amp; mathematics
        </div>
        <div className="text-center opacity-70">
          Built with FastAPI, scikit-learn &amp; React
        </div>
      </footer>
    </div>
  )
}
