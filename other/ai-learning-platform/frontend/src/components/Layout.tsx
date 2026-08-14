import { NavLink, Outlet, useLocation } from 'react-router-dom'
import { useTheme } from '../hooks/useTheme'
import { useI18n } from '../i18n/context'

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
