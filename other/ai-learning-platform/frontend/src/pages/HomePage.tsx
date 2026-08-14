import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { api } from '../api'
import type { Note, StatsResponse } from '../types'
import { Spinner, ErrorState } from '../components/States'
import CategoryBadge from '../components/CategoryBadge'
import { useI18n } from '../i18n/context'

const CATEGORY_ICONS: Record<string, string> = {
  math: 'functions',
  attention: 'psychology',
  hybrid: 'bolt',
  paper: 'description',
  general: 'article',
}

export default function HomePage() {
  const { lang, t } = useI18n()
  const [stats, setStats] = useState<StatsResponse | null>(null)
  const [recent, setRecent] = useState<Note[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const load = async () => {
    setLoading(true)
    setError(null)
    try {
      const [s, n] = await Promise.all([api.stats(lang), api.notes({ lang })])
      setStats(s)
      // "Journey" order: math → attention → hybrid → paper; otherwise stable.
      const order = ['math', 'attention', 'hybrid', 'paper', 'general']
      const sorted = [...n.notes].sort((a, b) => {
        const ai = order.indexOf(a.category.id)
        const bi = order.indexOf(b.category.id)
        return (ai === -1 ? 99 : ai) - (bi === -1 ? 99 : bi)
      })
      setRecent(sorted.slice(0, 6))
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    load()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [lang])

  if (loading) return <Spinner label={t.common.loading} />
  if (error || !stats) return <ErrorState message={error ?? t.common.loading} onRetry={load} />

  const totalHours = Math.round(stats.totalReadingMinutes / 60)

  return (
    <div className="flex flex-col gap-12">
      {/* Hero */}
      <header className="flex flex-col gap-4 pt-2 md:pt-6">
        <span className="inline-flex items-center gap-2 self-start text-caption font-semibold uppercase tracking-[0.15em] text-primary dark:text-inverse-primary">
          <span className="material-symbols-outlined" style={{ fontSize: 16 }}>auto_awesome</span>
          {t.home.badge}
        </span>
        <h1 className="font-headline text-headline-lg-mobile md:text-headline-xl text-on-surface dark:text-inverse-on-surface max-w-3xl leading-tight">
          {t.home.title1}{' '}
          <span className="italic text-primary dark:text-inverse-primary">{t.home.title2}</span>
        </h1>
        <p className="font-body-lg text-body-lg text-on-surface-variant dark:text-outline max-w-2xl leading-relaxed">
          {stats.totalNotes} {t.home.subtitle}
        </p>

        <div className="flex flex-wrap gap-3 mt-2">
          <Link
            to="/browse"
            className="inline-flex items-center gap-2 px-6 py-3 rounded-2xl bg-primary dark:bg-inverse-primary text-on-primary dark:text-inverse-surface font-label-md shadow-ambient hover:opacity-90 transition"
          >
            <span className="material-symbols-outlined" style={{ fontSize: 18 }}>menu_book</span>
            {t.home.browseCta}
          </Link>
          <Link
            to="/map"
            className="inline-flex items-center gap-2 px-6 py-3 rounded-2xl bg-surface-container dark:bg-dark-surface-elevated text-on-surface dark:text-dark-on-surface border border-outline-variant/60 dark:border-white/10 font-label-md hover:bg-surface-container-high transition"
          >
            <span className="material-symbols-outlined" style={{ fontSize: 18 }}>hub</span>
            {t.home.mapCta}
          </Link>
        </div>
      </header>

      {/* Stat cards */}
      <section className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <StatCard icon="description" value={stats.totalNotes.toString()} label={t.home.notes} />
        <StatCard icon="schedule" value={`${totalHours}h`} label={t.home.readingTime} />
        <StatCard icon="category" value={stats.categories.length.toString()} label={t.home.subjects} />
        <StatCard icon="functions" value={`${(stats.totalWords / 1000).toFixed(1)}k`} label={t.home.words} />
      </section>

      {/* Roadmap nodes — mirrors templates/learning_roadmap */}
      <section>
        <div className="flex items-end justify-between mb-6">
          <div>
            <h2 className="font-headline text-headline-lg-mobile text-on-surface dark:text-inverse-on-surface">
              {t.home.journey}
            </h2>
            <p className="text-body-md text-on-surface-variant dark:text-outline mt-1">
              {t.home.journeyDesc}
            </p>
          </div>
          <Link
            to="/path"
            className="hidden md:inline-flex items-center gap-1 text-label-md text-primary dark:text-inverse-primary font-semibold hover:opacity-80"
          >
            {t.home.fullPath}
            <span className="material-symbols-outlined" style={{ fontSize: 16 }}>arrow_forward</span>
          </Link>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-12 gap-6 relative">
          <div className="hidden md:block absolute top-[5.5rem] left-0 right-0 h-px bg-outline-variant dark:bg-white/10 -z-0" />

          {stats.categories.slice(0, 4).map((cat, i) => {
            const status = i === 0 ? t.home.startHere : i === 1 ? t.home.upNext : t.home.explore
            const accent = i === 0
            return (
              <Link
                key={cat.id}
                to={`/browse?category=${cat.id}`}
                className={`md:col-span-3 bg-surface-container-lowest dark:bg-dark-surface-elevated rounded-3xl p-6 shadow-ambient dark:shadow-dark-ambient border ${
                  accent
                    ? 'border-primary/40 dark:border-inverse-primary/30 ring-1 ring-primary/10'
                    : 'border-outline-variant/50 dark:border-white/10'
                } flex flex-col gap-4 group hover:-translate-y-1 hover:shadow-ambient-lg transition-all duration-300 relative overflow-hidden`}
              >
                {accent && (
                  <div className="absolute -top-16 -right-16 w-40 h-40 bg-primary/10 rounded-full blur-3xl pointer-events-none" />
                )}
                <div className="flex justify-between items-start relative">
                  <div
                    className={`w-12 h-12 rounded-2xl flex items-center justify-center ${
                      accent
                        ? 'bg-primary text-on-primary'
                        : 'bg-surface-variant dark:bg-white/5 text-on-surface-variant dark:text-outline'
                    }`}
                  >
                    <span className="material-symbols-outlined" style={{ fontSize: 24 }}>
                      {CATEGORY_ICONS[cat.id] ?? 'article'}
                    </span>
                  </div>
                  <span
                    className={`px-3 py-1 rounded-full text-caption font-semibold ${
                      accent
                        ? 'bg-primary/10 text-primary dark:bg-inverse-primary/15 dark:text-inverse-primary'
                        : 'bg-surface-variant dark:bg-white/5 text-on-surface-variant dark:text-outline'
                    }`}
                  >
                    {status}
                  </span>
                </div>
                <div className="relative">
                  <h3 className="font-headline text-headline-lg-mobile text-on-surface dark:text-inverse-on-surface mb-1.5">
                    {t.home.categoryNames[cat.id] ?? cat.en}
                  </h3>
                  <p className="text-body-md text-on-surface-variant dark:text-outline leading-relaxed">
                    {t.home.categoryBlurbs[cat.id] ?? t.home.categoryBlurbs.general}
                  </p>
                </div>
                <div className="mt-auto relative">
                  <div className="flex justify-between text-caption text-on-surface-variant dark:text-outline mb-2">
                    <span>{cat.count} {t.home.notes}</span>
                    <span className="inline-flex items-center gap-1 text-primary dark:text-inverse-primary font-semibold opacity-0 group-hover:opacity-100 transition">
                      {t.home.open}
                      <span className="material-symbols-outlined" style={{ fontSize: 14 }}>arrow_forward</span>
                    </span>
                  </div>
                </div>
              </Link>
            )
          })}
        </div>
      </section>

      {/* Recent / featured notes */}
      <section>
        <div className="flex items-end justify-between mb-6">
          <h2 className="font-headline text-headline-lg-mobile text-on-surface dark:text-inverse-on-surface">
            {t.home.startReading}
          </h2>
          <Link
            to="/browse"
            className="text-label-md text-primary dark:text-inverse-primary font-semibold hover:opacity-80 inline-flex items-center gap-1"
          >
            {t.home.viewAll}
            <span className="material-symbols-outlined" style={{ fontSize: 16 }}>arrow_forward</span>
          </Link>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
          {recent.map((note) => (
            <Link
              key={note.id}
              to={`/note/${note.id}`}
              className="group bg-surface-container-lowest dark:bg-dark-surface-elevated rounded-3xl p-6 shadow-ambient dark:shadow-dark-ambient border border-outline-variant/50 dark:border-white/10 hover:-translate-y-1 hover:shadow-ambient-lg transition-all duration-300 flex flex-col"
            >
              <div className="flex items-center justify-between mb-4">
                <CategoryBadge category={note.category} />
                <span className="text-caption text-outline inline-flex items-center gap-1">
                  <span className="material-symbols-outlined" style={{ fontSize: 13 }}>schedule</span>
                  {note.readingTime} {t.common.minutes}
                </span>
              </div>
              <h3 className="font-headline text-lg md:text-xl text-on-surface dark:text-inverse-on-surface mb-2 leading-snug group-hover:text-primary dark:group-hover:text-inverse-primary transition-colors">
                {note.title}
              </h3>
              <p className="text-body-md text-on-surface-variant dark:text-outline line-clamp-3 leading-relaxed flex-1">
                {note.description}
              </p>
              <div className="mt-5 pt-4 border-t border-outline-variant/40 dark:border-white/10 flex items-center justify-between">
                <span className="text-caption text-outline">{note.wordCount.toLocaleString()} {t.home.words}</span>
                <span className="inline-flex items-center gap-1 text-label-md font-semibold text-primary dark:text-inverse-primary">
                  {t.common.read}
                  <span className="material-symbols-outlined" style={{ fontSize: 15 }}>arrow_forward</span>
                </span>
              </div>
            </Link>
          ))}
        </div>
      </section>

      {/* Supporting resources bento */}
      <section>
        <h2 className="font-headline text-headline-lg-mobile text-on-surface dark:text-inverse-on-surface mb-6">
          {t.home.resources}
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
          <BentoCard
            icon="search"
            title={t.home.resSearchTitle}
            description={t.home.resSearchDesc}
            to="/browse"
          />
          <BentoCard
            icon="hub"
            title={t.home.resMapTitle}
            description={t.home.resMapDesc}
            to="/map"
          />
          <BentoCard
            icon="route"
            title={t.home.resPathTitle}
            description={t.home.resPathDesc}
            to="/path"
          />
        </div>
      </section>
    </div>
  )
}

function StatCard({ icon, value, label }: { icon: string; value: string; label: string }) {
  return (
    <div className="bg-surface-container-low dark:bg-dark-surface-elevated rounded-2xl p-5 border border-outline-variant/40 dark:border-white/10">
      <span className="material-symbols-outlined text-primary dark:text-inverse-primary mb-3" style={{ fontSize: 24 }}>
        {icon}
      </span>
      <div className="font-headline text-3xl font-bold text-on-surface dark:text-inverse-on-surface">{value}</div>
      <div className="text-caption text-on-surface-variant dark:text-outline uppercase tracking-wider mt-1">{label}</div>
    </div>
  )
}

function BentoCard({
  icon,
  title,
  description,
  to,
}: {
  icon: string
  title: string
  description: string
  to: string
}) {
  return (
    <Link
      to={to}
      className="group bg-surface-container dark:bg-dark-surface-elevated rounded-3xl p-7 flex flex-col items-start gap-4 border border-outline-variant/40 dark:border-white/10 hover:-translate-y-1 hover:shadow-ambient-lg transition-all duration-300"
    >
      <div className="w-14 h-14 rounded-full bg-secondary-container dark:bg-secondary flex items-center justify-center text-on-secondary-container dark:text-on-secondary">
        <span className="material-symbols-outlined" style={{ fontSize: 28 }}>{icon}</span>
      </div>
      <div>
        <h4 className="font-headline text-lg font-bold text-on-surface dark:text-inverse-on-surface mb-1">{title}</h4>
        <p className="text-body-md text-on-surface-variant dark:text-outline leading-relaxed">{description}</p>
      </div>
      <span className="mt-auto inline-flex items-center gap-1 text-label-md font-semibold text-primary dark:text-inverse-primary">
        Explore
        <span className="material-symbols-outlined group-hover:translate-x-1 transition-transform" style={{ fontSize: 16 }}>arrow_forward</span>
      </span>
    </Link>
  )
}
