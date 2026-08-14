import { Link } from 'react-router-dom'
import { useI18n } from '../i18n/context'

// Placeholder landing page for the Agent Engineering column. The full
// curriculum (tools, memory, planning, multi-agent) is still being written;
// this page keeps a visible home for the section and links into the lab.

const PLAN_ITEMS = [
  'tool', 'memory', 'planning', 'multi-agent',
]

export default function AgentPage() {
  const { t } = useI18n()
  const name = t.home.categoryNames.agent ?? 'Agent Engineering'
  const blurb = t.home.categoryBlurbs.agent ?? ''

  return (
    <div className="flex flex-col gap-8 pt-2 max-w-3xl">
      <header className="flex flex-col gap-4">
        <span className="inline-flex items-center gap-2 self-start text-caption font-semibold uppercase tracking-[0.15em] text-primary dark:text-inverse-primary">
          <span className="material-symbols-outlined" style={{ fontSize: 16 }}>smart_toy</span>
          {name}
        </span>
        <h1 className="font-headline text-headline-lg-mobile md:text-headline-xl text-on-surface dark:text-inverse-on-surface leading-tight">
          {name}
        </h1>
        <p className="font-body-lg text-body-lg text-on-surface-variant dark:text-outline leading-relaxed">
          {blurb}
        </p>
      </header>

      <section className="bg-surface-container-low dark:bg-dark-surface-elevated rounded-3xl p-6 border border-outline-variant/40 dark:border-white/10">
        <h2 className="font-headline text-lg text-on-surface dark:text-inverse-on-surface mb-4">
          {t.common.loading}
        </h2>
        <p className="text-body-md text-on-surface-variant dark:text-outline mb-6">
          {t.nav.lab} — {'transformer training & self-attention'}{' '}
          <Link to="/lab/attention" className="text-primary dark:text-inverse-primary font-semibold hover:underline">
            {t.lab.title}
          </Link>
        </p>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          {PLAN_ITEMS.map((k) => (
            <div
              key={k}
              className="rounded-2xl bg-surface-container dark:bg-dark-surface p-4 border border-outline-variant/40 dark:border-white/10 text-caption text-on-surface-variant dark:text-outline"
            >
              {k}
            </div>
          ))}
        </div>
      </section>
    </div>
  )
}
