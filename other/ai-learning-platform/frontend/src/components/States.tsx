export function Spinner({ label }: { label?: string }) {
  return (
    <div className="flex flex-col items-center justify-center py-20 gap-4 text-on-surface-variant dark:text-outline">
      <div className="w-10 h-10 rounded-full border-[3px] border-surface-variant dark:border-white/10 border-t-primary dark:border-t-inverse-primary animate-spin" />
      {label && <p className="text-body-md">{label}</p>}
    </div>
  )
}

export function ErrorState({ message, onRetry }: { message: string; onRetry?: () => void }) {
  return (
    <div className="flex flex-col items-center justify-center py-20 gap-4 text-center">
      <div className="w-14 h-14 rounded-full bg-error-container text-on-error-container flex items-center justify-center">
        <span className="material-symbols-outlined" style={{ fontSize: 28 }}>error_outline</span>
      </div>
      <p className="text-body-lg text-on-surface dark:text-dark-on-surface max-w-md">{message}</p>
      {onRetry && (
        <button
          onClick={onRetry}
          className="px-5 py-2.5 rounded-full bg-primary text-on-primary font-label-md hover:opacity-90 transition-opacity"
        >
          Retry
        </button>
      )}
    </div>
  )
}

export function EmptyState({ icon = 'search_off', title, subtitle }: { icon?: string; title: string; subtitle?: string }) {
  return (
    <div className="flex flex-col items-center justify-center py-20 gap-3 text-center text-on-surface-variant dark:text-outline">
      <span className="material-symbols-outlined" style={{ fontSize: 48 }}>{icon}</span>
      <h3 className="font-headline text-headline-lg-mobile text-on-surface dark:text-dark-on-surface">{title}</h3>
      {subtitle && <p className="text-body-md max-w-sm">{subtitle}</p>}
    </div>
  )
}
