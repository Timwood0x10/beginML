import type { LabControl, LabParams } from './types'

/**
 * A reusable, template-styled control row for the Math Lab sidebar.
 * Renders the correct input based on the control's `type`.
 */
export function ControlRow({
  control,
  label: labelOverride,
  value,
  onChange,
  onAction,
}: {
  control: LabControl
  label?: string
  value: LabParams[string] | undefined
  onChange: (key: string, value: string | number | boolean) => void
  onAction?: (key: string) => void
}) {
  const { key, type } = control
  const label = labelOverride ?? control.label

  if (type === 'action') {
    return (
      <button
        onClick={() => onAction?.(key)}
        className="w-full inline-flex items-center justify-center gap-2 px-4 py-2.5 rounded-xl bg-primary text-on-primary dark:bg-inverse-primary dark:text-inverse-surface font-label-md text-label-md hover:opacity-90 transition"
      >
        <span className="material-symbols-outlined" style={{ fontSize: 18 }}>restart_alt</span>
        {label}
      </button>
    )
  }

  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-center justify-between gap-2">
        <label className="text-label-md text-label-md text-on-surface dark:text-dark-on-surface font-semibold">
          {label}
        </label>
        {type === 'range' && (
          <span className="text-caption text-primary dark:text-inverse-primary font-mono tabular-nums bg-primary-fixed/60 dark:bg-white/10 px-2 py-0.5 rounded-md">
            {formatValue(Number(value), control.step)}
          </span>
        )}
      </div>

      {type === 'range' && (
        <input
          type="range"
          min={control.min}
          max={control.max}
          step={control.step}
          value={Number(value ?? control.default ?? 0)}
          onChange={(e) => onChange(key, parseFloat(e.target.value))}
          className="ailearn-range w-full"
        />
      )}

      {type === 'select' && (
        <div className="flex flex-wrap gap-1.5">
          {control.options?.map((opt) => {
            const active = (value ?? control.default) === opt
            return (
              <button
                key={opt}
                onClick={() => onChange(key, opt)}
                className={`px-3 py-1.5 rounded-lg text-caption font-semibold capitalize border transition-all ${
                  active
                    ? 'bg-primary text-on-primary border-primary dark:bg-inverse-primary dark:text-inverse-surface dark:border-inverse-primary'
                    : 'bg-surface-container-lowest dark:bg-dark-surface text-on-surface-variant dark:text-outline border-outline-variant/60 dark:border-white/10 hover:border-primary/50'
                }`}
              >
                {opt}
              </button>
            )
          })}
        </div>
      )}

      {type === 'toggle' && (
        <button
          role="switch"
          aria-checked={Boolean(value)}
          onClick={() => onChange(key, !value)}
          className={`relative w-11 h-6 rounded-full transition-colors self-start ${
            value
              ? 'bg-primary dark:bg-inverse-primary'
              : 'bg-surface-variant dark:bg-white/15'
          }`}
        >
          <span
            className={`absolute top-0.5 left-0.5 w-5 h-5 rounded-full bg-white shadow transition-transform ${
              value ? 'translate-x-5' : ''
            }`}
          />
        </button>
      )}
    </div>
  )
}

function formatValue(v: string | number | boolean | undefined, step?: number): string {
  if (typeof v !== 'number') return String(v ?? '')
  if (step && step < 0.01) return v.toFixed(4)
  if (step && step < 0.1) return v.toFixed(3)
  if (step && step < 1) return v.toFixed(2)
  return v.toString()
}

/** Build a default params object from a module's control definitions. */
export function defaultParams(controls: LabControl[]): LabParams {
  const p: LabParams = {}
  for (const c of controls) {
    if (c.type !== 'action' && c.default !== undefined) p[c.key] = c.default
  }
  return p
}
