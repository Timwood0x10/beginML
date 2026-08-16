// ABPanel — A/B snapshot comparison (interactivity layer).
//
// A lab can lock two parameter snapshots (A and B) and see their key
// metrics side by side: same axis, same scale — the difference becomes
// visible at a glance. The snapshot is captured from the current computed
// result by the experiment itself; ABPanel only renders the comparison.
// Copy lives here (shared UI) so any experiment can add A/B in one line.
import { useI18n } from "../i18n/context";

export interface ABMetric {
  key: string;
  label: string;
  value: number;
  /** Higher is better (green when A/B improves), else inverted. */
  higherBetter?: boolean;
}

export interface ABShot {
  name: string;
  metrics: ABMetric[];
}

const UI: Record<"zh" | "en", Record<string, string>> = {
  zh: {
    title: "A/B 对比",
    hint: "锁定两组参数快照，并排看关键指标的变化。",
    snapA: "锁定 A",
    snapB: "锁定 B",
    swap: "↔ 交换",
    clear: "清除",
    empty: "还没有快照。调好一组参数点「锁定 A」，再调另一组点「锁定 B」。",
    delta: "变化",
  },
  en: {
    title: "A/B Compare",
    hint: "Lock two parameter snapshots and see the key metrics side by side.",
    snapA: "Lock A",
    snapB: "Lock B",
    swap: "↔ Swap",
    clear: "Clear",
    empty:
      'No snapshots yet. Tune one set of parameters and hit "Lock A", tune another and hit "Lock B".',
    delta: "Delta",
  },
};

function MetricRow({ m, delta }: { m: ABMetric; delta: number | null }) {
  const dir =
    delta === null
      ? null
      : delta > 0.0005
        ? "up"
        : delta < -0.0005
          ? "down"
          : "flat";
  const good =
    dir === null ? null : m.higherBetter ? dir === "up" : dir === "down";
  return (
    <div className="flex items-center gap-2 text-caption">
      <span className="w-28 shrink-0 truncate text-outline">{m.label}</span>
      <span className="w-14 text-right font-mono text-on-surface dark:text-dark-on-surface tabular-nums">
        {m.value.toFixed(3)}
      </span>
      {delta !== null && (
        <span
          className={`font-mono tabular-nums ${good === null ? "text-outline" : good ? "text-[#2f6b3e] dark:text-[#9ed0a8]" : "text-[#C8604A] dark:text-[#f0b3a4]"}`}
        >
          {dir === "flat"
            ? "·"
            : `${dir === "up" ? "▲" : "▼"}${Math.abs(delta).toFixed(3)}`}
        </span>
      )}
    </div>
  );
}

/**
 * ABPanel — renders a locked A shot and a locked B shot side by side,
 * with per-metric deltas (A → B). Null shots show an empty hint.
 */
export function ABPanel({
  a,
  b,
  onSnapA,
  onSnapB,
  onSwap,
  onClear,
  disabled,
}: {
  a: ABShot | null;
  b: ABShot | null;
  onSnapA: () => void;
  onSnapB: () => void;
  onSwap: () => void;
  onClear: () => void;
  disabled?: boolean;
}) {
  const { lang } = useI18n();
  const ui = UI[lang];

  const metricsA = a?.metrics ?? [];
  const metricsB = b?.metrics ?? [];
  const byKey = new Map<string, number>();
  for (const m of metricsB) byKey.set(m.key, m.value);

  return (
    <div className="bg-surface-container-lowest dark:bg-dark-surface rounded-3xl p-5 border border-outline-variant/40 dark:border-white/10">
      <div className="flex items-center gap-2 mb-1">
        <span className="text-lg">⚖️</span>
        <h3 className="font-label-md text-label-md uppercase tracking-wider text-on-surface dark:text-dark-on-surface">
          {ui.title}
        </h3>
        <span className="ml-auto inline-flex gap-1.5">
          <button
            onClick={onSnapA}
            disabled={disabled}
            className="px-3 py-1.5 rounded-lg text-label-md font-semibold bg-surface-variant dark:bg-white/10 text-on-surface dark:text-dark-on-surface hover:opacity-80 transition disabled:opacity-40"
          >
            {ui.snapA}
          </button>
          <button
            onClick={onSnapB}
            disabled={disabled}
            className="px-3 py-1.5 rounded-lg text-label-md font-semibold bg-surface-variant dark:bg-white/10 text-on-surface dark:text-dark-on-surface hover:opacity-80 transition disabled:opacity-40"
          >
            {ui.snapB}
          </button>
          {(a || b) && (
            <>
              <button
                onClick={onSwap}
                className="px-3 py-1.5 rounded-lg text-label-md font-semibold text-on-surface-variant dark:text-outline hover:bg-surface-variant dark:hover:bg-white/10 transition"
              >
                {ui.swap}
              </button>
              <button
                onClick={onClear}
                className="px-3 py-1.5 rounded-lg text-label-md font-semibold text-on-surface-variant dark:text-outline hover:bg-surface-variant dark:hover:bg-white/10 transition"
              >
                {ui.clear}
              </button>
            </>
          )}
        </span>
      </div>
      <p className="text-caption text-on-surface-variant dark:text-outline mb-3">
        {ui.hint}
      </p>

      {!a && !b ? (
        <div className="rounded-2xl border border-dashed border-outline-variant/50 dark:border-white/15 p-4 text-center text-caption text-on-surface-variant dark:text-outline">
          {ui.empty}
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {/* A column */}
          <div className="rounded-2xl border border-outline-variant/40 dark:border-white/10 bg-surface-container dark:bg-white/5 p-3">
            <div className="font-label-md font-semibold text-primary dark:text-inverse-primary mb-2 truncate">
              {a?.name ?? "A"}
            </div>
            <div className="flex flex-col gap-1.5">
              {metricsA.map((m) => (
                <MetricRow key={m.key} m={m} delta={null} />
              ))}
            </div>
          </div>
          {/* B column with deltas */}
          <div className="rounded-2xl border border-outline-variant/40 dark:border-white/10 bg-surface-container dark:bg-white/5 p-3">
            <div className="font-label-md font-semibold text-[#2f6b3e] dark:text-[#9ed0a8] mb-2 truncate">
              {b?.name ?? "B"}
            </div>
            <div className="flex flex-col gap-1.5">
              {metricsB.map((m) => {
                const aVal = byKey.get(m.key);
                return (
                  <MetricRow
                    key={m.key}
                    m={m}
                    delta={aVal === undefined ? null : m.value - aVal}
                  />
                );
              })}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
