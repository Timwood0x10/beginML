import { useState } from "react";
import type { LabResult } from "../types";

interface AttentionResult extends LabResult {
  tokens: string[];
  n: number;
  d: number;
  temperature: number;
  causal: boolean;
  Q: number[][];
  K: number[][];
  scores: number[][];
  weights: number[][];
  output: number[][];
  entropy: number[];
  maxEntropy: number;
}

type View = "weights" | "scores";

export default function AttentionLab({ result }: { result: LabResult | null }) {
  const r = result as AttentionResult | null;
  const [view, setView] = useState<View>("weights");
  const [hover, setHover] = useState<{ row: number; col: number } | null>(null);

  if (!r) return null;

  const matrix = view === "weights" ? r.weights : r.scores;
  const flat = matrix.flat();
  const min = Math.min(...flat);
  const max = Math.max(...flat);
  const span = max - min || 1;

  const cellColor = (v: number) => {
    if (view === "weights") {
      // 0..1 cream -> clay
      const t = v;
      const r1 = Math.round(242 + (184 - 242) * t);
      const g1 = Math.round(237 + (96 - 237) * t);
      const b1 = Math.round(227 + (74 - 227) * t);
      return `rgb(${r1},${g1},${b1})`;
    }
    // scores: diverging around 0
    const t = (v - min) / span;
    if (t < 0.5) {
      const k = t / 0.5;
      return `rgb(${Math.round(242 + (255 - 242) * k)},${Math.round(237 + (255 - 237) * k)},${Math.round(227 + (255 - 227) * k)})`;
    }
    const k = (t - 0.5) / 0.5;
    return `rgb(${Math.round(255 - (255 - 90) * k)},${Math.round(255 - (255 - 90) * k)},${Math.round(255 - (255 - 90) * k)})`;
  };

  const formula = r.causal
    ? "Attention(Q,K,V) = softmax( QKᵀ·(T/√d) + causal_mask ) V"
    : "Attention(Q,K,V) = softmax( QKᵀ·(T/√d) ) V";

  return (
    <div className="flex flex-col gap-5">
      {/* Pipeline explanation */}
      <div className="bg-surface-container-lowest dark:bg-dark-surface rounded-3xl p-6 shadow-ambient dark:shadow-dark-ambient border border-outline-variant/40 dark:border-white/10">
        <div className="flex flex-wrap items-center justify-between gap-3 mb-4">
          <h3 className="font-headline text-lg text-on-surface dark:text-inverse-on-surface inline-flex items-center gap-2">
            <span
              className="material-symbols-outlined"
              style={{ fontSize: 20 }}
            >
              grid_on
            </span>
            Scaled dot-product attention
          </h3>
          <div className="inline-flex rounded-xl bg-surface-container dark:bg-white/5 p-1">
            {(["weights", "scores"] as View[]).map((v) => (
              <button
                key={v}
                onClick={() => setView(v)}
                className={`px-3 py-1.5 rounded-lg text-caption font-semibold capitalize transition ${
                  view === v
                    ? "bg-primary text-on-primary dark:bg-inverse-primary dark:text-inverse-surface"
                    : "text-on-surface-variant dark:text-outline"
                }`}
              >
                {v === "weights" ? "softmax weights" : "QKᵀ scores"}
              </button>
            ))}
          </div>
        </div>

        <code className="block font-mono text-body-md text-primary dark:text-inverse-primary bg-primary-fixed/40 dark:bg-white/5 rounded-xl px-4 py-3 mb-5 overflow-x-auto">
          {formula}
        </code>

        <div className="w-full overflow-x-auto">
          <div className="inline-block min-w-full">
            {/* column headers = keys */}
            <div className="flex ml-16">
              {r.tokens.map((t) => (
                <div
                  key={t}
                  className="w-12 md:w-14 text-center text-caption font-semibold text-on-surface-variant dark:text-outline pb-1"
                >
                  {t}
                </div>
              ))}
              <div className="w-16" />
            </div>
            {matrix.map((row, i) => (
              <div key={i} className="flex items-center">
                <div className="w-16 text-right pr-2 text-caption font-semibold text-on-surface-variant dark:text-outline shrink-0">
                  {r.tokens[i]}
                </div>
                {row.map((v, j) => {
                  const masked = r.causal && j > i;
                  const isHover = hover?.row === i && hover?.col === j;
                  return (
                    <div
                      key={j}
                      onMouseEnter={() => setHover({ row: i, col: j })}
                      onMouseLeave={() => setHover(null)}
                      className={`w-12 h-12 md:w-14 md:h-14 m-0.5 rounded-lg flex items-center justify-center font-mono text-caption transition-transform ${isHover ? "ring-2 ring-primary dark:ring-inverse-primary scale-105 z-10" : ""}`}
                      style={{
                        background: masked
                          ? "repeating-linear-gradient(45deg, rgba(125,118,109,0.12), rgba(125,118,109,0.12) 4px, transparent 4px, transparent 8px)"
                          : cellColor(v),
                        color:
                          view === "weights"
                            ? v > 0.55
                              ? "#fff"
                              : "#3B3023"
                            : "#3B3023",
                      }}
                      title={`${r.tokens[i]} → ${r.tokens[j]}: ${v.toFixed(3)}${masked ? " (masked)" : ""}`}
                    >
                      {masked ? "" : v.toFixed(2)}
                    </div>
                  );
                })}
                {/* entropy bar */}
                <div className="ml-3 w-16 h-2 bg-surface-variant dark:bg-white/10 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-primary dark:bg-inverse-primary rounded-full"
                    style={{ width: `${(r.entropy[i] / r.maxEntropy) * 100}%` }}
                  />
                </div>
              </div>
            ))}
            <div className="flex items-center mt-2 ml-16">
              <span className="text-caption text-outline w-[calc(100%-4rem)]">
                ← query attends over keys →
              </span>
              <span className="w-16 text-caption text-outline text-center">
                focus
              </span>
            </div>
          </div>
        </div>

        {hover && !(r.causal && hover.col > hover.row) && (
          <p className="mt-3 text-caption text-on-surface-variant dark:text-outline">
            <span className="font-mono text-primary dark:text-inverse-primary">
              {r.tokens[hover.row]}
            </span>
            {" assigns weight "}
            <span className="font-mono font-semibold">
              {r.weights[hover.row][hover.col].toFixed(3)}
            </span>
            {" to "}
            <span className="font-mono text-primary dark:text-inverse-primary">
              {r.tokens[hover.col]}
            </span>
          </p>
        )}
      </div>

      {/* Hover detail: Q/K vectors for the query row */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <VectorCard
          title="Query vector"
          name="Q"
          tokens={r.tokens}
          row={hover?.row}
          vectors={r.Q}
          d={r.d}
        />
        <VectorCard
          title="Key vector"
          name="K"
          tokens={r.tokens}
          row={hover?.col}
          vectors={r.K}
          d={r.d}
        />
      </div>
    </div>
  );
}

function VectorCard({
  title,
  name,
  tokens,
  row,
  vectors,
  d,
}: {
  title: string;
  name: string;
  tokens: string[];
  row?: number;
  vectors: number[][];
  d: number;
}) {
  const idx = row ?? 0;
  const vec = vectors[idx] ?? [];
  const vmax = Math.max(...vec.map(Math.abs), 0.01);
  return (
    <div className="bg-surface-container-lowest dark:bg-dark-surface rounded-2xl p-5 border border-outline-variant/40 dark:border-white/10">
      <h4 className="font-label-md text-label-md uppercase tracking-wider text-on-surface dark:text-dark-on-surface mb-3">
        {title}{" "}
        <span className="font-mono text-primary dark:text-inverse-primary normal-case">
          {name}
          {idx}
        </span>
        <span className="float-right text-caption normal-case text-outline">
          {tokens[idx]}
        </span>
      </h4>
      <div className="flex items-end gap-1 h-20">
        {Array.from({ length: d }).map((_, k) => {
          const v = vec[k] ?? 0;
          const h = (Math.abs(v) / vmax) * 100;
          return (
            <div
              key={k}
              className="flex-1 flex flex-col justify-end"
              title={`${name}[${k}]=${v.toFixed(3)}`}
            >
              <div
                className="rounded-t"
                style={{
                  height: `${h}%`,
                  background: v >= 0 ? "#7A5C36" : "#C8604A",
                  opacity: 0.5 + (Math.abs(v) / vmax) * 0.5,
                }}
              />
            </div>
          );
        })}
      </div>
    </div>
  );
}
