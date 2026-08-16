import { useMemo, useState } from "react";
import type { LabResult, LabParams } from "../types";
import { useI18n } from "../../i18n/context";
import { labTextsZh, labTextsEn, fmt } from "../../i18n/lab";
import { ExplainBox } from "../Journal";
import { VerificationPanel } from "../VerificationPanel";
import { verificationData } from "../verification";
import { NotesPanel } from "../NotesPanel";
import { QuestList, type Quest } from "../QuestList";

interface Prediction {
  token: string;
  prob: number;
}

interface Suspect {
  type: "head" | "feature" | "ffn" | "residual";
  name: string;
  score: number;
  detail: string;
}

interface FeatureInfo {
  name: string;
  activated: string[];
  strength: number;
  hypothesis: string;
}

interface DetectiveResult extends LabResult {
  case_id: string;
  sentence: string[];
  target: string;
  question: string;
  prediction: Prediction[];
  head_rows: Record<string, Record<string, number>>;
  head_rows_full: Record<string, Record<string, Record<string, number>>>;
  features: FeatureInfo[];
  ranking: Suspect[];
  explanation: string;
  explanation_en: string;
  n_cases: number;
  provenance: string;
}

export default function DetectiveLab({
  result,
  loading,
  params,
  onRecord,
}: {
  result: LabResult | null;
  loading: boolean;
  error: string | null;
  onAction: (k: string) => void;
  params: LabParams;
  setParams: (p: LabParams) => void;
  onRecord?: (entry: {
    question: string;
    prediction: string;
    correct: boolean;
    evidence: string;
    params: LabParams;
  }) => void;
}) {
  const { lang } = useI18n();
  const texts = (lang === "zh" ? labTextsZh : labTextsEn)[
    "transformer-detective"
  ];
  const r = result as DetectiveResult | null;
  const [answer, setAnswer] = useState<string | null>(null);
  const [submitted, setSubmitted] = useState(false);
  const [suspectIdx, setSuspectIdx] = useState(0);
  // collected evidence: set of suspect indices the user actively investigated
  const [collected, setCollected] = useState<Set<number>>(new Set());
  // attention trace: path of token indices the user clicked through
  const [trace, setTrace] = useState<number[] | null>(null);

  const correct = useMemo(() => {
    if (!r) return false;
    const top = r.ranking[0];
    return (
      answer ===
      (top.type === "feature" ? "feature" : top.type === "ffn" ? "ffn" : "head")
    );
  }, [r, answer]);

  if (!r) {
    return (
      <div className="p-10 text-on-surface-variant dark:text-outline">
        {loading ? texts.ui.computing : texts.ui.adjustControls}
      </div>
    );
  }

  const tokens = r.sentence;
  const targetIdx = tokens.indexOf(r.target);
  const focusIdx =
    (trace && trace.length > 0 ? trace[trace.length - 1] : targetIdx) ??
    targetIdx;

  const top = r.ranking[0];
  const suspect = r.ranking[Math.min(suspectIdx, r.ranking.length - 1)];
  const maxProb = Math.max(...r.prediction.map((p) => p.prob));
  const expl = lang === "zh" ? r.explanation : r.explanation_en;
  const maxScore = r.ranking[0].score || 1e-9;

  // attention row for the focused token under the selected head
  const headRow: { srcIdx: number; src: string; w: number }[] | null =
    suspect.type === "head" && r.head_rows_full[suspect.name]
      ? Object.entries(r.head_rows_full[suspect.name][String(focusIdx)] ?? {})
          .map(([sj, w]) => ({
            srcIdx: Number(sj),
            src: tokens[Number(sj)] ?? "?",
            w,
          }))
          .sort((a, b) => b.w - a.w)
      : null;

  const allCollected = collected.size >= r.ranking.length;

  // --- Exploration quests (derived from computed result + investigation) --
  const quests: Quest[] = [
    { id: "collect", label: texts.quests![0], done: allCollected },
    {
      id: "close",
      label: texts.quests![1],
      done: allCollected && submitted && correct,
    },
    { id: "trace", label: texts.quests![2], done: (trace?.length ?? 0) >= 2 },
  ];
  const collect = (i: number) => {
    setSuspectIdx(i);
    setCollected((prev) => new Set(prev).add(i));
    setTrace(null);
  };
  const traceTo = (srcIdx: number) => {
    setTrace((prev) => [...(prev ?? [targetIdx]), srcIdx]);
  };

  return (
    <div className="flex flex-col gap-5">
      {/* Exploration quests */}
      <QuestList
        quests={quests}
        labId="transformer-detective"
        onRecord={onRecord}
        params={params}
        evidence={`top suspect ${r.ranking[0].name} (${r.ranking[0].score.toFixed(2)})`}
      />

      {/* Question layer */}
      <div className="bg-surface-container-lowest dark:bg-dark-surface rounded-3xl p-5 border border-outline-variant/40 dark:border-white/10">
        <div className="flex items-start gap-3">
          <span className="text-xl mt-0.5">🔍</span>
          <div>
            <div className="text-caption uppercase tracking-wider font-semibold text-outline mb-1">
              {texts.ui.question}
            </div>
            <p className="font-headline text-lg text-on-surface dark:text-inverse-on-surface leading-snug">
              {texts.question}
            </p>
          </div>
        </div>
      </div>

      {/* Case file: sentence (click a token to trace IT) + prediction */}
      <div className="bg-surface-container-lowest dark:bg-dark-surface rounded-3xl p-4 md:p-6 shadow-ambient dark:shadow-dark-ambient border border-outline-variant/40 dark:border-white/10">
        <div className="flex flex-wrap items-center justify-between gap-3 mb-3">
          <h3 className="font-headline text-lg text-on-surface dark:text-inverse-on-surface inline-flex items-center gap-2">
            <span
              className="material-symbols-outlined"
              style={{ fontSize: 20 }}
            >
              search
            </span>
            {texts.ui.title}
          </h3>
          <span className="text-caption text-outline font-mono">
            case: {r.case_id}
          </span>
        </div>

        <div className="flex flex-wrap gap-1.5 mb-1">
          {tokens.map((w, i) => (
            <button
              key={i}
              onClick={() => traceTo(i)}
              className={`px-2 py-1 rounded-lg text-body-md transition-all ${
                i === targetIdx
                  ? "bg-primary text-on-primary dark:bg-inverse-primary dark:text-inverse-surface font-bold"
                  : i === focusIdx
                    ? "ring-2 ring-primary dark:ring-inverse-primary bg-surface-container dark:bg-white/5 text-on-surface dark:text-dark-on-surface"
                    : "bg-surface-container dark:bg-white/5 text-on-surface dark:text-dark-on-surface hover:bg-surface-variant dark:hover:bg-white/10"
              }`}
              title={`${texts.ui.trace}: ${w}`}
            >
              {w}
            </button>
          ))}
        </div>
        <p className="text-caption text-on-surface-variant dark:text-outline mb-4">
          {texts.ui.traceHint}
        </p>

        <div className="text-caption uppercase tracking-wider font-semibold text-outline mb-2">
          {texts.ui.prediction}
        </div>
        <div className="flex flex-col gap-1.5">
          {r.prediction.map((p) => {
            const isTop = p.token === r.prediction[0].token;
            return (
              <div key={p.token} className="flex items-center gap-3">
                <span
                  className={`w-20 text-right text-caption font-semibold truncate ${isTop ? "text-primary dark:text-inverse-primary" : "text-on-surface dark:text-dark-on-surface"}`}
                >
                  {p.token}
                </span>
                <div className="flex-1 h-4 bg-surface-container dark:bg-white/5 rounded-md overflow-hidden">
                  <div
                    className="h-full rounded-md bg-primary dark:bg-inverse-primary transition-all duration-300"
                    style={{
                      width: `${(p.prob / maxProb) * 100}%`,
                      opacity: isTop ? 0.9 : 0.45,
                    }}
                  />
                </div>
                <span className="w-12 text-right font-mono text-caption text-on-surface-variant dark:text-outline tabular-nums">
                  {p.prob.toFixed(2)}
                </span>
              </div>
            );
          })}
        </div>
      </div>

      {/* Investigation: suspects (collect evidence) + evidence (trace) */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Suspects with collect buttons + progress */}
        <div className="bg-surface-container-lowest dark:bg-dark-surface rounded-3xl p-4 md:p-6 shadow-ambient dark:shadow-dark-ambient border border-outline-variant/40 dark:border-white/10">
          <div className="flex items-center justify-between gap-2 mb-3">
            <h3 className="font-headline text-lg text-on-surface dark:text-inverse-on-surface inline-flex items-center gap-2">
              <span
                className="material-symbols-outlined"
                style={{ fontSize: 20 }}
              >
                policy
              </span>
              {texts.ui.suspects}
            </h3>
            {/* collection progress */}
            <span
              className={`inline-flex items-center gap-1.5 text-caption font-semibold px-2.5 py-1 rounded-full border ${
                allCollected
                  ? "text-[#2f6b3e] border-[#2f6b3e]/40 bg-[#2f6b3e]/10"
                  : "text-on-surface-variant dark:text-outline border-outline-variant/50 dark:border-white/10"
              }`}
            >
              {allCollected ? "✓ " : ""}
              {fmt(texts.ui.progress, {
                a: collected.size,
                n: r.ranking.length,
              })}
            </span>
          </div>
          <div className="flex flex-col gap-2">
            {r.ranking.map((s, i) => {
              const active = i === suspectIdx;
              const done = collected.has(i);
              return (
                <div
                  key={s.name}
                  className={`rounded-2xl px-4 py-3 border transition-all ${
                    active
                      ? "border-primary/50 dark:border-inverse-primary/50 bg-primary/10 dark:bg-inverse-primary/10"
                      : "border-outline-variant/40 dark:border-white/10 bg-surface-container dark:bg-white/5"
                  }`}
                >
                  <div className="flex items-center gap-2 mb-0.5">
                    <span
                      className={`text-label-md font-bold ${i === 0 ? "text-[#C8604A]" : "text-on-surface dark:text-dark-on-surface"}`}
                    >
                      {i === 0 ? "🥇 " : `${i + 1}. `}
                      {s.name}
                    </span>
                    <span className="ml-auto font-mono text-caption text-outline">
                      {s.type === "feature"
                        ? texts.ui.feature
                        : s.type === "ffn"
                          ? texts.ui.ffn
                          : texts.ui.head}
                    </span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="flex-1 h-2 bg-surface-container dark:bg-white/10 rounded-full overflow-hidden">
                      <div
                        className="h-full rounded-full bg-[#C8604A] transition-all duration-300"
                        style={{
                          width: `${(s.score / maxScore) * 100}%`,
                          opacity: 0.6 + 0.4 * (s.score / maxScore),
                        }}
                      />
                    </div>
                    <span className="font-mono text-caption text-primary dark:text-inverse-primary">
                      {s.score.toFixed(2)}
                    </span>
                    <button
                      onClick={() => collect(i)}
                      className={`inline-flex items-center gap-1 px-3 py-1.5 rounded-lg text-label-md font-semibold border transition-all ${
                        done
                          ? "bg-[#2f6b3e]/15 text-[#2f6b3e] dark:text-[#9ed0a8] border-[#2f6b3e]/40 cursor-default"
                          : "bg-primary text-on-primary dark:bg-inverse-primary dark:text-inverse-surface border-transparent hover:opacity-90"
                      }`}
                    >
                      <span
                        className="material-symbols-outlined"
                        style={{ fontSize: 15 }}
                      >
                        {done ? "check" : "search"}
                      </span>
                      {done ? texts.ui.collected : texts.ui.investigate}
                    </button>
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Evidence for the selected suspect + trace chain */}
        <div className="bg-surface-container-lowest dark:bg-dark-surface rounded-3xl p-4 md:p-6 shadow-ambient dark:shadow-dark-ambient border border-outline-variant/40 dark:border-white/10">
          <div className="flex items-center justify-between gap-2 mb-1">
            <h3 className="font-headline text-lg text-on-surface dark:text-inverse-on-surface inline-flex items-center gap-2">
              <span
                className="material-symbols-outlined"
                style={{ fontSize: 20 }}
              >
                fingerprint
              </span>
              {texts.ui.evidence}
            </h3>
            <button
              onClick={() => setTrace(null)}
              className="inline-flex items-center gap-1 text-caption text-on-surface-variant dark:text-outline hover:text-primary dark:hover:text-inverse-primary transition"
            >
              <span
                className="material-symbols-outlined"
                style={{ fontSize: 14 }}
              >
                home
              </span>
              {texts.ui.backToTarget}
            </button>
          </div>
          <div className="text-caption text-outline mb-1 font-mono">
            {suspect.name} · {suspect.detail}
          </div>

          {/* trace breadcrumb */}
          <div className="flex flex-wrap items-center gap-1 mb-3 text-caption">
            {texts.ui.trace}:{" "}
            {(trace ?? [targetIdx]).map((ti, i) => (
              <span key={i} className="inline-flex items-center gap-1">
                {i > 0 && <span className="text-outline">→</span>}
                <button
                  onClick={() =>
                    setTrace((trace ?? [targetIdx]).slice(0, i + 1))
                  }
                  className={`px-2 py-0.5 rounded-md font-mono ${
                    ti === targetIdx
                      ? "bg-primary/15 text-primary dark:text-inverse-primary"
                      : "bg-surface-container dark:bg-white/10 text-on-surface dark:text-dark-on-surface"
                  }`}
                >
                  {tokens[ti] ?? "?"}
                </button>
              </span>
            ))}
          </div>

          {headRow ? (
            <div className="flex flex-col gap-1.5">
              {headRow.map(({ srcIdx, src, w }) => (
                <div key={srcIdx} className="flex items-center gap-3">
                  <button
                    onClick={() => traceTo(srcIdx)}
                    className={`w-16 text-right text-caption font-semibold truncate rounded transition ${
                      srcIdx === focusIdx
                        ? "text-[#2f6b3e] dark:text-[#9ed0a8]"
                        : "text-on-surface dark:text-dark-on-surface hover:text-primary dark:hover:text-inverse-primary"
                    }`}
                    title={`${texts.ui.trace}: ${src}`}
                  >
                    {src}
                  </button>
                  <div className="flex-1 h-4 bg-surface-container dark:bg-white/5 rounded-md overflow-hidden">
                    <div
                      className="h-full rounded-md bg-[#5B6BB0] transition-all duration-300"
                      style={{
                        width: `${Math.max(w * 100, 1)}%`,
                        opacity: 0.35 + w * 0.65,
                      }}
                    />
                  </div>
                  <span className="w-12 text-right font-mono text-caption text-on-surface-variant dark:text-outline tabular-nums">
                    {w.toFixed(2)}
                  </span>
                </div>
              ))}
              <p className="mt-2 text-caption text-on-surface-variant dark:text-outline">
                ← {tokens[focusIdx]} {texts.ui.attendsOver}（
                {texts.ui.traceHint2}）
              </p>
            </div>
          ) : suspect.type === "feature" ? (
            <div className="flex flex-col gap-2">
              <div className="flex flex-wrap gap-1.5">
                {r.features
                  .filter((f) => f.name === suspect.name)
                  .flatMap((f) => f.activated)
                  .map((tok, i) => (
                    <span
                      key={i}
                      className="px-2 py-1 rounded-lg bg-[#C8604A]/15 text-[#C8604A] dark:text-[#f0b3a4] text-caption font-semibold"
                    >
                      {tok}
                    </span>
                  ))}
              </div>
              <p className="text-caption text-on-surface-variant dark:text-outline">
                {r.features.find((f) => f.name === suspect.name)?.hypothesis}
              </p>
            </div>
          ) : (
            <p className="text-caption text-on-surface-variant dark:text-outline">
              {suspect.detail}
            </p>
          )}
        </div>
      </div>

      {/* CASE CLOSED — gated until all evidence is collected */}
      <div
        className={`rounded-3xl p-4 md:p-6 border transition-all ${
          allCollected
            ? "border-[#8a3a35]/40 dark:border-white/10"
            : "border-outline-variant/40 dark:border-white/10 opacity-80"
        } bg-surface-container-lowest dark:bg-dark-surface`}
      >
        <div className="flex flex-wrap items-center gap-3 mb-2">
          <span className="text-xl">{allCollected ? "⚖️" : "🔒"}</span>
          <h3 className="font-headline text-lg text-on-surface dark:text-inverse-on-surface inline-flex items-center gap-2">
            <span
              className="material-symbols-outlined"
              style={{ fontSize: 20 }}
            >
              gavel
            </span>
            {texts.ui.caseClosed}
          </h3>
          {!allCollected && (
            <span className="text-caption text-on-surface-variant dark:text-outline ml-auto">
              {fmt(texts.ui.locked, { a: collected.size, n: r.ranking.length })}
            </span>
          )}
        </div>

        {allCollected ? (
          <>
            <div className="mb-3 text-caption text-outline font-mono">
              {fmt(texts.ui.verdictEvidence, {
                s: top.name,
                sc: top.score.toFixed(2),
              })}
            </div>
            <p className="text-body-md text-on-surface dark:text-inverse-on-surface leading-relaxed">
              {expl}
            </p>
            <div className="mt-3 text-caption text-on-surface-variant dark:text-outline">
              {texts.ui.yourEvidence}:{" "}
              {r.ranking
                .map((s, i) => (collected.has(i) ? s.name : null))
                .filter(Boolean)
                .join(" · ")}
            </div>
          </>
        ) : (
          <p className="text-caption text-on-surface-variant dark:text-outline">
            {texts.ui.lockedHint}
          </p>
        )}
      </div>

      {/* Challenge layer — independent of Controls */}
      <div className="bg-surface-container-lowest dark:bg-dark-surface rounded-3xl p-5 border border-outline-variant/40 dark:border-white/10">
        <div className="flex items-center gap-2 mb-1">
          <span className="text-lg">🧪</span>
          <h3 className="font-label-md text-label-md uppercase tracking-wider text-on-surface dark:text-dark-on-surface">
            {texts.ui.makePrediction}
          </h3>
        </div>
        <p className="text-body-md text-on-surface dark:text-inverse-on-surface mb-3">
          {texts.challengeQuestion}
        </p>
        <div className="flex flex-wrap gap-2 mb-3">
          {texts.challengeOptions.map((opt) => (
            <button
              key={opt.value}
              onClick={() => {
                setAnswer(opt.value);
                setSubmitted(false);
              }}
              className={`px-4 py-2 rounded-xl text-label-md font-semibold border transition-all ${
                answer === opt.value
                  ? "bg-primary text-on-primary border-primary dark:bg-inverse-primary dark:text-inverse-surface"
                  : "bg-surface-container-lowest dark:bg-dark-surface text-on-surface-variant dark:text-outline border-outline-variant/60 dark:border-white/10 hover:border-primary/50"
              }`}
            >
              {opt.label}
            </button>
          ))}
        </div>
        <button
          disabled={answer === null}
          onClick={() => {
            setSubmitted(true);
            if (onRecord && r) {
              const label =
                texts.challengeOptions.find((o) => o.value === answer)?.label ??
                answer ??
                "";
              const top = r.ranking[0];
              onRecord({
                question: texts.challengeQuestion,
                prediction: label,
                correct,
                evidence: `top suspect ${top.name} (${top.score.toFixed(2)})`,
                params: { ...params },
              });
            }
          }}
          className="inline-flex items-center gap-2 px-5 py-2 rounded-xl bg-on-surface text-on-primary dark:bg-inverse-surface dark:text-inverse-surface font-label-md text-label-md hover:opacity-90 transition disabled:opacity-40"
        >
          {texts.ui.runExperiment}
        </button>

        {submitted && answer !== null && (
          <div
            className={`mt-4 rounded-2xl p-4 border ${correct ? "border-[#2f6b3e]/40" : "border-[#C8604A]/40"} bg-surface-container dark:bg-white/5`}
          >
            <div className="flex items-center gap-2 mb-2">
              <span className="text-lg">{correct ? "✅" : "❌"}</span>
              <span
                className={`font-label-md font-bold ${correct ? "text-[#2f6b3e] dark:text-[#9ed0a8]" : "text-[#C8604A] dark:text-[#f0b3a4]"}`}
              >
                {correct ? texts.ui.correct : texts.ui.notQuite}
              </span>
              <span className="text-caption text-outline ml-auto font-mono">
                {fmt(texts.ui.verdictEvidence, {
                  s: top.name,
                  sc: top.score.toFixed(2),
                })}
              </span>
            </div>
            <ul className="text-caption text-on-surface-variant dark:text-outline space-y-1">
              {texts.explanation.map((line) => (
                <li key={line}>• {line}</li>
              ))}
            </ul>
          </div>
        )}
      </div>

      {/* Knowledge verification — multi-source */}
      <VerificationPanel entry={verificationData["transformer-detective"]} />

      {/* Related notes */}
      <NotesPanel notes={texts.notes} />

      {/* L3 Explain — learner-owned, never machine-graded */}
      <ExplainBox
        onRecord={onRecord}
        evidence={`top suspect ${r.ranking[0].name} (${r.ranking[0].score.toFixed(2)})`}
        params={params}
      />

      {/* Related Notes */}
      <div className="bg-surface-container-lowest dark:bg-dark-surface rounded-3xl p-5 border border-outline-variant/40 dark:border-white/10">
        <h3 className="font-label-md text-label-md uppercase tracking-wider text-on-surface dark:text-dark-on-surface mb-3">
          📚 {texts.ui.relatedNotes}
        </h3>
        <div className="flex flex-col gap-2">
          {texts.notes.map((n) => (
            <div
              key={n.src}
              className="flex items-center gap-3 px-4 py-2.5 rounded-xl bg-surface-container dark:bg-white/5 border border-outline-variant/40 dark:border-white/10"
            >
              <span
                className="material-symbols-outlined text-outline"
                style={{ fontSize: 18 }}
              >
                menu_book
              </span>
              <div className="min-w-0">
                <div className="text-body-md text-on-surface dark:text-dark-on-surface truncate">
                  {n.title}
                </div>
                <div className="text-caption text-outline font-mono truncate">
                  {n.src}
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
