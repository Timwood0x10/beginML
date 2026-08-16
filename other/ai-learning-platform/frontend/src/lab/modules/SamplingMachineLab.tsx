import { useEffect, useMemo, useRef, useState } from "react";
import type { LabResult, LabParams } from "../types";
import { useI18n } from "../../i18n/context";
import { labTextsZh, labTextsEn, fmt } from "../../i18n/lab";
import { QuestList, type Quest } from "../QuestList";
import { VerificationPanel } from "../VerificationPanel";
import { verificationData } from "../verification";
import { NotesPanel } from "../NotesPanel";

interface SamplingResult extends LabResult {
  tokens: string[];
  raw_logits: number[];
  scaled_logits: number[];
  probs: number[];
  mask: boolean[];
  filtered_probs: number[];
  counts: number[];
  samples: number;
  temperature: number;
  gate: string;
  top_k: number;
  top_p: number;
  seed: number;
  entropy: number;
  max_entropy: number;
  provenance: string;
}

// Frozen pipeline (see plan §10.1): temperature rescales the softmax INPUT
// as logits/T — the raw logits themselves never change.
const PIPELINE = [
  { icon: "tune", label: "LOGITS", sub: "raw" },
  { icon: "thermostat", label: "TEMPERATURE", sub: "÷ T" },
  { icon: "functions", label: "SOFTMAX", sub: "prob" },
  { icon: "filter_alt", label: "FILTER GATE", sub: "top-k / top-p" },
  { icon: "casino", label: "SAMPLE", sub: "🎲" },
];

const CHALLENGE_CORRECT = "diverse";

export default function SamplingMachineLab({
  result,
  loading,
  onAction,
  onRecord,
  params,
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
    insight?: string;
  }) => void;
}) {
  const { lang } = useI18n();
  const texts = (lang === "zh" ? labTextsZh : labTextsEn)["sampling-machine"];
  const r = result as SamplingResult | null;
  const [answer, setAnswer] = useState<string | null>(null);
  const [submitted, setSubmitted] = useState(false);
  const correct = answer === CHALLENGE_CORRECT;

  // --- L2 Manipulate Challenge ------------------------------------------
  // Goal picked by the learner; success is DERIVED from the current result
  // (entropy is a consequence of the controls, never set directly). The
  // learner must tune temperature / gate / top-k / top-p until it passes.
  const [l2Goal, setL2Goal] = useState<"diverse" | "concentrated" | null>(null);
  const recordedL2 = useRef<string | null>(null);

  // --- L3 Explain: learner's own words (No-LLM, no auto-judgment) --------
  const [l3Draft, setL3Draft] = useState("");
  const [l3Saved, setL3Saved] = useState(false);

  const l2Achieved =
    l2Goal === "diverse"
      ? (r?.entropy ?? 0) >= 0.8 * (r?.max_entropy ?? 1)
      : l2Goal === "concentrated"
        ? (r?.entropy ?? 0) <= 0.25 * (r?.max_entropy ?? 1)
        : false;

  // --- Exploration quests (derived from computed result + L2 state) ------
  const quests = useMemo<Quest[]>(() => {
    if (!r) return [];
    return [
      {
        id: "diverse",
        label: texts.quests![0],
        done: r.entropy >= 0.8 * r.max_entropy,
      },
      {
        id: "peaked",
        label: texts.quests![1],
        done: r.entropy <= 0.3 * r.max_entropy,
      },
      {
        id: "l2",
        label: texts.quests![2],
        done: l2Goal !== null && l2Achieved,
      },
    ];
  }, [r, texts, l2Goal, l2Achieved]);

  // Record the achievement into the Journal exactly once per goal.
  useEffect(() => {
    if (r && l2Goal && l2Achieved && recordedL2.current !== l2Goal) {
      recordedL2.current = l2Goal;
      onRecord?.({
        question:
          l2Goal === "diverse" ? texts.ui.l2Diverse : texts.ui.l2Concentrated,
        prediction: "manual tuning",
        correct: true,
        evidence: `entropy=${r.entropy.toFixed(2)}bits, T=${r.temperature}`,
        params: { ...params },
      });
    }
  }, [l2Goal, l2Achieved, r, onRecord, texts, params]);

  if (!r) {
    return (
      <div className="p-10 text-on-surface-variant dark:text-outline">
        {loading ? texts.ui.computing : texts.ui.adjustControls}
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-5">
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

      {/* Exploration quests */}
      <QuestList
        quests={quests}
        labId="sampling-machine"
        onRecord={onRecord}
        params={params}
        evidence={`entropy=${r.entropy.toFixed(2)}bits, T=${r.temperature}`}
      />

      {/* Pipeline */}
      <div className="bg-surface-container-lowest dark:bg-dark-surface rounded-3xl p-5 shadow-ambient dark:shadow-dark-ambient border border-outline-variant/40 dark:border-white/10">
        <div className="flex items-center justify-between gap-2 mb-4 overflow-x-auto pb-1">
          {PIPELINE.map((stage, i) => (
            <div key={stage.label} className="flex items-center gap-2 shrink-0">
              {i > 0 && <span className="text-outline">→</span>}
              <div
                className={`flex flex-col items-center gap-0.5 px-3 py-2 rounded-xl border ${
                  stage.label === "SAMPLE"
                    ? "bg-primary/10 dark:bg-inverse-primary/10 border-primary/40 dark:border-inverse-primary/40"
                    : "bg-surface-container dark:bg-white/5 border-outline-variant/50 dark:border-white/10"
                }`}
              >
                <span
                  className="material-symbols-outlined text-primary dark:text-inverse-primary"
                  style={{ fontSize: 18 }}
                >
                  {stage.icon}
                </span>
                <span className="text-[10px] font-bold tracking-wider text-on-surface dark:text-dark-on-surface">
                  {stage.label}
                </span>
                <span className="text-[9px] text-outline">{stage.sub}</span>
              </div>
            </div>
          ))}
        </div>

        {/* Bar charts: raw -> temperature -> softmax -> gate */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
          <ChartBlock
            title={texts.ui.rawLogits}
            hint={texts.ui.rawLogitsHint}
            color="#C8604A"
          >
            <Bars values={r.raw_logits} labels={r.tokens} color="#C8604A" />
          </ChartBlock>
          <ChartBlock
            title={texts.ui.afterTemp}
            hint={fmt(texts.ui.afterTempHint, { t: r.temperature })}
            color="#7A5C36"
          >
            <Bars values={r.scaled_logits} labels={r.tokens} color="#7A5C36" />
          </ChartBlock>
          <ChartBlock
            title={texts.ui.softmax}
            hint={fmt(texts.ui.softmaxHint, {
              e: r.entropy.toFixed(2),
              m: r.max_entropy.toFixed(2),
            })}
            color="#5B6BB0"
          >
            <Bars values={r.probs} labels={r.tokens} color="#5B6BB0" />
          </ChartBlock>
          <ChartBlock
            title={texts.ui.gate}
            hint={
              r.gate === "none"
                ? texts.ui.gateHintNone
                : fmt(texts.ui.gateHint, {
                    k: r.top_k,
                    p: r.top_p,
                    kept: r.mask.filter(Boolean).length,
                    total: r.tokens.length,
                  })
            }
            color="#2f6b3e"
          >
            <div className="flex items-end gap-1">
              {r.tokens.map((t, i) => {
                const kept = r.mask[i];
                const v = r.filtered_probs[i];
                const max = Math.max(...r.filtered_probs, 1e-9);
                return (
                  <div
                    key={t}
                    className="flex-1 flex flex-col justify-end items-center min-w-0"
                    title={`${t}: ${kept ? v.toFixed(3) : "filtered out"}`}
                  >
                    <div
                      className="w-full rounded-t transition-all duration-300"
                      style={{
                        height: `${Math.max((v / max) * 100, 1)}%`,
                        background: kept
                          ? "#2f6b3e"
                          : "repeating-linear-gradient(45deg, rgba(125,118,109,0.15), rgba(125,118,109,0.15) 3px, transparent 3px, transparent 6px)",
                        opacity: kept ? 0.75 : 0.4,
                      }}
                    />
                  </div>
                );
              })}
            </div>
            <div className="flex mt-1">
              {r.tokens.map((t) => (
                <span
                  key={t}
                  className="flex-1 text-center text-[9px] text-outline truncate"
                >
                  {t}
                </span>
              ))}
            </div>
            <div className="mt-2 text-[10px] text-outline text-center">
              {texts.ui.gateCaption}
            </div>
          </ChartBlock>
        </div>
      </div>

      {/* Sampler */}
      <div className="bg-surface-container-lowest dark:bg-dark-surface rounded-3xl p-5 shadow-ambient dark:shadow-dark-ambient border border-outline-variant/40 dark:border-white/10">
        <div className="flex flex-wrap items-center justify-between gap-3 mb-4">
          <h3 className="font-headline text-lg text-on-surface dark:text-inverse-on-surface inline-flex items-center gap-2">
            <span
              className="material-symbols-outlined"
              style={{ fontSize: 20 }}
            >
              casino
            </span>
            {texts.ui.theSampler}
          </h3>
          <button
            onClick={() => onAction("sample")}
            className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl bg-primary text-on-primary dark:bg-inverse-primary dark:text-inverse-surface font-label-md text-label-md hover:opacity-90 transition shadow-sm"
          >
            <span
              className="material-symbols-outlined"
              style={{ fontSize: 18 }}
            >
              casino
            </span>
            {fmt(texts.ui.sampleButton, { seed: r.seed })}
          </button>
        </div>
        <BallSlots tokens={r.tokens} counts={r.counts} seed={r.seed} />
        <div className="mt-3 flex flex-wrap gap-3 text-caption text-on-surface-variant dark:text-outline">
          <span>
            {texts.ui.seedLabel}{" "}
            <span className="font-mono text-primary dark:text-inverse-primary">
              {r.seed}
            </span>
          </span>
          <span>
            {texts.ui.drawsLabel} <span className="font-mono">{r.samples}</span>
          </span>
          <span>
            {texts.ui.simLabel}{" "}
            <span className="font-mono">{r.provenance}</span>
          </span>
        </div>
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
              onRecord({
                question: texts.challengeQuestion,
                prediction: label,
                correct: answer === CHALLENGE_CORRECT,
                evidence: `entropy=${r.entropy.toFixed(2)}bits, T=${r.temperature}`,
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
                {r.temperature > 1
                  ? fmt(texts.ui.tFlat, { t: r.temperature })
                  : fmt(texts.ui.tSharp, { t: r.temperature })}
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

      {/* L2 Manipulate Challenge — independent of Controls, independent of Predict */}
      <div className="bg-surface-container-lowest dark:bg-dark-surface rounded-3xl p-5 border border-outline-variant/40 dark:border-white/10">
        <div className="flex items-center gap-2 mb-1">
          <span className="text-lg">🎛️</span>
          <h3 className="font-label-md text-label-md uppercase tracking-wider text-on-surface dark:text-dark-on-surface">
            {texts.ui.l2Title}
          </h3>
        </div>
        <p className="text-body-md text-on-surface dark:text-inverse-on-surface mb-3">
          {texts.ui.l2Tagline}
        </p>

        <div className="flex flex-wrap gap-2 mb-3">
          <button
            onClick={() => {
              setL2Goal("diverse");
              recordedL2.current = null;
            }}
            className={`px-4 py-2 rounded-xl text-label-md font-semibold border transition-all ${
              l2Goal === "diverse"
                ? "bg-primary text-on-primary border-primary dark:bg-inverse-primary dark:text-inverse-surface"
                : "bg-surface-container-lowest dark:bg-dark-surface text-on-surface-variant dark:text-outline border-outline-variant/60 dark:border-white/10 hover:border-primary/50"
            }`}
          >
            {texts.ui.l2Diverse}
          </button>
          <button
            onClick={() => {
              setL2Goal("concentrated");
              recordedL2.current = null;
            }}
            className={`px-4 py-2 rounded-xl text-label-md font-semibold border transition-all ${
              l2Goal === "concentrated"
                ? "bg-primary text-on-primary border-primary dark:bg-inverse-primary dark:text-inverse-surface"
                : "bg-surface-container-lowest dark:bg-dark-surface text-on-surface-variant dark:text-outline border-outline-variant/60 dark:border-white/10 hover:border-primary/50"
            }`}
          >
            {texts.ui.l2Concentrated}
          </button>
        </div>

        {l2Goal && (
          <div>
            <div className="flex items-center gap-3 mb-2">
              <span className="text-caption text-on-surface-variant dark:text-outline">
                {texts.ui.l2Entropy}
              </span>
              <div className="flex-1 h-4 bg-surface-container dark:bg-white/5 rounded-md overflow-hidden">
                <div
                  className="h-full rounded-md bg-primary dark:bg-inverse-primary transition-all duration-300"
                  style={{
                    width: `${Math.min(100, (r.entropy / r.max_entropy) * 100)}%`,
                  }}
                />
              </div>
              <span className="font-mono text-caption text-on-surface-variant dark:text-outline tabular-nums">
                {r.entropy.toFixed(2)} / {r.max_entropy.toFixed(2)} bits
              </span>
            </div>
            <div
              className={`rounded-2xl p-3 border ${
                l2Achieved
                  ? "border-[#2f6b3e]/40 bg-[#2f6b3e]/10 text-[#2f6b3e] dark:text-[#9ed0a8]"
                  : "border-outline-variant/40 dark:border-white/10 bg-surface-container dark:bg-white/5 text-on-surface-variant dark:text-outline"
              }`}
            >
              <div className="text-label-md font-semibold mb-1">
                {l2Achieved ? texts.ui.l2Success : texts.ui.l2NotYet}
              </div>
              <div className="text-caption">{texts.ui.l2Hint}</div>
            </div>
            <button
              onClick={() => {
                setL2Goal(null);
                recordedL2.current = null;
              }}
              className="mt-3 inline-flex items-center gap-1.5 text-caption text-on-surface-variant dark:text-outline hover:text-primary dark:hover:text-inverse-primary transition"
            >
              <span
                className="material-symbols-outlined"
                style={{ fontSize: 15 }}
              >
                restart_alt
              </span>
              {texts.ui.l2Reset}
            </button>
          </div>
        )}
      </div>

      {/* Knowledge verification — multi-source */}
      <VerificationPanel entry={verificationData["sampling-machine"]} />

      {/* Related notes */}
      <NotesPanel notes={texts.notes} />

      {/* L3 Explain — independent, learner-owned, never machine-graded */}
      <div className="bg-surface-container-lowest dark:bg-dark-surface rounded-3xl p-5 border border-outline-variant/40 dark:border-white/10">
        <div className="flex items-center gap-2 mb-1">
          <span className="text-lg">✍️</span>
          <h3 className="font-label-md text-label-md uppercase tracking-wider text-on-surface dark:text-dark-on-surface">
            {texts.ui.l3Title}
          </h3>
        </div>
        <p className="text-body-md text-on-surface dark:text-inverse-on-surface mb-3">
          {texts.ui.l3Placeholder}
        </p>
        <textarea
          value={l3Draft}
          onChange={(ev) => {
            setL3Draft(ev.target.value);
            setL3Saved(false);
          }}
          rows={3}
          placeholder={texts.ui.l3Placeholder}
          className="w-full rounded-2xl px-4 py-3 bg-surface-container dark:bg-white/5 border border-outline-variant/50 dark:border-white/15 text-body-md text-on-surface dark:text-dark-on-surface placeholder:text-outline focus:outline-none focus:border-primary/50 resize-none"
        />
        <div className="mt-3 flex flex-wrap items-center gap-3">
          <button
            disabled={l3Draft.trim() === ""}
            onClick={() => {
              if (onRecord && r) {
                onRecord({
                  question: texts.ui.l3Title,
                  prediction: "",
                  correct: true,
                  evidence: `entropy=${r.entropy.toFixed(2)}bits, T=${r.temperature}`,
                  params: { ...params },
                  insight: l3Draft.trim(),
                });
                setL3Saved(true);
              }
            }}
            className="inline-flex items-center gap-2 px-5 py-2 rounded-xl bg-on-surface text-on-primary dark:bg-inverse-surface dark:text-inverse-surface font-label-md text-label-md hover:opacity-90 transition disabled:opacity-40"
          >
            <span
              className="material-symbols-outlined"
              style={{ fontSize: 17 }}
            >
              save
            </span>
            {texts.ui.l3Save}
          </button>
          {l3Saved && (
            <span className="text-caption text-[#2f6b3e] dark:text-[#9ed0a8]">
              {texts.ui.l3Saved}
            </span>
          )}
        </div>
      </div>

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

function ChartBlock({
  title,
  hint,
  color,
  children,
}: {
  title: string;
  hint: string;
  color: string;
  children: React.ReactNode;
}) {
  return (
    <div>
      <div className="flex items-baseline justify-between gap-2 mb-2">
        <h4
          className="text-[11px] font-bold tracking-wider text-on-surface dark:text-dark-on-surface"
          style={{ color }}
        >
          {title}
        </h4>
        <span className="text-[10px] text-outline text-right">{hint}</span>
      </div>
      {children}
    </div>
  );
}

function Bars({
  values,
  labels,
  color,
}: {
  values: number[];
  labels: string[];
  color: string;
}) {
  const max = Math.max(...values, 1e-9);
  const min = Math.min(...values, 0);
  const span = max - min || 1e-9;
  return (
    <div>
      <div className="flex items-end gap-1 h-28">
        {values.map((v, i) => (
          <div
            key={i}
            className="flex-1 flex justify-center items-end min-w-0"
            title={`${labels[i]}: ${v.toFixed(3)}`}
          >
            <div
              className="w-full max-w-6 rounded-t-sm transition-all duration-300"
              style={{
                height: `${((v - min) / span) * 100}%`,
                background: color,
                opacity: 0.7,
              }}
            />
          </div>
        ))}
      </div>
      <div className="flex mt-1">
        {labels.map((l) => (
          <span
            key={l}
            className="flex-1 text-center text-[9px] text-outline truncate"
          >
            {l}
          </span>
        ))}
      </div>
    </div>
  );
}

function BallSlots({
  tokens,
  counts,
  seed,
}: {
  tokens: string[];
  counts: number[];
  seed: number;
}) {
  return (
    <div className="flex gap-1">
      {tokens.map((t, i) => (
        <div key={t} className="flex-1 flex flex-col items-center min-w-0">
          <div className="relative w-full h-32 bg-surface-container dark:bg-white/5 rounded-lg overflow-hidden flex items-end">
            {Array.from({ length: Math.min(counts[i], 14) }).map((_, k) => (
              <div
                key={`${seed}-${k}`}
                className="absolute rounded-full"
                style={{
                  left: "50%",
                  marginLeft: -5,
                  bottom: 6 + k * 12,
                  width: 10,
                  height: 10,
                  background: "#C8604A",
                  animation: `ailearn-drop 0.5s ${k * 0.04}s ease-out both`,
                }}
              />
            ))}
          </div>
          <span className="text-[9px] text-outline mt-1 truncate w-full text-center">
            {t}
          </span>
          <span className="text-caption font-mono text-primary dark:text-inverse-primary tabular-nums">
            {counts[i]}
          </span>
        </div>
      ))}
    </div>
  );
}
