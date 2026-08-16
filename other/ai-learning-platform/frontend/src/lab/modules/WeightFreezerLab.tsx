import { useEffect, useMemo, useRef, useState } from "react";
import type { LabResult, LabParams } from "../types";
import { useI18n } from "../../i18n/context";
import { labTextsZh, labTextsEn, fmt } from "../../i18n/lab";
import { ExplainBox } from "../Journal";
import { VerificationPanel } from "../VerificationPanel";
import { verificationData } from "../verification";
import { NotesPanel } from "../NotesPanel";
import { QuestList, type Quest } from "../QuestList";

interface LevelInfo {
  label: string;
  bits: number;
  error: number;
  mse: number;
  memory_mb: number;
}

interface FreezerResult extends LabResult {
  points: number[][];
  quantized: number[][];
  current_label: string;
  current_bits: number;
  current_error: number;
  levels: LevelInfo[];
  bit_width: number;
  ternary: boolean;
  n_points: number;
  seed: number;
  provenance: string;
}

const LEVEL_COLORS = ["#5B6BB0", "#2f6b3e", "#7A5C36", "#C8604A", "#8a3a35"];

export default function WeightFreezerLab({
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
  const texts = (lang === "zh" ? labTextsZh : labTextsEn)["weight-freezer"];
  const r = result as FreezerResult | null;
  const [answer, setAnswer] = useState<string | null>(null);
  const [submitted, setSubmitted] = useState(false);

  // Evidence-based verdict: the lowest-bits level that stays near-lossless.
  const correct = useMemo(() => {
    if (!r) return false;
    const lossless = r.levels.filter(
      (l) => l.label !== "fp32" && l.error < 0.001,
    );
    const target = lossless.length
      ? lossless[lossless.length - 1].label
      : "ternary";
    return answer === target;
  }, [r, answer]);

  // --- L2 Manipulate Challenge ------------------------------------------
  // Success is DERIVED from the current quantization (error and level are
  // results, not controls). The learner must pick bit width / 1.58b mode
  // until the freezer lands in the target state.
  const [l2Goal, setL2Goal] = useState<"lossless" | "sweet" | null>(null);
  const recordedL2 = useRef<string | null>(null);

  const l2Achieved =
    l2Goal === "lossless"
      ? (r?.current_error ?? 1) < 0.001
      : l2Goal === "sweet"
        ? r?.current_label === "int4" || r?.current_label === "int8"
        : false;

  // Record the achievement into the Journal exactly once per goal.
  useEffect(() => {
    if (r && l2Goal && l2Achieved && recordedL2.current !== l2Goal) {
      recordedL2.current = l2Goal;
      onRecord?.({
        question:
          l2Goal === "lossless" ? texts.ui.l2Lossless : texts.ui.l2Sweet,
        prediction: "manual tuning",
        correct: true,
        evidence: `${r.current_label} err=${(r.current_error * 100).toFixed(2)}%`,
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

  // --- Exploration quests (derived from computed result) -----------------
  const quests: Quest[] = [
    {
      id: "lossless",
      label: texts.quests![0],
      done: r.current_label === "int8" && r.current_error < 0.001,
    },
    {
      id: "sweet",
      label: texts.quests![1],
      done: r.current_label === "int4" || r.current_label === "int8",
    },
    {
      id: "break",
      label: texts.quests![2],
      done: r.current_label === "ternary",
    },
  ];

  // --- weight cloud view -------------------------------------------------
  const W = 420,
    H = 340;
  const pad = 26;
  const all = [...r.points.flat(), ...r.quantized.flat()];
  const lo = Math.min(...all);
  const hi = Math.max(...all);
  const span = hi - lo || 1;
  const px = (v: number) => pad + ((v - lo) / span) * (W - 2 * pad);
  const py = (v: number) => pad + (1 - (v - lo) / span) * (H - 2 * pad);

  // --- BREAK IT staircase -----------------------------------------------
  const levelMap = new Map(r.levels.map((l) => [l.label, l]));
  const staircase = ["fp32", "int8", "int4", "int2", "ternary"];
  const curIdx = staircase.indexOf(r.current_label);

  // --- Pareto curve ------------------------------------------------------
  const PW = 420,
    PH = 220;
  const plo = 30,
    pt = 14,
    pr = 12,
    pb = 26;
  const bitsMax = 34;
  const errMin = 0.00001; // log floor
  const errMax = Math.max(...r.levels.map((l) => l.error), 0.01);
  const lpx = (b: number) => plo + (1 - b / bitsMax) * (PW - plo - pr); // bits descend left→right
  // Log y-scale. Clamp to [0,1]: log(0) is -Infinity (fp32 error is exactly
  // 0), so a zero-error level must snap to the top of the chart instead of
  // producing -Infinity coordinates.
  const lpy = (e: number) => {
    const safe = Math.max(e, errMin / 10);
    const t = Math.log(safe / errMax) / Math.log(errMin / errMax);
    return pt + (1 - Math.min(1, Math.max(0, t))) * (PH - pt - pb);
  };

  return (
    <div className="flex flex-col gap-5">
      {/* Exploration quests */}
      <QuestList
        quests={quests}
        labId="weight-freezer"
        onRecord={onRecord}
        params={params}
        evidence={`${r.current_label} err=${(r.current_error * 100).toFixed(2)}%`}
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

      {/* Weight cloud */}
      <div className="bg-surface-container-lowest dark:bg-dark-surface rounded-3xl p-4 md:p-6 shadow-ambient dark:shadow-dark-ambient border border-outline-variant/40 dark:border-white/10">
        <div className="flex flex-wrap items-center justify-between gap-3 mb-3">
          <h3 className="font-headline text-lg text-on-surface dark:text-inverse-on-surface inline-flex items-center gap-2">
            <span
              className="material-symbols-outlined"
              style={{ fontSize: 20 }}
            >
              ac_unit
            </span>
            {texts.ui.title}
          </h3>
          <div className="inline-flex items-center gap-3 text-caption text-on-surface-variant dark:text-outline">
            <span>
              {texts.ui.current}:{" "}
              <span className="font-mono text-primary dark:text-inverse-primary">
                {r.current_label}
              </span>
            </span>
            <span>
              {texts.ui.memory}:{" "}
              <span className="font-mono">{r.current_bits} bit</span>
            </span>
            <span>
              {texts.ui.error}:{" "}
              <span className="font-mono">
                {(r.current_error * 100).toFixed(2)}%
              </span>
            </span>
          </div>
        </div>

        <div className="flex justify-center">
          <svg
            viewBox={`0 0 ${W} ${H}`}
            className="w-full max-w-[420px]"
            role="img"
            aria-label="weight cloud"
          >
            {/* FP32 cloud (faint) */}
            {r.points.map((p, i) => (
              <circle
                key={`o${i}`}
                cx={px(p[0])}
                cy={py(p[1])}
                r="2.6"
                fill="#7d766d"
                opacity="0.22"
              />
            ))}
            {/* quantized cloud (solid) */}
            {r.quantized.map((p, i) => (
              <circle
                key={`q${i}`}
                cx={px(p[0])}
                cy={py(p[1])}
                r="2.8"
                fill={r.ternary ? "#8a3a35" : "#5B6BB0"}
                opacity="0.8"
              />
            ))}
          </svg>
        </div>
        <div className="mt-2 flex flex-wrap gap-4 text-caption text-on-surface-variant dark:text-outline">
          <span className="inline-flex items-center gap-1.5">
            <span className="w-3 h-3 rounded-full bg-[#7d766d] opacity-40" />{" "}
            {texts.ui.fp32}
          </span>
          <span className="inline-flex items-center gap-1.5">
            <span className="w-3 h-3 rounded-full bg-[#5B6BB0]" />{" "}
            {texts.ui.quantized}
          </span>
          {r.ternary && (
            <span className="inline-flex items-center gap-1.5">
              <span className="w-3 h-3 rounded-full bg-[#8a3a35]" />{" "}
              {texts.ui.ternary}
            </span>
          )}
        </div>
      </div>

      {/* BREAK IT staircase */}
      <div className="bg-surface-container-lowest dark:bg-dark-surface rounded-3xl p-4 md:p-6 shadow-ambient dark:shadow-dark-ambient border border-outline-variant/40 dark:border-white/10">
        <h3 className="font-headline text-lg text-on-surface dark:text-inverse-on-surface mb-4 inline-flex items-center gap-2">
          <span className="material-symbols-outlined" style={{ fontSize: 20 }}>
            format_list_numbered
          </span>
          BREAK IT: FP32 → INT8 → INT4 → INT2 → TERNARY
        </h3>
        <div className="flex flex-col gap-2">
          {staircase.map((label, i) => {
            const lv = levelMap.get(label);
            if (!lv) return null;
            const active = i === curIdx;
            return (
              <div
                key={label}
                className={`flex items-center gap-3 rounded-xl px-4 py-2.5 border transition-all ${
                  active
                    ? "border-primary/50 dark:border-inverse-primary/50 bg-primary/10 dark:bg-inverse-primary/10"
                    : "border-outline-variant/40 dark:border-white/10 bg-surface-container dark:bg-white/5"
                }`}
              >
                <span className="w-14 font-mono text-label-md font-bold text-on-surface dark:text-dark-on-surface">
                  {label}
                </span>
                <div className="flex-1 h-4 bg-surface-container dark:bg-white/10 rounded-md overflow-hidden">
                  <div
                    className="h-full rounded-md transition-all duration-300"
                    style={{
                      width: `${Math.min(100, (lv.bits / 32) * 100)}%`,
                      background: LEVEL_COLORS[i % LEVEL_COLORS.length],
                      opacity: 0.65,
                    }}
                  />
                </div>
                <span className="w-24 text-right font-mono text-caption text-on-surface-variant dark:text-outline tabular-nums">
                  {lv.bits} bit
                </span>
                <span
                  className="w-28 text-right font-mono text-caption tabular-nums"
                  style={{ color: i >= 3 ? "#C8604A" : "#2f6b3e" }}
                >
                  err {(lv.error * 100).toFixed(2)}%
                </span>
              </div>
            );
          })}
        </div>
        <p className="mt-3 text-caption text-on-surface-variant dark:text-outline">
          💡 {texts.ui.sweetSpot}
        </p>
      </div>

      {/* Pareto sweet spot */}
      <div className="bg-surface-container-lowest dark:bg-dark-surface rounded-3xl p-4 md:p-6 shadow-ambient dark:shadow-dark-ambient border border-outline-variant/40 dark:border-white/10">
        <h3 className="font-headline text-lg text-on-surface dark:text-inverse-on-surface mb-3 inline-flex items-center gap-2">
          <span className="material-symbols-outlined" style={{ fontSize: 20 }}>
            show_chart
          </span>
          {texts.ui.pareto}
        </h3>
        <svg
          viewBox={`0 0 ${PW} ${PH}`}
          className="w-full"
          role="img"
          aria-label="pareto"
        >
          {/* y grid (log) */}
          {[1, 0.1, 0.01, 0.001, 0.0001].map((v) => (
            <g key={v}>
              <line
                x1={plo}
                y1={lpy(v)}
                x2={PW - pr}
                y2={lpy(v)}
                stroke="rgba(125,118,109,0.14)"
                strokeDasharray="2 4"
              />
              <text
                x={plo - 6}
                y={lpy(v) + 3}
                textAnchor="end"
                fontSize="9"
                className="fill-outline"
              >
                {v}
              </text>
            </g>
          ))}
          {/* curve */}
          <polyline
            points={r.levels
              .map(
                (l) => `${lpx(l.bits).toFixed(1)},${lpy(l.error).toFixed(1)}`,
              )
              .join(" ")}
            fill="none"
            stroke="#7A5C36"
            strokeWidth="2"
          />
          {/* points */}
          {r.levels.map((l) => (
            <g key={l.label}>
              <circle
                cx={lpx(l.bits)}
                cy={lpy(l.error)}
                r={l.label === r.current_label ? 5 : 3.5}
                fill={l.label === r.current_label ? "#C8604A" : "#7A5C36"}
                stroke="#fff"
                strokeWidth="1.2"
              />
              <text
                x={lpx(l.bits)}
                y={lpy(l.error) - 8}
                textAnchor="middle"
                fontSize="9"
                className="fill-outline"
              >
                {l.label}
              </text>
            </g>
          ))}
          {/* sweet spot band: INT4..INT8 */}
          <rect
            x={lpx(8)}
            y={pt}
            width={lpx(4) - lpx(8)}
            height={PH - pt - pb}
            fill="rgba(47,107,62,0.10)"
            stroke="rgba(47,107,62,0.3)"
            strokeDasharray="3 3"
          />
          <text
            x={(lpx(8) + lpx(4)) / 2}
            y={pt + 12}
            textAnchor="middle"
            fontSize="9"
            fill="#2f6b3e"
          >
            {texts.ui.sweetSpot}
          </text>
          <text x={plo} y={PH - 8} fontSize="9" className="fill-outline">
            {texts.ui.memory} (bit) →
          </text>
          <text
            x={plo - 34}
            y={(pt + PH - pb) / 2}
            fontSize="9"
            className="fill-outline"
            transform={`rotate(-90 ${plo - 30} ${(pt + PH - pb) / 2})`}
          >
            {texts.ui.error}
          </text>
        </svg>
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
                correct,
                evidence: `${r.current_label} err=${(r.current_error * 100).toFixed(2)}%`,
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
                  e0: (levelMap.get("fp32")?.error ?? 0).toFixed(4),
                  e4: (levelMap.get("int4")?.error ?? 0).toFixed(4),
                  e2: (levelMap.get("int2")?.error ?? 0).toFixed(4),
                })}
              </span>
            </div>
            <ul className="text-caption text-on-surface-variant dark:text-outline space-y-1">
              {texts.explanation.map((line) => (
                <li key={line}>
                  •{" "}
                  {fmt(line, {
                    e0: (levelMap.get("fp32")?.error ?? 0).toFixed(4),
                    e4: (levelMap.get("int4")?.error ?? 0).toFixed(4),
                    e2: (levelMap.get("int2")?.error ?? 0).toFixed(4),
                  })}
                </li>
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
              setL2Goal("lossless");
              recordedL2.current = null;
            }}
            className={`px-4 py-2 rounded-xl text-label-md font-semibold border transition-all ${
              l2Goal === "lossless"
                ? "bg-primary text-on-primary border-primary dark:bg-inverse-primary dark:text-inverse-surface"
                : "bg-surface-container-lowest dark:bg-dark-surface text-on-surface-variant dark:text-outline border-outline-variant/60 dark:border-white/10 hover:border-primary/50"
            }`}
          >
            {texts.ui.l2Lossless}
          </button>
          <button
            onClick={() => {
              setL2Goal("sweet");
              recordedL2.current = null;
            }}
            className={`px-4 py-2 rounded-xl text-label-md font-semibold border transition-all ${
              l2Goal === "sweet"
                ? "bg-primary text-on-primary border-primary dark:bg-inverse-primary dark:text-inverse-surface"
                : "bg-surface-container-lowest dark:bg-dark-surface text-on-surface-variant dark:text-outline border-outline-variant/60 dark:border-white/10 hover:border-primary/50"
            }`}
          >
            {texts.ui.l2Sweet}
          </button>
        </div>

        {l2Goal && (
          <div>
            <div className="mb-2 text-caption text-on-surface-variant dark:text-outline">
              {texts.ui.l2Current}:{" "}
              <span className="font-mono text-primary dark:text-inverse-primary">
                {r.current_label}
              </span>
              {" · "}err{" "}
              <span className="font-mono">
                {(r.current_error * 100).toFixed(2)}%
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
      <VerificationPanel entry={verificationData["weight-freezer"]} />

      {/* Related notes */}
      <NotesPanel notes={texts.notes} />

      {/* L3 Explain — learner-owned, never machine-graded */}
      <ExplainBox
        onRecord={onRecord}
        evidence={`${r.current_label} err=${(r.current_error * 100).toFixed(2)}%`}
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
