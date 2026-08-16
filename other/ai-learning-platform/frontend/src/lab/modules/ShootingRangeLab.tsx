import { useEffect, useMemo, useRef, useState } from "react";
import type { LabResult, LabParams } from "../types";
import { useI18n } from "../../i18n/context";
import { labTextsZh, labTextsEn, fmt } from "../../i18n/lab";
import { ExplainBox } from "../Journal";
import { VerificationPanel } from "../VerificationPanel";
import { verificationData } from "../verification";
import { NotesPanel } from "../NotesPanel";
import { QuestList, type Quest } from "../QuestList";

interface RangeResult extends LabResult {
  x: number[];
  f: number[];
  xs: number[];
  ys: number[];
  fits: number[][];
  x_star: number;
  true_at_star: number;
  shots: number[];
  mean_shot: number;
  bias2: number;
  variance: number;
  noise2: number;
  mse: number;
  quadrant: "high-high" | "bias-dominated" | "variance-dominated" | "balanced";
  complexity: number;
  samples: number;
  noise: number;
  seed: number;
  provenance: string;
}

const GOLDEN_ANGLE = 2.39996323; // golden angle in radians (uniform-ish spread)

export default function ShootingRangeLab({
  result,
  loading,
  params,
  setParams,
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
  const texts = (lang === "zh" ? labTextsZh : labTextsEn)["shooting-range"];
  const r = result as RangeResult | null;
  const [answer, setAnswer] = useState<string | null>(null);
  const [submitted, setSubmitted] = useState(false);

  const correct = useMemo(() => {
    if (!r) return false;
    // Evidence-based verdict: the largest term is the dominant one.
    const dom =
      r.bias2 >= r.variance && r.bias2 >= r.noise2
        ? "bias"
        : r.variance >= r.noise2
          ? "variance"
          : "noise";
    return answer === dom;
  }, [r, answer]);

  // --- L2 Manipulate Challenge ------------------------------------------
  // Success is DERIVED from the computed quadrant (bias²/variance/noise²
  // are results, not controls). The learner must tune complexity / samples
  // / noise until the model lands in the target quadrant.
  const [l2Goal, setL2Goal] = useState<"bias" | "balanced" | null>(null);
  const recordedL2 = useRef<string | null>(null);

  const l2Achieved =
    l2Goal === "bias"
      ? r?.quadrant === "bias-dominated"
      : l2Goal === "balanced"
        ? r?.quadrant === "balanced"
        : false;

  // Record the achievement into the Journal exactly once per goal.
  useEffect(() => {
    if (r && l2Goal && l2Achieved && recordedL2.current !== l2Goal) {
      recordedL2.current = l2Goal;
      onRecord?.({
        question: l2Goal === "bias" ? texts.ui.l2Bias : texts.ui.l2Balanced,
        prediction: "manual tuning",
        correct: true,
        evidence: `quadrant=${r.quadrant}, Bias²=${r.bias2.toFixed(3)}, Var=${r.variance.toFixed(3)}`,
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
  const mse = r.bias2 + r.variance + r.noise2;
  const quests: Quest[] = [
    {
      id: "balanced",
      label: texts.quests![0],
      done: r.quadrant === "balanced",
    },
    {
      id: "bias",
      label: texts.quests![1],
      done: r.quadrant === "bias-dominated",
    },
    { id: "lowmse", label: texts.quests![2], done: mse < 0.05 },
  ];

  // --- target view: shots mapped to a 2D target ------------------------
  const T = 240; // target canvas size
  const cx = T / 2,
    cy = T / 2;
  const spread = Math.max(
    Math.abs(r.true_at_star),
    Math.max(...r.shots.map((s) => Math.abs(s - r.true_at_star))),
    0.3,
  );
  const K = (T / 2 - 14) / spread;
  const rings = [1, 0.6, 0.3]; // target rings (fraction of max radius)
  const shotPts = r.shots.map((s, i) => {
    const a = i * GOLDEN_ANGLE;
    const rad = (s - r.true_at_star) * K;
    return { x: cx + rad * Math.cos(a), y: cy + rad * Math.sin(a) };
  });
  const meanRad = (r.mean_shot - r.true_at_star) * K;

  // --- fit wireframe (function view) -----------------------------------
  const W = 300,
    H = 220;
  const pad = { l: 34, r: 10, t: 10, b: 24 };
  const xMin = Math.min(...r.x),
    xMax = Math.max(...r.x);
  const allY = [...r.f, ...r.ys];
  const yMin = Math.min(...allY),
    yMax = Math.max(...allY);
  const ySpan = yMax - yMin || 1;
  const px = (v: number) =>
    pad.l + ((v - xMin) / (xMax - xMin)) * (W - pad.l - pad.r);
  const py = (v: number) =>
    pad.t + (1 - (v - yMin) / ySpan) * (H - pad.t - pad.b);
  const truePath = r.x
    .map(
      (v, i) =>
        `${i === 0 ? "M" : "L"}${px(v).toFixed(1)},${py(r.f[i]).toFixed(1)}`,
    )
    .join(" ");
  const quadLabel = {
    "bias-dominated": texts.ui.quadrantBias,
    "variance-dominated": texts.ui.quadrantVariance,
    "high-high": texts.ui.quadrantHighHigh,
    balanced: texts.ui.quadrantBalanced,
  }[r.quadrant];

  const mseTotal = r.mse || 1e-9;
  const segBias = (r.bias2 / mseTotal) * 100;
  const segVar = (r.variance / mseTotal) * 100;
  const segNoise = 100 - segBias - segVar;

  return (
    <div className="flex flex-col gap-5">
      {/* Exploration quests */}
      <QuestList
        quests={quests}
        labId="shooting-range"
        onRecord={onRecord}
        params={params}
        evidence={`Bias²=${r.bias2.toFixed(4)}, Var=${r.variance.toFixed(4)}, Noise²=${r.noise2.toFixed(4)}`}
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

      {/* Target + function side by side */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Target */}
        <div className="bg-surface-container-lowest dark:bg-dark-surface rounded-3xl p-4 md:p-6 shadow-ambient dark:shadow-dark-ambient border border-outline-variant/40 dark:border-white/10">
          <h3 className="font-headline text-lg text-on-surface dark:text-inverse-on-surface mb-2 inline-flex items-center gap-2">
            <span
              className="material-symbols-outlined"
              style={{ fontSize: 20 }}
            >
              track_changes
            </span>
            {texts.ui.title}
          </h3>
          <div className="flex justify-center">
            <svg
              viewBox={`0 0 ${T} ${T}`}
              className="w-full max-w-[260px]"
              role="img"
              aria-label="target"
            >
              {rings.map((f) => (
                <circle
                  key={f}
                  cx={cx}
                  cy={cy}
                  r={(T / 2 - 10) * f}
                  fill="none"
                  stroke="rgba(125,118,109,0.25)"
                  strokeWidth="1.5"
                  strokeDasharray="3 4"
                />
              ))}
              {/* noise floor ring */}
              <circle
                cx={cx}
                cy={cy}
                r={Math.min(T / 2 - 10, Math.sqrt(r.noise2) * K)}
                fill="none"
                stroke="rgba(200,96,74,0.35)"
                strokeWidth="1"
                strokeDasharray="2 3"
              />
              {/* shots */}
              {shotPts.map((p, i) => (
                <circle
                  key={i}
                  cx={p.x}
                  cy={p.y}
                  r="3"
                  fill="#C8604A"
                  opacity="0.55"
                />
              ))}
              {/* mean shot marker */}
              <circle
                cx={cx + meanRad}
                cy={cy}
                r="5"
                fill="#2f6b3e"
                stroke="#fff"
                strokeWidth="1.5"
              />
              {/* bullseye */}
              <circle
                cx={cx}
                cy={cy}
                r="4"
                fill="#54483A"
                stroke="#fff"
                strokeWidth="1.5"
              />
              <circle cx={cx} cy={cy} r="1.5" fill="#fff" />
            </svg>
          </div>
          <div className="mt-3 flex flex-col gap-1 text-caption text-on-surface-variant dark:text-outline">
            <span>
              🎯 {texts.ui.target} ={" "}
              <span className="font-mono text-primary dark:text-inverse-primary">
                {r.true_at_star.toFixed(3)}
              </span>
            </span>
            <span>🔴 {texts.ui.shots}</span>
            <span>
              🟢 {texts.ui.meanShot} ={" "}
              <span className="font-mono">{r.mean_shot.toFixed(3)}</span>（偏差
              = {(r.mean_shot - r.true_at_star).toFixed(3)}）
            </span>
          </div>
        </div>

        {/* Function view */}
        <div className="bg-surface-container-lowest dark:bg-dark-surface rounded-3xl p-4 md:p-6 shadow-ambient dark:shadow-dark-ambient border border-outline-variant/40 dark:border-white/10">
          <h3 className="font-headline text-lg text-on-surface dark:text-inverse-on-surface mb-2 inline-flex items-center gap-2">
            <span
              className="material-symbols-outlined"
              style={{ fontSize: 20 }}
            >
              show_chart
            </span>
            {texts.ui.trueFn}
          </h3>
          <svg
            viewBox={`0 0 ${W} ${H}`}
            className="w-full"
            role="img"
            aria-label="function"
          >
            {/* fits band */}
            {r.fits.map((fit, i) => (
              <path
                key={i}
                d={fit
                  .map(
                    (v, j) =>
                      `${j === 0 ? "M" : "L"}${px(r.x[j]).toFixed(1)},${py(v).toFixed(1)}`,
                  )
                  .join(" ")}
                fill="none"
                stroke="rgba(91,107,176,0.28)"
                strokeWidth="1.4"
              />
            ))}
            {/* data points */}
            {r.xs.map((v, i) => (
              <circle
                key={i}
                cx={px(v)}
                cy={py(r.ys[i])}
                r="2.5"
                fill="#C8604A"
                opacity="0.6"
              />
            ))}
            {/* true function */}
            <path
              d={truePath}
              fill="none"
              stroke="#2f6b3e"
              strokeWidth="2.4"
              strokeLinejoin="round"
            />
          </svg>
          <div className="mt-2 flex flex-wrap gap-3 text-caption text-on-surface-variant dark:text-outline">
            <span className="inline-flex items-center gap-1.5">
              <span className="w-3 h-0.5 rounded bg-[#2f6b3e]" />{" "}
              {texts.ui.trueFn}
            </span>
            <span className="inline-flex items-center gap-1.5">
              <span className="w-3 h-0.5 rounded bg-[#C8604A]" />{" "}
              {texts.ui.dataPoints}
            </span>
            <span className="inline-flex items-center gap-1.5">
              <span className="w-3 h-0.5 rounded bg-[#5B6BB0] opacity-60" />{" "}
              {texts.ui.fits}
            </span>
          </div>
        </div>
      </div>

      {/* Quadrant + MSE decomposition */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div
          className={`rounded-3xl p-5 border bg-surface-container-lowest dark:bg-dark-surface ${
            r.quadrant === "variance-dominated"
              ? "border-[#C8604A]/40"
              : r.quadrant === "bias-dominated"
                ? "border-[#5B6BB0]/40"
                : r.quadrant === "high-high"
                  ? "border-[#8a3a35]/40"
                  : "border-[#2f6b3e]/40"
          }`}
        >
          <h3 className="font-label-md text-label-md uppercase tracking-wider text-on-surface dark:text-dark-on-surface mb-2">
            📌 {texts.ui.quadrant}
          </h3>
          <div className="font-headline text-lg text-on-surface dark:text-inverse-on-surface">
            {quadLabel}
          </div>
          <div className="mt-2 font-mono text-caption text-on-surface-variant dark:text-outline">
            {fmt(texts.ui.verdictEvidence, {
              b: r.bias2.toFixed(3),
              v: r.variance.toFixed(3),
              n: r.noise2.toFixed(3),
            })}
          </div>
        </div>

        <div className="bg-surface-container-lowest dark:bg-dark-surface rounded-3xl p-5 border border-outline-variant/40 dark:border-white/10">
          <h3 className="font-label-md text-label-md uppercase tracking-wider text-on-surface dark:text-dark-on-surface mb-3">
            🧮 {texts.ui.mse}
          </h3>
          <div className="h-5 w-full flex rounded-lg overflow-hidden">
            <div
              className="h-full bg-[#5B6BB0] transition-all duration-300"
              style={{ width: `${segBias}%` }}
              title={`Bias² = ${r.bias2.toFixed(4)}`}
            />
            <div
              className="h-full bg-[#C8604A] transition-all duration-300"
              style={{ width: `${segVar}%` }}
              title={`Variance = ${r.variance.toFixed(4)}`}
            />
            <div
              className="h-full bg-[#7A5C36] transition-all duration-300"
              style={{ width: `${segNoise}%` }}
              title={`Noise² = ${r.noise2.toFixed(4)}`}
            />
          </div>
          <div className="mt-3 flex flex-wrap gap-4 text-caption text-on-surface-variant dark:text-outline">
            <span className="inline-flex items-center gap-1.5">
              <span className="w-3 h-3 rounded-sm bg-[#5B6BB0]" />{" "}
              {texts.ui.bias2} ={" "}
              <span className="font-mono">{r.bias2.toFixed(3)}</span>
            </span>
            <span className="inline-flex items-center gap-1.5">
              <span className="w-3 h-3 rounded-sm bg-[#C8604A]" />{" "}
              {texts.ui.variance} ={" "}
              <span className="font-mono">{r.variance.toFixed(3)}</span>
            </span>
            <span className="inline-flex items-center gap-1.5">
              <span className="w-3 h-3 rounded-sm bg-[#7A5C36]" />{" "}
              {texts.ui.noise2} ={" "}
              <span className="font-mono">{r.noise2.toFixed(3)}</span>
            </span>
            <span className="ml-auto inline-flex items-center gap-1.5 font-semibold">
              {texts.ui.mseTotal} ={" "}
              <span className="font-mono">{r.mse.toFixed(3)}</span>
            </span>
          </div>
        </div>
      </div>

      {/* DATASET quick action */}
      <div className="bg-surface-container-lowest dark:bg-dark-surface rounded-3xl p-4 border border-outline-variant/40 dark:border-white/10 flex flex-wrap items-center gap-3">
        <span className="text-lg">📦</span>
        <span className="text-body-md text-on-surface dark:text-inverse-on-surface">
          样本 {r.samples} → 500：
          <span className="font-mono text-primary dark:text-inverse-primary">
            variance {r.variance.toFixed(3)}
          </span>
        </span>
        <button
          onClick={() => setParams({ ...params, samples: "500" })}
          className="ml-auto inline-flex items-center gap-2 px-4 py-2 rounded-xl bg-primary text-on-primary dark:bg-inverse-primary dark:text-inverse-surface font-label-md text-label-md hover:opacity-90 transition"
        >
          <span className="material-symbols-outlined" style={{ fontSize: 18 }}>
            add_chart
          </span>
          500 samples
        </button>
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
                evidence: `Bias²=${r.bias2.toFixed(4)}, Var=${r.variance.toFixed(4)}, Noise²=${r.noise2.toFixed(4)}`,
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
                  b: r.bias2.toFixed(3),
                  v: r.variance.toFixed(3),
                  n: r.noise2.toFixed(3),
                })}
              </span>
            </div>
            <ul className="text-caption text-on-surface-variant dark:text-outline space-y-1">
              {texts.explanation.map((line) => (
                <li key={line}>
                  •{" "}
                  {fmt(line, {
                    b: r.bias2.toFixed(3),
                    v: r.variance.toFixed(3),
                    n: r.noise2.toFixed(3),
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
              setL2Goal("bias");
              recordedL2.current = null;
            }}
            className={`px-4 py-2 rounded-xl text-label-md font-semibold border transition-all ${
              l2Goal === "bias"
                ? "bg-primary text-on-primary border-primary dark:bg-inverse-primary dark:text-inverse-surface"
                : "bg-surface-container-lowest dark:bg-dark-surface text-on-surface-variant dark:text-outline border-outline-variant/60 dark:border-white/10 hover:border-primary/50"
            }`}
          >
            {texts.ui.l2Bias}
          </button>
          <button
            onClick={() => {
              setL2Goal("balanced");
              recordedL2.current = null;
            }}
            className={`px-4 py-2 rounded-xl text-label-md font-semibold border transition-all ${
              l2Goal === "balanced"
                ? "bg-primary text-on-primary border-primary dark:bg-inverse-primary dark:text-inverse-surface"
                : "bg-surface-container-lowest dark:bg-dark-surface text-on-surface-variant dark:text-outline border-outline-variant/60 dark:border-white/10 hover:border-primary/50"
            }`}
          >
            {texts.ui.l2Balanced}
          </button>
        </div>

        {l2Goal && (
          <div>
            <div className="mb-2 text-caption text-on-surface-variant dark:text-outline">
              {texts.ui.l2Quadrant}:{" "}
              <span className="font-mono text-primary dark:text-inverse-primary">
                {r.quadrant}
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
      <VerificationPanel entry={verificationData["shooting-range"]} />

      {/* Related notes */}
      <NotesPanel notes={texts.notes} />

      {/* L3 Explain — learner-owned, never machine-graded */}
      <ExplainBox
        onRecord={onRecord}
        evidence={`Bias²=${r.bias2.toFixed(4)}, Var=${r.variance.toFixed(4)}, Noise²=${r.noise2.toFixed(4)}`}
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
