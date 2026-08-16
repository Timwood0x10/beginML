// PaperPage — Executable Paper: Equation → Implementation → Experiment.
//
// Layout is THREE layers, not three columns:
//   PAPER        (left)        the PDF as itself, formulas highlighted
//   IMPLEMENTATION (right top) formula ↔ code correspondence (code_map)
//   EXPERIMENT   (bottom)      input controls + RUN + observation/evidence
//
// The backend does everything: formula detection/location (paper_formulas),
// semantic anchors (anchors), formula→source→experiment mappings
// (mappings). The frontend only renders.
import { useEffect, useMemo, useState } from "react";
import { api } from "../api";
import { useI18n } from "../i18n/context";

interface PageImage {
  page: number;
  image: string;
  width: number;
  height: number;
}

interface FormulaZone {
  id: string;
  page: number;
  bbox: number[];
  text: string;
  section_id: string | null;
  section_title: string;
  anchor?: {
    type: string;
    section: string;
    label: string;
    concept: string;
  };
  implementation?: {
    file: string;
    symbols: string[];
    lines: number[];
  };
  experiment?: {
    runner: string;
    inputs: Record<
      string,
      { label: string; min: number; max: number; step: number; default: number }
    >;
    observation_zh: string;
    observation_en: string;
  };
  code_map?: { paper: string; code: string }[];
}

const UI: Record<"zh" | "en", Record<string, string>> = {
  zh: {
    title: "论文 ↔ 实现 ↔ 实验",
    subtitle:
      "Executable Paper——点公式，看它对应的实现，改参数，亲眼看到结果变化。",
    clickHint: "琥珀色高亮 = 论文公式：点击查看实现与实验。",
    paper: "PAPER",
    implementation: "IMPLEMENTATION",
    experiment: "EXPERIMENT",
    runBtn: "RUN EXPERIMENT",
    loading: "加载中…",
    empty: "点击左侧 PDF 中高亮的公式。",
    noImpl: "暂无实现",
    concept: "概念",
    file: "文件",
    symbols: "符号",
    lines: "行",
    paperCol: "Paper",
    codeCol: "Code",
    observation: "OBSERVATION",
    evidence: "EVIDENCE",
  },
  en: {
    title: "Paper ↔ Implementation ↔ Experiment",
    subtitle:
      "Executable Paper — click an equation, see its implementation, tweak a parameter, watch the result change.",
    clickHint:
      "Amber = paper formula: click to see implementation + experiment.",
    paper: "PAPER",
    implementation: "IMPLEMENTATION",
    experiment: "EXPERIMENT",
    runBtn: "RUN EXPERIMENT",
    loading: "Loading…",
    empty: "Click a highlighted formula in the PDF on the left.",
    noImpl: "no implementation",
    concept: "Concept",
    file: "File",
    symbols: "Symbols",
    lines: "Lines",
    paperCol: "Paper",
    codeCol: "Code",
    observation: "OBSERVATION",
    evidence: "EVIDENCE",
  },
};

const HAS_IMPL = new Set([
  "s1",
  "s2",
  "s4",
  "s7",
  "s8",
  "s9",
  "s10",
  "s12",
  "s15",
]);

function fmtVal(v: unknown): string {
  if (typeof v === "number")
    return Number.isInteger(v) ? String(v) : v.toFixed(6);
  if (Array.isArray(v)) return `[${v.map(fmtVal).join(", ")}]`;
  if (typeof v === "boolean") return String(v);
  return String(v ?? "");
}

export default function PaperPage() {
  const { lang } = useI18n();
  const ui = UI[lang];
  const [meta, setMeta] = useState<{
    title: string;
    author: string;
    pages: number;
    zoom: number;
  } | null>(null);
  const [images, setImages] = useState<PageImage[]>([]);
  const [formulas, setFormulas] = useState<FormulaZone[]>([]);
  const [selected, setSelected] = useState<FormulaZone | null>(null);
  const [source, setSource] = useState<{ title: string; code: string } | null>(
    null,
  );
  const [result, setResult] = useState<Record<string, unknown> | null>(null);
  const [running, setRunning] = useState(false);
  const [params, setParams] = useState<Record<string, number>>({});

  const byPage = useMemo(() => {
    const m = new Map<number, FormulaZone[]>();
    for (const fm of formulas) {
      if (!m.has(fm.page)) m.set(fm.page, []);
      m.get(fm.page)!.push(fm);
    }
    return m;
  }, [formulas]);

  useEffect(() => {
    let alive = true;
    Promise.all([api.lab.paperView(), api.lab.paperFormulas()])
      .then(async ([v, f]) => {
        if (!alive) return;
        setMeta({
          title: v.title,
          author: v.author,
          pages: v.pages,
          zoom: v.zoom,
        });
        setImages(v.images);
        setFormulas(f.formulas);
        const first = f.formulas.find(
          (fm) => fm.section_id && HAS_IMPL.has(fm.section_id),
        );
        if (first) await select(first);
      })
      .catch(() => {});
    return () => {
      alive = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const select = async (fm: FormulaZone) => {
    setSelected(fm);
    const sid = fm.section_id;
    if (!sid) return;
    // reset experiment params from the formula's experiment definition
    const def = fm.experiment?.inputs ?? {};
    const init: Record<string, number> = {};
    for (const [k, v] of Object.entries(def)) init[k] = v.default;
    setParams(init);
    setRunning(true);
    try {
      const [src, run] = await Promise.all([
        api.lab.paperSource(sid),
        api.lab.paperRun(sid, init),
      ]);
      setSource({ title: src.title, code: src.code });
      setResult(run.result);
    } catch {
      setSource(null);
      setResult(null);
    } finally {
      setRunning(false);
    }
  };

  const run = async () => {
    const sid = selected?.section_id;
    if (!sid) return;
    setRunning(true);
    try {
      const res = await api.lab.paperRun(sid, params);
      setResult(res.result);
    } catch {
      /* keep previous */
    } finally {
      setRunning(false);
    }
  };

  const fzone = (fm: FormulaZone, img: PageImage) => {
    const z = meta?.zoom ?? 2;
    const pad = 3;
    const left = (((fm.bbox[0] - pad) * z) / img.width) * 100;
    const top = (((fm.bbox[1] - pad) * z) / img.height) * 100;
    const width = (((fm.bbox[2] - fm.bbox[0] + pad * 2) * z) / img.width) * 100;
    const height =
      (((fm.bbox[3] - fm.bbox[1] + pad * 2) * z) / img.height) * 100;
    const impl = fm.section_id !== null && HAS_IMPL.has(fm.section_id);
    const active = selected?.id === fm.id;
    return { left, top, width, height, impl, active };
  };

  const fm = selected;
  const exp = fm?.experiment;
  const impl = fm?.implementation;
  const anchor = fm?.anchor;

  return (
    <div className="flex flex-col gap-4 pt-2">
      <header>
        <h1 className="font-headline text-headline-lg-mobile md:text-headline-xl text-on-surface dark:text-inverse-on-surface inline-flex items-center gap-2">
          <span
            className="material-symbols-outlined text-primary dark:text-inverse-primary"
            style={{ fontSize: 28 }}
          >
            memory
          </span>
          {ui.title}
        </h1>
        <p className="text-body-md text-on-surface-variant dark:text-outline mt-1">
          {ui.subtitle}
        </p>
        {meta && (
          <p className="text-caption text-outline mt-0.5">
            {meta.title} · {meta.author} · {meta.pages}{" "}
            {lang === "zh" ? "页" : "pages"} · {formulas.length}{" "}
            {lang === "zh" ? "个公式" : "formulas"} · {ui.clickHint}
          </p>
        )}
      </header>

      {/* PAPER / IMPLEMENTATION row */}
      <div className="grid grid-cols-1 lg:grid-cols-[minmax(0,1fr)_460px] gap-4 items-start">
        {/* PAPER — the PDF as itself, formulas clickable */}
        <section className="rounded-3xl border border-outline-variant/40 dark:border-white/10 bg-surface-container-lowest dark:bg-dark-surface p-4 min-w-0">
          <h2 className="font-label-md text-label-md uppercase tracking-wider text-primary dark:text-inverse-primary mb-3">
            📄 {ui.paper}
          </h2>
          {images.length === 0 ? (
            <div className="text-caption text-outline p-4">{ui.loading}</div>
          ) : (
            <div className="flex flex-col gap-4 max-h-[70vh] overflow-y-auto pr-1">
              {images.map((img) => {
                const zones = byPage.get(img.page) ?? [];
                return (
                  <div
                    key={img.page}
                    className="relative mx-auto"
                    style={{ width: "100%", maxWidth: img.width }}
                  >
                    <img
                      src={`data:image/png;base64,${img.image}`}
                      alt={`${meta?.title} p.${img.page}`}
                      className="w-full h-auto rounded-xl border border-outline-variant/40 dark:border-white/10"
                      draggable={false}
                    />
                    {zones.map((fz) => {
                      if (!fz.section_id) return null;
                      const z = fzone(fz, img);
                      return (
                        <button
                          key={fz.id}
                          onClick={() => select(fz)}
                          title={`${fz.anchor?.label ?? fz.id} · ${
                            fz.anchor?.concept ?? fz.section_title
                          }`}
                          className={`group absolute rounded-md transition-all ${
                            z.active
                              ? "bg-[#7A5C36]/55 ring-2 ring-[#7A5C36] dark:bg-[#c8a86a]/40"
                              : z.impl
                                ? "bg-[#7A5C36]/25 hover:bg-[#7A5C36]/45 dark:bg-[#c8a86a]/20 dark:hover:bg-[#c8a86a]/35"
                                : "bg-outline-variant/10 hover:bg-outline-variant/25 dark:bg-white/5 dark:hover:bg-white/15"
                          }`}
                          style={{
                            left: `${z.left}%`,
                            top: `${z.top}%`,
                            width: `${z.width}%`,
                            height: `${z.height}%`,
                          }}
                        >
                          <span className="sr-only">{fz.id}</span>
                        </button>
                      );
                    })}
                  </div>
                );
              })}
            </div>
          )}
        </section>

        {/* IMPLEMENTATION — formula ↔ code correspondence */}
        <section className="rounded-3xl border border-outline-variant/40 dark:border-white/10 bg-surface-container-lowest dark:bg-dark-surface p-4 min-w-0 lg:sticky lg:top-24">
          <h2 className="font-label-md text-label-md uppercase tracking-wider text-[#5B6BB0] dark:text-[#aab3e8] mb-3">
            🧩 {ui.implementation}
          </h2>

          {!fm || !impl ? (
            <div className="text-caption text-on-surface-variant dark:text-outline p-2">
              {ui.empty}
            </div>
          ) : (
            <>
              {/* semantic anchor */}
              {anchor && (
                <div className="rounded-xl border border-outline-variant/40 dark:border-white/10 bg-surface-container dark:bg-white/5 px-3 py-2 mb-3">
                  <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-caption">
                    <span className="text-outline">{ui.concept}:</span>
                    <span className="font-semibold text-on-surface dark:text-dark-on-surface">
                      {anchor.concept}
                    </span>
                    <span className="text-outline ml-auto">{anchor.label}</span>
                  </div>
                  <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-caption mt-1">
                    <span className="text-outline">{ui.file}:</span>
                    <span className="font-mono text-on-surface-variant dark:text-outline">
                      {impl.file}
                    </span>
                    <span className="text-outline">{ui.symbols}:</span>
                    <span className="font-mono text-on-surface-variant dark:text-outline">
                      {impl.symbols.join(", ")}
                    </span>
                    <span className="text-outline">{ui.lines}:</span>
                    <span className="font-mono text-on-surface-variant dark:text-outline">
                      {impl.lines[0]}–{impl.lines[1]}
                    </span>
                  </div>
                </div>
              )}

              {/* paper ↔ code correspondence */}
              <div className="text-caption uppercase tracking-wider text-outline font-semibold mb-2">
                {ui.paperCol} ↔ {ui.codeCol}
              </div>
              <div className="flex flex-col gap-1.5 mb-3">
                {(fm.code_map ?? []).map((row, i) => (
                  <div
                    key={i}
                    className="grid grid-cols-2 gap-2 rounded-xl border border-outline-variant/40 dark:border-white/10 bg-surface-container dark:bg-white/5 px-3 py-2"
                  >
                    <div className="text-caption font-semibold text-[#7A5C36] dark:text-[#c8a86a] font-mono break-words">
                      {row.paper}
                    </div>
                    <div className="text-caption font-mono text-[#2f6b3e] dark:text-[#9ed0a8] break-words">
                      {row.code}
                    </div>
                  </div>
                ))}
                {!fm.code_map?.length && (
                  <div className="text-caption text-outline">
                    {ui.noImpl} · {fm.section_title}
                  </div>
                )}
              </div>

              {/* full source */}
              {source && (
                <>
                  <div className="text-caption uppercase tracking-wider text-outline font-semibold mb-2">
                    {impl.file}
                  </div>
                  <pre className="text-caption font-mono leading-relaxed text-on-surface dark:text-dark-on-surface bg-surface-container dark:bg-white/5 rounded-xl p-3 overflow-x-auto max-h-[30vh] overflow-y-auto whitespace-pre">
                    {source.code}
                  </pre>
                </>
              )}
            </>
          )}
        </section>
      </div>

      {/* EXPERIMENT — inputs + RUN + observation/evidence */}
      <section className="rounded-3xl border border-outline-variant/40 dark:border-white/10 bg-surface-container-lowest dark:bg-dark-surface p-4">
        <div className="flex items-center justify-between mb-3">
          <h2 className="font-label-md text-label-md uppercase tracking-wider text-[#2f6b3e] dark:text-[#9ed0a8]">
            ⚙️ {ui.experiment}
          </h2>
          <button
            onClick={run}
            disabled={!selected || running}
            className="inline-flex items-center gap-1.5 px-4 py-2 rounded-xl bg-primary text-on-primary dark:bg-inverse-primary dark:text-inverse-surface text-label-md font-semibold hover:opacity-90 transition disabled:opacity-50"
          >
            <span
              className="material-symbols-outlined"
              style={{ fontSize: 16 }}
            >
              play_arrow
            </span>
            {ui.runBtn}
          </button>
        </div>

        {!fm || !exp ? (
          <div className="text-caption text-on-surface-variant dark:text-outline">
            {running ? ui.loading : ui.empty}
          </div>
        ) : (
          <div className="grid grid-cols-1 lg:grid-cols-[280px_minmax(0,1fr)] gap-6">
            {/* input controls */}
            <div className="flex flex-col gap-3">
              {Object.keys(exp.inputs).length === 0 && (
                <div className="text-caption text-outline">
                  {lang === "zh"
                    ? "本实验无参数——直接运行观察结果。"
                    : "No inputs — just run and observe."}
                </div>
              )}
              {Object.entries(exp.inputs).map(([key, def]) => (
                <label key={key} className="flex flex-col gap-1">
                  <span className="text-caption text-on-surface-variant dark:text-outline flex justify-between">
                    <span>{def.label}</span>
                    <span className="font-mono text-primary dark:text-inverse-primary">
                      {params[key] ?? def.default}
                    </span>
                  </span>
                  <input
                    type="range"
                    min={def.min}
                    max={def.max}
                    step={def.step}
                    value={params[key] ?? def.default}
                    onChange={(e) =>
                      setParams((p) => ({
                        ...p,
                        [key]: Number(e.target.value),
                      }))
                    }
                    className="w-full accent-[#2f6b3e]"
                  />
                </label>
              ))}
            </div>

            {/* observation + evidence */}
            <div className="min-w-0">
              <div className="text-caption uppercase tracking-wider text-outline font-semibold mb-1">
                🔭 {ui.observation}
              </div>
              <p className="text-body-md text-on-surface-variant dark:text-outline mb-3">
                {lang === "zh" ? exp.observation_zh : exp.observation_en}
              </p>

              <div className="text-caption uppercase tracking-wider text-outline font-semibold mb-1">
                📊 {ui.evidence}
              </div>
              {!result ? (
                <div className="text-caption text-outline">{ui.loading}</div>
              ) : (
                <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-2">
                  {Object.entries(result).map(([k, v]) => (
                    <div
                      key={k}
                      className="rounded-xl border border-outline-variant/40 dark:border-white/10 bg-surface-container dark:bg-white/5 px-3 py-2"
                    >
                      <div className="text-caption text-outline mb-0.5">
                        {k}
                      </div>
                      <div className="font-mono text-body-md text-on-surface dark:text-dark-on-surface break-all tabular-nums">
                        {fmtVal(v)}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        )}
      </section>
    </div>
  );
}
