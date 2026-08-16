// PaperPage — Executable Paper: Paper → Mechanism → Code → Experiment.
//
// One paper is one executable world (merged with the old mechanism visual lab):
//   MECHANISMS  (tabs)    curated per paper; picking one loads its data flow
//   VISUALIZE   (left)    attention heatmap / data-flow steps; clicking an
//                         element highlights the source line that implements it
//   SOURCE      (right)   numbered, syntax-highlighted implementation with the
//                         active line lit up (diagram ↔ source linkage)
//   PDF         (entry)   the original paper as itself, formulas clickable —
//                         an entry point, not the main interface
//   EXPERIMENT  (bottom)  input controls + RUN + observation/evidence
//
// The backend does everything: formula detection/location (paper_formulas),
// semantic anchors (anchors), formula→source→experiment mappings
// (mappings). The frontend only renders.
import { useEffect, useMemo, useRef, useState } from "react";
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
    concept_zh?: string;
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

// per-paper presentation: the mechanisms worth exploring + the code keywords
// each visual element maps to (drives the diagram ↔ source highlighting)
interface Mech {
  id: string;
  title: string;
  kind: "attn" | "flow";
  keywords: string[];
}

const MECHS: Record<string, Mech[]> = {
  transformer: [
    {
      id: "s2",
      title: "Scaled Dot-Product Attention",
      kind: "attn",
      keywords: ["softmax"],
    },
    {
      id: "s5",
      title: "Causal Self-Attention",
      kind: "attn",
      keywords: ["mask"],
    },
    { id: "s14", title: "Single-Query Row", kind: "attn", keywords: ["row"] },
    {
      id: "s9",
      title: "RoPE (Rotary Position)",
      kind: "flow",
      keywords: ["rope(x, 4)", "rope(x, 8)"],
    },
    {
      id: "s15",
      title: "Residual Connection",
      kind: "flow",
      keywords: ["F =", "x_next"],
    },
  ],
  "attention-residuals": [
    {
      id: "s6",
      title: "Sequence-Depth Duality",
      kind: "flow",
      keywords: ["h_res", "h_plain"],
    },
    {
      id: "s7",
      title: "Depth Stability (α scaling)",
      kind: "flow",
      keywords: ["alpha", "np.linalg.norm"],
    },
  ],
};

const UI: Record<"zh" | "en", Record<string, string>> = {
  zh: {
    title: "可执行论文实验室",
    subtitle:
      "一篇论文一个可执行世界——点机制看数据流，点可视化元素高亮源码，改参数运行实验。",
    pickPaper: "选择论文",
    chooseMech: "机制",
    visualize: "可视化（点击元素高亮代码）",
    source: "源码",
    viewPdf: "查看原始 PDF",
    hidePdf: "收起 PDF",
    loading: "加载中…",
    attnHint: "点击热力图单元格 → 高亮 softmax/掩码代码",
    flowHint: "点击步骤 → 高亮对应代码",
    empty: "选择上方一个机制，或点击 PDF 中的公式。",
    noVisual: "该机制暂无可视化——源码与实验仍然可用。",
    experiment: "EXPERIMENT",
    runBtn: "RUN EXPERIMENT",
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
    title: "Executable Paper Lab",
    subtitle:
      "One paper, one executable world — pick a mechanism, trace its data flow, click a visual element to highlight its code, tweak a parameter and run the experiment.",
    pickPaper: "Paper",
    chooseMech: "Mechanism",
    visualize: "Visualize (click an element to highlight code)",
    source: "Source",
    viewPdf: "View original PDF",
    hidePdf: "Hide PDF",
    loading: "Loading…",
    attnHint: "Click a heatmap cell → highlights the softmax/mask code",
    flowHint: "Click a step → highlights the matching code",
    empty: "Pick a mechanism above, or click a formula in the PDF.",
    noVisual:
      "No visualization for this mechanism — source and experiment still work.",
    experiment: "EXPERIMENT",
    runBtn: "RUN EXPERIMENT",
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

interface VisualData {
  type?: string;
  matrix?: number[][];
  rows?: number;
  cols?: number;
  steps?: { label: string; shape: string; value: unknown }[];
}

// ---- python highlighter (line-based, marker technique) ---------------------
const PY_KEYWORDS = new Set([
  "def", "class", "return", "for", "in", "if", "else", "elif", "while",
  "import", "from", "as", "lambda", "with", "try", "except", "finally",
  "raise", "pass", "break", "continue", "assert", "global", "nonlocal",
  "del", "yield", "and", "or", "not", "is", "None", "True", "False",
]);
const PY_BUILTINS = new Set([
  "print", "len", "sum", "max", "min", "abs", "round", "int", "float",
  "str", "list", "dict", "set", "tuple", "bool", "range", "enumerate",
  "zip", "map", "filter", "sorted", "reversed", "open", "type", "isinstance",
]);
const C_KW = "#c792ea";
const C_FN = "#82aaff";
const C_STR = "#c3a03a";
const C_NUM = "#f78c6c";
const C_CMT = "#6b7a5a";
const C_BLT = "#f07178";
const C_DEC = "#ffcb6b";

/** Colorize one line of Python. Tokens are stashed into placeholders first so
 * later regex passes never re-color text inside already-colored spans. */
function pyLine(line: string): string {
  let s = line
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
  const saved: string[] = [];
  const stash = (html: string) => {
    saved.push(html);
    return `\u0000${saved.length - 1}\u0000`;
  };
  // strings (single/double/triple, with optional f/r/b prefixes)
  s = s.replace(
    /([frbFRB]{0,2}("""[\s\S]*?"""|'''[\s\S]*?'''|"(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*'))/g,
    (m) => stash(`<span style="color:${C_STR}">${m}</span>`),
  );
  // comments (after strings so `#` inside a string stays untouched)
  s = s.replace(/(#[^\u0000]*)/, (m) =>
    stash(`<span style="color:${C_CMT};font-style:italic">${m}</span>`),
  );
  // decorators
  s = s.replace(/^(\s*@\w+)/, (m) => stash(`<span style="color:${C_DEC}">${m}</span>`));
  // def/class name pairs
  s = s.replace(
    /\b(def|class)\s+(\w+)/g,
    (_, k: string, name: string) =>
      stash(`<span style="color:${C_KW}">${k}</span> <span style="color:${C_FN}">${name}</span>`),
  );
  // keywords
  const kwRe = new RegExp(`\\b(${[...PY_KEYWORDS].join("|")})\\b`, "g");
  s = s.replace(kwRe, (m) => stash(`<span style="color:${C_KW}">${m}</span>`));
  // builtins
  const blRe = new RegExp(`\\b(${[...PY_BUILTINS].join("|")})\\b`, "g");
  s = s.replace(blRe, (m) => stash(`<span style="color:${C_BLT}">${m}</span>`));
  // self + library aliases
  s = s.replace(/\bself\b/g, (m) => stash(`<span style="color:${C_BLT}">${m}</span>`));
  s = s.replace(/\b(np|rng|nn)\./g, (_, a: string) =>
    stash(`<span style="color:${C_BLT}">${a}</span>.`),
  );
  // numbers
  s = s.replace(/\b(\d+\.?\d*)\b/g, (m) => stash(`<span style="color:${C_NUM}">${m}</span>`));
  return s.replace(/\u0000(\d+)\u0000/g, (_, n: string) => saved[Number(n)] ?? "");
}

/** First line (0-based) containing `keyword`, or null. */
function findLine(code: string, keyword: string): number | null {
  const lines = code.split("\n");
  for (let i = 0; i < lines.length; i++) {
    if (lines[i].includes(keyword)) return i;
  }
  return null;
}

/** 1-based [start, end] inclusive → 0-based line indices, or null. */
function rangeOf(a?: number, b?: number): number[] | null {
  if (a == null || b == null || b < a) return null;
  const out: number[] = [];
  for (let i = a; i <= b; i++) out.push(i - 1);
  return out;
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

// ---- visual components -----------------------------------------------------
/** Attention weight matrix as a heatmap; cells show their value, clicking a
 * cell highlights the implementing source line. */
function Heatmap({
  matrix,
  onCell,
  activeCell,
}: {
  matrix: number[][];
  onCell: (i: number, j: number) => void;
  activeCell?: [number, number] | null;
}) {
  const cell = 30;
  const gap = 3;
  const rows = matrix.length;
  const cols = matrix[0]?.length ?? 1;
  const max = Math.max(...matrix.flat(), 1e-9);
  return (
    <div className="overflow-x-auto">
      <svg
        width={cols * cell}
        height={rows * cell}
        viewBox={`0 0 ${cols * cell} ${rows * cell}`}
        className="min-w-full h-auto"
      >
        {matrix.map((row, i) =>
          row.map((v, j) => {
            const a = v > 0.001 ? 0.12 + 0.88 * (v / max) : 0;
            const active = activeCell?.[0] === i && activeCell?.[1] === j;
            return (
              <g key={`${i}-${j}`} onClick={() => onCell(i, j)}>
                <rect
                  x={j * cell}
                  y={i * cell}
                  width={cell - gap}
                  height={cell - gap}
                  rx={3}
                  fill={
                    v > 0.001
                      ? `rgba(122, 92, 54, ${a.toFixed(3)})`
                      : "#e9e5de"
                  }
                  className={`cursor-pointer transition-all duration-150 ${
                    active
                      ? "stroke-[#5B6BB0] stroke-2"
                      : "hover:opacity-80"
                  }`}
                >
                  <title>{`${i} → ${j}: ${v.toFixed(4)}`}</title>
                </rect>
                {v > 0.001 && (
                  <text
                    x={j * cell + (cell - gap) / 2}
                    y={i * cell + (cell - gap) / 2 + 1}
                    textAnchor="middle"
                    dominantBaseline="middle"
                    fontSize={9}
                    fontWeight={600}
                    fill={a > 0.55 ? "#fffaf0" : "#7A5C36"}
                    pointerEvents="none"
                    className="select-none font-mono"
                  >
                    {v >= 0.1 ? v.toFixed(1) : v.toFixed(2)}
                  </text>
                )}
              </g>
            );
          }),
        )}
      </svg>
    </div>
  );
}

/** Data-flow steps (input → transform → output) with shapes and values;
 * clicking a step highlights the implementing source line. */
function FlowSteps({
  steps,
  onStep,
  lang,
  activeIndex,
}: {
  steps: { label: string; shape: string; value: unknown }[];
  onStep: (i: number) => void;
  lang: string;
  activeIndex?: number | null;
}) {
  return (
    <div className="flex flex-col">
      {steps.map((s, i) => (
        <div key={i} className="flex flex-col">
          <button
            onClick={() => onStep(i)}
            className={`text-left rounded-xl border px-3 py-2 transition-all duration-200 ${
              activeIndex === i
                ? "border-[#5B6BB0] bg-[#5B6BB0]/10 dark:border-[#aab3e8] dark:bg-[#aab3e8]/10 shadow-[0_0_0_3px_rgba(91,107,176,0.15)]"
                : "border-outline-variant/40 dark:border-white/10 bg-surface-container dark:bg-white/5 hover:bg-[#7A5C36]/10 hover:border-[#7A5C36]/50"
            }`}
          >
            <div className="flex items-center gap-2 text-caption">
              <span
                className={`inline-flex items-center justify-center w-5 h-5 rounded-full font-mono text-[11px] shrink-0 ${
                  activeIndex === i
                    ? "bg-[#5B6BB0] text-white dark:bg-[#aab3e8] dark:text-[#0f172a]"
                    : "bg-[#7A5C36]/20 text-[#7A5C36] dark:bg-[#c8a86a]/20 dark:text-[#c8a86a]"
                }`}
              >
                {i + 1}
              </span>
              <span className="font-semibold text-on-surface dark:text-dark-on-surface truncate">
                {s.label}
              </span>
              <span className="ml-auto text-outline shrink-0 font-mono text-[10px] px-1.5 py-px rounded bg-outline-variant/30 dark:bg-white/10">
                {s.shape}
              </span>
            </div>
            <div className="font-mono text-caption text-on-surface-variant dark:text-outline truncate mt-1 ml-7">
              {Array.isArray(s.value)
                ? `[${s.value.slice(0, 8).join(", ")}${s.value.length > 8 ? "…" : ""}]`
                : String(s.value)}
            </div>
          </button>
          {i < steps.length - 1 && (
            <div className="flex justify-center py-0.5 text-[#7A5C36]/60 dark:text-[#c8a86a]/50">
              <span
                className="material-symbols-outlined"
                style={{ fontSize: 16 }}
              >
                arrow_downward
              </span>
            </div>
          )}
        </div>
      ))}
    </div>
  );
}

// ---- page ------------------------------------------------------------------
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
  const [paperId, setPaperId] = useState<string>("transformer");
  const [papers, setPapers] = useState<
    { id: string; title: string; author: string }[]
  >([]);
  const [mechIdx, setMechIdx] = useState(0);
  const [highlight, setHighlight] = useState<number[] | null>(null);
  const [activeCell, setActiveCell] = useState<[number, number] | null>(null);
  const [activeStep, setActiveStep] = useState<number | null>(null);
  const [playing, setPlaying] = useState(false);
  const [showPdf, setShowPdf] = useState(false);
  const codeRef = useRef<HTMLDivElement | null>(null);
  const playToken = useRef(0);

  const mechs = MECHS[paperId] ?? [];
  const mech = mechs[mechIdx];

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
    api.lab
      .papers()
      .then((p) => {
        if (alive) setPapers(p.papers);
      })
      .catch(() => {});
    loadPaper("transformer");
    return () => {
      alive = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  /** Load a paper: pages + formulas, then start on its first curated mechanism. */
  const loadPaper = async (pid: string) => {
    stopPlay();
    setPaperId(pid);
    setSource(null);
    setResult(null);
    setHighlight(null);
    setRunning(true);
    try {
      const [v, f] = await Promise.all([
        api.lab.paperView(pid),
        api.lab.paperFormulas(pid),
      ]);
      setMeta({
        title: v.title,
        author: v.author,
        pages: v.pages,
        zoom: v.zoom,
      });
      setImages(v.images);
      setFormulas(f.formulas);
      const curated = MECHS[pid] ?? [];
      if (curated[0]) {
        setMechIdx(0);
        await loadSection(curated[0].id, pid, f.formulas);
      } else {
        const first = f.formulas.find(
          (fm) => fm.section_id && HAS_IMPL.has(fm.section_id),
        );
        if (first) {
          setMechIdx(-1);
          await loadSection(first.section_id!, pid, f.formulas);
        }
      }
    } catch {
      /* keep previous on error */
    } finally {
      setRunning(false);
    }
  };

  /** Load source + run output for one section, syncing the formula zone. */
  const loadSection = async (
    sectionId: string,
    pid: string,
    fs: FormulaZone[],
  ) => {
    stopPlay();
    const fm = fs.find((x) => x.section_id === sectionId) ?? null;
    setSelected(fm);
    setHighlight(null);
    // reset experiment params from the formula's experiment definition
    const def = fm?.experiment?.inputs ?? {};
    const init: Record<string, number> = {};
    for (const [k, v] of Object.entries(def)) init[k] = v.default;
    setParams(init);
    setRunning(true);
    try {
      const [src, run] = await Promise.all([
        api.lab.paperSource(sectionId, pid),
        api.lab.paperRun(sectionId, init, pid),
      ]);
      if (!src.code) {
        // no hand-authored implementation — show "暂无实现" instead of a blank
        setSource({ title: src.title || fm?.section_title || "", code: "" });
        setResult(null);
      } else {
        setSource({ title: src.title, code: src.code });
        setResult(run.result);
        // light up the whole implementation line range by default
        const il = fm?.implementation?.lines;
        setHighlight(il ? rangeOf(il[0], il[1]) : null);
      }
    } catch {
      setSource(null);
      setResult(null);
    } finally {
      setRunning(false);
    }
  };

  const selectMech = (idx: number) => {
    const m = mechs[idx];
    if (!m) return;
    setMechIdx(idx);
    loadSection(m.id, paperId, formulas);
  };

  const selectFormula = async (fm: FormulaZone, pid?: string) => {
    const sid = fm.section_id;
    if (!sid) return;
    const i = mechs.findIndex((m) => m.id === sid);
    if (i >= 0) setMechIdx(i);
    await loadSection(sid, pid ?? paperId, formulas);
  };

  const run = async () => {
    const sid = selected?.section_id;
    if (!sid) return;
    setRunning(true);
    try {
      const res = await api.lab.paperRun(sid, params, paperId);
      setResult(res.result);
    } catch {
      /* keep previous */
    } finally {
      setRunning(false);
    }
  };

  const fm = selected;
  const exp = fm?.experiment;
  const impl = fm?.implementation;
  const anchor = fm?.anchor;
  const visual = (result ?? {})["visual"] as VisualData | undefined;
  const codeLines = useMemo(() => (source?.code ?? "").split("\n"), [source]);

  /** Clicking a visual element highlights the source line implementing it. */
  const onVisualClick = (kind: string, index?: number, cell?: [number, number]) => {
    if (!mech || !source?.code) return;
    let line: number | null = null;
    if (kind === "attn") {
      line = findLine(source.code, mech.keywords[0] ?? "");
      setActiveCell(cell ?? null);
      setActiveStep(null);
    } else if (kind === "flow" && index !== undefined) {
      const kw = mech.keywords[Math.min(index, mech.keywords.length - 1)];
      line = findLine(source.code, kw);
      setActiveStep(index);
      setActiveCell(null);
    }
    setHighlight(line !== null ? [line] : null);
  };

  /** Play the visualization step by step, pulsing each element and scrolling
   * its implementing line into view — microGPT-style playback. */
  const play = async () => {
    if (!mech || !source?.code || !visual || playing) return;
    const token = ++playToken.current;
    setPlaying(true);
    const kw = mech.keywords;
    try {
      if (visual.type === "attn" && visual.matrix) {
        const mat = visual.matrix;
        for (let i = 0; i < mat.length && token === playToken.current; i++) {
          for (let j = 0; j < mat[i].length && token === playToken.current; j++) {
            if (mat[i][j] <= 0.001) continue;
            const line = findLine(source.code, kw[0] ?? "");
            setActiveCell([i, j]);
            setHighlight(line !== null ? [line] : null);
            await sleep(120);
          }
        }
      } else if (visual.type === "flow" && visual.steps) {
        const n = visual.steps.length;
        for (let i = 0; i < n && token === playToken.current; i++) {
          const kw2 = kw[Math.min(i, kw.length - 1)];
          const line = findLine(source.code, kw2);
          setActiveStep(i);
          setHighlight(line !== null ? [line] : null);
          await sleep(650);
        }
      }
    } finally {
      if (token === playToken.current) {
        setActiveCell(null);
        setActiveStep(null);
        setPlaying(false);
      }
    }
  };

  /** Stop any running playback (e.g. when switching mechanism). */
  const stopPlay = () => {
    playToken.current++;
    setActiveCell(null);
    setActiveStep(null);
    setPlaying(false);
  };

  /** Smooth-scroll the first highlighted line into view inside the code pane. */
  const scrollToLine = (lines: number[]) => {
    const first = lines[0];
    const el = codeRef.current?.querySelector(`[data-line="${first}"]`);
    el?.scrollIntoView({ behavior: "smooth", block: "center" });
  };

  // auto-scroll the code pane to the first highlighted line when it changes
  useEffect(() => {
    if (highlight && highlight.length > 0) {
      scrollToLine(highlight);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [highlight]);

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
          <span className="bg-gradient-to-r from-[#7A5C36] to-[#5B6BB0] dark:from-[#c8a86a] dark:to-[#aab3e8] bg-clip-text text-transparent">
            {ui.title}
          </span>
        </h1>
        <p className="text-body-md text-on-surface-variant dark:text-outline mt-1">
          {ui.subtitle}
        </p>
        {papers.length > 0 && (
          <div className="mt-2 flex flex-wrap items-center gap-2">
            <span
              className="material-symbols-outlined text-primary dark:text-inverse-primary"
              style={{ fontSize: 18 }}
            >
              description
            </span>
            <select
              value={paperId}
              onChange={(e) => loadPaper(e.target.value)}
              className="rounded-xl border border-outline-variant/40 dark:border-white/10 bg-surface-container dark:bg-white/5 px-3 py-1.5 text-body-md text-on-surface dark:text-dark-on-surface font-semibold"
            >
              {papers.map((p) => (
                <option key={p.id} value={p.id}>
                  {p.title}
                </option>
              ))}
            </select>
            <button
              onClick={() => setShowPdf((s) => !s)}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-label-md font-semibold text-[#5B6BB0] dark:text-[#aab3e8] border border-[#5B6BB0]/30 dark:border-[#aab3e8]/30 hover:bg-[#5B6BB0]/10 transition-colors"
            >
              <span
                className="material-symbols-outlined"
                style={{ fontSize: 16 }}
              >
                picture_as_pdf
              </span>
              {showPdf ? ui.hidePdf : ui.viewPdf}
            </button>
          </div>
        )}
        {meta && (
          <p className="text-caption text-outline mt-0.5">
            {meta.title} · {meta.author} · {meta.pages}{" "}
            {lang === "zh" ? "页" : "pages"} · {formulas.length}{" "}
            {lang === "zh" ? "个公式" : "formulas"}
          </p>
        )}
      </header>

      {/* PDF — the original paper as itself; an entry point, formulas clickable */}
      {showPdf && (
        <section className="rounded-3xl border border-outline-variant/40 dark:border-white/10 bg-surface-container-lowest dark:bg-dark-surface p-4 min-w-0">
          <h2 className="font-label-md text-label-md uppercase tracking-wider text-primary dark:text-inverse-primary mb-3">
            📄 {ui.viewPdf}
          </h2>
          {images.length === 0 ? (
            <div className="text-caption text-outline p-4">{ui.loading}</div>
          ) : (
            <div className="flex flex-col gap-4 max-h-[60vh] overflow-y-auto pr-1">
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
                          onClick={() => selectFormula(fz)}
                          title={`${fz.anchor?.label ?? fz.id} · ${
                            lang === "zh"
                              ? fz.anchor?.concept_zh || fz.anchor?.concept
                              : fz.anchor?.concept
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
      )}

      {/* mechanism tabs */}
      <div className="flex flex-wrap items-center gap-x-4 gap-y-1 border-b border-outline-variant/40 dark:border-white/10">
        <span className="text-caption uppercase tracking-wider text-outline font-semibold pb-2 mr-1">
          🧭 {ui.chooseMech}
        </span>
        {mechs.map((m, i) => (
          <button
            key={m.id}
            onClick={() => selectMech(i)}
            className={`pb-2 -mb-px border-b-2 text-label-md font-semibold transition-colors ${
              i === mechIdx
                ? "border-[#7A5C36] text-[#7A5C36] dark:border-[#c8a86a] dark:text-[#c8a86a]"
                : "border-transparent text-outline hover:text-on-surface-variant dark:hover:text-outline"
            }`}
          >
            {m.title}
          </button>
        ))}
      </div>

      {/* split pane: interactive visualization (left) + source (right) */}
      <div className="grid grid-cols-1 lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)] gap-4 items-start">
        {/* LEFT — visualization, click an element to highlight its code */}
        <section className="rounded-3xl border border-outline-variant/40 dark:border-white/10 bg-surface-container-lowest dark:bg-dark-surface p-4 min-w-0">
          <h2 className="font-label-md text-label-md uppercase tracking-wider text-[#7A5C36] dark:text-[#c8a86a] mb-2">
            📊 {ui.visualize}
          </h2>
          {running ? (
            <div className="text-caption text-outline">{ui.loading}</div>
          ) : !source ? (
            <div className="text-caption text-outline">{ui.empty}</div>
          ) : !visual ? (
            <div className="text-caption text-outline">{ui.noVisual}</div>
          ) : visual.type === "attn" && visual.matrix ? (
            <>
              <div className="flex items-center gap-2 mb-2">
                <button
                  onClick={play}
                  disabled={playing}
                  className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-label-md font-semibold bg-[#7A5C36]/15 text-[#7A5C36] dark:bg-[#c8a86a]/15 dark:text-[#c8a86a] hover:bg-[#7A5C36]/25 dark:hover:bg-[#c8a86a]/25 transition-colors disabled:opacity-50"
                >
                  <span
                    className="material-symbols-outlined"
                    style={{ fontSize: 16 }}
                  >
                    {playing ? "pause" : "play_arrow"}
                  </span>
                  {lang === "zh" ? "播放" : "Play"}
                </button>
                <span className="text-caption text-outline">
                  {ui.attnHint}
                </span>
              </div>
              <Heatmap
                matrix={visual.matrix}
                onCell={(i, j) => onVisualClick("attn", undefined, [i, j])}
                activeCell={activeCell}
              />
            </>
          ) : visual.type === "flow" && visual.steps ? (
            <>
              <div className="flex items-center gap-2 mb-2">
                <button
                  onClick={play}
                  disabled={playing}
                  className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-label-md font-semibold bg-[#7A5C36]/15 text-[#7A5C36] dark:bg-[#c8a86a]/15 dark:text-[#c8a86a] hover:bg-[#7A5C36]/25 dark:hover:bg-[#c8a86a]/25 transition-colors disabled:opacity-50"
                >
                  <span
                    className="material-symbols-outlined"
                    style={{ fontSize: 16 }}
                  >
                    {playing ? "pause" : "play_arrow"}
                  </span>
                  {lang === "zh" ? "播放数据流" : "Play flow"}
                </button>
                <span className="text-caption text-outline">
                  {ui.flowHint}
                </span>
              </div>
              <FlowSteps
                steps={visual.steps}
                onStep={(i) => onVisualClick("flow", i)}
                lang={lang}
                activeIndex={activeStep}
              />
            </>
          ) : (
            <div className="text-caption text-outline">{ui.noVisual}</div>
          )}
        </section>

        {/* RIGHT — numbered, syntax-highlighted source with active line */}
        <section className="rounded-3xl border border-outline-variant/40 dark:border-white/10 bg-surface-container-lowest dark:bg-dark-surface p-4 min-w-0">
          <h2 className="font-label-md text-label-md uppercase tracking-wider text-[#5B6BB0] dark:text-[#aab3e8] mb-2">
            🧩 {ui.source}
          </h2>

          {/* semantic anchor */}
          {anchor && (
            <div className="rounded-xl border border-outline-variant/40 dark:border-white/10 bg-surface-container dark:bg-white/5 px-3 py-2 mb-3">
              <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-caption">
                <span className="text-outline">{ui.concept}:</span>
                <span className="font-semibold text-on-surface dark:text-dark-on-surface">
                  {lang === "zh"
                    ? anchor.concept_zh || anchor.concept
                    : anchor.concept}
                </span>
                <span className="text-outline ml-auto">{anchor.label}</span>
              </div>
              {impl && (
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
              )}
            </div>
          )}

          {/* paper ↔ code correspondence */}
          {fm?.code_map?.length ? (
            <>
              <div className="text-caption uppercase tracking-wider text-outline font-semibold mb-2">
                {ui.paperCol} ↔ {ui.codeCol}
              </div>
              <div className="flex flex-col gap-1.5 mb-3">
                {fm.code_map.map((row, i) => (
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
              </div>
            </>
          ) : null}

          {!source ? (
            <div className="text-caption text-on-surface-variant dark:text-outline p-2">
              {ui.empty}
            </div>
          ) : !source.code ? (
            <div className="rounded-xl border border-dashed border-outline-variant/60 dark:border-white/15 px-3 py-2.5 text-caption text-on-surface-variant dark:text-outline">
              📄 {ui.noImpl} · {source.title}
            </div>
          ) : (
            <div className="rounded-xl overflow-hidden border border-outline-variant/40 dark:border-white/10">
              {/* file header bar */}
              <div className="flex items-center gap-2 px-3 py-1.5 bg-surface-container dark:bg-white/5 border-b border-outline-variant/40 dark:border-white/10">
                <span
                  className="material-symbols-outlined text-[#5B6BB0] dark:text-[#aab3e8]"
                  style={{ fontSize: 15 }}
                >
                  code
                </span>
                <span className="font-mono text-caption text-on-surface-variant dark:text-outline truncate">
                  {impl?.file ?? source.title}
                </span>
                {impl && (
                  <span className="font-mono text-[10px] text-outline ml-auto shrink-0">
                    {impl.lines[0]}–{impl.lines[1]}
                  </span>
                )}
              </div>
              <div
                ref={codeRef}
                className="bg-surface-container dark:bg-white/5 overflow-x-auto max-h-[60vh] overflow-y-auto"
              >
                {codeLines.map((ln, i) => {
                  const lit = highlight?.includes(i) ?? false;
                  return (
                    <div
                      key={i}
                      data-line={i}
                      className={`flex px-2 py-px font-mono text-caption leading-relaxed transition-colors duration-150 ${
                        lit
                          ? "bg-[#5B6BB0]/15 dark:bg-[#aab3e8]/15 border-l-2 border-[#5B6BB0] dark:border-[#aab3e8]"
                          : "odd:bg-transparent even:bg-black/[0.03] dark:even:bg-white/[0.03] border-l-2 border-transparent"
                      }`}
                    >
                      <span className="min-w-[2.5rem] text-right pr-3 text-outline select-none shrink-0">
                        {i + 1}
                      </span>
                      <span
                        className="text-on-surface dark:text-dark-on-surface whitespace-pre"
                        dangerouslySetInnerHTML={{ __html: pyLine(ln) || " " }}
                      />
                    </div>
                  );
                })}
              </div>
            </div>
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