// PaperPage — the paper PDF rendered AS ITSELF, clickable in place.
//
// The backend renders every PDF page to a PNG and returns each section's
// page + title bbox (PDF points). This page shows the pages as images and
// overlays transparent clickable regions on the sections: hover highlights
// the region, clicking loads that section's numpy implementation (middle /
// right) and its computed output.
import { useEffect, useMemo, useState } from "react";
import { api } from "../api";
import { useI18n } from "../i18n/context";

interface ViewSection {
  id: string;
  level: number;
  title: string;
  text: string;
  math: boolean;
  page: number;
  bbox: number[];
}

interface PageImage {
  page: number;
  image: string;
  width: number;
  height: number;
}

const UI: Record<"zh" | "en", Record<string, string>> = {
  zh: {
    title: "论文 ↔ 源码 ↔ 运行",
    subtitle:
      "论文 PDF 原样呈现——点击论文上的任意标题，查看该部分的源码实现与运行结果。",
    clickHint: "把鼠标移到 PDF 上高亮的标题区域，点击即可。",
    source: "该部分源码实现",
    run: "运行结果",
    runBtn: "▶ 重新运行",
    loading: "加载中…",
    empty: "点击左侧 PDF 中的高亮标题查看实现。",
    formula: "公式",
  },
  en: {
    title: "Paper ↔ Source ↔ Run",
    subtitle:
      "The paper PDF as itself — click any highlighted heading to see that section's implementation and output.",
    clickHint: "Hover the highlighted heading regions in the PDF and click.",
    source: "Section implementation",
    run: "Run output",
    runBtn: "▶ Re-run",
    loading: "Loading…",
    empty: "Click a highlighted heading in the PDF on the left.",
    formula: "formula",
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
  const [sections, setSections] = useState<ViewSection[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [source, setSource] = useState<{ title: string; code: string } | null>(
    null,
  );
  const [result, setResult] = useState<Record<string, unknown> | null>(null);
  const [running, setRunning] = useState(false);

  const byPage = useMemo(() => {
    const m = new Map<number, ViewSection[]>();
    for (const s of sections) {
      if (!m.has(s.page)) m.set(s.page, []);
      m.get(s.page)!.push(s);
    }
    return m;
  }, [sections]);

  useEffect(() => {
    let alive = true;
    api.lab
      .paperView()
      .then(async (v) => {
        if (!alive) return;
        setMeta({
          title: v.title,
          author: v.author,
          pages: v.pages,
          zoom: v.zoom,
        });
        setImages(v.images);
        setSections(v.sections);
        const first = v.sections.find((s) => HAS_IMPL.has(s.id));
        if (first) {
          setSelected(first.id);
          await loadSection(first.id);
        }
      })
      .catch(() => {});
    return () => {
      alive = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const loadSection = async (sid: string) => {
    setRunning(true);
    try {
      const [src, run] = await Promise.all([
        api.lab.paperSource(sid),
        api.lab.paperRun(sid, {}),
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

  const click = (sid: string) => {
    setSelected(sid);
    loadSection(sid);
  };

  const rerun = async () => {
    if (!selected) return;
    setRunning(true);
    try {
      const run = await api.lab.paperRun(selected, {});
      setResult(run.result);
    } catch {
      /* keep previous */
    } finally {
      setRunning(false);
    }
  };

  const zone = (s: ViewSection, img: PageImage) => {
    // bbox is in PDF points; zoom maps it to rendered pixels; % keeps it
    // aligned no matter how the image is scaled by CSS.
    const z = meta?.zoom ?? 2;
    const left = ((s.bbox[0] * z) / img.width) * 100;
    const top = ((s.bbox[1] * z) / img.height) * 100;
    const width = (((s.bbox[2] - s.bbox[0]) * z) / img.width) * 100;
    const height = (((s.bbox[3] - s.bbox[1]) * z) / img.height) * 100;
    const impl = HAS_IMPL.has(s.id);
    const active = selected === s.id;
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
          {ui.title}
        </h1>
        <p className="text-body-md text-on-surface-variant dark:text-outline mt-1">
          {ui.subtitle}
        </p>
        {meta && (
          <p className="text-caption text-outline mt-0.5">
            {meta.title} · {meta.author} · {meta.pages}{" "}
            {lang === "zh" ? "页" : "pages"} · {ui.clickHint}
          </p>
        )}
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-[minmax(0,1fr)_440px] gap-4 items-start">
        {/* LEFT — the PDF rendered as itself, clickable */}
        <section className="rounded-3xl border border-outline-variant/40 dark:border-white/10 bg-surface-container-lowest dark:bg-dark-surface p-4 min-w-0">
          {images.length === 0 ? (
            <div className="text-caption text-outline p-4">{ui.loading}</div>
          ) : (
            <div className="flex flex-col gap-4 max-h-[78vh] overflow-y-auto pr-1">
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
                    {zones.map((s) => {
                      const z = zone(s, img);
                      if (!z.impl) return null;
                      return (
                        <button
                          key={s.id}
                          onClick={() => click(s.id)}
                          title={`${s.id} ${s.title}`}
                          className={`absolute rounded-sm border transition-all ${
                            z.active
                              ? "border-primary bg-primary/25 dark:bg-inverse-primary/25 ring-2 ring-primary/40"
                              : "border-primary/50 bg-primary/10 hover:bg-primary/20 dark:bg-inverse-primary/15 dark:hover:bg-inverse-primary/25"
                          }`}
                          style={{
                            left: `${z.left}%`,
                            top: `${z.top}%`,
                            width: `${z.width}%`,
                            height: `${z.height}%`,
                          }}
                        />
                      );
                    })}
                  </div>
                );
              })}
            </div>
          )}
        </section>

        {/* RIGHT — implementation + output of the clicked section */}
        <section className="rounded-3xl border border-outline-variant/40 dark:border-white/10 bg-surface-container-lowest dark:bg-dark-surface p-4 min-w-0 lg:sticky lg:top-24">
          <div className="flex items-center justify-between mb-2">
            <h2 className="font-label-md text-label-md uppercase tracking-wider text-[#5B6BB0] dark:text-[#aab3e8]">
              🧩 {ui.source}
            </h2>
            <button
              onClick={rerun}
              disabled={!selected || running}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-primary text-on-primary dark:bg-inverse-primary dark:text-inverse-surface text-label-md font-semibold hover:opacity-90 transition disabled:opacity-50"
            >
              <span
                className="material-symbols-outlined"
                style={{ fontSize: 15 }}
              >
                play_arrow
              </span>
              {ui.runBtn}
            </button>
          </div>

          {!source ? (
            <div className="text-caption text-on-surface-variant dark:text-outline p-2">
              {running ? ui.loading : ui.empty}
            </div>
          ) : (
            <>
              <div className="text-caption text-outline font-mono mb-2">
                {source.title}
              </div>
              <pre className="text-caption font-mono leading-relaxed text-on-surface dark:text-dark-on-surface bg-surface-container dark:bg-white/5 rounded-xl p-3 overflow-x-auto max-h-[38vh] overflow-y-auto whitespace-pre">
                {source.code}
              </pre>

              <h2 className="font-label-md text-label-md uppercase tracking-wider text-[#2f6b3e] dark:text-[#9ed0a8] mt-4 mb-2">
                ⚙️ {ui.run}
              </h2>
              {!result ? (
                <div className="text-caption text-outline">{ui.loading}</div>
              ) : (
                <div className="flex flex-col gap-2 max-h-[34vh] overflow-y-auto pr-1">
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
            </>
          )}
        </section>
      </div>
    </div>
  );
}
