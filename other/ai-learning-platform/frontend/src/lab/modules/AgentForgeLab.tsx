import { useEffect, useMemo, useState } from 'react'
import type { LabResult } from '../types'

// Agent Forge — an executable cognitive-system workbench.
// Layout is vertical so nothing is squeezed: a top toolbar, a collapsible
// primitive strip, a full-width wiring canvas, and the inspector below it.

interface PrimitiveMeta {
  label: string
  group: string
  icon: string
  ports: { in: string[]; out: string[] }
  desc: string
  why: string
  tradeoffs: string[]
  note: string
}

interface TraceStep {
  node: string
  label: string
  icon: string
  latency: number
  tokens: number
  calls: number
  llm: boolean
  failed: boolean
  detail: string
}

interface ForgeResult extends LabResult {
  preset: string
  presetName: string
  primitives: Record<string, PrimitiveMeta>
  groups: string[]
  graph: { name: string; nodes: { id: string }[]; edges: { from: string; to: string; port: string }[] }
  run: {
    task: string
    trace: TraceStep[]
    totals: { latency: number; tokens: number; calls: number; llm_calls: number }
    failures: { node: string; label: string; mode: string; message: string; recovery: string }[]
    recovered: number
    resilience: number
    chaos: string[]
  }
  yaml: string
  compare: {
    active: boolean
    a: { latency: number; tokens: number; calls: number; llm_calls: number }
    b: { latency: number; tokens: number; calls: number; llm_calls: number }
    diffs: { metric: string; a: number; b: number; direction: string }[]
  }
}

// Rough canvas grid for the wiring view.
const COLS = 3
const NODE_W = 148
const NODE_H = 92
const GAP = 44

export default function AgentForgeLab({ result }: { result: LabResult | null }) {
  const r = result as ForgeResult | null
  const [selected, setSelected] = useState<string | null>(null)
  const [playing, setPlaying] = useState(false)
  const [step, setStep] = useState(0)
  const [mode, setMode] = useState<'run' | 'yaml' | 'compare'>('run')
  const [paletteOpen, setPaletteOpen] = useState(true)

  // Position nodes on a grid (topo order from backend).
  const layout = useMemo(() => {
    type LayoutNode = { id: string; x: number; y: number }
    type LayoutEdge = { from: string; to: string; port: string }
    if (!r) return { nodes: [] as LayoutNode[], edges: [] as LayoutEdge[], width: 0, height: 0 }
    const ids = r.graph.nodes.map((n) => n.id)
    const cols = Math.min(COLS, Math.max(1, ids.length))
    const rows = Math.ceil(ids.length / cols)
    const width = cols * (NODE_W + GAP) + GAP
    const height = rows * (NODE_H + GAP) + GAP
    const nodes: LayoutNode[] = ids.map((id, i) => ({
      id,
      x: GAP + (i % cols) * (NODE_W + GAP),
      y: GAP + Math.floor(i / cols) * (NODE_H + GAP),
    }))
    return { nodes, edges: r.graph.edges as LayoutEdge[], width, height }
  }, [r])

  // Playback: reveal one trace step at a time.
  useEffect(() => {
    if (!playing || !r) return
    if (step >= r.run.trace.length) {
      setPlaying(false)
      return
    }
    const t = setTimeout(() => setStep((s) => s + 1), 420)
    return () => clearTimeout(t)
  }, [playing, step, r])

  if (!r) return null

  const traceNodes = new Set(r.run.trace.slice(0, step).map((s) => s.node))
  const failedNodes = new Set(r.run.failures.map((f) => f.node))
  const sel = selected ? r.primitives[selected] : null

  const runTrace = r.run.trace
  const chaosActive = r.run.chaos.length > 0

  return (
    <div className="flex flex-col gap-5">
      {/* Top toolbar: mode tabs + run controls */}
      <div className="flex flex-wrap items-center gap-3">
        <div className="flex items-center gap-1.5">
          {(['run', 'yaml', 'compare'] as const).map((m) => (
            <button
              key={m}
              onClick={() => setMode(m)}
              className={`px-3.5 py-1.5 rounded-xl text-caption font-semibold capitalize transition ${
                mode === m ? 'bg-primary text-on-primary dark:bg-inverse-primary dark:text-inverse-surface'
                : 'bg-surface-container dark:bg-white/5 text-on-surface-variant dark:text-outline'
              }`}
            >
              {m === 'run' ? '▶ Run' : m === 'yaml' ? 'YAML' : 'Compare'}
            </button>
          ))}
        </div>

        <button
          onClick={() => {
            if (playing) { setPlaying(false); setStep(r.run.trace.length) }
            else { setStep(0); setPlaying(true) }
          }}
          className="inline-flex items-center gap-1.5 px-3.5 py-1.5 rounded-xl bg-primary text-on-primary dark:bg-inverse-primary dark:text-inverse-surface font-label-md text-label-md hover:opacity-90 transition"
        >
          <span className="material-symbols-outlined" style={{ fontSize: 17 }}>
            {playing ? 'stop' : 'play_arrow'}
          </span>
          {playing ? 'Pause' : 'Replay run'}
        </button>

        <span className="text-caption text-outline truncate max-w-[45%]">
          task: “{r.run.task.slice(0, 26)}{r.run.task.length > 26 ? '…' : ''}”
        </span>

        <div className="ml-auto flex items-center gap-2">
          {chaosActive && (
            <span className="text-caption text-[#C8604A]">⚠ chaos</span>
          )}
          <span className={`text-caption font-mono font-semibold ${chaosActive ? 'text-[#C8604A]' : 'text-primary dark:text-inverse-primary'}`}>
            resilience {r.run.resilience}
          </span>
        </div>
      </div>

      {/* Collapsible primitive strip */}
      <div className="rounded-2xl bg-surface-container-lowest dark:bg-dark-surface border border-outline-variant/40 dark:border-white/10 px-4 py-3">
        <button
          onClick={() => setPaletteOpen((o) => !o)}
          className="flex items-center gap-2 text-caption font-semibold uppercase tracking-wider text-on-surface dark:text-dark-on-surface"
        >
          <span className="material-symbols-outlined" style={{ fontSize: 16 }}>{paletteOpen ? 'expand_less' : 'expand_more'}</span>
          Cognitive primitives
        </button>
        {paletteOpen && (
          <div className="mt-3 flex flex-wrap gap-x-6 gap-y-2">
            {r.groups.map((g) => (
              <div key={g} className="flex items-center gap-1.5">
                <span className="text-[10px] uppercase tracking-wider text-outline font-semibold mr-0.5">{g}</span>
                {Object.entries(r.primitives)
                  .filter(([, p]) => p.group === g)
                  .map(([id, p]) => (
                    <button
                      key={id}
                      onClick={() => setSelected(selected === id ? null : id)}
                      title={p.desc}
                      className={`inline-flex items-center gap-1 px-2 py-1 rounded-lg text-caption transition ${
                        selected === id
                          ? 'bg-primary text-on-primary dark:bg-inverse-primary dark:text-inverse-surface font-semibold'
                          : 'bg-surface-container dark:bg-white/5 text-on-surface-variant dark:text-outline hover:bg-surface-variant dark:hover:bg-white/10'
                      }`}
                    >
                      <span className="material-symbols-outlined" style={{ fontSize: 14 }}>{p.icon}</span>
                      {p.label}
                    </button>
                  ))}
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Full-width wiring canvas */}
      <div className="bg-surface-container-lowest dark:bg-dark-surface rounded-3xl p-5 shadow-ambient dark:shadow-dark-ambient border border-outline-variant/40 dark:border-white/10 overflow-x-auto">
        <div className="mb-3 flex items-center justify-between">
          <h3 className="font-headline text-lg text-on-surface dark:text-inverse-on-surface inline-flex items-center gap-2">
            <span className="material-symbols-outlined" style={{ fontSize: 20 }}>construction</span>
            {r.presetName}
          </h3>
          <span className="text-caption text-outline">{r.graph.nodes.length} nodes · {r.graph.edges.length} edges</span>
        </div>
        <div className="relative" style={{ width: layout.width, height: layout.height }}>
          {/* Edges */}
          <svg className="absolute inset-0 w-full h-full pointer-events-none" width={layout.width} height={layout.height}>
            {layout.edges.map((e, i) => {
              const a = layout.nodes.find((n) => n.id === e.from)
              const b = layout.nodes.find((n) => n.id === e.to)
              if (!a || !b) return null
              const x1 = a.x + NODE_W, y1 = a.y + NODE_H / 2
              const x2 = b.x, y2 = b.y + NODE_H / 2
              const mx = (x1 + x2) / 2
              const active = traceNodes.has(e.from) && traceNodes.has(e.to)
              return (
                <g key={i}>
                  <path
                    d={`M${x1},${y1} C${mx},${y1} ${mx},${y2} ${x2},${y2}`}
                    fill="none"
                    stroke={active ? '#A8382A' : 'rgba(140,122,107,0.35)'}
                    strokeWidth={active ? 2 : 1.2}
                  />
                  <circle cx={x2 - 8} cy={y2} r="3" fill={active ? '#A8382A' : 'rgba(140,122,107,0.45)'} />
                </g>
              )
            })}
          </svg>
          {/* Nodes */}
          {layout.nodes.map((n) => {
            const meta = r.primitives[n.id]
            const isSel = selected === n.id
            const executed = traceNodes.has(n.id)
            const failed = failedNodes.has(n.id)
            const stepInfo = runTrace.find((s) => s.node === n.id)
            return (
              <button
                key={n.id}
                onClick={() => setSelected(n.id)}
                className={`absolute rounded-2xl border p-2.5 text-left transition-all ${
                  isSel ? 'ring-2 ring-primary dark:ring-inverse-primary scale-105 z-10'
                  : failed ? 'border-[#C8604A]'
                  : executed ? 'border-primary/60 dark:border-inverse-primary/60'
                  : 'border-outline-variant/60 dark:border-white/10'
                } ${failed ? 'bg-[#f3dfdc] dark:bg-[#3d2a28]'
                  : executed ? 'bg-primary-fixed/40 dark:bg-white/5'
                  : 'bg-surface-container dark:bg-dark-surface'}`}
                style={{ left: n.x, top: n.y, width: NODE_W, minHeight: NODE_H }}
              >
                <div className="flex items-center gap-1.5">
                  <span className="material-symbols-outlined" style={{ fontSize: 15, color: failed ? '#C8604A' : undefined }}>
                    {failed ? 'error' : meta?.icon ?? 'crop_square'}
                  </span>
                  <span className="font-semibold text-caption text-on-surface dark:text-dark-on-surface truncate">
                    {meta?.label ?? n.id}
                  </span>
                </div>
                {stepInfo && executed && (
                  <div className="mt-1 text-[10px] font-mono text-outline leading-tight">
                    {stepInfo.latency}ms{stepInfo.tokens ? ` · ${stepInfo.tokens}t` : ''}{stepInfo.calls ? ` · ×${stepInfo.calls}` : ''}
                  </div>
                )}
                {failed && (
                  <div className="mt-1 text-[10px] text-[#C8604A] font-semibold">✗ failed</div>
                )}
                <div className="mt-1.5 flex items-center justify-between">
                  <PortDots types={meta?.ports.in ?? []} side="in" />
                  <PortDots types={meta?.ports.out ?? []} side="out" />
                </div>
              </button>
            )
          })}
        </div>
        <HintLine graph={r.graph} primitives={r.primitives} />
      </div>

      {/* Inspector below the canvas */}
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-5">
        <div className="flex flex-col gap-5 min-w-0">
          {mode === 'run' && <RunInspector run={r.run} step={step} />}
          {mode === 'yaml' && <YamlCard yaml={r.yaml} />}
          {mode === 'compare' && <CompareCard compare={r.compare} />}
        </div>
        <AnatomyCard meta={sel} id={selected} />
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

function PortDots({ types, side }: { types: string[]; side: 'in' | 'out' }) {
  if (types.length === 0) return <span className="text-[9px] text-outline">{side === 'in' ? 'in:—' : ''}</span>
  return (
    <span className="flex gap-1">
      {types.map((t) => (
        <span
          key={t}
          title={`${side} port: ${t}`}
          className={`w-2 h-2 rounded-full ${side === 'in' ? 'bg-[#3A6B58]' : 'bg-[#C88A35]'}`}
        />
      ))}
    </span>
  )
}

// Semantic contract checker: finds edges that violate port compatibility.
function HintLine({ graph, primitives }: { graph: ForgeResult['graph']; primitives: Record<string, PrimitiveMeta> }) {
  const problems = graph.edges
    .map((e) => {
      const dst = primitives[e.to]
      if (!dst) return null
      if (e.port && !dst.ports.in.includes(e.port)) {
        return { from: e.from, to: e.to, port: e.port, expects: dst.ports.in }
      }
      return null
    })
    .filter(Boolean) as { from: string; to: string; port: string; expects: string[] }[]

  if (problems.length === 0) {
    return (
      <div className="mt-3 text-caption text-[#3A6B58] dark:text-[#a8d3dc]">
        ✓ all edges satisfy semantic port contracts
      </div>
    )
  }
  return (
    <div className="mt-3 rounded-xl bg-[#f3dfdc] dark:bg-[#3d2a28] text-[#8a3a35] dark:text-[#e9b8b2] px-3 py-2 text-caption">
      {problems.map((p, i) => (
        <div key={i}>
          ⚠ incompatible contracts: {primitives[p.from]?.label}.{p.port} → {primitives[p.to]?.label} expects [{p.expects.join(', ')}]
        </div>
      ))}
    </div>
  )
}

// Run trace: each step with latency / tokens / calls and failure recovery.
function RunInspector({ run, step }: { run: ForgeResult['run']; step: number }) {
  return (
    <div className="rounded-2xl bg-surface-container-lowest dark:bg-dark-surface border border-outline-variant/40 dark:border-white/10 p-4">
      <h4 className="font-label-md text-label-md uppercase tracking-wider text-on-surface dark:text-dark-on-surface mb-3 inline-flex items-center gap-2">
        <span className="material-symbols-outlined" style={{ fontSize: 16 }}>graphic_eq</span>
        Run trace
      </h4>
      <div className="flex flex-col gap-1.5 max-h-72 overflow-y-auto pr-1">
        {run.trace.map((s, i) => (
          <div
            key={i}
            className={`rounded-lg px-3 py-2 border ${
              i < step ? 'border-primary/40 dark:border-inverse-primary/40 bg-primary-fixed/30 dark:bg-white/5'
              : 'border-outline-variant/40 dark:border-white/10'
            }`}
          >
            <div className="flex items-center justify-between gap-2">
              <span className="text-caption font-semibold text-on-surface dark:text-dark-on-surface inline-flex items-center gap-1.5">
                <span className="material-symbols-outlined" style={{ fontSize: 14 }}>{s.icon}</span>
                {s.label}
              </span>
              <span className="text-[10px] font-mono text-outline">
                {s.latency}ms{s.tokens ? ` · ${s.tokens}t` : ''}
              </span>
            </div>
            <div className="text-[10px] text-outline mt-0.5">{s.detail}</div>
            {s.failed && (
              <div className="mt-1 text-[10px] text-[#C8604A]">✗ {s.label} hit a failure</div>
            )}
          </div>
        ))}
      </div>
      <div className="mt-3 flex items-center justify-between text-caption text-outline">
        <span className="font-mono">
          {run.totals.latency.toFixed(1)}ms · {run.totals.tokens}t · {run.totals.llm_calls} LLM calls
        </span>
        <span className="font-semibold text-primary dark:text-inverse-primary">resilience {run.resilience}</span>
      </div>
      {run.failures.length > 0 && (
        <div className="mt-2 rounded-xl bg-[#f3dfdc] dark:bg-[#3d2a28] text-[#8a3a35] dark:text-[#e9b8b2] px-3 py-2 text-caption">
          {run.failures.map((f, i) => (
            <div key={i} className="mb-1 last:mb-0">
              <b>{f.label}</b>: {f.recovery}
            </div>
          ))}
          <div className="mt-1 font-semibold">Agent survived · {run.recovered} recovery actions</div>
        </div>
      )}
    </div>
  )
}

// Compiled YAML output.
function YamlCard({ yaml }: { yaml: string }) {
  return (
    <div className="rounded-2xl bg-surface-container-lowest dark:bg-dark-surface border border-outline-variant/40 dark:border-white/10 p-4">
      <h4 className="font-label-md text-label-md uppercase tracking-wider text-on-surface dark:text-dark-on-surface mb-3 inline-flex items-center gap-2">
        <span className="material-symbols-outlined" style={{ fontSize: 16 }}>data_object</span>
        Compiled YAML
      </h4>
      <pre className="bg-surface-container dark:bg-white/5 rounded-xl p-3 overflow-x-auto text-[11px] font-mono text-primary dark:text-inverse-primary leading-relaxed">
        {yaml}
      </pre>
    </div>
  )
}

// Compare view: baseline vs current architecture behavior.
function CompareCard({ compare }: { compare: ForgeResult['compare'] }) {
  const rows = [
    ['latency', 'latency (ms)'],
    ['tokens', 'tokens'],
    ['calls', 'tool calls'],
    ['llm_calls', 'LLM calls'],
  ] as const
  return (
    <div className="rounded-2xl bg-surface-container-lowest dark:bg-dark-surface border border-outline-variant/40 dark:border-white/10 p-4">
      <h4 className="font-label-md text-label-md uppercase tracking-wider text-on-surface dark:text-dark-on-surface mb-3 inline-flex items-center gap-2">
        <span className="material-symbols-outlined" style={{ fontSize: 16 }}>compare_arrows</span>
        Baseline vs current
      </h4>
      <table className="w-full text-caption">
        <thead>
          <tr className="text-outline text-left">
            <th className="pb-1 font-semibold">metric</th>
            <th className="pb-1 font-semibold text-right">baseline</th>
            <th className="pb-1 font-semibold text-right">current</th>
          </tr>
        </thead>
        <tbody>
          {rows.map(([key, label]) => {
            const a = compare.a[key]
            const b = compare.b[key]
            const diff = compare.diffs.find((d) => d.metric === key)
            return (
              <tr key={key} className="border-t border-outline-variant/30 dark:border-white/10">
                <td className="py-1 text-on-surface-variant dark:text-outline">{label}</td>
                <td className="py-1 text-right font-mono">{typeof a === 'number' ? a.toFixed(1) : a}</td>
                <td className={`py-1 text-right font-mono ${diff ? (diff.direction === 'up' ? 'text-[#C8604A]' : 'text-[#3A6B58]') : 'text-on-surface dark:text-dark-on-surface'}`}>
                  {typeof b === 'number' ? b.toFixed(1) : b}
                  {diff ? (diff.direction === 'up' ? ' ▲' : ' ▼') : ''}
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
      <div className="mt-2 text-caption text-outline">
        {compare.active ? 'current run includes chaos injections' : 'enable “compare with baseline” to see behavior diff'}
      </div>
    </div>
  )
}

// Anatomy: why / trade-offs / linked ARES note for the selected primitive.
function AnatomyCard({ meta, id }: { meta: PrimitiveMeta | null; id: string | null }) {
  if (!meta || !id) {
    return (
      <div className="rounded-2xl bg-surface-container-lowest dark:bg-dark-surface border border-outline-variant/40 dark:border-white/10 p-4 text-caption text-outline">
        Select a primitive to inspect its anatomy — why it exists, its trade-offs, and the ARES note behind it.
      </div>
    )
  }
  return (
    <div className="rounded-2xl bg-surface-container-lowest dark:bg-dark-surface border border-outline-variant/40 dark:border-white/10 p-4">
      <h4 className="font-label-md text-label-md uppercase tracking-wider text-on-surface dark:text-dark-on-surface mb-2 inline-flex items-center gap-2">
        <span className="material-symbols-outlined" style={{ fontSize: 16 }}>stethoscope</span>
        {meta.label}
      </h4>
      <p className="text-caption text-on-surface-variant dark:text-outline leading-relaxed">{meta.desc}</p>
      <div className="mt-3">
        <div className="text-[10px] uppercase tracking-wider text-outline font-semibold mb-1">Why</div>
        <p className="text-caption text-on-surface dark:text-dark-on-surface leading-relaxed">{meta.why}</p>
      </div>
      <div className="mt-3">
        <div className="text-[10px] uppercase tracking-wider text-outline font-semibold mb-1">Trade-offs</div>
        <div className="flex flex-col gap-0.5">
          {meta.tradeoffs.map((t, i) => (
            <span key={i} className="text-caption text-on-surface-variant dark:text-outline">{t}</span>
          ))}
        </div>
      </div>
      <div className="mt-3 rounded-xl bg-surface-container dark:bg-white/5 px-3 py-2">
        <div className="text-[10px] uppercase tracking-wider text-outline font-semibold mb-0.5">ARES note</div>
        <div className="text-caption font-mono text-primary dark:text-inverse-primary">{meta.note}</div>
      </div>
    </div>
  )
}
