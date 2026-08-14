import type { LabResult } from '../types'

// Agent builder lab: the sidebar controls pick one option per category
// (memory / tools / planning / multi-agent). This component renders the
// assembled agent as a layered block diagram plus a plain-English summary
// and the generated YAML config.

interface BuilderNode {
  category: string
  categoryLabel: string
  icon: string
  choice: string
  label: string
  why: string
  behavior: string
}

interface BuilderResult extends LabResult {
  diagram: BuilderNode[]
  architecture: string
  config: string
}

// Render order for the block diagram: LLM core at the bottom, then the four
// capability layers on top of it.
const LAYER_ORDER = ['memory', 'tools', 'planning', 'multi']

export default function AgentBuilderLab({ result }: { result: LabResult | null }) {
  const r = result as BuilderResult | null
  if (!r || !Array.isArray(r.diagram)) return null

  const byCategory: Record<string, BuilderNode> = {}
  for (const node of r.diagram) byCategory[node.category] = node
  const layers = LAYER_ORDER.map((c) => byCategory[c]).filter(Boolean)

  return (
    <div className="flex flex-col gap-5">
      {/* Block diagram */}
      <div className="bg-surface-container-lowest dark:bg-dark-surface rounded-3xl p-6 shadow-ambient dark:shadow-dark-ambient border border-outline-variant/40 dark:border-white/10">
        <h3 className="font-headline text-lg text-on-surface dark:text-inverse-on-surface inline-flex items-center gap-2 mb-5">
          <span className="material-symbols-outlined" style={{ fontSize: 20 }}>widgets</span>
          Assembled agent
        </h3>
        <div className="flex flex-col items-center gap-2">
          {layers.map((node, i) => (
            <div key={node.category} className="w-full max-w-xl">
              <LayerCard node={node} last={i === layers.length - 1} />
            </div>
          ))}
          {/* LLM brain at the base */}
          <div className="w-full max-w-xl rounded-2xl bg-[#3D322C] text-[#F4EAE1] dark:bg-[#201B13] dark:text-[#D9CDB8] border border-outline-variant/60 dark:border-white/15 px-5 py-3 flex items-center gap-3">
            <span className="material-symbols-outlined" style={{ fontSize: 20 }}>psychology</span>
            <div className="min-w-0">
              <div className="font-semibold text-body-md">LLM brain</div>
              <div className="text-caption opacity-75">routes every request: reads memory, calls tools, plans steps</div>
            </div>
          </div>
        </div>
      </div>

      {/* Architecture summary */}
      <div className="bg-surface-container-lowest dark:bg-dark-surface rounded-3xl p-6 shadow-ambient dark:shadow-dark-ambient border border-outline-variant/40 dark:border-white/10">
        <h3 className="font-headline text-lg text-on-surface dark:text-inverse-on-surface inline-flex items-center gap-2 mb-3">
          <span className="material-symbols-outlined" style={{ fontSize: 20 }}>architecture</span>
          How it fits together
        </h3>
        <p className="text-body-md text-on-surface-variant dark:text-outline leading-relaxed">{r.architecture}</p>
      </div>

      {/* Generated config */}
      <div className="bg-surface-container-lowest dark:bg-dark-surface rounded-3xl p-6 shadow-ambient dark:shadow-dark-ambient border border-outline-variant/40 dark:border-white/10">
        <h3 className="font-headline text-lg text-on-surface dark:text-inverse-on-surface inline-flex items-center gap-2 mb-3">
          <span className="material-symbols-outlined" style={{ fontSize: 20 }}>data_object</span>
          Config sketch (YAML)
        </h3>
        <pre className="bg-surface-container dark:bg-dark-surface-elevated dark:bg-white/5 rounded-2xl p-4 overflow-x-auto text-caption font-mono text-primary dark:text-inverse-primary leading-relaxed">
          {r.config}
        </pre>
      </div>
    </div>
  )
}

// One selectable layer in the block diagram: category name + chosen option.
function LayerCard({ node, last }: { node: BuilderNode; last: boolean }) {
  return (
    <div className="flex flex-col gap-1 w-full">
      <div className="rounded-2xl bg-surface-container dark:bg-dark-surface-elevated dark:bg-white/5 border border-outline-variant/60 dark:border-white/10 px-5 py-3 flex items-center gap-3">
        <span
          className={`material-symbols-outlined ${
            last ? 'text-primary dark:text-inverse-primary' : 'text-outline'
          }`}
          style={{ fontSize: 20 }}
        >
          {node.icon}
        </span>
        <div className="min-w-0 flex-1">
          <div className="text-caption uppercase tracking-wider text-outline font-semibold">
            {node.categoryLabel}
          </div>
          <div className="font-semibold text-body-md text-on-surface dark:text-dark-on-surface">
            {node.label}
          </div>
        </div>
        <span className="text-caption text-outline text-right max-w-[40%] hidden sm:block">{node.why}</span>
      </div>
      <div className="text-caption text-outline px-5 pb-2 -mt-1">{node.behavior}</div>
    </div>
  )
}
