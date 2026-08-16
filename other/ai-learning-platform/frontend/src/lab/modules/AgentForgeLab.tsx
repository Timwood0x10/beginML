import { useEffect, useMemo, useRef, useState } from "react";
import type { LabResult } from "../types";

// Agent Forge — a cognitive-system workbench.
// The canvas is the desktop (75%+ of the space): drag primitives from the
// toolbox, wire their semantic ports, and compile the agent. Trace /
// inspector / knowledge open as drawers instead of occupying the canvas.
// Build, Run and Experiment are the three top-level modes.

interface PrimitiveMeta {
  label: string;
  group: string;
  icon: string;
  ports: { in: string[]; out: string[] };
  desc: string;
  why: string;
  tradeoffs: string[];
  note: string;
  class: string;
  rarity: string;
}

interface BrickClassMeta {
  label: string;
  icon: string;
  color: string;
  rarity: string;
}

interface RecoveryBrickMeta {
  label: string;
  icon: string;
  desc: string;
  recovers: string[];
}

interface BreakBrickMeta {
  id: string;
  label: string;
  icon: string;
  targets: string[];
  desc: string;
}

interface SkillBoxMeta {
  id: string;
  label: string;
  icon: string;
  desc: string;
  tools: string[];
  inner: { id: string; label: string; icon: string }[];
}

interface TraceStep {
  node: string;
  label: string;
  icon: string;
  latency: number;
  tokens: number;
  calls: number;
  llm: boolean;
  failed: boolean;
  detail: string;
}

interface ForgeRun {
  task: string;
  trace: TraceStep[];
  totals: { latency: number; tokens: number; calls: number; llm_calls: number };
  failures: {
    node: string;
    label: string;
    icon: string;
    mode: string;
    message: string;
    impact: string;
    recovery: string;
    recovered: boolean;
    needs: string | null;
  }[];
  recovered: number;
  unrecovered: number;
  resilience: number;
  status: string;
  chaos: string[];
  recoveries: string[];
  discovery: { phase: string; item: string; kind: string }[];
}

interface ForgeResult extends LabResult {
  preset: string;
  presetName: string;
  primitives: Record<string, PrimitiveMeta>;
  brickClasses: Record<string, BrickClassMeta>;
  breakBricks: BreakBrickMeta[];
  recoveryBricks: Record<string, RecoveryBrickMeta>;
  skillBoxes: SkillBoxMeta[];
  discoveryRegistry: { id: string; kind: string; icon: string; desc: string }[];
  groups: string[];
  examples: { id: string; name: string; desc: string }[];
  graph: {
    name: string;
    nodes: { id: string }[];
    edges: { from: string; to: string; port: string }[];
  };
  run: ForgeRun;
  yaml: string;
  validation: {
    valid: boolean;
    violations: {
      kind: string;
      from: string;
      to: string;
      port: string;
      expected: string[];
      message: string;
    }[];
  };
  compare: {
    active: boolean;
    a: { latency: number; tokens: number; calls: number; llm_calls: number };
    b: { latency: number; tokens: number; calls: number; llm_calls: number };
    diffs: { metric: string; a: number; b: number; direction: string }[];
  };
  // experiment mode payload
  baseline?: {
    name: string;
    success: boolean;
    resilience: number;
    llm_calls: number;
    tokens: number;
    latency: number;
    recovery: number;
    failures: number;
  };
  variant?: {
    name: string;
    success: boolean;
    resilience: number;
    llm_calls: number;
    tokens: number;
    latency: number;
    recovery: number;
    failures: number;
  };
  // evolve mode payload
  history?: { gen: number; best: number; avg: number }[];
  best?: {
    fitness: number;
    graph: {
      name: string;
      nodes: { id: string }[];
      edges: { from: string; to: string; port: string }[];
    };
  };
  fittest?: { name: string; fitness: number }[];
}

interface CanvasNode {
  id: string;
  type: string;
  x: number;
  y: number;
}
interface CanvasEdge {
  from: string;
  to: string;
  port: string;
}

const NODE_W = 150;
const NODE_H = 96;
const GROUP_GAP = 26;

// Helper: pick a color per port type — semantic system variables so the
// hue-meaning mapping (task=brain, context=memory, action=amber,
// result=agent) follows every theme.
const PORT_COLORS: Record<string, string> = {
  task: "var(--ailearn-semantic-brain)",
  context: "var(--ailearn-semantic-memory)",
  action: "var(--ailearn-semantic-action)",
  result: "var(--ailearn-semantic-agent)",
};

export default function AgentForgeLab({
  result,
  params,
  setParams,
}: {
  result: LabResult | null;
  params: Record<string, unknown>;
  setParams: (p: Record<string, unknown>) => void;
}) {
  const r = result as ForgeResult | null;
  const [mode, setMode] = useState<"build" | "run" | "experiment">("build");
  const [nodes, setNodes] = useState<CanvasNode[]>([]);
  const [edges, setEdges] = useState<CanvasEdge[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [drawer, setDrawer] = useState<
    null | "trace" | "inspect" | "knowledge"
  >(null);
  const [playing, setPlaying] = useState(false);
  const [step, setStep] = useState(0);
  const [dragType, setDragType] = useState<string | null>(null);
  const [dragPos, setDragPos] = useState<{ x: number; y: number } | null>(null);
  const [wire, setWire] = useState<{
    from: string;
    port: string;
    x: number;
    y: number;
  } | null>(null);
  const [mismatch, setMismatch] = useState<string | null>(null);
  const [openSkill, setOpenSkill] = useState<string | null>(null);
  // BREAK mode: nodeId -> failure mode ("pull a brick apart").
  const [breaks, setBreaks] = useState<Record<string, string>>({});
  const [breakDrag, setBreakDrag] = useState<string | null>(null);
  // Attached Recovery bricks (pluggable self-healing).
  const [recoveries, setRecoveries] = useState<string[]>([]);
  // Installed skills (a skill that has been loaded into the canvas).
  const [installedSkills, setInstalledSkills] = useState<string[]>([]);
  const lastPreset = useRef<string | null>(null);
  const canvasRef = useRef<HTMLDivElement | null>(null);

  // Sync canvas from the backend graph when the preset changes (not on every
  // compute — local edits must survive a RUN round-trip).
  useEffect(() => {
    if (!r) return;
    if (r.preset !== lastPreset.current) {
      lastPreset.current = r.preset;
      setNodes(
        r.graph.nodes.map((n, i) => ({
          id: n.id,
          type: n.id,
          x: gridX(i),
          y: gridY(i),
        })),
      );
      setEdges(
        r.graph.edges.map((e) => ({ from: e.from, to: e.to, port: e.port })),
      );
      setSelected(null);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [r?.preset]);

  // Live contract validation using local edges.
  const validation = useMemo(() => {
    const violations: {
      from: string;
      to: string;
      port: string;
      expected: string[];
      message: string;
    }[] = [];
    for (const e of edges) {
      const dst = r?.primitives[e.to];
      if (!dst) continue;
      if (!dst.ports.in.includes(e.port)) {
        violations.push({
          from: e.from,
          to: e.to,
          port: e.port,
          expected: dst.ports.in,
          message: `${r?.primitives[e.from]?.label}.${e.port} → ${dst.label} expects [${dst.ports.in.join(", ")}]`,
        });
      }
    }
    return { valid: violations.length === 0, violations };
  }, [edges, r]);

  // Replay playback.
  useEffect(() => {
    if (!playing || !r) return;
    if (step >= r.run.trace.length) {
      setPlaying(false);
      return;
    }
    const t = setTimeout(() => setStep((s) => s + 1), 420);
    return () => clearTimeout(t);
  }, [playing, step, r]);

  if (!r) return null;

  const traceNodes = new Set(r.run.trace.slice(0, step).map((s) => s.node));
  const failedNodes = new Set(r.run.failures.map((f) => f.node));

  const runNow = () => {
    // Push the local graph to the backend via params (triggers recompute).
    setParams({
      ...params,
      graph: { nodes: nodes.map((n) => ({ id: n.type })), edges },
      breaks,
      recoveries,
      runToken: Date.now(),
    });
    setMode("run");
    setStep(0);
    setPlaying(true);
  };

  const loadExample = (id: string) => {
    setParams({
      ...params,
      preset: id,
      graph: undefined,
      runToken: Date.now(),
    });
  };

  const addPrimitive = (type: string, x: number, y: number) => {
    const id = `${type}-${nodes.length + 1}`;
    setNodes((ns) => [...ns, { id, type, x, y }]);
    setSelected(id);
  };

  // Expand a Skill box into its inner tools on the canvas ("big brick
  // contains small bricks"): each inner tool becomes its own node.
  const expandSkill = (skill: SkillBoxMeta) => {
    const base = nodes.length;
    const next = skill.inner.map((t, i) => ({
      id: `${t.id}-${base + i + 1}`,
      type: t.id,
      x: 24 + (i % 3) * (NODE_W + 20),
      y: 24 + Math.floor(i / 3) * (NODE_H + GROUP_GAP),
    }));
    setNodes((ns) => [...ns, ...next]);
    setOpenSkill(null);
  };

  // INSTALL: mark a skill as loaded into the agent (and expand it).
  const installSkill = (skill: SkillBoxMeta) => {
    if (!installedSkills.includes(skill.id)) {
      setInstalledSkills((s) => [...s, skill.id]);
    }
    expandSkill(skill);
  };

  // REMOVE: drop the skill and pull its inner bricks off the canvas.
  const removeSkill = (skill: SkillBoxMeta) => {
    const types = new Set(skill.inner.map((t) => t.id));
    setNodes((ns) => ns.filter((n) => !types.has(n.type)));
    setEdges((es) => {
      const ids = new Set(
        nodes.filter((n) => types.has(n.type)).map((n) => n.id),
      );
      return es.filter((e) => !ids.has(e.from) && !ids.has(e.to));
    });
    setInstalledSkills((s) => s.filter((id) => id !== skill.id));
    if (openSkill === skill.id) setOpenSkill(null);
  };

  // -- canvas pointer handlers -------------------------------------------

  const onCanvasDrop = (e: React.DragEvent) => {
    e.preventDefault();
    if (!dragType) return;
    const rect = (e.currentTarget as HTMLElement).getBoundingClientRect();
    addPrimitive(
      dragType,
      e.clientX - rect.left - NODE_W / 2,
      e.clientY - rect.top - NODE_H / 2,
    );
    setDragType(null);
    setDragPos(null);
  };

  const onCanvasDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    const rect = (e.currentTarget as HTMLElement).getBoundingClientRect();
    setDragPos({ x: e.clientX - rect.left, y: e.clientY - rect.top });
  };

  const moveNode = (
    id: string,
    originX: number,
    originY: number,
    dx: number,
    dy: number,
  ) => {
    // Absolute position = where the drag started + total delta. Using the
    // node's current x/y here would compound each mousemove and cause the
    // node to drift far ahead of the cursor.
    setNodes((ns) =>
      ns.map((n) =>
        n.id === id ? { ...n, x: originX + dx, y: originY + dy } : n,
      ),
    );
  };

  const startWire = (
    from: string,
    port: string,
    clientX: number,
    clientY: number,
  ) => {
    setWire({ from, port, x: clientX, y: clientY });
    const onMove = (ev: MouseEvent) =>
      setWire((w) => (w ? { ...w, x: ev.clientX, y: ev.clientY } : w));
    const onUp = (ev: MouseEvent) => {
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
      // Hit-test: which node is under the cursor right now?
      const canvas = canvasRef.current;
      if (!canvas) {
        setWire(null);
        return;
      }
      const rect = canvas.getBoundingClientRect();
      const cx = ev.clientX - rect.left;
      const cy = ev.clientY - rect.top;
      const target = nodes.find(
        (n) =>
          cx >= n.x && cx <= n.x + NODE_W && cy >= n.y && cy <= n.y + NODE_H,
      );
      endWire(target ? target.id : "");
    };
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
  };

  const endWire = (to: string) => {
    if (!wire) return;
    if (!to) {
      setWire(null);
      return;
    }
    const dst = r.primitives[to];
    if (!dst) {
      setWire(null);
      return;
    }
    if (dst.ports.in.includes(wire.port)) {
      setEdges((es) => {
        const rest = es.filter((e) => !(e.from === wire.from && e.to === to));
        return [...rest, { from: wire.from, to, port: wire.port }];
      });
      setMismatch(null);
    } else {
      setMismatch(
        `${r.primitives[wire.from]?.label}.${wire.port} → ${dst.label} expects [${dst.ports.in.join(", ")}]`,
      );
      setTimeout(() => setMismatch(null), 2600);
    }
    setWire(null);
  };

  const deleteNode = (id: string) => {
    setNodes((ns) => ns.filter((n) => n.id !== id));
    setEdges((es) => es.filter((e) => e.from !== id && e.to !== id));
    if (selected === id) setSelected(null);
  };

  const drawerContent = () => {
    if (drawer === "trace") return <TraceDrawer run={r.run} step={step} />;
    if (drawer === "inspect")
      return (
        <InspectDrawer
          meta={selected ? r.primitives[selected] : null}
          id={selected}
        />
      );
    if (drawer === "knowledge")
      return <KnowledgeDrawer primitives={r.primitives} groups={r.groups} />;
    return null;
  };

  return (
    <div className="flex flex-col gap-4">
      {/* Top bar: modes + run + status */}
      <div className="flex flex-wrap items-center gap-2">
        {(["build", "run", "experiment"] as const).map((m) => (
          <button
            key={m}
            onClick={() => {
              setMode(m);
              if (m !== "run") {
                setPlaying(false);
                setStep(0);
              }
            }}
            className={`px-4 py-1.5 rounded-xl text-caption font-bold uppercase tracking-wider transition ${
              mode === m
                ? "bg-primary text-on-primary dark:bg-inverse-primary dark:text-inverse-surface"
                : "bg-surface-container dark:bg-white/5 text-on-surface-variant dark:text-outline hover:bg-surface-variant dark:hover:bg-white/10"
            }`}
          >
            {m}
          </button>
        ))}

        {mode === "build" && (
          <button
            onClick={runNow}
            className="inline-flex items-center gap-1.5 px-4 py-1.5 rounded-xl bg-primary text-on-primary dark:bg-inverse-primary dark:text-inverse-surface font-label-md text-label-md hover:opacity-90 transition"
          >
            <span
              className="material-symbols-outlined"
              style={{ fontSize: 17 }}
            >
              build
            </span>
            Build & run
          </button>
        )}
        {mode === "run" && (
          <button
            onClick={() => {
              if (playing) {
                setPlaying(false);
                setStep(r.run.trace.length);
              } else {
                setStep(0);
                setPlaying(true);
              }
            }}
            className="inline-flex items-center gap-1.5 px-4 py-1.5 rounded-xl bg-primary text-on-primary dark:bg-inverse-primary dark:text-inverse-surface font-label-md text-label-md hover:opacity-90 transition"
          >
            <span
              className="material-symbols-outlined"
              style={{ fontSize: 17 }}
            >
              {playing ? "stop" : "play_arrow"}
            </span>
            {playing ? "Pause" : "Replay"}
          </button>
        )}

        <span className="text-caption text-outline truncate max-w-[30%] ml-1">
          {r.presetName}
        </span>

        <div className="ml-auto flex items-center gap-2">
          {mode === "build" && (
            <span
              className={`inline-flex items-center gap-1.5 text-caption font-semibold ${validation.valid ? "text-[#3A6B58] dark:text-[#a8d3dc]" : "text-[#C8604A]"}`}
            >
              <span
                className={`w-2 h-2 rounded-full ${validation.valid ? "bg-[#3A6B58]" : "bg-[#C8604A]"}`}
              />
              {validation.valid
                ? "Agent valid"
                : `${validation.violations.length} contract violation${validation.violations.length > 1 ? "s" : ""}`}
            </span>
          )}
          {mode === "run" && (
            <span className="text-caption font-mono font-semibold text-primary dark:text-inverse-primary">
              resilience {r.run.resilience}
            </span>
          )}
        </div>
      </div>

      {/* Body: toolbox + canvas */}
      <div className="flex gap-4">
        {/* TOOLBOX */}
        <aside className="w-52 shrink-0 flex flex-col gap-4 max-h-[calc(100vh-12rem)] overflow-y-auto pr-1">
          <div className="rounded-2xl bg-surface-container-lowest dark:bg-dark-surface border border-outline-variant/40 dark:border-white/10 p-3">
            <h4 className="text-[10px] uppercase tracking-wider text-outline font-semibold mb-2">
              Examples
            </h4>
            <div className="flex flex-col gap-1">
              {r.examples.map((ex) => (
                <button
                  key={ex.id}
                  onClick={() => loadExample(ex.id)}
                  title={ex.desc}
                  className={`text-left px-2.5 py-1.5 rounded-lg text-caption transition ${
                    r.preset === ex.id && !params.graph
                      ? "bg-primary text-on-primary dark:bg-inverse-primary dark:text-inverse-surface font-semibold"
                      : "text-on-surface-variant dark:text-outline hover:bg-surface-variant dark:hover:bg-white/5"
                  }`}
                >
                  <span className="font-semibold">{ex.name}</span>
                  <span className="block text-[10px] opacity-75 truncate">
                    {ex.desc}
                  </span>
                </button>
              ))}
            </div>
          </div>

          <div className="rounded-2xl bg-surface-container-lowest dark:bg-dark-surface border border-outline-variant/40 dark:border-white/10 p-3">
            <h4 className="text-[10px] uppercase tracking-wider text-outline font-semibold mb-2">
              Bricks <span className="normal-case">(drag to canvas)</span>
            </h4>
            <div className="flex flex-col gap-2.5">
              {(
                [
                  "brain",
                  "memory",
                  "action",
                  "capability",
                  "recovery",
                  "agent",
                ] as const
              ).map((cls) => {
                const clsMeta = r.brickClasses[cls];
                if (!clsMeta) return null;
                // Recovery bricks are pluggable self-healing blocks.
                const isRecovery = cls === "recovery";
                const bricks = isRecovery
                  ? Object.entries(r.recoveryBricks).map(([id, b]) => ({
                      id,
                      label: b.label,
                      icon: b.icon,
                      desc: b.desc,
                    }))
                  : Object.entries(r.primitives)
                      .filter(([, p]) => p.class === cls)
                      .map(([id, p]) => ({
                        id,
                        label: p.label,
                        icon: p.icon,
                        desc: p.desc,
                      }));
                if (bricks.length === 0) return null;
                return (
                  <div key={cls}>
                    <div className="flex items-center gap-1.5 mb-1">
                      <span
                        className="w-2 h-2 rounded-full"
                        style={{ background: clsMeta.color }}
                      />
                      <span className="text-[9px] uppercase tracking-wider font-semibold text-on-surface-variant dark:text-outline">
                        {clsMeta.label}
                      </span>
                      <span className="text-[8px] px-1 rounded bg-surface-variant dark:bg-white/10 text-outline">
                        {clsMeta.rarity}
                      </span>
                    </div>
                    <div className="flex flex-col gap-0.5">
                      {bricks.map((b) => {
                        const attached = recoveries.includes(b.id);
                        return (
                          <div
                            key={b.id}
                            draggable
                            onDragStart={() => setDragType(b.id)}
                            onClick={() => {
                              if (isRecovery) {
                                // Recovery bricks are pluggable: click to
                                // attach / detach self-healing capability.
                                setRecoveries((rs) =>
                                  attached
                                    ? rs.filter((r) => r !== b.id)
                                    : [...rs, b.id],
                                );
                                return;
                              }
                              setSelected(selected === b.id ? null : b.id);
                            }}
                            className={`flex items-center gap-1.5 px-2 py-1 rounded-lg text-caption transition ${
                              attached
                                ? "bg-[#3A6B58] text-white dark:bg-[#3A6B58] font-semibold"
                                : selected === b.id
                                  ? "bg-primary text-on-primary dark:bg-inverse-primary dark:text-inverse-surface font-semibold"
                                  : "cursor-grab active:cursor-grabbing text-on-surface-variant dark:text-outline hover:bg-surface-variant dark:hover:bg-white/5"
                            }`}
                            title={b.desc}
                            style={
                              selected === b.id || attached
                                ? undefined
                                : { borderLeft: `3px solid ${clsMeta.color}` }
                            }
                          >
                            <span
                              className="material-symbols-outlined"
                              style={{ fontSize: 14 }}
                            >
                              {attached ? "check_circle" : b.icon}
                            </span>
                            {b.label}
                            {isRecovery && attached && (
                              <span className="ml-auto text-[8px] uppercase tracking-wider opacity-80">
                                attached
                              </span>
                            )}
                          </div>
                        );
                      })}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Skill boxes — a brick that contains bricks */}
          <div className="rounded-2xl bg-surface-container-lowest dark:bg-dark-surface border border-outline-variant/40 dark:border-white/10 p-3">
            <h4 className="text-[10px] uppercase tracking-wider text-outline font-semibold mb-2">
              Skill boxes{" "}
              <span className="normal-case">(a brick of bricks)</span>
            </h4>
            <div className="flex flex-col gap-1">
              {r.skillBoxes.map((sk) => (
                <div
                  key={sk.id}
                  className="rounded-lg border border-outline-variant/40 dark:border-white/10"
                >
                  <button
                    onClick={() =>
                      setOpenSkill(openSkill === sk.id ? null : sk.id)
                    }
                    className="w-full flex items-center gap-1.5 px-2 py-1.5 text-left text-caption text-on-surface dark:text-dark-on-surface hover:bg-surface-variant dark:hover:bg-white/5 transition rounded-t-lg"
                  >
                    <span
                      className="material-symbols-outlined"
                      style={{ fontSize: 15, color: "#7A5C8E" }}
                    >
                      {sk.icon}
                    </span>
                    <span className="font-semibold flex-1">{sk.label}</span>
                    <span
                      className="material-symbols-outlined text-outline"
                      style={{ fontSize: 13 }}
                    >
                      {openSkill === sk.id ? "expand_less" : "expand_more"}
                    </span>
                  </button>
                  {openSkill === sk.id && (
                    <div className="px-2 pb-2">
                      <div className="text-[10px] text-outline mb-1.5">
                        {sk.desc}
                      </div>
                      <div className="flex flex-col gap-0.5">
                        {sk.inner.map((t) => (
                          <div
                            key={t.id}
                            className="flex items-center gap-1.5 px-1.5 py-0.5 text-caption text-on-surface-variant dark:text-outline"
                          >
                            <span
                              className="material-symbols-outlined"
                              style={{ fontSize: 13 }}
                            >
                              {t.icon}
                            </span>
                            {t.label}
                          </div>
                        ))}
                      </div>
                      <div className="mt-1 text-[9px] text-outline">
                        {sk.inner.length} tools · MCP deps: {sk.tools.length}
                      </div>
                      <div className="mt-1.5 flex gap-1">
                        <button
                          onClick={() => installSkill(sk)}
                          className={`flex-1 inline-flex items-center justify-center gap-1 px-1.5 py-1 rounded-lg text-caption font-semibold transition ${
                            installedSkills.includes(sk.id)
                              ? "bg-[#3A6B58] text-white"
                              : "bg-[#3A6B58]/15 dark:bg-[#3A6B58]/25 text-[#2f6068] dark:text-[#a8d3dc] hover:opacity-90"
                          }`}
                        >
                          <span
                            className="material-symbols-outlined"
                            style={{ fontSize: 12 }}
                          >
                            download
                          </span>
                          Install
                        </button>
                        <button
                          onClick={() => expandSkill(sk)}
                          className="flex-1 inline-flex items-center justify-center gap-1 px-1.5 py-1 rounded-lg bg-primary/10 dark:bg-white/10 text-primary dark:text-inverse-primary text-caption font-semibold hover:opacity-90 transition"
                        >
                          <span
                            className="material-symbols-outlined"
                            style={{ fontSize: 12 }}
                          >
                            call_split
                          </span>
                          Open
                        </button>
                        <button
                          onClick={() => removeSkill(sk)}
                          disabled={!installedSkills.includes(sk.id)}
                          className="flex-1 inline-flex items-center justify-center gap-1 px-1.5 py-1 rounded-lg bg-[#C8604A]/10 dark:bg-[#C8604A]/20 text-[#C8604A] text-caption font-semibold hover:opacity-90 transition disabled:opacity-30"
                        >
                          <span
                            className="material-symbols-outlined"
                            style={{ fontSize: 12 }}
                          >
                            close
                          </span>
                          Remove
                        </button>
                      </div>
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>

          {/* BREAK bricks — drag a failure onto a node to break it */}
          <div className="rounded-2xl bg-surface-container-lowest dark:bg-dark-surface border border-outline-variant/40 dark:border-white/10 p-3">
            <h4 className="text-[10px] uppercase tracking-wider text-[#C8604A] font-semibold mb-2">
              💥 Break it{" "}
              <span className="normal-case">(drag onto a brick)</span>
            </h4>
            <div className="flex flex-col gap-1">
              {r.breakBricks.map((b) => (
                <div
                  key={b.id}
                  draggable
                  onDragStart={() => setBreakDrag(b.id)}
                  title={b.desc}
                  className={`flex items-center gap-1.5 px-2 py-1 rounded-lg text-caption cursor-grab active:cursor-grabbing transition ${
                    breakDrag === b.id
                      ? "bg-[#C8604A] text-white font-semibold"
                      : "text-on-surface-variant dark:text-outline hover:bg-[#C8604A]/10 dark:hover:bg-[#C8604A]/20"
                  }`}
                >
                  <span
                    className="material-symbols-outlined"
                    style={{ fontSize: 14 }}
                  >
                    {b.icon}
                  </span>
                  {b.label}
                </div>
              ))}
            </div>
          </div>
        </aside>

        {/* CANVAS — the desktop */}
        <main
          ref={canvasRef}
          className="flex-1 min-w-0 relative bg-surface-container-lowest dark:bg-dark-surface rounded-3xl border border-outline-variant/40 dark:border-white/10 overflow-hidden"
          style={{ minHeight: 560 }}
          onDrop={onCanvasDrop}
          onDragOver={onCanvasDragOver}
        >
          <div
            className="absolute inset-0 pointer-events-none"
            style={{
              backgroundImage:
                "radial-gradient(rgba(140,122,107,0.14) 1px, transparent 1px)",
              backgroundSize: "22px 22px",
            }}
          />
          <div className="absolute inset-0">
            {/* edges */}
            <svg className="absolute inset-0 w-full h-full pointer-events-none">
              {edges.map((e, i) => {
                const a = nodes.find((n) => n.id === e.from);
                const b = nodes.find((n) => n.id === e.to);
                if (!a || !b) return null;
                const x1 = a.x + NODE_W,
                  y1 = a.y + NODE_H / 2;
                const x2 = b.x,
                  y2 = b.y + NODE_H / 2;
                const mx = (x1 + x2) / 2;
                const active = traceNodes.has(e.from) && traceNodes.has(e.to);
                return (
                  <g key={i}>
                    <path
                      d={`M${x1},${y1} C${mx},${y1} ${mx},${y2} ${x2},${y2}`}
                      fill="none"
                      stroke={active ? "#A8382A" : "rgba(140,122,107,0.4)"}
                      strokeWidth={active ? 2 : 1.2}
                    />
                    <circle
                      cx={x2 - 7}
                      cy={y2}
                      r="3"
                      fill={active ? "#A8382A" : "rgba(140,122,107,0.5)"}
                    />
                  </g>
                );
              })}
              {wire &&
                (() => {
                  const a = nodes.find((n) => n.id === wire.from);
                  if (!a) return null;
                  const x1 = a.x + NODE_W,
                    y1 = a.y + NODE_H / 2;
                  const mx = (x1 + wire.x) / 2;
                  return (
                    <path
                      d={`M${x1},${y1} C${mx},${y1} ${mx},${wire.y} ${wire.x},${wire.y}`}
                      fill="none"
                      stroke="rgba(200,96,74,0.7)"
                      strokeWidth="1.6"
                      strokeDasharray="5 4"
                    />
                  );
                })()}
            </svg>

            {/* nodes */}
            {nodes.map((n) => {
              const meta = r.primitives[n.type];
              const isSel = selected === n.id;
              const executed = traceNodes.has(n.id);
              const failed = failedNodes.has(n.id);
              const failure = r.run.failures.find((f) => f.node === n.id);
              const stepInfo = r.run.trace.find((s) => s.node === n.id);
              const broken = breaks[n.id];
              const breakMeta = broken
                ? r.breakBricks.find((b) => b.id === broken)
                : null;
              return (
                <div
                  key={n.id}
                  onMouseDown={(e) => {
                    const sx = e.clientX,
                      sy = e.clientY,
                      nx = n.x,
                      ny = n.y;
                    const onMove = (ev: MouseEvent) =>
                      moveNode(n.id, nx, ny, ev.clientX - sx, ev.clientY - sy);
                    const onUp = () => {
                      window.removeEventListener("mousemove", onMove);
                      window.removeEventListener("mouseup", onUp);
                    };
                    window.addEventListener("mousemove", onMove);
                    window.addEventListener("mouseup", onUp);
                  }}
                  onDragOver={(e) => {
                    e.preventDefault();
                    e.stopPropagation();
                  }}
                  onDrop={(e) => {
                    // "Break it": drop a failure brick onto this node.
                    e.preventDefault();
                    e.stopPropagation();
                    if (breakDrag) {
                      setBreaks((b) => ({ ...b, [n.id]: breakDrag }));
                      setBreakDrag(null);
                    }
                  }}
                  onClick={(e) => e.stopPropagation()}
                  className={`absolute rounded-2xl border p-2.5 select-none ${
                    isSel
                      ? "ring-2 ring-primary dark:ring-inverse-primary z-10"
                      : broken
                        ? "border-[#C8604A]"
                        : failed
                          ? "border-[#C8604A] forge-node-failed"
                          : executed
                            ? "border-primary/60 dark:border-inverse-primary/60"
                            : "border-outline-variant/60 dark:border-white/10"
                  } ${
                    broken
                      ? "bg-[#f3dfdc] dark:bg-[#3d2a28]"
                      : failed
                        ? "bg-[#f3dfdc] dark:bg-[#3d2a28]"
                        : executed
                          ? "bg-primary-fixed/40 dark:bg-white/5"
                          : "bg-surface-container dark:bg-dark-surface"
                  }`}
                  style={{
                    left: n.x,
                    top: n.y,
                    width: NODE_W,
                    minHeight: NODE_H,
                    cursor: "move",
                  }}
                >
                  <div className="flex items-center justify-between">
                    <span className="font-semibold text-caption text-on-surface dark:text-dark-on-surface inline-flex items-center gap-1">
                      <span
                        className="material-symbols-outlined"
                        style={{
                          fontSize: 14,
                          color: failed || broken ? "#C8604A" : undefined,
                        }}
                      >
                        {failed || broken
                          ? "error"
                          : (meta?.icon ?? "crop_square")}
                      </span>
                      {meta?.label ?? n.type}
                    </span>
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        deleteNode(n.id);
                      }}
                      className="text-outline hover:text-[#C8604A] transition"
                      aria-label="delete node"
                    >
                      <span
                        className="material-symbols-outlined"
                        style={{ fontSize: 13 }}
                      >
                        close
                      </span>
                    </button>
                  </div>
                  {broken && breakMeta && (
                    <div
                      className="mt-0.5 inline-flex items-center gap-1 rounded bg-[#C8604A]/15 dark:bg-[#C8604A]/30 px-1.5 py-0.5 text-[9px] text-[#C8604A] font-semibold cursor-pointer"
                      title="click to remove break"
                      onClick={(e) => {
                        e.stopPropagation();
                        setBreaks((b) => {
                          const nb = { ...b };
                          delete nb[n.id];
                          return nb;
                        });
                      }}
                    >
                      <span
                        className="material-symbols-outlined"
                        style={{ fontSize: 10 }}
                      >
                        bolt
                      </span>
                      {breakMeta.label}
                    </div>
                  )}
                  {stepInfo && executed && (
                    <div className="mt-0.5 text-[10px] font-mono text-outline">
                      {stepInfo.latency}ms
                      {stepInfo.tokens ? ` · ${stepInfo.tokens}t` : ""}
                    </div>
                  )}
                  {failed && failure && (
                    <div className="mt-0.5 flex items-center gap-1 text-[9px] max-w-full">
                      {/* 💥 the failure */}
                      <span className="inline-flex items-center gap-0.5 rounded bg-[#C8604A]/15 dark:bg-[#C8604A]/30 px-1 py-0.5 text-[#C8604A] font-semibold">
                        <span
                          className="material-symbols-outlined"
                          style={{ fontSize: 9 }}
                        >
                          bolt
                        </span>
                        {failure.mode.replace("_", " ")}
                      </span>
                      <span className="text-[#C8604A]">✗</span>
                      {failure.recovered ? (
                        /* 🩹 recovered: the healing action won */
                        <span className="inline-flex items-center gap-0.5 rounded bg-[#3A6B58]/10 dark:bg-[#3A6B58]/25 px-1 py-0.5 text-[#2f6068] dark:text-[#a8d3dc]">
                          <span
                            className="material-symbols-outlined"
                            style={{ fontSize: 9 }}
                          >
                            healing
                          </span>
                          <span className="truncate">{failure.recovery}</span>
                        </span>
                      ) : (
                        /* no recovery brick attached → lethal */
                        <span className="rounded bg-[#C8604A]/10 px-1 py-0.5 text-[#C8604A]">
                          no recovery (
                          {failure.needs?.replace("recovery-", "") ?? "?"})
                        </span>
                      )}
                    </div>
                  )}
                  {/* ports — Brick Connectors: snap (green) vs reject (red) */}
                  <div className="mt-1.5 flex items-start justify-between">
                    <div className="flex flex-col gap-1">
                      {(meta?.ports.in ?? []).map((pt) => {
                        const canSnap = wire !== null && wire.port === pt;
                        const rejected =
                          wire !== null && !meta?.ports.in.includes(wire.port);
                        return (
                          <span
                            key={pt}
                            className={`inline-flex items-center gap-1 ${
                              wire
                                ? canSnap
                                  ? "forge-port-snap"
                                  : rejected
                                    ? "forge-port-reject"
                                    : "opacity-40"
                                : ""
                            }`}
                            title={
                              wire
                                ? canSnap
                                  ? `snap: accepts ${wire.port}`
                                  : rejected
                                    ? `cannot connect ${wire.port}`
                                    : undefined
                                : undefined
                            }
                          >
                            <span
                              className={`w-2 h-2 rounded-full transition-all ${
                                canSnap
                                  ? "ring-2 ring-[#3A6B58] scale-125"
                                  : rejected
                                    ? "bg-[#C8604A]"
                                    : ""
                              }`}
                              style={{
                                background: PORT_COLORS[pt] ?? "#8a8376",
                              }}
                            />
                            <span className="text-[8px] text-outline">
                              {pt}
                            </span>
                          </span>
                        );
                      })}
                    </div>
                    <div className="flex flex-col items-end gap-1">
                      {(meta?.ports.out ?? []).map((pt) => (
                        <span
                          key={pt}
                          className="inline-flex items-center gap-1 cursor-crosshair"
                          onMouseDown={(e) => {
                            e.stopPropagation();
                            startWire(n.id, pt, e.clientX, e.clientY);
                          }}
                          title={`drag to connect ${pt}`}
                        >
                          <span className="text-[8px] text-outline">{pt}</span>
                          <span
                            className="w-2 h-2 rounded-full"
                            style={{ background: PORT_COLORS[pt] ?? "#8a8376" }}
                          />
                        </span>
                      ))}
                    </div>
                  </div>
                </div>
              );
            })}

            {/* drag preview */}
            {dragType && dragPos && (
              <div
                className="absolute pointer-events-none rounded-2xl border-2 border-dashed border-primary/50 dark:border-inverse-primary/50 bg-primary/10 dark:bg-white/10 px-3 py-2 text-caption font-semibold text-primary dark:text-inverse-primary"
                style={{ left: dragPos.x - 40, top: dragPos.y - 14 }}
              >
                + {r.primitives[dragType]?.label}
              </div>
            )}

            {/* CONTRACT MISMATCH toast */}
            {mismatch && (
              <div className="absolute left-1/2 -translate-x-1/2 bottom-4 z-20 rounded-xl bg-[#C8604A] text-white px-4 py-2.5 shadow-lg max-w-[90%]">
                <div className="text-[10px] uppercase tracking-wider opacity-80 font-semibold">
                  Contract mismatch
                </div>
                <div className="text-caption font-mono">{mismatch}</div>
              </div>
            )}

            {/* empty state */}
            {nodes.length === 0 && (
              <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
                <div className="text-center text-outline">
                  <div
                    className="material-symbols-outlined"
                    style={{ fontSize: 40 }}
                  >
                    add_circle
                  </div>
                  <div className="text-caption mt-2">
                    Drag primitives here to assemble your agent
                  </div>
                </div>
              </div>
            )}
          </div>
        </main>
      </div>

      {/* CONTRACT mismatch summary in build mode */}
      {mode === "build" && !validation.valid && (
        <div className="rounded-xl bg-[#f3dfdc] dark:bg-[#3d2a28] text-[#8a3a35] dark:text-[#e9b8b2] px-3 py-2 text-caption">
          <div className="font-semibold mb-1">
            BUILD FAILED — {validation.violations.length} contract violation
            {validation.violations.length > 1 ? "s" : ""}
          </div>
          {validation.violations.map((v, i) => (
            <div key={i} className="font-mono text-[11px]">
              {v.message}
            </div>
          ))}
        </div>
      )}

      {/* EXPERIMENT mode */}
      {mode === "experiment" && (
        <ExperimentPanel result={r} params={params} setParams={setParams} />
      )}

      {/* RUN / EVENT STREAM — a persistent strip while the agent runs */}
      {mode === "run" && (
        <div className="rounded-2xl bg-surface-container-lowest dark:bg-dark-surface border border-outline-variant/40 dark:border-white/10 p-3 overflow-x-auto">
          <div className="flex items-center gap-0.5 flex-nowrap min-w-max">
            {r.run.trace.map((s, i) => {
              const done = i < step;
              const isCurrent = i === step;
              const failure = r.run.failures.find((f) => f.node === s.node);
              return (
                <span key={i} className="flex items-center gap-0.5 flex-nowrap">
                  <span
                    className={`px-2 py-1 rounded-lg text-caption font-semibold inline-flex items-center gap-1 whitespace-nowrap transition-all ${
                      s.failed
                        ? "bg-[#C8604A] text-white"
                        : done
                          ? "bg-primary text-on-primary dark:bg-inverse-primary dark:text-inverse-surface"
                          : isCurrent
                            ? "bg-primary/30 dark:bg-white/15 text-on-surface dark:text-dark-on-surface ring-1 ring-primary dark:ring-inverse-primary"
                            : "bg-surface-container dark:bg-white/5 text-outline"
                    }`}
                  >
                    <span
                      className="material-symbols-outlined"
                      style={{ fontSize: 13 }}
                    >
                      {s.failed ? "bolt" : s.icon}
                    </span>
                    {s.label}
                    {done && (
                      <span className="font-mono text-[9px] opacity-80">
                        {s.latency}ms
                      </span>
                    )}
                  </span>
                  {i < r.run.trace.length - 1 && (
                    <span className="text-outline mx-0.5">
                      {s.failed ? "✗" : "→"}
                    </span>
                  )}
                  {s.failed && failure && done && (
                    <span
                      className={`mr-1 inline-flex items-center gap-0.5 rounded px-1.5 py-0.5 text-[9px] font-semibold whitespace-nowrap ${
                        failure.recovered
                          ? "bg-[#3A6B58]/15 dark:bg-[#3A6B58]/30 text-[#2f6068] dark:text-[#a8d3dc]"
                          : "bg-[#C8604A]/15 dark:bg-[#C8604A]/30 text-[#C8604A]"
                      }`}
                    >
                      <span
                        className="material-symbols-outlined"
                        style={{ fontSize: 9 }}
                      >
                        {failure.recovered ? "healing" : "heart_broken"}
                      </span>
                      {failure.recovered
                        ? failure.recovery
                        : `no recovery (${(failure.needs ?? "?").replace("recovery-", "")})`}
                    </span>
                  )}
                </span>
              );
            })}
            {!playing && step >= r.run.trace.length && (
              <span
                className={`ml-2 text-caption font-semibold ${r.run.status === "failed" ? "text-[#C8604A]" : "text-[#3A6B58]"}`}
              >
                {r.run.status === "failed"
                  ? "✗ agent down"
                  : "✓ agent survived"}{" "}
                · resilience {r.run.resilience}
              </span>
            )}
          </div>

          {/* Runtime capability discovery strip */}
          {r.run.discovery.length > 0 && (
            <div className="mt-2 pt-2 border-t border-outline-variant/30 dark:border-white/10 flex items-center gap-0.5 flex-nowrap min-w-max">
              <span className="mr-1 text-[10px] uppercase tracking-wider text-[#7A5C8E] font-semibold inline-flex items-center gap-1">
                <span
                  className="material-symbols-outlined"
                  style={{ fontSize: 12 }}
                >
                  search
                </span>
                discover
              </span>
              {r.run.discovery.map((d, i) => (
                <span key={i} className="flex items-center gap-0.5 flex-nowrap">
                  <span
                    className={`px-1.5 py-0.5 rounded text-[10px] font-semibold whitespace-nowrap ${
                      d.phase === "execute"
                        ? "bg-[#3A6B58]/15 dark:bg-[#3A6B58]/30 text-[#2f6068] dark:text-[#a8d3dc]"
                        : d.phase === "unload"
                          ? "bg-surface-container dark:bg-white/5 text-outline"
                          : "bg-[#7A5C8E]/15 dark:bg-[#7A5C8E]/30 text-[#5a3d85] dark:text-[#c4b5e6]"
                    }`}
                  >
                    {d.phase === "discover" && "🔍"}
                    {d.phase === "load" && "📦"}
                    {d.phase === "execute" && "⚙️"}
                    {d.phase === "unload" && "♻️"} {d.phase} {d.item}
                  </span>
                  <span className="text-outline mx-0.5">·</span>
                </span>
              ))}
            </div>
          )}
        </div>
      )}

      {/* bottom drawer */}
      {mode !== "experiment" && (
        <>
          <div className="flex items-center gap-2">
            {(["trace", "inspect", "knowledge"] as const).map((d) => (
              <button
                key={d}
                onClick={() => setDrawer(drawer === d ? null : d)}
                className={`px-3.5 py-1.5 rounded-xl text-caption font-semibold uppercase tracking-wider transition ${
                  drawer === d
                    ? "bg-primary text-on-primary dark:bg-inverse-primary dark:text-inverse-surface"
                    : "bg-surface-container dark:bg-white/5 text-on-surface-variant dark:text-outline hover:bg-surface-variant dark:hover:bg-white/10"
                }`}
              >
                {d}
              </button>
            ))}
            <span className="text-caption text-outline ml-auto">
              {nodes.length} nodes · {edges.length} contracts
            </span>
          </div>
          {drawer && (
            <div className="rounded-2xl bg-surface-container-lowest dark:bg-dark-surface border border-outline-variant/40 dark:border-white/10 p-4 max-h-[45vh] overflow-y-auto">
              {drawerContent()}
            </div>
          )}
        </>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Grid placement helpers
// ---------------------------------------------------------------------------

function gridX(i: number): number {
  const col = i % 3;
  return 24 + col * (NODE_W + 20);
}

function gridY(i: number): number {
  const row = Math.floor(i / 3);
  return 24 + row * (NODE_H + GROUP_GAP);
}

// ---------------------------------------------------------------------------
// Drawers
// ---------------------------------------------------------------------------

function TraceDrawer({ run, step }: { run: ForgeRun; step: number }) {
  return (
    <div>
      <h4 className="font-label-md text-label-md uppercase tracking-wider text-on-surface dark:text-dark-on-surface mb-3">
        Run trace
      </h4>
      <div className="flex items-center gap-0 flex-wrap">
        {run.trace.map((s, i) => (
          <span key={i} className="flex items-center">
            <span
              className={`px-2 py-1 rounded-lg text-caption font-semibold inline-flex items-center gap-1 ${
                i < step
                  ? "bg-primary text-on-primary dark:bg-inverse-primary dark:text-inverse-surface"
                  : "bg-surface-container dark:bg-white/5 text-outline"
              }`}
            >
              <span
                className="material-symbols-outlined"
                style={{ fontSize: 13 }}
              >
                {s.icon}
              </span>
              {s.label}
              <span className="font-mono text-[9px] opacity-80">
                {s.latency}ms
              </span>
            </span>
            {i < run.trace.length - 1 && (
              <span className="text-outline mx-1">→</span>
            )}
          </span>
        ))}
      </div>
      <div className="mt-3 flex items-center justify-between text-caption text-outline">
        <span className="font-mono">
          {run.totals.latency.toFixed(1)}ms · {run.totals.tokens}t ·{" "}
          {run.totals.llm_calls} LLM calls
        </span>
        <span className="font-semibold text-primary dark:text-inverse-primary">
          resilience {run.resilience}
        </span>
      </div>
      {run.failures.length > 0 && (
        <div className="mt-2 rounded-xl bg-[#f3dfdc] dark:bg-[#3d2a28] text-[#8a3a35] dark:text-[#e9b8b2] px-3 py-2 text-caption">
          {run.failures.map((f, i) => (
            <div key={i} className="mb-1 last:mb-0">
              <b>{f.label}</b>: {f.recovery}{" "}
              <span className="opacity-70">({f.impact})</span>
            </div>
          ))}
          <div className="mt-1 font-semibold">
            {run.status === "failed" ? "Agent down" : "Agent survived"} ·{" "}
            {run.recovered} recovery actions
          </div>
        </div>
      )}
    </div>
  );
}

function InspectDrawer({
  meta,
  id,
}: {
  meta: PrimitiveMeta | null;
  id: string | null;
}) {
  if (!meta || !id) {
    return (
      <div className="text-caption text-outline">
        Select a primitive to inspect its anatomy.
      </div>
    );
  }
  return (
    <div>
      <h4 className="font-label-md text-label-md uppercase tracking-wider text-on-surface dark:text-dark-on-surface mb-2">
        {meta.label}
      </h4>
      <p className="text-caption text-on-surface-variant dark:text-outline leading-relaxed">
        {meta.desc}
      </p>
      <div className="mt-3 grid grid-cols-1 md:grid-cols-2 gap-4">
        <div>
          <div className="text-[10px] uppercase tracking-wider text-outline font-semibold mb-1">
            Why
          </div>
          <p className="text-caption text-on-surface dark:text-dark-on-surface leading-relaxed">
            {meta.why}
          </p>
        </div>
        <div>
          <div className="text-[10px] uppercase tracking-wider text-outline font-semibold mb-1">
            Trade-offs
          </div>
          <div className="flex flex-col gap-0.5">
            {meta.tradeoffs.map((t, i) => (
              <span
                key={i}
                className="text-caption text-on-surface-variant dark:text-outline"
              >
                {t}
              </span>
            ))}
          </div>
        </div>
      </div>
      <div className="mt-3 rounded-xl bg-surface-container dark:bg-white/5 px-3 py-2 inline-block">
        <div className="text-[10px] uppercase tracking-wider text-outline font-semibold">
          ARES note
        </div>
        <div className="text-caption font-mono text-primary dark:text-inverse-primary">
          {meta.note}
        </div>
      </div>
    </div>
  );
}

function KnowledgeDrawer({
  primitives,
  groups,
}: {
  primitives: Record<string, PrimitiveMeta>;
  groups: string[];
}) {
  return (
    <div>
      <h4 className="font-label-md text-label-md uppercase tracking-wider text-on-surface dark:text-dark-on-surface mb-3">
        Architecture evidence
      </h4>
      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-3">
        {groups.map((g) =>
          Object.entries(primitives)
            .filter(([, p]) => p.group === g)
            .map(([id, p]) => (
              <div
                key={id}
                className="rounded-xl border border-outline-variant/40 dark:border-white/10 bg-surface-container dark:bg-white/5 p-3"
              >
                <div className="text-[9px] uppercase tracking-wider text-outline font-semibold">
                  {g}
                </div>
                <div className="font-semibold text-caption text-on-surface dark:text-dark-on-surface mt-0.5">
                  {p.label}
                </div>
                <div className="text-[11px] text-on-surface-variant dark:text-outline mt-1">
                  {p.why}
                </div>
                <div className="text-[10px] font-mono text-primary dark:text-inverse-primary mt-1.5">
                  {p.note}
                </div>
              </div>
            )),
        )}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Experiment panel
// ---------------------------------------------------------------------------

function ExperimentPanel({
  result,
  params,
  setParams,
}: {
  result: ForgeResult;
  params: Record<string, unknown>;
  setParams: (p: Record<string, unknown>) => void;
}) {
  const [tab, setTab] = useState<"compare" | "evolve">("compare");
  const [baseline, setBaseline] = useState("simple");
  const [variant, setVariant] = useState("rag");
  const [chaos, setChaos] = useState<Record<string, boolean>>({
    chaos_mcp: true,
  });
  const [generations, setGenerations] = useState(4);
  const [population, setPopulation] = useState(6);

  const runExperiment = () => {
    setParams({
      ...params,
      experiment: true,
      baseline,
      variant,
      chaos_memory: !!chaos.chaos_memory,
      chaos_tool: !!chaos.chaos_tool,
      chaos_mcp: !!chaos.chaos_mcp,
      chaos_llm: !!chaos.chaos_llm,
      chaos_context: !!chaos.chaos_context,
      runToken: Date.now(),
    });
  };

  const runEvolve = () => {
    setParams({
      ...params,
      evolve: true,
      generations,
      population,
      runToken: Date.now(),
    });
  };

  const rows = [
    ["success", "Success"],
    ["resilience", "Resilience"],
    ["llm_calls", "LLM calls"],
    ["tokens", "Tokens"],
    ["latency", "Latency (ms)"],
    ["recovery", "Recovery"],
  ] as const;

  const maxFitness = Math.max(...(result.history ?? []).map((h) => h.best), 1);

  return (
    <div className="rounded-2xl bg-surface-container-lowest dark:bg-dark-surface border border-outline-variant/40 dark:border-white/10 p-4">
      <div className="flex items-center gap-2 mb-4">
        {(["compare", "evolve"] as const).map((tb) => (
          <button
            key={tb}
            onClick={() => setTab(tb)}
            className={`px-3.5 py-1.5 rounded-xl text-caption font-semibold uppercase tracking-wider transition ${
              tab === tb
                ? "bg-primary text-on-primary dark:bg-inverse-primary dark:text-inverse-surface"
                : "bg-surface-container dark:bg-white/5 text-on-surface-variant dark:text-outline hover:bg-surface-variant dark:hover:bg-white/10"
            }`}
          >
            {tb === "compare" ? "Baseline vs Variant" : "🧬 Evolve"}
          </button>
        ))}
      </div>

      {tab === "compare" ? (
        <>
          <h4 className="font-label-md text-label-md uppercase tracking-wider text-on-surface dark:text-dark-on-surface mb-4">
            Controlled experiment — baseline vs variant
          </h4>
          <div className="flex flex-wrap items-end gap-4 mb-4">
            <div>
              <div className="text-[10px] uppercase tracking-wider text-outline font-semibold mb-1">
                Baseline
              </div>
              <select
                value={baseline}
                onChange={(e) => setBaseline(e.target.value)}
                className="rounded-lg bg-surface-container dark:bg-white/5 border border-outline-variant/60 dark:border-white/10 px-3 py-1.5 text-caption text-on-surface dark:text-dark-on-surface"
              >
                {result.examples.map((ex) => (
                  <option key={ex.id} value={ex.id}>
                    {ex.name}
                  </option>
                ))}
              </select>
            </div>
            <div>
              <div className="text-[10px] uppercase tracking-wider text-outline font-semibold mb-1">
                Variant
              </div>
              <select
                value={variant}
                onChange={(e) => setVariant(e.target.value)}
                className="rounded-lg bg-surface-container dark:bg-white/5 border border-outline-variant/60 dark:border-white/10 px-3 py-1.5 text-caption text-on-surface dark:text-dark-on-surface"
              >
                {result.examples.map((ex) => (
                  <option key={ex.id} value={ex.id}>
                    {ex.name}
                  </option>
                ))}
              </select>
            </div>
            <div className="flex flex-wrap gap-3">
              {(
                [
                  "chaos_memory",
                  "chaos_tool",
                  "chaos_mcp",
                  "chaos_llm",
                  "chaos_context",
                ] as const
              ).map((k) => (
                <label
                  key={k}
                  className="inline-flex items-center gap-1.5 text-caption text-on-surface-variant dark:text-outline cursor-pointer"
                >
                  <input
                    type="checkbox"
                    checked={!!chaos[k]}
                    onChange={(e) =>
                      setChaos((c) => ({ ...c, [k]: e.target.checked }))
                    }
                    className="accent-[#C8604A]"
                  />
                  {k.replace("chaos_", "")}
                </label>
              ))}
            </div>
            <button
              onClick={runExperiment}
              className="inline-flex items-center gap-1.5 px-4 py-2 rounded-xl bg-primary text-on-primary dark:bg-inverse-primary dark:text-inverse-surface font-label-md text-label-md hover:opacity-90 transition"
            >
              <span
                className="material-symbols-outlined"
                style={{ fontSize: 16 }}
              >
                science
              </span>
              Run experiment
            </button>
          </div>

          {result.baseline && result.variant && (
            <div className="overflow-x-auto">
              <table className="w-full text-caption">
                <thead>
                  <tr className="text-outline text-left">
                    <th className="pb-2 font-semibold">metric</th>
                    <th className="pb-2 font-semibold text-right">
                      {result.baseline.name}
                    </th>
                    <th className="pb-2 font-semibold text-right">
                      {result.variant.name}
                    </th>
                    <th className="pb-2 font-semibold text-right">Δ</th>
                  </tr>
                </thead>
                <tbody>
                  {rows.map(([key, label]) => {
                    const a = result.baseline![key];
                    const b = result.variant![key];
                    const d =
                      typeof a === "number" && typeof b === "number"
                        ? b - a
                        : null;
                    const fmt = (v: unknown) =>
                      typeof v === "number"
                        ? Number.isInteger(v)
                          ? v
                          : v.toFixed(1)
                        : String(v);
                    return (
                      <tr
                        key={key}
                        className="border-t border-outline-variant/30 dark:border-white/10"
                      >
                        <td className="py-1.5 text-on-surface-variant dark:text-outline">
                          {label}
                        </td>
                        <td className="py-1.5 text-right font-mono">
                          {fmt(a)}
                        </td>
                        <td
                          className={`py-1.5 text-right font-mono ${d && d < 0 ? "text-[#3A6B58]" : d && d > 0 ? "text-[#C8604A]" : "text-on-surface dark:text-dark-on-surface"}`}
                        >
                          {fmt(b)}
                        </td>
                        <td className="py-1.5 text-right font-mono text-outline">
                          {d === null || d === 0
                            ? "—"
                            : (d > 0 ? "+" : "") + fmt(Math.abs(d))}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
        </>
      ) : (
        <>
          <h4 className="font-label-md text-label-md uppercase tracking-wider text-on-surface dark:text-dark-on-surface mb-4">
            🧬 Evolve — let architectures compete, mutate and recombine
          </h4>
          <div className="flex flex-wrap items-end gap-4 mb-4">
            <div>
              <div className="text-[10px] uppercase tracking-wider text-outline font-semibold mb-1">
                Generations
              </div>
              <input
                type="number"
                min={2}
                max={10}
                value={generations}
                onChange={(e) => setGenerations(Number(e.target.value))}
                className="w-20 rounded-lg bg-surface-container dark:bg-white/5 border border-outline-variant/60 dark:border-white/10 px-3 py-1.5 text-caption text-on-surface dark:text-dark-on-surface"
              />
            </div>
            <div>
              <div className="text-[10px] uppercase tracking-wider text-outline font-semibold mb-1">
                Population
              </div>
              <input
                type="number"
                min={4}
                max={12}
                value={population}
                onChange={(e) => setPopulation(Number(e.target.value))}
                className="w-20 rounded-lg bg-surface-container dark:bg-white/5 border border-outline-variant/60 dark:border-white/10 px-3 py-1.5 text-caption text-on-surface dark:text-dark-on-surface"
              />
            </div>
            <button
              onClick={runEvolve}
              className="inline-flex items-center gap-1.5 px-4 py-2 rounded-xl bg-primary text-on-primary dark:bg-inverse-primary dark:text-inverse-surface font-label-md text-label-md hover:opacity-90 transition"
            >
              <span
                className="material-symbols-outlined"
                style={{ fontSize: 16 }}
              >
                genetics
              </span>
              Run evolution
            </button>
          </div>

          {result.history && (
            <>
              {/* per-generation fitness bars */}
              <div className="mb-3">
                <div className="text-[10px] uppercase tracking-wider text-outline font-semibold mb-1.5">
                  Fitness per generation
                </div>
                <div className="flex flex-col gap-1">
                  {result.history.map((h) => (
                    <div key={h.gen} className="flex items-center gap-2">
                      <span className="w-14 text-caption font-mono text-outline">
                        G{h.gen}
                      </span>
                      <div className="flex-1 h-3 bg-surface-variant dark:bg-white/10 rounded-full overflow-hidden">
                        <div
                          className="h-full bg-primary dark:bg-inverse-primary"
                          style={{ width: `${(h.best / maxFitness) * 100}%` }}
                        />
                      </div>
                      <span className="w-20 text-caption font-mono text-on-surface dark:text-dark-on-surface text-right">
                        best {h.best}{" "}
                        <span className="text-outline">/ avg {h.avg}</span>
                      </span>
                    </div>
                  ))}
                </div>
              </div>

              {/* fittest architectures */}
              {result.fittest && result.fittest.length > 0 && (
                <div className="mb-3">
                  <div className="text-[10px] uppercase tracking-wider text-outline font-semibold mb-1.5">
                    Fittest
                  </div>
                  <div className="flex flex-col gap-0.5">
                    {result.fittest.map((f, i) => (
                      <div
                        key={i}
                        className="flex items-center justify-between text-caption"
                      >
                        <span className="text-on-surface-variant dark:text-outline">
                          #{i + 1} {f.name}
                        </span>
                        <span className="font-mono text-primary dark:text-inverse-primary">
                          {f.fitness}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* evolved architecture nodes */}
              {result.best && result.best.graph.nodes.length > 0 && (
                <div>
                  <div className="text-[10px] uppercase tracking-wider text-outline font-semibold mb-1.5">
                    🏆 Evolved agent
                  </div>
                  <div className="flex flex-wrap gap-1.5">
                    {result.best.graph.nodes.map((n) => {
                      const meta = result.primitives[n.id];
                      return (
                        <span
                          key={n.id}
                          className="inline-flex items-center gap-1 rounded-lg bg-surface-container dark:bg-white/5 px-2 py-1 text-caption text-on-surface dark:text-dark-on-surface"
                        >
                          <span
                            className="material-symbols-outlined"
                            style={{ fontSize: 13 }}
                          >
                            {meta?.icon ?? "crop_square"}
                          </span>
                          {meta?.label ?? n.id}
                        </span>
                      );
                    })}
                  </div>
                  <div className="mt-1 text-caption text-outline">
                    {result.best.graph.edges.length} contracts · fitness{" "}
                    {result.best.fitness}
                  </div>
                </div>
              )}
            </>
          )}
        </>
      )}
    </div>
  );
}
