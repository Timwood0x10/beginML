"""
Agent Forge — a cognitive-system workbench.

Instead of a config form, Forge treats an agent as an executable graph of
"cognitive primitives". Every primitive declares semantic ports (task /
context / action / result); edges are only valid when the port contracts
match, so a wrong wiring is not a UI refusal but a semantic error.

The module simulates a run over the assembled graph, can inject failures
(CHAOS mode) with recovery actions and a resilience score, compiles the
graph into an agent YAML, and can compare two architectures side by side.

Rules kept on purpose: functions <= 120 lines, file <= 1000 lines, English
comments only.
"""

from typing import Any
import time

# ---------------------------------------------------------------------------
# Primitive metadata
# ---------------------------------------------------------------------------

# Semantic port types. An edge from A to B is valid when A's output port
# type appears in B's input port types.
PORT_TYPES = ["task", "context", "action", "result"]

# Groups express the agent's capability space (not a config form).
GROUPS = ["THINK", "REMEMBER", "ACT", "PLAN", "COORDINATE"]

# id -> primitive definition.
# ports: {"in": [...], "out": [...]} — the semantic contracts.
# cost: {"latency": ms, "tokens": per call, "calls": max calls} used by the
#       simulator; "llm": True marks brain calls that dominate cost.
PRIMITIVES: dict[str, dict[str, Any]] = {
    "input": {
        "label": "Input", "group": "THINK", "icon": "input",
        "ports": {"in": [], "out": ["task"]},
        "desc": "The user task enters the system.",
        "why": "Every agent run starts with a task that must be parsed.",
        "tradeoffs": ["+ explicit task boundary", "- no structure without parsing"],
        "note": "00-goagentx-intro.md",
        "cost": {"latency": 2, "tokens": 0, "calls": 1, "llm": False},
    },
    "llm": {
        "label": "LLM Brain", "group": "THINK", "icon": "psychology",
        "ports": {"in": ["context", "result", "task"], "out": ["result"]},
        "desc": "The reasoning core: reads context, produces results.",
        "why": "The brain decides what to do next given memory and tools.",
        "tradeoffs": ["+ flexible reasoning", "- latency & cost dominate"],
        "note": "20-llm-client-layer.md",
        "cost": {"latency": 320, "tokens": 1200, "calls": 2, "llm": True},
    },
    "memory": {
        "label": "Memory", "group": "REMEMBER", "icon": "memory",
        "ports": {"in": ["task", "context"], "out": ["context"]},
        "desc": "Recall past sessions and distilled experience.",
        "why": "Memory distillation turns raw logs into reusable experience.",
        "tradeoffs": ["+ persistence", "+ personalization", "- retrieval cost"],
        "note": "03-memory-distillation-deep-dive.md",
        "cost": {"latency": 18, "tokens": 0, "calls": 1, "llm": False},
    },
    "retrieve": {
        "label": "Retrieve", "group": "REMEMBER", "icon": "manage_search",
        "ports": {"in": ["task"], "out": ["context"]},
        "desc": "Hybrid search: vector + full-text + structured scoring.",
        "why": "The right context decides the answer; more is not better.",
        "tradeoffs": ["+ grounded answers", "+ less hallucination", "- index overhead"],
        "note": "10-retrieval-system-deep-dive.md",
        "cost": {"latency": 42, "tokens": 0, "calls": 1, "llm": False},
    },
    "planner": {
        "label": "Planner", "group": "PLAN", "icon": "route",
        "ports": {"in": ["task", "context"], "out": ["task", "action"]},
        "desc": "Decompose the task into steps; replan when needed.",
        "why": "ReAct becomes a DAG: orchestrated, interruptible, retryable.",
        "tradeoffs": ["+ long-task handling", "+ HITL hooks", "- plan drift"],
        "note": "04-workflow-engine-deep-dive.md",
        "cost": {"latency": 150, "tokens": 600, "calls": 1, "llm": True},
    },
    "tool": {
        "label": "Tool", "group": "ACT", "icon": "handyman",
        "ports": {"in": ["action"], "out": ["result"]},
        "desc": "Execute a registered tool with validated arguments.",
        "why": "Four invocation paths, one safety net for LLM errors.",
        "tradeoffs": ["+ capability", "+ deterministic fallback", "- schema burden"],
        "note": "05-tool-system-deep-dive.md",
        "cost": {"latency": 210, "tokens": 0, "calls": 2, "llm": False},
    },
    "mcp": {
        "label": "MCP Tool", "group": "ACT", "icon": "extension",
        "ports": {"in": ["action"], "out": ["result"]},
        "desc": "Discover and call external tools via MCP protocol.",
        "why": "One protocol, many servers; automatic service discovery.",
        "tradeoffs": ["+ tool ecosystem", "+ dynamic discovery", "- transport setup"],
        "note": "15-mcp-integration-deep-dive.md",
        "cost": {"latency": 318, "tokens": 0, "calls": 2, "llm": False},
    },
    "reflect": {
        "label": "Reflect", "group": "THINK", "icon": "history_edu",
        "ports": {"in": ["result"], "out": ["result"]},
        "desc": "Critique the draft and produce an improved version.",
        "why": "Self-critique raises quality on hard tasks.",
        "tradeoffs": ["+ quality", "- extra LLM calls"],
        "note": "11-autonomous-evolution-deep-dive.md",
        "cost": {"latency": 260, "tokens": 900, "calls": 1, "llm": True},
    },
    "verify": {
        "label": "Verify", "group": "PLAN", "icon": "verified",
        "ports": {"in": ["result"], "out": ["result"]},
        "desc": "Check the result against the evaluation framework.",
        "why": "Evaluation turns performance into a comparable signal.",
        "tradeoffs": ["+ measurable quality", "+ gates evolution", "- setup cost"],
        "note": "21-evaluation-framework.md",
        "cost": {"latency": 90, "tokens": 300, "calls": 1, "llm": True},
    },
    "delegate": {
        "label": "Delegate", "group": "COORDINATE", "icon": "groups",
        "ports": {"in": ["task"], "out": ["task", "result"]},
        "desc": "Dispatch subtasks to worker agents via a protocol.",
        "why": "Harmony protocol gives collaborators a common language.",
        "tradeoffs": ["+ parallelism", "+ specialization", "- coordination cost"],
        "note": "02-agent-harmony-protocol.md",
        "cost": {"latency": 130, "tokens": 500, "calls": 2, "llm": True},
    },
    "critic": {
        "label": "Critic", "group": "COORDINATE", "icon": "rate_review",
        "ports": {"in": ["result"], "out": ["result"]},
        "desc": "A second voice reviews the output before it ships.",
        "why": "Generator + critic converges when the critic approves.",
        "tradeoffs": ["+ adversarial quality", "+ fewer regressions", "- more calls"],
        "note": "24.1-ga-deep-dive.md",
        "cost": {"latency": 240, "tokens": 800, "calls": 1, "llm": True},
    },
    "output": {
        "label": "Output", "group": "THINK", "icon": "output",
        "ports": {"in": ["result"], "out": []},
        "desc": "The final answer leaves the system.",
        "why": "A terminal node that finalizes the trace.",
        "tradeoffs": ["+ clean boundary", "- nothing"],
        "note": "00-goagentx-intro.md",
        "cost": {"latency": 1, "tokens": 0, "calls": 1, "llm": False},
    },
}


def primitive(id_: str) -> dict[str, Any]:
    """Return a primitive definition (fallback: input)."""
    return PRIMITIVES.get(id_, PRIMITIVES["input"])


def port_compatible(out_type: str, target: dict[str, Any]) -> bool:
    """True when an output port type fits a target primitive's input ports."""
    return out_type in target["ports"]["in"]


# ---------------------------------------------------------------------------
# Graph helpers
# ---------------------------------------------------------------------------


def check_graph(nodes: list[dict[str, Any]], edges: list[dict[str, Any]]) -> list[str]:
    """Validate wiring: node ids exist and every edge matches port contracts."""
    errors: list[str] = []
    ids = [n["id"] for n in nodes]
    for e in edges:
        if e.get("from") not in ids or e.get("to") not in ids:
            errors.append(f"edge references unknown node: {e.get('from')} -> {e.get('to')}")
            continue
        src = primitive(e["from"])
        dst = primitive(e["to"])
        if not port_compatible(e.get("port", "result"), dst):
            errors.append(
                f"incompatible contracts: {src['label']}.{e.get('port','result')} "
                f"→ {dst['label']} expects {dst['ports']['in']}"
            )
    return errors


def _successors(edges: list[dict[str, Any]], node_id: str) -> list[str]:
    """Ordered downstream node ids of a node."""
    return [e["to"] for e in edges if e["from"] == node_id]


def topo_order(nodes: list[dict[str, Any]], edges: list[dict[str, Any]]) -> list[str]:
    """Return node ids in execution order (Kahn's algorithm)."""
    indeg = {n["id"]: 0 for n in nodes}
    for e in edges:
        if e["to"] in indeg:
            indeg[e["to"]] += 1
    queue = [n["id"] for n in nodes if indeg[n["id"]] == 0]
    order: list[str] = []
    while queue:
        cur = queue.pop(0)
        order.append(cur)
        for nxt in _successors(edges, cur):
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                queue.append(nxt)
    for n in nodes:
        if n["id"] not in order:
            order.append(n["id"])
    return order


# ---------------------------------------------------------------------------
# Run simulator
# ---------------------------------------------------------------------------

# CHAOS failure presets: which primitives they can hit and their recovery.
CHAOS_MODES: dict[str, dict[str, Any]] = {
    "memory_unavailable": {
        "label": "Memory unavailable",
        "targets": ["memory", "retrieve"],
        "recovery": "Memory fallback → session context only",
        "impact": {"latency": 2, "tokens": 0},
    },
    "tool_timeout": {
        "label": "Tool timeout",
        "targets": ["tool", "mcp"],
        "recovery": "Tool retry ×2 with backoff",
        "impact": {"latency": 1.8, "tokens": 0},
    },
    "mcp_failure": {
        "label": "MCP failure",
        "targets": ["mcp"],
        "recovery": "Planner replanned around unavailable server",
        "impact": {"latency": 1.5, "tokens": 300},
    },
    "llm_retry": {
        "label": "LLM retry",
        "targets": ["llm", "planner", "reflect", "verify", "delegate", "critic"],
        "recovery": "LLM retry with exponential backoff",
        "impact": {"latency": 1.6, "tokens": 0},
    },
    "context_overflow": {
        "label": "Context overflow",
        "targets": ["llm", "planner", "reflect"],
        "recovery": "Context window truncation → distilled summary",
        "impact": {"latency": 1.3, "tokens": -400},
    },
}


def simulate_run(nodes: list[dict[str, Any]], edges: list[dict[str, Any]],
                 task: str, chaos: list[str] | None = None) -> dict[str, Any]:
    """Simulate a run over the graph, emitting a replayable trace.

    Each visited primitive records latency (ms), token usage and call count;
    every step also carries a short narrative so the frontend can "watch the
    agent think". CHAOS modes inject failures on matching primitives; the
    failure record lists the recovery action that would be taken.
    """
    chaos = chaos or []
    active = {m: set(CHAOS_MODES[m]["targets"]) for m in chaos if m in CHAOS_MODES}
    order = topo_order(nodes, edges)
    trace: list[dict[str, Any]] = []
    totals = {"latency": 0, "tokens": 0, "calls": 0, "llm_calls": 0}
    failures: list[dict[str, str]] = []
    rng_seed = int(time.time() * 1000) % 1000

    for nid in order:
        meta = primitive(nid)
        cost = meta["cost"]
        latency = cost["latency"]
        tokens = cost["tokens"]
        calls = cost["calls"]

        # Jitter to feel alive (±15%).
        jitter = 1.0 + ((rng_seed + len(trace) * 7) % 30 - 15) / 100.0
        latency = round(latency * jitter, 1)

        hit_mode = None
        for mode in chaos:
            if nid in active.get(mode, set()):
                hit_mode = mode
                break
        if hit_mode:
            impact = CHAOS_MODES[hit_mode]["impact"]
            latency = round(latency * impact["latency"], 1)
            tokens = max(0, tokens + impact["tokens"])
            failures.append({
                "node": nid,
                "label": meta["label"],
                "mode": hit_mode,
                "message": f"{meta['label']} hit {CHAOS_MODES[hit_mode]['label']}",
                "recovery": CHAOS_MODES[hit_mode]["recovery"],
            })

        totals["latency"] += latency
        totals["tokens"] += tokens
        totals["calls"] += calls
        if cost["llm"]:
            totals["llm_calls"] += 1

        trace.append({
            "node": nid,
            "label": meta["label"],
            "icon": meta["icon"],
            "latency": latency,
            "tokens": tokens,
            "calls": calls,
            "llm": cost["llm"],
            "failed": hit_mode is not None,
            "detail": _step_detail(nid, meta, task, calls),
        })

    # Resilience: failures that have a recovery path are survivable.
    recovered = len(failures)
    resilience = max(0, 100 - recovered * 22) if failures else 100

    return {
        "task": task,
        "trace": trace,
        "totals": totals,
        "failures": failures,
        "recovered": recovered,
        "resilience": resilience,
        "chaos": chaos,
    }


def _step_detail(node_id: str, meta: dict[str, Any], task: str, calls: int) -> str:
    """One-line narrative for a trace step."""
    label = meta["label"]
    t = (task[:24] + "…") if len(task) > 24 else task
    texts = {
        "input": f"Parsing task: “{t}”",
        "llm": f"Reasoning over context for “{t}”",
        "memory": f"Recalling {calls} distilled memories",
        "retrieve": f"Hybrid search over corpus for “{t}”",
        "planner": f"Decomposing into {calls} steps",
        "tool": f"Executing tool ({calls} calls)",
        "mcp": f"Calling MCP server ({calls} calls)",
        "reflect": "Critiquing draft and revising",
        "verify": f"Running {calls} evaluation checks",
        "delegate": f"Dispatching to {calls} workers",
        "critic": "Reviewing with a second voice",
        "output": "Finalizing answer",
    }
    return texts.get(node_id, f"{label} step")


# ---------------------------------------------------------------------------
# Agent Compiler + Compare
# ---------------------------------------------------------------------------


def compile_agent(nodes: list[dict[str, Any]], edges: list[dict[str, Any]],
                  name: str = "my_agent") -> str:
    """Compile the graph into an agent YAML (a build artifact, not input)."""
    ids = [n["id"] for n in nodes]
    lines = [f"# Compiled from Agent Forge graph", f"agent:", f"  name: {name}"]
    has_memory = any(i in ("memory", "retrieve") for i in ids)
    has_planner = "planner" in ids
    has_tools = [i for i in ("tool", "mcp") if i in ids]
    has_reflect = any(i in ("reflect", "critic", "verify") for i in ids)
    has_delegate = "delegate" in ids

    if has_memory:
        memory_type = "tiered" if "memory" in ids else "vector"
        lines.append("memory:")
        lines.append(f"  type: {memory_type}")
        if "retrieve" in ids:
            lines.append("  retrieval:")
            lines.append("    top_k: 5")
            lines.append("    hybrid: true")
    if has_planner:
        lines.append("planner:")
        lines.append("  type: react")
        lines.append("  max_steps: 8")
    if has_tools:
        lines.append("tools:")
        for t in has_tools:
            if t == "mcp":
                lines.append("  - mcp: filesystem")
                lines.append("  - mcp: git")
            else:
                lines.append("  - builtin: registry")
    if has_reflect:
        lines.append("reflection:")
        lines.append("  enabled: true")
        lines.append("  rounds: 2")
    if has_delegate:
        lines.append("multi_agent:")
        lines.append("  pattern: orchestrator")
        lines.append("  workers: 4")
    return "\n".join(lines)


def compare_agents(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """Run two architectures over the same task and diff their behavior."""
    ra = simulate_run(a["nodes"], a["edges"], a["task"], a.get("chaos"))
    rb = simulate_run(b["nodes"], b["edges"], b["task"], b.get("chaos"))

    def summary(r: dict[str, Any]) -> dict[str, Any]:
        t = r["totals"]
        return {
            "latency": round(t["latency"], 1),
            "tokens": t["tokens"],
            "calls": t["calls"],
            "llm_calls": t["llm_calls"],
            "resilience": r["resilience"],
            "failures": len(r["failures"]),
            "steps": len(r["trace"]),
        }

    sa, sb = summary(ra), summary(rb)
    diffs = []
    for key in ("latency", "tokens", "calls", "llm_calls", "resilience", "failures", "steps"):
        if sa[key] != sb[key]:
            direction = "up" if sb[key] > sa[key] else "down"
            diffs.append({
                "metric": key,
                "a": sa[key], "b": sb[key], "direction": direction,
            })
    return {"a": sa, "b": sb, "diffs": diffs}


# ---------------------------------------------------------------------------
# Preset architectures
# ---------------------------------------------------------------------------

PRESETS: dict[str, dict[str, Any]] = {
    "simple": {
        "name": "Simple ReAct",
        "nodes": [{"id": "input"}, {"id": "planner"}, {"id": "llm"}, {"id": "output"}],
        "edges": [
            {"from": "input", "to": "planner", "port": "task"},
            {"from": "planner", "to": "llm", "port": "action"},
            {"from": "llm", "to": "output", "port": "result"},
        ],
    },
    "rag": {
        "name": "RAG Agent",
        "nodes": [{"id": "input"}, {"id": "retrieve"}, {"id": "planner"}, {"id": "tool"}, {"id": "llm"}, {"id": "output"}],
        "edges": [
            {"from": "input", "to": "retrieve", "port": "task"},
            {"from": "retrieve", "to": "planner", "port": "context"},
            {"from": "planner", "to": "tool", "port": "action"},
            {"from": "planner", "to": "llm", "port": "task"},
            {"from": "tool", "to": "llm", "port": "result"},
            {"from": "llm", "to": "output", "port": "result"},
        ],
    },
    "tiered": {
        "name": "Tiered Memory",
        "nodes": [{"id": "input"}, {"id": "memory"}, {"id": "retrieve"}, {"id": "planner"}, {"id": "mcp"}, {"id": "llm"}, {"id": "reflect"}, {"id": "output"}],
        "edges": [
            {"from": "input", "to": "memory", "port": "task"},
            {"from": "memory", "to": "retrieve", "port": "context"},
            {"from": "retrieve", "to": "planner", "port": "context"},
            {"from": "planner", "to": "mcp", "port": "action"},
            {"from": "planner", "to": "llm", "port": "task"},
            {"from": "mcp", "to": "llm", "port": "result"},
            {"from": "llm", "to": "reflect", "port": "result"},
            {"from": "reflect", "to": "output", "port": "result"},
        ],
    },
    "multi-agent": {
        "name": "Multi-Agent",
        "nodes": [{"id": "input"}, {"id": "delegate"}, {"id": "planner"}, {"id": "tool"}, {"id": "llm"}, {"id": "critic"}, {"id": "output"}],
        "edges": [
            {"from": "input", "to": "delegate", "port": "task"},
            {"from": "delegate", "to": "planner", "port": "task"},
            {"from": "planner", "to": "tool", "port": "action"},
            {"from": "planner", "to": "llm", "port": "task"},
            {"from": "tool", "to": "llm", "port": "result"},
            {"from": "llm", "to": "critic", "port": "result"},
            {"from": "critic", "to": "output", "port": "result"},
        ],
    },
    "empty": {"name": "Blank canvas", "nodes": [], "edges": []},
}


def compute(params: dict[str, Any]) -> dict[str, Any]:
    """Lab entry point: assemble, run, inject chaos, compile, compare."""
    preset_id = str(params.get("preset", "simple"))
    preset = PRESETS.get(preset_id, PRESETS["simple"])
    task = str(params.get("task", "分析这个项目的 FFI 安全问题"))

    # CHAOS toggles -> active failure modes.
    chaos = []
    for key, mode in (("chaos_memory", "memory_unavailable"),
                      ("chaos_tool", "tool_timeout"),
                      ("chaos_mcp", "mcp_failure"),
                      ("chaos_llm", "llm_retry"),
                      ("chaos_context", "context_overflow")):
        if params.get(key):
            chaos.append(mode)

    run = simulate_run(preset["nodes"], preset["edges"], task, chaos)
    yaml = compile_agent(preset["nodes"], preset["edges"], name=preset["name"].lower().replace(" ", "-"))

    # Baseline (no chaos) for the compare view.
    baseline = simulate_run(preset["nodes"], preset["edges"], task, [])
    compare = {
        "active": bool(params.get("compare")),
        "a": baseline["totals"],
        "b": run["totals"],
        "diffs": _diff_summaries(baseline, run),
    }

    return {
        "preset": preset_id,
        "presetName": preset["name"],
        "primitives": {
            pid: {
                "label": p["label"], "group": p["group"], "icon": p["icon"],
                "ports": p["ports"], "desc": p["desc"], "why": p["why"],
                "tradeoffs": p["tradeoffs"], "note": p["note"],
            }
            for pid, p in PRIMITIVES.items()
        },
        "groups": GROUPS,
        "graph": preset,
        "run": run,
        "yaml": yaml,
        "compare": compare,
    }


def _diff_summaries(base: dict[str, Any], run: dict[str, Any]) -> list[dict[str, Any]]:
    """Diff totals between a baseline run and the (possibly chaotic) run."""
    diffs = []
    for key in ("latency", "tokens", "calls", "llm_calls"):
        a = base["totals"][key]
        b = run["totals"][key]
        if a != b:
            diffs.append({"metric": key, "a": a, "b": b,
                          "direction": "up" if b > a else "down"})
    return diffs
