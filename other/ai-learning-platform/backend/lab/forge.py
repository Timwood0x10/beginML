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
import numpy as np

# ---------------------------------------------------------------------------
# Primitive metadata
# ---------------------------------------------------------------------------

# Semantic port types. An edge from A to B is valid when A's output port
# type appears in B's input port types.
PORT_TYPES = ["task", "context", "action", "result"]

# Groups express the agent's capability space (not a config form).
GROUPS = ["THINK", "REMEMBER", "ACT", "PLAN", "COORDINATE"]

# Brick classes — the LEGO-like categories every primitive belongs to.
# rarity tiers (matching the "rarety" ladder): primitive < capability <
# skill < agent < system. Color hints are used by the UI palette — they are
# CSS variables of the semantic system (--ailearn-semantic-*), so the same
# hue-meaning mapping holds across every theme: Brain=fog indigo,
# Memory=bamboo green, Action=clay amber, Recovery=coral, Agent=dusty
# violet, Capability=teal.
BRICK_CLASSES: dict[str, dict[str, Any]] = {
    "brain": {"label": "Brain", "icon": "psychology", "color": "var(--ailearn-semantic-brain)", "rarity": "primitive"},
    "memory": {"label": "Memory", "icon": "memory", "color": "var(--ailearn-semantic-memory)", "rarity": "primitive"},
    "action": {"label": "Action", "icon": "handyman", "color": "var(--ailearn-semantic-action)", "rarity": "primitive"},
    "capability": {"label": "Capability", "icon": "extension", "color": "var(--ailearn-semantic-capability)", "rarity": "capability"},
    "recovery": {"label": "Recovery", "icon": "healing", "color": "var(--ailearn-semantic-chaos)", "rarity": "capability"},
    "agent": {"label": "Agent", "icon": "smart_toy", "color": "var(--ailearn-semantic-agent)", "rarity": "agent"},
}

# primitive id -> brick class (kept separate so PRIMITIVES stays readable).
_CLASS_OF: dict[str, str] = {
    "input": "brain", "llm": "brain", "planner": "brain", "reflect": "brain",
    "verify": "brain", "critic": "brain", "output": "brain",
    "memory": "memory", "retrieve": "memory",
    "tool": "action", "mcp": "action", "discover": "capability",
    "delegate": "agent",
}

# Recovery bricks — pluggable self-healing capabilities. When attached to
# the agent they make matching failures recoverable; when unplugged the same
# failure can take the agent down.
RECOVERY_BRICKS: dict[str, dict[str, Any]] = {
    "recovery-retry": {
        "label": "Retry", "icon": "replay",
        "desc": "Retry a failed call with exponential backoff.",
        "recovers": ["tool_timeout", "llm_retry"],
    },
    "recovery-fallback": {
        "label": "Fallback", "icon": "alt_route",
        "desc": "Fall back to a degraded but working path (e.g. session-only memory).",
        "recovers": ["memory_unavailable", "context_overflow"],
    },
    "recovery-replan": {
        "label": "Replan", "icon": "route",
        "desc": "Replan around the failed capability (e.g. discover an alternative tool).",
        "recovers": ["mcp_failure"],
    },
}

# BREAK bricks — dragable failures for the "break it" mode.
BREAK_BRICKS: list[dict[str, Any]] = [
    {"id": "memory_unavailable", "label": "Memory failure", "icon": "memory_slash",
     "targets": ["memory", "retrieve"], "desc": "Recall fails — context lost."},
    {"id": "tool_timeout", "label": "Tool timeout", "icon": "timer_off",
     "targets": ["tool", "mcp"], "desc": "A tool call hangs past its deadline."},
    {"id": "mcp_failure", "label": "MCP disconnect", "icon": "link_off",
     "targets": ["mcp"], "desc": "The MCP server goes away mid-run."},
    {"id": "llm_retry", "label": "LLM failure", "icon": "sync_problem",
     "targets": ["llm", "planner", "reflect", "verify", "delegate", "critic"],
     "desc": "The brain fails or times out."},
    {"id": "context_overflow", "label": "Context overflow", "icon": "data_exploration",
     "targets": ["llm", "planner", "reflect"], "desc": "The window overflows."},
]

# Skill boxes — "a brick that contains bricks". Dropping one on the canvas
# expands into its inner capability graph (discovery → tools → executor).
SKILL_BOXES: list[dict[str, Any]] = [
    {
        "id": "skill-coding",
        "label": "Coding Skill",
        "icon": "code_blocks",
        "desc": "File, shell and git tools wrapped by an executor.",
        "tools": ["file", "shell", "git"],
        "inner": [
            {"id": "file", "label": "File Tool", "icon": "description"},
            {"id": "shell", "label": "Shell Tool", "icon": "terminal"},
            {"id": "git", "label": "Git Tool", "icon": "commit"},
        ],
    },
    {
        "id": "skill-research",
        "label": "Web Research Skill",
        "icon": "travel_explore",
        "desc": "Discover web tools, search, browse and summarize.",
        "tools": ["search", "browser", "summarize"],
        "inner": [
            {"id": "search", "label": "Search", "icon": "search"},
            {"id": "browser", "label": "Browser", "icon": "language"},
            {"id": "summarize", "label": "Summarizer", "icon": "summarize"},
        ],
    },
]

# Runtime capability registry — what a Discover brick can find at runtime.
# The agent does not own these; it discovers, loads, executes and unloads
# them on demand.
DISCOVERY_REGISTRY: list[dict[str, Any]] = [
    {"id": "github.search", "kind": "mcp", "icon": "hub",
     "desc": "Search repositories and code via MCP."},
    {"id": "github.file", "kind": "mcp", "icon": "description",
     "desc": "Read a file from a repository via MCP."},
    {"id": "gh", "kind": "local", "icon": "terminal",
     "desc": "Local GitHub CLI executable."},
    {"id": "web.search", "kind": "skill", "icon": "travel_explore",
     "desc": "Web research skill (search + browse + summarize)."},
    {"id": "code.analyze", "kind": "skill", "icon": "code_blocks",
     "desc": "Code analysis skill (file + shell + git)."},
    {"id": "fs.read", "kind": "local", "icon": "folder_open",
     "desc": "Local filesystem read executable."},
]

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
    "discover": {
        "label": "Discover", "group": "ACT", "icon": "search",
        "ports": {"in": ["task", "action"], "out": ["action", "result"]},
        "desc": "Find a capability at runtime: MCP / Skill / local executable.",
        "why": "The agent doesn't own tools — it owns the ability to find them.",
        "tradeoffs": ["+ on-demand loading", "+ no static registry", "- discovery latency"],
        "note": "15-mcp-integration-deep-dive.md",
        "cost": {"latency": 60, "tokens": 200, "calls": 1, "llm": False},
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
        "recovers_with": "recovery-fallback",
        "impact": {"latency": 2, "tokens": 0},
    },
    "tool_timeout": {
        "label": "Tool timeout",
        "targets": ["tool", "mcp"],
        "recovery": "Tool retry ×2 with backoff",
        "recovers_with": "recovery-retry",
        "impact": {"latency": 1.8, "tokens": 0},
    },
    "mcp_failure": {
        "label": "MCP failure",
        "targets": ["mcp"],
        "recovery": "Planner replanned around unavailable server",
        "recovers_with": "recovery-replan",
        "impact": {"latency": 1.5, "tokens": 300},
    },
    "llm_retry": {
        "label": "LLM retry",
        "targets": ["llm", "planner", "reflect", "verify", "delegate", "critic"],
        "recovery": "LLM retry with exponential backoff",
        "recovers_with": "recovery-retry",
        "impact": {"latency": 1.6, "tokens": 0},
    },
    "context_overflow": {
        "label": "Context overflow",
        "targets": ["llm", "planner", "reflect"],
        "recovery": "Context window truncation → distilled summary",
        "recovers_with": "recovery-fallback",
        "impact": {"latency": 1.3, "tokens": -400},
    },
}


def simulate_run(nodes: list[dict[str, Any]], edges: list[dict[str, Any]],
                 task: str, chaos: list[str] | None = None,
                 recoveries: list[str] | None = None,
                 breaks: dict[str, str] | None = None) -> dict[str, Any]:
    """Simulate a run over the graph, emitting a replayable trace.

    Each visited primitive records latency (ms), token usage and call count;
    every step also carries a short narrative so the frontend can "watch the
    agent think". CHAOS modes inject failures on matching primitives; the
    failure record lists the recovery action that would be taken.

    `recoveries` lists the attached Recovery bricks. A failure only counts
    as recovered when the matching Recovery brick is attached — pull it off
    and the same failure becomes lethal.

    `breaks` maps a specific node id -> failure mode ("break this brick").
    A node in `breaks` fails no matter what type it is — the LEGO "pull the
    brick apart" interaction.
    """
    chaos = chaos or []
    recoveries = set(recoveries or [])
    breaks = breaks or {}
    active = {m: set(CHAOS_MODES[m]["targets"]) for m in chaos if m in CHAOS_MODES}
    order = topo_order(nodes, edges)
    trace: list[dict[str, Any]] = []
    totals = {"latency": 0, "tokens": 0, "calls": 0, "llm_calls": 0}
    failures: list[dict[str, str]] = []
    discovery: list[dict[str, str]] = []
    rng_seed = int(time.time() * 1000) % 1000
    # Which registry entries have been loaded (per run).
    loaded_caps: set[str] = set()

    for nid in order:
        meta = primitive(nid)
        cost = meta["cost"]
        latency = cost["latency"]
        tokens = cost["tokens"]
        calls = cost["calls"]

        # Runtime capability discovery: the Discover brick finds, loads,
        # executes and unloads capabilities on demand.
        if nid == "discover":
            picks = DISCOVERY_REGISTRY[:2]
            for pick in picks:
                if pick["id"] in loaded_caps:
                    discovery.append({"phase": "execute", "item": pick["id"], "kind": pick["kind"]})
                    continue
                discovery.append({"phase": "discover", "item": pick["id"], "kind": pick["kind"]})
                discovery.append({"phase": "load", "item": pick["id"], "kind": pick["kind"]})
                discovery.append({"phase": "execute", "item": pick["id"], "kind": pick["kind"]})
                discovery.append({"phase": "unload", "item": pick["id"], "kind": pick["kind"]})
                loaded_caps.add(pick["id"])
            latency += 60
            tokens += 200

        # Jitter to feel alive (±15%).
        jitter = 1.0 + ((rng_seed + len(trace) * 7) % 30 - 15) / 100.0
        latency = round(latency * jitter, 1)

        # A specific break on this node wins over global chaos modes.
        hit_mode = breaks.get(nid)
        if hit_mode is None:
            for mode in chaos:
                if nid in active.get(mode, set()):
                    hit_mode = mode
                    break
        if hit_mode and hit_mode in CHAOS_MODES:
            impact = CHAOS_MODES[hit_mode]["impact"]
            latency = round(latency * impact["latency"], 1)
            tokens = max(0, tokens + impact["tokens"])
            # Human-readable impact summary for the chaos report.
            impact_parts = []
            if impact["latency"] != 1.0:
                pct = round((impact["latency"] - 1.0) * 100)
                impact_parts.append(f"+{pct}% latency")
            if impact["tokens"] > 0:
                impact_parts.append(f"+{impact['tokens']} tokens (replan)")
            # Recovery only works if the matching Recovery brick is attached.
            needed = CHAOS_MODES[hit_mode].get("recovers_with")
            recovered_ok = needed in recoveries
            failures.append({
                "node": nid,
                "label": meta["label"],
                "icon": meta["icon"],
                "mode": hit_mode,
                "message": f"{meta['label']} hit {CHAOS_MODES[hit_mode]['label']}",
                "impact": ", ".join(impact_parts) or "no direct cost change",
                "recovery": CHAOS_MODES[hit_mode]["recovery"],
                "recovered": recovered_ok,
                "needs": needed,
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

    # Resilience: only failures whose Recovery brick is attached heal.
    recovered = sum(1 for f in failures if f["recovered"])
    unrecovered = len(failures) - recovered
    # Unrecovered failures hit hard; recovered ones cost a bit.
    resilience = 100
    if failures:
        resilience = max(0, 100 - unrecovered * 40 - recovered * 12)
    status = "failed" if resilience <= 0 else "survived"

    return {
        "task": task,
        "trace": trace,
        "totals": totals,
        "failures": failures,
        "recovered": recovered,
        "unrecovered": unrecovered,
        "resilience": resilience,
        "status": status,
        "chaos": chaos,
        "recoveries": sorted(recoveries),
        "discovery": discovery,
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

# EXAMPLES metadata — presets reframed as architecture examples the user
# can load into the canvas and then edit, not as a "mode selector".
EXAMPLES: list[dict[str, str]] = [
    {"id": "simple", "name": "Simple ReAct",
     "desc": "Minimal think-act-observe loop: planner routes to the LLM."},
    {"id": "rag", "name": "RAG Agent",
     "desc": "Retrieve grounded context first, then plan with tools."},
    {"id": "tiered", "name": "Tiered Memory Agent",
     "desc": "Memory → retrieval → MCP tools, with reflection at the end."},
    {"id": "multi-agent", "name": "Multi-Agent Debate",
     "desc": "Delegate subtasks, then a critic reviews the final output."},
    {"id": "empty", "name": "Blank canvas",
     "desc": "Start from scratch — drag primitives in and wire them."},
]


def compute(params: dict[str, Any]) -> dict[str, Any]:
    """Lab entry point: assemble, run, inject chaos, compile, compare.

    The frontend may send a custom graph (nodes/edges) it assembled on the
    canvas; when absent we fall back to a preset example. Passing
    `experiment: true` routes to the controlled baseline-vs-variant runner.
    """
    if params.get("experiment"):
        return experiment(params)
    if params.get("evolve"):
        return evolve(params)
    preset_id = str(params.get("preset", "simple"))
    preset = PRESETS.get(preset_id, PRESETS["simple"])
    custom = params.get("graph")
    if isinstance(custom, dict) and isinstance(custom.get("nodes"), list):
        graph = custom
    else:
        graph = preset
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

    # Attached Recovery bricks (pluggable self-healing).
    raw_recoveries = params.get("recoveries")
    recoveries = [r for r in (raw_recoveries or []) if r in RECOVERY_BRICKS]

    # Per-node breaks: {"<node_id>": "<failure_mode>"} from the "break it"
    # drag interaction. Maps to the LEGO "pull a brick apart" gesture.
    raw_breaks = params.get("breaks")
    breaks: dict[str, str] = {}
    if isinstance(raw_breaks, dict):
        breaks = {str(k): str(v) for k, v in raw_breaks.items() if str(v) in CHAOS_MODES}
    elif isinstance(raw_breaks, list):
        for item in raw_breaks:
            if isinstance(item, dict) and str(item.get("mode")) in CHAOS_MODES:
                breaks[str(item.get("node"))] = str(item["mode"])

    run = simulate_run(graph["nodes"], graph["edges"], task, chaos, recoveries, breaks)
    name = graph.get("name") or (custom and "custom-agent") or preset["name"]
    yaml = compile_agent(graph["nodes"], graph["edges"], name=str(name).lower().replace(" ", "-"))
    validation = validate_graph(graph["nodes"], graph["edges"])

    # Baseline (no chaos) for the compare view.
    baseline = simulate_run(graph["nodes"], graph["edges"], task, [], recoveries)
    compare = {
        "active": bool(params.get("compare")),
        "a": baseline["totals"],
        "b": run["totals"],
        "diffs": _diff_summaries(baseline, run),
    }

    return {
        "preset": preset_id,
        "presetName": name,
        "primitives": {
            pid: {
                "label": p["label"], "group": p["group"], "icon": p["icon"],
                "ports": p["ports"], "desc": p["desc"], "why": p["why"],
                "tradeoffs": p["tradeoffs"], "note": p["note"],
                "class": _CLASS_OF.get(pid, "brain"),
                "rarity": BRICK_CLASSES[_CLASS_OF.get(pid, "brain")]["rarity"],
            }
            for pid, p in PRIMITIVES.items()
        },
        "brickClasses": BRICK_CLASSES,
        "breakBricks": BREAK_BRICKS,
        "recoveryBricks": RECOVERY_BRICKS,
        "skillBoxes": SKILL_BOXES,
        "discoveryRegistry": DISCOVERY_REGISTRY,
        "groups": GROUPS,
        "examples": EXAMPLES,
        "graph": graph,
        "run": run,
        "yaml": yaml,
        "compare": compare,
        "validation": validation,
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


def validate_graph(nodes: list[dict[str, Any]], edges: list[dict[str, Any]]) -> dict[str, Any]:
    """Validate a graph's wiring; returns valid flag + a violation list.

    Unlike check_graph (which returns raw strings), this returns structured
    violations so the UI can render a CONTRACT MISMATCH card with the exact
    ports involved and what the target actually accepts.
    """
    ids = [n["id"] for n in nodes]
    violations = []
    for e in edges:
        src_id, dst_id = e.get("from"), e.get("to")
        if src_id not in ids or dst_id not in ids:
            violations.append({
                "kind": "unknown_node",
                "from": src_id, "to": dst_id, "port": e.get("port", ""),
                "expected": [], "message": f"unknown node: {src_id} -> {dst_id}",
            })
            continue
        dst = primitive(dst_id)
        if e.get("port") and e["port"] not in dst["ports"]["in"]:
            src = primitive(src_id)
            violations.append({
                "kind": "contract",
                "from": src_id, "to": dst_id, "port": e["port"],
                "expected": dst["ports"]["in"],
                "message": (
                    f"{src['label']}.{e['port']} → {dst['label']} expects "
                    f"[{', '.join(dst['ports']['in'])}]"
                ),
            })
    return {"valid": len(violations) == 0, "violations": violations}


def experiment(params: dict[str, Any]) -> dict[str, Any]:
    """Run a controlled experiment: baseline vs variant, with failure
    injection on the variant. Unifies Run + Chaos + Compare into one view."""
    task = str(params.get("task", "分析这个项目的 FFI 安全问题"))
    baseline_id = str(params.get("baseline", "simple"))
    variant_id = str(params.get("variant", "rag"))
    base = PRESETS.get(baseline_id, PRESETS["simple"])
    var = PRESETS.get(variant_id, PRESETS["rag"])

    chaos = []
    for key, mode in (("chaos_memory", "memory_unavailable"),
                      ("chaos_tool", "tool_timeout"),
                      ("chaos_mcp", "mcp_failure"),
                      ("chaos_llm", "llm_retry"),
                      ("chaos_context", "context_overflow")):
        if params.get(key):
            chaos.append(mode)

    rb = simulate_run(base["nodes"], base["edges"], task, [])
    rv = simulate_run(var["nodes"], var["edges"], task, chaos)

    def row(run: dict[str, Any]) -> dict[str, Any]:
        t = run["totals"]
        return {
            "success": run["status"] == "survived",
            "resilience": run["resilience"],
            "llm_calls": t["llm_calls"],
            "tokens": t["tokens"],
            "latency": round(t["latency"], 1),
            "recovery": run["recovered"] if run["failures"] else 0,
            "failures": len(run["failures"]),
        }

    return {
        "task": task,
        "baseline": {"name": base["name"], **row(rb)},
        "variant": {"name": var["name"], **row(rv)},
        "chaos": chaos,
        "diffs": _diff_summaries(rb, rv),
    }


# ---------------------------------------------------------------------------
# Evolution — the LEGO that rebuilds itself
# ---------------------------------------------------------------------------

# The genotype is a graph (nodes/edges). Fitness measures how well it runs:
# survival dominates, then resilience, minus cost (latency/tokens) and a
# small structural penalty for bloat.
def _fitness(graph: dict[str, Any], task: str) -> float:
    """Score an architecture by running it without chaos."""
    run = simulate_run(graph["nodes"], graph["edges"], task, [])
    t = run["totals"]
    score = 100.0 if run["status"] == "survived" else 30.0
    score += run["resilience"] * 0.2
    score -= t["latency"] * 0.02
    score -= t["tokens"] * 0.001
    score -= len(graph["nodes"]) * 1.5  # bloat penalty
    return round(max(0.0, score), 2)


def _mutate(graph: dict[str, Any], rng: np.random.Generator) -> dict[str, Any]:
    """Randomly add a node, drop a node, or rewire an edge."""
    nodes = [dict(n) for n in graph["nodes"]]
    edges = [dict(e) for e in graph["edges"]]
    pool = ["memory", "retrieve", "tool", "mcp", "reflect", "verify", "critic"]
    # Drop a non-essential node (keep input/output/llm).
    droppable = [n["id"] for n in nodes if n["id"] not in ("input", "output", "llm")]
    if droppable and rng.random() < 0.4:
        drop = rng.choice(droppable)
        nodes = [n for n in nodes if n["id"] != drop]
        edges = [e for e in edges if e["from"] != drop and e["to"] != drop]
    # Add a node wired from planner/llm to output-ish sink.
    if rng.random() < 0.4:
        new_id = str(rng.choice(pool))
        source = "planner" if any(n["id"] == "planner" for n in nodes) else "llm"
        if new_id not in [n["id"] for n in nodes]:
            nodes.append({"id": new_id})
            edges.append({"from": source, "to": new_id, "port": "task"})
            edges.append({"from": new_id, "to": "llm", "port": "result"})
    # Rewire one edge.
    if edges and rng.random() < 0.3:
        e = edges[rng.integers(0, len(edges))]
        e["to"] = "llm"
        e["port"] = "result"
    return {"name": f"mutant-{rng.integers(100, 999)}", "nodes": nodes, "edges": edges}


def _crossover(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """Merge two architectures: union of nodes, prefer a's edges, keep valid."""
    ids = set(n["id"] for n in a["nodes"]) | set(n["id"] for n in b["nodes"])
    nodes = [{"id": i} for i in ids]
    edges = list(a["edges"])
    seen = {(e["from"], e["to"]) for e in edges}
    for e in b["edges"]:
        if (e["from"], e["to"]) not in seen and e["from"] in ids and e["to"] in ids:
            edges.append(dict(e))
    return {"name": "crossover", "nodes": nodes, "edges": edges}


def evolve(params: dict[str, Any]) -> dict[str, Any]:
    """Run a tiny genetic algorithm over agent architectures.

    Generations: evaluate the population, keep the best, then create the
    next generation via crossover + mutation. Returns per-generation best /
    average fitness plus the evolved (fittest) architecture.
    """
    task = str(params.get("task", "分析这个项目的 FFI 安全问题"))
    generations = int(params.get("generations", 4))
    pop_size = int(params.get("population", 6))
    seed = int(params.get("seed", 7))
    rng = np.random.default_rng(seed)

    # Initial population from the presets (excluding the empty canvas).
    population = [dict(p) for k, p in PRESETS.items() if k != "empty"]
    while len(population) < pop_size:
        parent = population[rng.integers(0, len(population))]
        population.append(_mutate(parent, rng))

    history = []
    for gen in range(generations):
        scored = [( _fitness(g, task), g) for g in population]
        scored.sort(key=lambda x: x[0], reverse=True)
        best_fit, best_graph = scored[0]
        avg_fit = round(sum(s for s, _ in scored) / len(scored), 2)
        history.append({"gen": gen + 1, "best": best_fit, "avg": avg_fit})

        # Selection: keep the top half.
        keep = scored[: max(2, pop_size // 2)]
        next_pop = [dict(g) for _, g in keep]
        # Crossover + mutation to refill.
        while len(next_pop) < pop_size:
            a = keep[rng.integers(0, len(keep))][1]
            b = keep[rng.integers(0, len(keep))][1]
            child = _crossover(a, b)
            if rng.random() < 0.6:
                child = _mutate(child, rng)
            next_pop.append(child)
        population = next_pop

    # Final population evaluation.
    scored = [( _fitness(g, task), g) for g in population]
    scored.sort(key=lambda x: x[0], reverse=True)
    best_fit, best_graph = scored[0]

    return {
        "task": task,
        "generations": generations,
        "population": pop_size,
        "history": history,
        "best": {"fitness": best_fit, "graph": best_graph},
        "fittest": [{"name": g.get("name", "agent"), "fitness": s}
                    for s, g in scored[:5]],
    }
