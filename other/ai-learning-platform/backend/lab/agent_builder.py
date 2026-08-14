"""
Agent builder: a block-assembly model of an AI agent.

The user picks one option from each of four categories (memory, tools,
planning, multi-agent). The module explains how those choices fit together,
returns a rendered YAML config sketch and a layered architecture diagram so
the frontend can show the assembled agent as clickable blocks.

All copy is English; functions are short; the file follows the project rules
(each function <= 120 lines, file <= 1000 lines).
"""

from typing import Any

# Category order used by the UI and the diagram (top -> bottom).
CATEGORY_ORDER = ["memory", "tools", "planning", "multi"]

# Each category: label + icon (Material Symbols) + options.
# Each option: label, one-line "why", behavior note, YAML config snippet.
COMPONENTS: dict[str, dict[str, Any]] = {
    "memory": {
        "label": "Memory",
        "icon": "memory",
        "options": {
            "none": {
                "label": "Stateless",
                "why": "Every request starts fresh — simplest, no data to keep.",
                "behavior": "Each turn is independent; the model only sees the prompt.",
                "config": "memory:\n  enabled: false",
            },
            "context": {
                "label": "Context window",
                "why": "Recent chat history is packed into the prompt.",
                "behavior": "Works for short sessions; cost grows with history.",
                "config": "memory:\n  enabled: true\n  type: context\n  max_tokens: 8192",
            },
            "vector": {
                "label": "Vector RAG",
                "why": "Retrieve relevant snippets from a corpus by embedding.",
                "behavior": "Scales to large knowledge bases; needs an index & retriever.",
                "config": "memory:\n  enabled: true\n  type: vector\n  top_k: 5\n  embedding: text-embedding-3-small",
            },
            "longterm": {
                "label": "Long-term store",
                "why": "Persistent database / file memory across sessions.",
                "behavior": "Remembers users and facts; needs schema & dedup logic.",
                "config": "memory:\n  enabled: true\n  type: longterm\n  store: sqlite://agents.db",
            },
            "hybrid": {
                "label": "Tiered memory",
                "why": "Short-term context + vector retrieval + long-term store.",
                "behavior": "The production pattern: fast recent history, semantic recall, durable facts.",
                "config": "memory:\n  enabled: true\n  type: hybrid\n  tiers: [context, vector, longterm]",
            },
        },
    },
    "tools": {
        "label": "Tools",
        "icon": "handyman",
        "options": {
            "none": {
                "label": "No tools",
                "why": "Pure text in / text out.",
                "behavior": "Cannot touch the outside world; fine for chat-only agents.",
                "config": "tools:\n  enabled: false",
            },
            "mcp": {
                "label": "MCP protocol",
                "why": "Discover and call external tools through the Model Context Protocol.",
                "behavior": "One protocol for many servers; the modern way to plug in tools.",
                "config": "tools:\n  enabled: true\n  type: mcp\n  servers:\n    - name: filesystem\n      url: http://localhost:3001/sse",
            },
            "search": {
                "label": "Web search",
                "why": "Query the open web for up-to-date facts.",
                "behavior": "Good for research agents; needs an API key & result parsing.",
                "config": "tools:\n  enabled: true\n  type: search\n  provider: bing",
            },
            "code": {
                "label": "Code execution",
                "why": "Run generated code in a sandbox.",
                "behavior": "Enables the agent to compute, plot and verify — with safety limits.",
                "config": "tools:\n  enabled: true\n  type: code\n  sandbox: docker\n  timeout_seconds: 30",
            },
            "custom": {
                "label": "Function calling",
                "why": "Hand-written functions registered as callable tools.",
                "behavior": "Simple and precise; you maintain the schema and the executor.",
                "config": "tools:\n  enabled: true\n  type: function_calling\n  registry: my_functions.py",
            },
        },
    },
    "planning": {
        "label": "Planning",
        "icon": "route",
        "options": {
            "none": {
                "label": "Direct answer",
                "why": "No planning loop; answer immediately.",
                "behavior": "Fast for simple tasks; struggles with multi-step problems.",
                "config": "planning:\n  enabled: false",
            },
            "react": {
                "label": "ReAct",
                "why": "Interleaved think -> act -> observe loop.",
                "behavior": "The classic pattern; each observation feeds the next thought.",
                "config": "planning:\n  enabled: true\n  type: react\n  max_steps: 8",
            },
            "plan": {
                "label": "Plan-and-execute",
                "why": "Draft a step list first, then carry it out.",
                "behavior": "Better for long tasks; plans can be revised mid-way.",
                "config": "planning:\n  enabled: true\n  type: plan_execute\n  max_steps: 12",
            },
            "reflect": {
                "label": "Reflection",
                "why": "Self-critique the output, then improve it.",
                "behavior": "Raises quality on hard tasks at the cost of extra calls.",
                "config": "planning:\n  enabled: true\n  type: reflection\n  rounds: 2",
            },
            "tree": {
                "label": "Tree search",
                "why": "Explore several candidate action paths, keep the best.",
                "behavior": "Strong but expensive; used in coding & math agents.",
                "config": "planning:\n  enabled: true\n  type: tree_search\n  beam: 4\n  depth: 6",
            },
        },
    },
    "multi": {
        "label": "Multi-agent",
        "icon": "groups",
        "options": {
            "single": {
                "label": "Single agent",
                "why": "One agent does everything.",
                "behavior": "Simplest to build and debug.",
                "config": "multi_agent:\n  enabled: false",
            },
            "orchestrator": {
                "label": "Orchestrator-worker",
                "why": "A planner dispatches subtasks to worker agents.",
                "behavior": "Scales to parallel work; needs task splitting & result merging.",
                "config": "multi_agent:\n  enabled: true\n  pattern: orchestrator\n  workers: 4",
            },
            "blackboard": {
                "label": "Blackboard",
                "why": "Agents share a scratchpad they all read & write.",
                "behavior": "Good for incremental assembly; risk of conflicting edits.",
                "config": "multi_agent:\n  enabled: true\n  pattern: blackboard\n  shared: scratchpad.json",
            },
            "review": {
                "label": "Generator + critic",
                "why": "One agent drafts, another critiques until accepted.",
                "behavior": "Raises quality; converges when the critic approves.",
                "config": "multi_agent:\n  enabled: true\n  pattern: reviewer\n  max_rounds: 3",
            },
        },
    },
}


def _option(category: str, choice: str) -> dict[str, str]:
    """Return a single option's metadata, falling back to the first one."""
    opts = COMPONENTS[category]["options"]
    return opts.get(choice, next(iter(opts.values())))


def _id_for_label(category: str, label: str) -> str:
    """Map a friendly display label back to its option id."""
    for key, meta in COMPONENTS[category]["options"].items():
        if meta["label"] == label:
            return key
    return ""


def option_labels(category: str) -> list[str]:
    """Friendly labels for the frontend select control."""
    return [meta["label"] for meta in COMPONENTS[category]["options"].values()]


def _build_config(selections: dict[str, str]) -> str:
    """Assemble the per-category YAML snippets into one config sketch."""
    lines = ["# Generated agent config (sketch)", "agent:", "  name: my_agent"]
    for cat in CATEGORY_ORDER:
        snippet = _option(cat, selections.get(cat, ""))["config"]
        for i, ln in enumerate(snippet.split("\n")):
            indent = "  " if i == 0 else "    "
            lines.append(f"{indent}{ln}")
    return "\n".join(lines)


def _describe(selections: dict[str, str]) -> str:
    """One-sentence summary of how the chosen blocks fit together."""
    mem = _option("memory", selections.get("memory", ""))["label"]
    tool = _option("tools", selections.get("tools", ""))["label"]
    plan = _option("planning", selections.get("planning", ""))["label"]
    multi = _option("multi", selections.get("multi", ""))["label"]
    return (
        f"An agent with {mem} memory, {tool} tools, {plan} planning and "
        f"{multi} coordination: the LLM brain routes through memory for "
        "context, calls tools to touch the world, plans the steps, and "
        "coordinates other agents when the task needs it."
    )


def _diagram(selections: dict[str, str]) -> list[dict[str, Any]]:
    """Layered architecture nodes for the frontend block diagram."""
    nodes = []
    for cat in CATEGORY_ORDER:
        meta = COMPONENTS[cat]
        opt = _option(cat, selections.get(cat, ""))
        nodes.append({
            "category": cat,
            "categoryLabel": meta["label"],
            "icon": meta["icon"],
            "choice": selections.get(cat, ""),
            "label": opt["label"],
            "why": opt["why"],
            "behavior": opt["behavior"],
        })
    return nodes


def compute(params: dict[str, Any]) -> dict[str, Any]:
    """Lab entry point: turn four block choices into an agent blueprint.

    The frontend sends friendly labels (e.g. "Vector RAG"); we resolve them
    back to stable ids so the diagram and config use canonical keys.
    """
    clean = {}
    for cat in CATEGORY_ORDER:
        label = str(params.get(cat, ""))
        clean[cat] = _id_for_label(cat, label)
    return {
        "categories": [
            {"id": cat, "label": COMPONENTS[cat]["label"], "icon": COMPONENTS[cat]["icon"]}
            for cat in CATEGORY_ORDER
        ],
        "selections": clean,
        "diagram": _diagram(clean),
        "architecture": _describe(clean),
        "config": _build_config(clean),
    }
