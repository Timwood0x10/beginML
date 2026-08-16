import { Link } from "react-router-dom";
import { useI18n } from "../i18n/context";

// Landing page for the Agent Engineering column. Points to the interactive
// ARES Agent Lab, the Transformer training lab, and the note library
// filtered to this category.

const INTERACTIVES = [
  {
    to: "/lab/agent-forge",
    icon: "construction",
    titleKey: "forgeTitle",
    descKey: "forgeDesc",
  },
  {
    to: "/lab/transformer-training",
    icon: "model_training",
    titleKey: "trainTitle",
    descKey: "trainDesc",
  },
  {
    to: "/lab/attention",
    icon: "grid_on",
    titleKey: "attnTitle",
    descKey: "attnDesc",
  },
] as const;

export default function AgentPage() {
  const { lang, t } = useI18n();
  const name = t.home.categoryNames.agent ?? "Agent Engineering";
  const blurb = t.home.categoryBlurbs.agent ?? "";

  const copy: Record<string, { title: string; desc: string }> = {
    forgeTitle: {
      title: "ARES Agent Lab",
      desc: "像乐高一样搭 Agent：拖认知积木、吸附语义端口、注入故障、观察自愈与进化。",
    },
    forgeDesc: { title: "", desc: "" },
    trainTitle: {
      title: "Transformer 训练",
      desc: "从零训练迷你 Transformer，观察损失曲线与注意力热图逐步成形。",
    },
    trainDesc: { title: "", desc: "" },
    attnTitle: {
      title: "自注意力",
      desc: "一步步构建 Q·Kᵀ/√d 注意力权重矩阵，观察 softmax 的锐化。",
    },
    attnDesc: { title: "", desc: "" },
  };
  const copyEn: Record<string, { title: string; desc: string }> = {
    forgeTitle: {
      title: "ARES Agent Lab",
      desc: "Build an agent like Lego: drag cognitive blocks, snap semantic ports, inject faults, watch self-heal and evolve.",
    },
    forgeDesc: { title: "", desc: "" },
    trainTitle: {
      title: "Transformer Training",
      desc: "Train a tiny transformer from scratch; watch loss and attention emerge.",
    },
    trainDesc: { title: "", desc: "" },
    attnTitle: {
      title: "Self-Attention",
      desc: "Build Q·Kᵀ/√d attention weights step by step and see softmax sharpen.",
    },
    attnDesc: { title: "", desc: "" },
  };
  const cc = lang === "zh" ? copy : copyEn;

  return (
    <div className="flex flex-col gap-8 pt-2 max-w-3xl">
      <header className="flex flex-col gap-4">
        <span className="inline-flex items-center gap-2 self-start text-caption font-semibold uppercase tracking-[0.15em] text-primary dark:text-inverse-primary">
          <span className="material-symbols-outlined" style={{ fontSize: 16 }}>
            smart_toy
          </span>
          {name}
        </span>
        <h1 className="font-headline text-headline-lg-mobile md:text-headline-xl text-on-surface dark:text-inverse-on-surface leading-tight">
          {name}
        </h1>
        <p className="font-body-lg text-body-lg text-on-surface-variant dark:text-outline leading-relaxed">
          {blurb}
        </p>
      </header>

      {/* Interactive labs */}
      <section>
        <h2 className="font-headline text-lg text-on-surface dark:text-inverse-on-surface mb-4 inline-flex items-center gap-2">
          <span className="material-symbols-outlined" style={{ fontSize: 20 }}>
            science
          </span>
          {lang === "zh" ? "可交互实验室" : "Interactive Labs"}
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {INTERACTIVES.map((item) => (
            <Link
              key={item.to}
              to={item.to}
              className="group rounded-2xl bg-surface-container-low dark:bg-dark-surface-elevated border border-outline-variant/40 dark:border-white/10 p-5 hover:border-primary/60 dark:hover:border-inverse-primary/60 transition-colors"
            >
              <span
                className="material-symbols-outlined text-primary dark:text-inverse-primary"
                style={{ fontSize: 26 }}
              >
                {item.icon}
              </span>
              <div className="mt-3 font-semibold text-body-md text-on-surface dark:text-dark-on-surface group-hover:text-primary dark:group-hover:text-inverse-primary transition-colors">
                {cc[item.titleKey].title}
              </div>
              <div className="mt-1 text-caption text-on-surface-variant dark:text-outline leading-relaxed">
                {cc[item.descKey].desc}
              </div>
            </Link>
          ))}
        </div>
      </section>

      {/* Notes in this category */}
      <section>
        <h2 className="font-headline text-lg text-on-surface dark:text-inverse-on-surface mb-4 inline-flex items-center gap-2">
          <span className="material-symbols-outlined" style={{ fontSize: 20 }}>
            menu_book
          </span>
          {lang === "zh" ? "笔记" : "Notes"}
        </h2>
        <div className="rounded-2xl bg-surface-container-low dark:bg-dark-surface-elevated border border-outline-variant/40 dark:border-white/10 p-5">
          <p className="text-body-md text-on-surface-variant dark:text-outline leading-relaxed mb-4">
            {lang === "zh"
              ? "goagent（ares 框架）架构拆解系列与融合改写笔记，从零到一构建 Agent 的完整工程脉络。"
              : "A series of goagent (ARES framework) architecture breakdowns and merge-rewrite notes — the full engineering path from zero to a working Agent."}
          </p>
          <Link
            to="/browse?category=agent"
            className="inline-flex items-center gap-2 px-4 py-2.5 rounded-xl bg-primary text-on-primary dark:bg-inverse-primary dark:text-inverse-surface font-label-md text-label-md hover:opacity-90 transition"
          >
            <span
              className="material-symbols-outlined"
              style={{ fontSize: 18 }}
            >
              arrow_forward
            </span>
            {lang === "zh" ? "进入 Agent 笔记库" : "Open the Agent note library"}
          </Link>
        </div>
      </section>
    </div>
  );
}
