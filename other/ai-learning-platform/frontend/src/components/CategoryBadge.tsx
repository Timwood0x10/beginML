import type { Category } from "../types";

const CATEGORY_BADGE_STYLES: Record<string, string> = {
  math: "bg-[#f3dfdc] text-[#8a3a35] dark:bg-[#3d2a28] dark:text-[#e9b8b2]",
  attention:
    "bg-[#e1e4f3] text-[#3a3f85] dark:bg-[#262b45] dark:text-[#b6bce8]",
  hybrid: "bg-[#e9e1f3] text-[#5a3d85] dark:bg-[#2c2640] dark:text-[#c4b5e6]",
  paper: "bg-[#dcecef] text-[#2f6068] dark:bg-[#20333a] dark:text-[#a8d3dc]",
  agent: "bg-[#ece1f3] text-[#6a3d85] dark:bg-[#352640] dark:text-[#cdb5e6]",
  general:
    "bg-surface-variant text-on-surface-variant dark:bg-white/10 dark:text-dark-on-surface",
};

export default function CategoryBadge({
  category,
  size = "sm",
}: {
  category: Category;
  size?: "sm" | "md";
}) {
  const styles =
    CATEGORY_BADGE_STYLES[category.id] ?? CATEGORY_BADGE_STYLES.general;
  const padding =
    size === "md" ? "px-3.5 py-1.5 text-label-md" : "px-2.5 py-1 text-caption";
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full font-semibold ${padding} ${styles}`}
    >
      <span
        className="material-symbols-outlined"
        style={{ fontSize: size === "md" ? 16 : 13 }}
      >
        {category.icon}
      </span>
      {category.en}
    </span>
  );
}

export { CATEGORY_BADGE_STYLES };
