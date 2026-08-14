// UI string dictionaries for the AIScope interface (zh / en).
// All user-facing copy lives here; components pull via useI18n().t().

export type Lang = 'zh' | 'en'

export interface UIDict {
  nav: {
    roadmap: string
    library: string
    path: string
    map: string
    agent: string
    lab: string
  }
  common: {
    read: string
    minutes: string
    words: string
    back: string
    retry: string
    loading: string
    clear: string
    theme: string
  }
  home: {
    badge: string
    title1: string
    title2: string
    subtitle: string
    browseCta: string
    mapCta: string
    notes: string
    readingTime: string
    subjects: string
    journey: string
    journeyDesc: string
    fullPath: string
    startHere: string
    upNext: string
    explore: string
    open: string
    startReading: string
    viewAll: string
    resources: string
    resSearchTitle: string
    resSearchDesc: string
    resMapTitle: string
    resMapDesc: string
    resPathTitle: string
    resPathDesc: string
    go: string
    words: string
    categoryNames: Record<string, string>
    categoryBlurbs: Record<string, string>
  }
  browse: {
    title: string
    subtitle: string
    searchPlaceholder: string
    all: string
    showing: string
    of: string
    rankedBy: string
    clearFilters: string
    noResults: string
    noResultsSub: string
    tryOther: string
  }
  path: {
    badge: string
    title: string
    subtitle: string
    notes: string
    startHere: string
    phases: Record<string, { phase: string; title: string; blurb: string }>
  }
  map: {
    title: string
    subtitle: string
    reset: string
    all: string
    hint: string
    open: string
  }
  lab: {
    title: string
    subtitle: string
    controls: string
    computing: string
  }
  note: {
    contents: string
    related: string
    similar: string
    backToLibrary: string
    minRead: string
    min: string
  }
}

export const zh: UIDict = {
  nav: { roadmap: '学习路线', library: '资料库', path: '学习路径', map: '知识地图', agent: 'Agent', lab: '数学实验室' },
  common: { read: '阅读', minutes: '分钟', words: '字', back: '返回', retry: '重试', loading: '加载中…', clear: '清除', theme: '主题' },
  home: {
    badge: '你的个人知识图谱',
    title1: '一间安静的书斋，',
    title2: '通向深入理解。',
    subtitle: '围绕 AI 的数学、架构与前沿的交互式笔记——可搜索、互联互通，由 scikit-learn 构建语义地图。',
    browseCta: '浏览资料库',
    mapCta: '打开知识地图',
    notes: '篇笔记',
    readingTime: '阅读时长',
    subjects: '学科',
    journey: '你的学习旅程',
    journeyDesc: '从第一性原理走向前沿架构。',
    fullPath: '完整路径',
    startHere: '从这里开始',
    upNext: '下一步',
    explore: '探索',
    open: '打开',
    startReading: '开始阅读',
    viewAll: '查看全部',
    resources: '辅助资源',
    resSearchTitle: '语义搜索',
    resSearchDesc: '由 scikit-learn 驱动的 TF-IDF 搜索，按含义而非关键词找笔记。',
    resMapTitle: '知识地图',
    resMapDesc: '基于内容的多维缩放，把每篇笔记放在二维地图上。',
    resPathTitle: '引导路径',
    resPathDesc: '沿着时间线浏览笔记——从微积分到混合架构。',
    go: '探索',
    words: '字数',
    categoryNames: { math: '数学', attention: '自注意力', hybrid: '混合架构', paper: '研究论文', agent: '智能体工程', general: '综合' },
    categoryBlurbs: {
      math: '微积分、线性代数、优化与深度学习的几何。',
      attention: '自注意力、多头、编码器、解码器、RoPE 与推理。',
      hybrid: 'MoE、Mamba、MLA、量化与 Transformer 之后的边界。',
      paper: '带批注的研究论文与细读。',
      agent: '从零构建 Agent：工具调用、记忆、规划与多智能体协作。',
      general: '跨 AI 领域的基础笔记。',
    },
  },
  browse: {
    title: '资料库',
    subtitle: '由 scikit-learn 驱动的语义搜索，在多个学科间浏览全部笔记。',
    searchPlaceholder: '按概念搜索，例如 梯度下降、RoPE、MoE…',
    all: '全部',
    showing: '显示',
    of: '共',
    rankedBy: '按相关性排序',
    clearFilters: '清除筛选',
    noResults: '没有符合条件的笔记',
    noResultsSub: '换个概念再试试，或清空搜索。',
    tryOther: '换个主题试试。',
  },
  path: {
    badge: '引导式课程',
    title: '一条深思熟虑的笔记之路',
    subtitle: '四个阶段排列的笔记——先打地基，再看架构，最后抵达前沿。',
    notes: '篇笔记',
    startHere: '从这里开始',
    phases: {
      math: { phase: '第一阶段', title: '数学基础', blurb: '微积分、线性代数、概率与优化——每个模型之下的基石。' },
      attention: { phase: '第二阶段', title: '自注意力与 Transformer', blurb: '从缩放点积注意力到多头、编码器、解码器与 RoPE。' },
      hybrid: { phase: '第三阶段', title: '混合与前沿架构', blurb: 'MoE、Mamba/SSM、MLA、MTP、量化与 Transformer 之后的推理。' },
      paper: { phase: '第四阶段', title: '研究论文细读', blurb: '带批注的论文与实证研究，与基础并读。' },
      agent: { phase: '第五阶段', title: 'Agent 工程', blurb: '从零构建 Agent：工具调用、记忆、规划与多智能体协作。' },
      general: { phase: '参考', title: '综合笔记', blurb: '基础综述与参考资料。' },
    },
  },
  map: {
    title: '知识地图',
    subtitle: '每篇笔记按内容的 TF-IDF 向量做多维缩放定位——相邻即相似。拖拽平移，滚轮缩放，点击阅读。',
    reset: '重置视图',
    all: '全部',
    hint: '拖拽平移 · 滚轮缩放',
    open: '点击打开',
  },
  lab: {
    title: '数学实验室',
    subtitle: '交互式可视化',
    controls: '控件',
    computing: '计算中…',
  },
  note: {
    contents: '目录',
    related: '相关笔记',
    similar: '相似度',
    backToLibrary: '返回资料库',
    minRead: '分钟阅读',
    min: '分钟',
  },
}

export const en: UIDict = {
  nav: { roadmap: 'Roadmap', library: 'Library', path: 'Learning Path', map: 'Knowledge Map', agent: 'Agent', lab: 'Math Lab' },
  common: { read: 'Read', minutes: 'min', words: 'words', back: 'Back', retry: 'Retry', loading: 'Loading…', clear: 'Clear', theme: 'Theme' },
  home: {
    badge: 'Your personal knowledge atlas',
    title1: 'A quiet library for',
    title2: 'deep understanding.',
    subtitle: 'Interactive notes on the mathematics, architecture and frontiers of modern AI — searchable, interconnected, mapped with scikit-learn.',
    browseCta: 'Browse the library',
    mapCta: 'Open the knowledge map',
    notes: 'notes',
    readingTime: 'reading time',
    subjects: 'subjects',
    journey: 'Your learning journey',
    journeyDesc: 'Follow the path from first principles to frontier architectures.',
    fullPath: 'Full path',
    startHere: 'Start here',
    upNext: 'Up next',
    explore: 'Explore',
    open: 'Open',
    startReading: 'Start reading',
    viewAll: 'View all',
    resources: 'Supporting resources',
    resSearchTitle: 'Semantic search',
    resSearchDesc: 'TF-IDF search powered by scikit-learn finds notes by meaning, not just keywords.',
    resMapTitle: 'Knowledge map',
    resMapDesc: 'A 2D map of every note, positioned by multi-dimensional scaling of their content.',
    resPathTitle: 'Guided path',
    resPathDesc: 'A timeline through the notes — from calculus to hybrid architectures.',
    go: 'Explore',
    words: 'words',
    categoryNames: { math: 'Mathematics', attention: 'Self-Attention', hybrid: 'Hybrid Models', paper: 'Research Papers', agent: 'Agent Engineering', general: 'General' },
    categoryBlurbs: {
      math: 'Calculus, linear algebra, optimization & the geometry of deep learning.',
      attention: 'Self-attention, multi-head, encoders, decoders, RoPE & inference.',
      hybrid: 'MoE, Mamba, MLA, quantization and the post-Transformer frontier.',
      paper: 'Annotated research papers and close readings.',
      agent: 'Building an Agent from scratch: tools, memory, planning & multi-agent collaboration.',
      general: 'Foundational notes across the AI landscape.',
    },
  },
  browse: {
    title: 'Library',
    subtitle: 'Semantic search powered by scikit-learn, browsing every note across subjects.',
    searchPlaceholder: 'Search by concept, e.g. gradient descent, RoPE, MoE…',
    all: 'All',
    showing: 'Showing',
    of: 'of',
    rankedBy: 'ranked by relevance',
    clearFilters: 'Clear filters',
    noResults: 'No notes match your filters',
    noResultsSub: 'Try another concept, or clear the search.',
    tryOther: 'Try a different subject.',
  },
  path: {
    badge: 'Guided curriculum',
    title: 'A deliberate path through the notes',
    subtitle: 'Notes arranged in four phases — foundations first, then architectures, then the frontier.',
    notes: 'notes',
    startHere: 'Start here',
    phases: {
      math: { phase: 'Phase I', title: 'Mathematical foundations', blurb: 'Calculus, linear algebra, probability and optimization — the bedrock beneath every model.' },
      attention: { phase: 'Phase II', title: 'Self-attention & Transformers', blurb: 'From scaled dot-product attention to multi-head, encoders, decoders and RoPE.' },
      hybrid: { phase: 'Phase III', title: 'Hybrid & frontier architectures', blurb: 'MoE, Mamba/SSMs, MLA, MTP, quantization and post-Transformer reasoning.' },
      paper: { phase: 'Phase IV', title: 'Research close-readings', blurb: 'Annotated papers and empirical studies, read alongside the foundations.' },
      agent: { phase: 'Phase V', title: 'Agent engineering', blurb: 'Building an Agent from scratch: tools, memory, planning & multi-agent collaboration.' },
      general: { phase: 'Reference', title: 'General notes', blurb: 'Foundational overviews and reference material.' },
    },
  },
  map: {
    title: 'Knowledge map',
    subtitle: 'Every note is placed by multi-dimensional scaling of its TF-IDF vector — nearby notes discuss similar ideas. Drag to pan, scroll to zoom, click to read.',
    reset: 'Reset view',
    all: 'All',
    hint: 'Drag to pan · scroll to zoom',
    open: 'click to open',
  },
  lab: {
    title: 'Math Lab',
    subtitle: 'Interactive visualizations',
    controls: 'Controls',
    computing: 'computing…',
  },
  note: {
    contents: 'Contents',
    related: 'Related notes',
    similar: 'similar',
    backToLibrary: 'Back to library',
    minRead: 'min read',
    min: 'min',
  },
}
