# AIScope — AI 学习平台

面向 AI 学习的交互式平台：笔记库、知识地图、**可执行论文实验室**（Paper Lab）与数学实验实验室（Math Lab）。核心卖点是「论文 → 机制 → 源码 → 实验」的可执行论文：每篇论文解析出真实章节大纲，点章节看实现、改参数跑实验、亲手观察结果。

## ✨ 亮点：可执行论文（Paper Lab）

| 能力 | 说明 |
|---|---|
| **真实大纲** | PDF 解析出论文完整章节（编号 + 标题 + 页码），不靠 LLM 猜测 |
| **公式 ↔ 源码联动** | 点击公式/热力图/数据流步骤，右侧源码行自动高亮并滚动 |
| **机制导航** | 每篇论文精选核心机制（注意力、RoPE、残差…），一键切入 |
| **可运行实现** | 每个章节带独立可运行的 numpy 实现 + 参数实验 + 观察/证据 |

目前内置两篇论文：`transformer.pdf`（16 节 / 231 公式）、`attention_residuals.pdf`（7 节）。

## 🚀 快速开始

**依赖**：`uv`（Python 包管理）、`node`/`npm`、`lsof`（可选，用于端口检查）。

```bash
./start.sh        # 一键启动：安装依赖 + 启动后端(:8000) + 前端(:5173)
./start.sh --stop # 停止
```

启动后自动打开浏览器：

- 前端：<http://localhost:5173>
- 后端健康检查：<http://127.0.0.1:8000/api/health>
- 日志：`backend.log` / `frontend.log`

**手动启动**（可选）：

```bash
# 后端
cd backend && uv sync && uv run uvicorn app:app --host 127.0.0.1 --port 8000
# 前端（另一终端）
cd frontend && npm install && npm run dev
```

## 🧰 开发命令

```bash
make start   # 等同 ./start.sh
make stop    # 停止服务
make check   # 前端 prettier+tsc，后端 ruff+compile
make format  # 格式化前后端
```

## 📁 目录结构

```
├── start.sh            # 一键启动脚本
├── Makefile            # 启动/停止/检查/格式化
├── backend/
│   ├── app.py          # FastAPI 入口（笔记/地图/论文 API）
│   └── lab/            # 数学实验室模块 + 论文解析器
│       ├── pdf_paper.py        # PDF → 章节（含标题合并）
│       ├── paper_formulas.py   # 公式提取（CM 字体法）
│       ├── mappings.py         # 公式→实现→实验 人工映射
│       ├── paper_sections.py   # transformer 可运行实现
│       └── paper_sections_ar.py# attention-residuals 实现
└── frontend/
    └── src/pages/PaperPage.tsx # 可执行论文页面（大纲+可视化+源码+实验）
```

## ⚠️ 笔记数据位置

后端笔记库的根目录是**仓库的父目录**（`backend/app.py` 中 `NOTES_ROOT = parents[2]`），笔记按 `zh/`、`en/`、`images/` 组织。**克隆仓库后笔记库为空是正常的**——本地演示请在父目录放置笔记数据；Paper Lab 与 Math Lab 完全自包含（PDF 与实现都在仓库内），不受影响。

## 🛠 技术栈

- 后端：FastAPI / numpy / scikit-learn / pymupdf（PDF 解析）
- 前端：React 18 / TypeScript / Vite / Tailwind
- 无 LLM 参与：所有数字、公式、章节全部由确定性代码计算/解析
