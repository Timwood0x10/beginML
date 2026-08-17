# AIScope — AI Learning Platform

An interactive platform for AI learning: notes library, knowledge map, **Executable Paper Lab** (Paper Lab), and Math Lab. The core feature is "Paper → Mechanism → Source Code → Experiment" executable papers: each paper is parsed into real chapter outlines, click a chapter to see implementation, modify parameters to run experiments, and observe results firsthand.

## ✨ Highlights: Executable Paper Lab

| Capability | Description |
|---|---|
| **Real Outlines** | PDF parsing extracts complete paper chapters (numbered + titles + pages), no LLM guessing |
| **Formula ↔ Code Linking** | Click formulas/heatmaps/data flow steps, right-side source code auto-highlights and scrolls |
| **Mechanism Navigation** | Each paper selects core mechanisms (attention, RoPE, residual...), one-click entry |
| **Runnable Implementations** | Each chapter comes with independent runnable numpy implementations + parameter experiments + observations/evidence |

Currently includes two papers: `transformer.pdf` (16 sections / 231 formulas), `attention_residuals.pdf` (7 sections).

## 🚀 Quick Start

**Dependencies**: `uv` (Python package manager), `node`/`npm`, `lsof` (optional, for port checking).

```bash
./start.sh        # One-click start: install dependencies + start backend(:8000) + frontend(:5173)
./start.sh --stop # Stop services
```

After starting, browser opens automatically:

- Frontend: <http://localhost:5173>
- Backend health check: <http://127.0.0.1:8000/api/health>
- Logs: `backend.log` / `frontend.log`

**Manual Start** (optional):

```bash
# Backend
cd backend && uv sync && uv run uvicorn app:app --host 127.0.0.1 --port 8000
# Frontend (another terminal)
cd frontend && npm install && npm run dev
```

## 🧰 Development Commands

```bash
make start   # Same as ./start.sh
make stop    # Stop services
make check   # Frontend prettier+tsc, backend ruff+compile
make format  # Format both frontend and backend
```

## 📁 Project Structure

```
├── start.sh            # One-click launcher
├── Makefile            # Start/stop/check/format
├── backend/
│   ├── app.py          # FastAPI entry (notes/map/paper API)
│   └── lab/            # Math lab modules + paper parser
│       ├── pdf_paper.py        # PDF → chapters (with title merging)
│       ├── paper_formulas.py   # Formula extraction (CM font method)
│       ├── mappings.py         # Manual mappings: formula→implementation→experiment
│       ├── paper_sections.py   # Transformer runnable implementation
│       └── paper_sections_ar.py# Attention-residuals implementation
└── frontend/
    └── src/pages/PaperPage.tsx # Executable paper page (outline+visualization+source+experiments)
```

## ⚠️ Notes Data Location

The backend notes library root directory is the **parent directory of the repository** (`backend/app.py` uses `NOTES_ROOT = parents[2]`), with notes organized by `zh/`, `en/`, `images/`. **After cloning the repo, the notes library will be empty** — for local demo, place note data in the parent directory. Paper Lab and Math Lab are fully self-contained (PDFs and implementations are within the repo) and are not affected.

## 🛠 Tech Stack

- Backend: FastAPI / numpy / scikit-learn / pymupdf (PDF parsing)
- Frontend: React 18 / TypeScript / Vite / Tailwind
- No LLM involvement: All numbers, formulas, and chapters are computed/parsed by deterministic code

## 🤖 Development Note

This project was developed with assistance from [mimo 2.5](https://github.com/mimo). The UI was independently designed by the author.

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
