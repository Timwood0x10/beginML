# AIScope — AI Interactive Learning Platform

> **Papers too hard to read? Formulas too abstract? Models too complex to run?**
> AIScope turns AI knowledge into interactive labs: experiment hands-on, observe results, understand deeply.

## What is this?

A platform that makes AI concepts "come alive." No more static documentation — instead:

```
Theory → Interactive experiments → Parameter tuning → Real-time visualization → Deep understanding
```

## Core Features

### 1. 25+ Interactive Labs

| Category | Labs | What You Learn |
|---|---|---|
| **Classic Math** | Gradient Descent, Activations, Losses, Convolution, Matrix Transforms, Distributions, Entropy, Neural Network, PCA, Regularization, SVM | ML fundamentals |
| **Model Behavior** | Sampling Machine, Rotary Observatory, Token Society, Transformer Detective | How models think |
| **Learning Dynamics** | Dangerous Mountain, Shooting Range, Representation River, Transformer MRI | Training process secrets |
| **Model Efficiency** | Weight Freezer, Mamba Memory Race, Expert Routing Room, Feature Hunt | Compression & acceleration |
| **Paper Lab** | Transformer, Attention Residuals | Paper → Code → Experiments |
| **Agent Lab** | ARES Agent Lab | Build, break, evolve agents |

### 2. Executable Papers (Paper Lab)

| Traditional Reading | AIScope |
|---|---|
| Read PDF, see formulas, guess meanings | Click formula, right-side code auto-highlights |
| Read code, don't know which paper section | Click chapter, jump directly to implementation |
| Want to run experiments? Build from scratch | Change a parameter, see results immediately |
| Read and forget | Hands-on experiments, deep understanding |

**Built-in Papers**:
- `transformer.pdf`: 16 sections / 231 formulas / Full runnable implementations
- `attention_residuals.pdf`: 7 sections / Residual connection mechanism explained

### 3. Formula ↔ Source Code Bidirectional Linking

```
Click attention formula → Right-side code auto-scrolls to softmax implementation
Modify num_heads parameter → Heatmap updates in real-time
Click causal mask → See mask matrix visualization
```

### 4. Mechanism Navigation

Each paper selects core mechanisms, one-click entry:
- Multi-Head Attention
- Rotary Position Embedding (RoPE)
- Residual Connection
- Layer Normalization
- ...

### 5. Experiment Challenge System

Every lab has challenge tasks:
- **Predict → Run → Verify**: Guess first, then see evidence
- **Manipulation challenges**: Achieve specific goals (e.g., "make distribution more diverse")
- **Discovery chain**: Complete one experiment to unlock the next

### 6. Knowledge Map

TF-IDF based semantic similarity note map to discover related topics.

### 7. ARES Agent Lab

Build agents like LEGO:
- Drag cognitive bricks
- Snap semantic ports
- Expand skill boxes
- Attach recovery bricks
- RUN to watch thinking → BREAK to destroy → watch self-healing

## Quick Start

```bash
# One-click start
./start.sh

# Open browser
# Frontend: http://localhost:5173
# Backend: http://127.0.0.1:8000/api/health
```

**Dependencies**: `uv` (Python), `node`/`npm`

## Technical Highlights

- **No LLM**: All numbers, formulas, chapters computed/parsed by deterministic code
- **PDF Parsing**: Real chapter extraction, no LLM guessing
- **Formula Extraction**: Precise identification based on font features
- **Real-time Computation**: numpy implementations, change parameters → immediate results
- **Bilingual**: Chinese and English interfaces

## Project Structure

```
ai-learning-platform/
├── backend/
│   ├── app.py              # FastAPI entry
│   └── lab/
│       ├── pdf_paper.py    # PDF → Chapter parsing
│       ├── paper_formulas.py # Formula extraction
│       ├── paper_sections.py # Paper implementations
│       ├── attention.py    # Attention experiments
│       ├── transformer.py  # Transformer experiments
│       ├── moe.py          # MoE experiments
│       ├── mamba.py        # Mamba experiments
│       └── ...             # More experiment modules
├── frontend/
│   └── src/
│       ├── pages/
│       │   └── PaperPage.tsx # Executable paper page
│       └── lab/
│           └── modules/      # 25+ lab components
├── start.sh                # One-click start
└── README.md
```

## Who is this for?

- **AI Learners**: Papers too painful? Try interactive reading
- **Researchers**: Want to quickly validate paper ideas? Tweak parameters and run experiments
- **Developers**: Want to understand paper implementations? See runnable code directly
- **Educators**: Want students to experiment? This is a ready-made lab environment
- **Agent Developers**: Want to build agents? Just drag and drop bricks

## Development Note

This project was developed with assistance from mimo 2.5. The UI was independently designed by the author.

## License

Apache License 2.0

---

**中文**: [Chinese Version](README.md)
