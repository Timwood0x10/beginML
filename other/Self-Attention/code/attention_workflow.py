"""
多头注意力工作流程可视化
不使用sklearn，手动创建词向量，详细展示单个头的工作流程，然后展示多头拼接
"""

import numpy as np
import json
import os

# 创建输出目录
output_dir = "attention_mechanism"
os.makedirs(output_dir, exist_ok=True)

print("=" * 60)
print("🧠 多头注意力工作流程可视化")
print("=" * 60)

# ============================================
# 1. 手动创建词向量（不使用sklearn）
# ============================================

# 输入句子
sentence_words = ["The", "cat", "sits", "on", "the", "mat"]
n_words = len(sentence_words)
d_model = 8  # 模型维度

print(f"句子: {' '.join(sentence_words)}")
print(f"词数: {n_words}, 模型维度: {d_model}")

# 手工设计词向量（让每个词有明显的特征）
np.random.seed(42)

# 创建有意义的词向量
word_vectors = {
    "The": np.array([1.0, 0.0, 0.0, 0.5, 0.2, 0.1, 0.3, 0.0]),  # 定冠词特征
    "cat": np.array([0.0, 1.0, 0.0, 0.8, 0.6, 0.4, 0.0, 0.2]),  # 动物特征
    "sits": np.array([0.0, 0.0, 1.0, 0.3, 0.7, 0.5, 0.2, 0.4]),  # 动词特征
    "on": np.array([0.5, 0.0, 0.0, 0.0, 0.3, 0.2, 0.8, 0.1]),  # 介词特征
    "the": np.array(
        [1.0, 0.0, 0.0, 0.4, 0.1, 0.2, 0.2, 0.0]
    ),  # 定冠词特征（与The相似）
    "mat": np.array([0.0, 0.8, 0.0, 0.6, 0.3, 0.7, 0.0, 0.5]),  # 物体特征
}

# 构建输入矩阵 X
X = np.array([word_vectors[word] for word in sentence_words])

print(f"\n输入词向量矩阵 X 形状: {X.shape}")
print(f"X =\n{np.round(X, 3)}")

# ============================================
# 2. 多头注意力参数设置
# ============================================

num_heads = 4
d_k = d_model // num_heads

print(f"\n多头注意力配置:")
print(f"词数: {n_words}")
print(f"模型维度: {d_model}")
print(f"注意力头数: {num_heads}")
print(f"每头维度: {d_k}")

# 初始化权重矩阵（手工设计，让计算结果更有意义）
np.random.seed(42)

# 头1的权重（关注语法关系）
W_Q1 = np.array(
    [
        [1.0, 0.0],
        [0.0, 1.0],
        [0.5, 0.5],
        [0.0, 0.0],
        [0.3, 0.7],
        [0.7, 0.3],
        [0.2, 0.8],
        [0.8, 0.2],
    ]
)

W_K1 = np.array(
    [
        [0.9, 0.1],
        [0.1, 0.9],
        [0.6, 0.4],
        [0.4, 0.6],
        [0.2, 0.8],
        [0.8, 0.2],
        [0.3, 0.7],
        [0.7, 0.3],
    ]
)

W_V1 = np.array(
    [
        [1.0, 0.0],
        [0.0, 1.0],
        [0.4, 0.6],
        [0.6, 0.4],
        [0.5, 0.5],
        [0.5, 0.5],
        [0.2, 0.8],
        [0.8, 0.2],
    ]
)

# 其他头的权重（随机生成但有规律）
W_Q = [W_Q1]
W_K = [W_K1]
W_V = [W_V1]

for i in range(1, num_heads):
    W_Q.append(np.random.randn(d_model, d_k) * 0.3)
    W_K.append(np.random.randn(d_model, d_k) * 0.3)
    W_V.append(np.random.randn(d_model, d_k) * 0.3)

# 输出权重
W_O = np.random.randn(num_heads * d_k, d_model) * 0.2

# ============================================
# 3. 详细计算第一个头的工作流程
# ============================================

print(f"\n{'='*20} 头1详细工作流程 {'='*20}")

head_idx = 0

# 步骤1: 计算Q, K, V
Q1 = X @ W_Q[head_idx]
K1 = X @ W_K[head_idx]
V1 = X @ W_V[head_idx]

print(f"\n步骤1: 投影到Q,K,V空间")
print(f"Q1 = X @ W_Q[0] 形状: {Q1.shape}")
print(f"Q1 =\n{np.round(Q1, 3)}")
print(f"K1 = X @ W_K[0] 形状: {K1.shape}")
print(f"K1 =\n{np.round(K1, 3)}")
print(f"V1 = X @ W_V[0] 形状: {V1.shape}")
print(f"V1 =\n{np.round(V1, 3)}")

# 步骤2: 计算注意力分数
scores1 = Q1 @ K1.T / np.sqrt(d_k)
print(f"\n步骤2: 计算注意力分数")
print(f"Scores1 = Q1 @ K1.T / sqrt({d_k}) 形状: {scores1.shape}")
print(f"Scores1 =\n{np.round(scores1, 3)}")

# 步骤3: Softmax归一化
exp_scores1 = np.exp(scores1)
sum_exp1 = np.sum(exp_scores1, axis=1, keepdims=True)
attention1 = exp_scores1 / sum_exp1

print(f"\n步骤3: Softmax归一化")
print(f"Attention1 = softmax(Scores1) 形状: {attention1.shape}")
print(f"Attention1 =\n{np.round(attention1, 3)}")
print(f"每行和: {np.sum(attention1, axis=1)}")

# 步骤4: 计算Z矩阵（加权和）
Z1 = attention1 @ V1
print(f"\n步骤4: 计算Z矩阵")
print(f"Z1 = Attention1 @ V1 形状: {Z1.shape}")
print(f"Z1 =\n{np.round(Z1, 3)}")

# ============================================
# 4. 计算所有头
# ============================================

print(f"\n{'='*20} 计算所有头 {'='*20}")

all_heads = []
for i in range(num_heads):
    Q = X @ W_Q[i]
    K = X @ W_K[i]
    V = X @ W_V[i]
    scores = Q @ K.T / np.sqrt(d_k)
    attention = np.exp(scores) / np.exp(scores).sum(axis=1, keepdims=True)
    Z = attention @ V

    all_heads.append(
        {
            "head_idx": i,
            "Q": Q.tolist(),
            "K": K.tolist(),
            "V": V.tolist(),
            "scores": scores.tolist(),
            "attention": attention.tolist(),
            "Z": Z.tolist(),
        }
    )

    print(f"\n头{i+1}: Z矩阵形状 {Z.shape}")
    print(f"Z{i+1} =\n{np.round(Z, 3)}")

# ============================================
# 5. 拼接所有头的输出
# ============================================

print(f"\n{'='*20} 拼接多头输出 {'='*20}")

# 将所有头的Z矩阵拼接
concatenated_Z = np.concatenate([np.array(head["Z"]) for head in all_heads], axis=1)
print(f"拼接后的Z矩阵形状: {concatenated_Z.shape}")
print(f"Concatenated Z =\n{np.round(concatenated_Z, 3)}")

# 最终输出
final_output = concatenated_Z @ W_O
print(f"\n最终输出 = Concatenated Z @ W_O")
print(f"最终输出形状: {final_output.shape}")
print(f"Final Output =\n{np.round(final_output, 3)}")

# ============================================
# 6. 准备可视化数据
# ============================================

workflow_data = {
    "sentence": " ".join(sentence_words),
    "words": sentence_words,
    "word_embeddings": {
        "vocabulary": list(word_vectors.keys()),
        "embeddings": {word: vec.tolist() for word, vec in word_vectors.items()},
        "selected_words": sentence_words,
        "selected_embeddings": X.tolist(),
        "shape": X.shape,
        "description": "手工设计的词向量，每个词有不同的语义特征",
    },
    "single_head_workflow": {
        "head_idx": 0,
        "input_X": X.tolist(),
        "W_Q": W_Q[0].tolist(),
        "W_K": W_K[0].tolist(),
        "W_V": W_V[0].tolist(),
        "Q": Q1.tolist(),
        "K": K1.tolist(),
        "V": V1.tolist(),
        "scores": scores1.tolist(),
        "attention": attention1.tolist(),
        "Z": Z1.tolist(),
        "steps": [
            {
                "step": 1,
                "title": "输入词向量",
                "description": "原始词向量矩阵 X，每个词用8维向量表示",
                "matrix": X.tolist(),
                "shape": X.shape,
                "formula": "X = [word_vectors]",
                "details": "手工设计的词向量，包含语法和语义特征",
            },
            {
                "step": 2,
                "title": "投影到查询空间",
                "description": "Q = X @ W_Q，将词向量投影到查询空间",
                "matrix": Q1.tolist(),
                "shape": Q1.shape,
                "weights": W_Q[0].tolist(),
                "formula": "Q = X × W_Q",
                "details": f"形状变换: ({X.shape[0]}×{X.shape[1]}) × ({W_Q[0].shape[0]}×{W_Q[0].shape[1]}) = ({Q1.shape[0]}×{Q1.shape[1]})",
            },
            {
                "step": 3,
                "title": "投影到键空间",
                "description": "K = X @ W_K，将词向量投影到键空间",
                "matrix": K1.tolist(),
                "shape": K1.shape,
                "weights": W_K[0].tolist(),
                "formula": "K = X × W_K",
                "details": f"形状变换: ({X.shape[0]}×{X.shape[1]}) × ({W_K[0].shape[0]}×{W_K[0].shape[1]}) = ({K1.shape[0]}×{K1.shape[1]})",
            },
            {
                "step": 4,
                "title": "投影到值空间",
                "description": "V = X @ W_V，将词向量投影到值空间",
                "matrix": V1.tolist(),
                "shape": V1.shape,
                "weights": W_V[0].tolist(),
                "formula": "V = X × W_V",
                "details": f"形状变换: ({X.shape[0]}×{X.shape[1]}) × ({W_V[0].shape[0]}×{W_V[0].shape[1]}) = ({V1.shape[0]}×{V1.shape[1]})",
            },
            {
                "step": 5,
                "title": "计算注意力分数",
                "description": "Scores = Q @ K.T / sqrt(d_k)，计算词与词之间的关联分数",
                "matrix": scores1.tolist(),
                "shape": scores1.shape,
                "formula": f"Scores = Q × K.T / √{d_k}",
                "details": f"形状变换: ({Q1.shape[0]}×{Q1.shape[1]}) × ({K1.shape[0]}×{K1.shape[1]}).T = ({scores1.shape[0]}×{scores1.shape[1]})",
            },
            {
                "step": 6,
                "title": "Softmax归一化",
                "description": "Attention = softmax(Scores)，将分数转换为概率分布",
                "matrix": attention1.tolist(),
                "shape": attention1.shape,
                "formula": "Attention = softmax(Scores)",
                "details": f"每行经过softmax后和为1，形状: {attention1.shape}",
            },
            {
                "step": 7,
                "title": "计算Z矩阵",
                "description": "Z = Attention @ V，用注意力权重对V进行加权求和",
                "matrix": Z1.tolist(),
                "shape": Z1.shape,
                "formula": "Z = Attention × V",
                "details": f"形状变换: ({attention1.shape[0]}×{attention1.shape[1]}) × ({V1.shape[0]}×{V1.shape[1]}) = ({Z1.shape[0]}×{Z1.shape[1]})",
            },
        ],
    },
    "all_heads": all_heads,
    "concatenation": {
        "individual_Z": [head["Z"] for head in all_heads],
        "concatenated_Z": concatenated_Z.tolist(),
        "W_O": W_O.tolist(),
        "final_output": final_output.tolist(),
        "formula": "Final = Concat(Z₁, Z₂, Z₃, Z₄) × W_O",
        "details": f"拼接形状: ({n_words}×{d_k}) × {num_heads} = ({n_words}×{num_heads*d_k})",
    },
    "parameters": {
        "n_words": n_words,
        "d_model": d_model,
        "num_heads": num_heads,
        "d_k": d_k,
    },
}

# ============================================
# 7. 生成HTML可视化页面（内嵌JSON数据）
# ============================================

html_content = f"""
<!DOCTYPE html>
<html lang="zh">
<head>
    <meta charset="utf-8">
    <title>多头注意力工作流程可视化</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}
        
        .container {{
            max-width: 1600px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }}
        
        header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }}
        
        h1 {{
            font-size: 36px;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        }}
        
        .subtitle {{
            font-size: 18px;
            opacity: 0.9;
        }}
        
        .main-content {{
            padding: 30px;
        }}
        
        .workflow-section {{
            margin-bottom: 40px;
            background: white;
            border-radius: 10px;
            padding: 25px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        
        .section-title {{
            font-size: 24px;
            color: #495057;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid #667eea;
        }}
        
        .step-container {{
            margin: 30px 0;
        }}
        
        .step-header {{
            display: flex;
            align-items: center;
            margin-bottom: 15px;
        }}
        
        .step-number {{
            width: 40px;
            height: 40px;
            background: #667eea;
            color: white;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            margin-right: 15px;
        }}
        
        .step-title {{
            font-size: 20px;
            font-weight: bold;
            color: #495057;
        }}
        
        .step-description {{
            color: #6c757d;
            margin-bottom: 15px;
            font-style: italic;
        }}
        
        .step-details {{
            background: #e3f2fd;
            border-left: 4px solid #2196f3;
            padding: 15px;
            margin: 15px 0;
            border-radius: 5px;
            font-size: 14px;
        }}
        
        .matrix-container {{
            background: #f8f9fa;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
        }}
        
        .matrix-info {{
            display: flex;
            justify-content: space-between;
            margin-bottom: 10px;
            font-size: 14px;
            color: #6c757d;
        }}
        
        .formula {{
            background: #fff3cd;
            border: 1px solid #ffeaa7;
            border-radius: 5px;
            padding: 15px;
            margin: 15px 0;
            font-family: monospace;
            text-align: center;
            font-size: 16px;
        }}
        
        .plot-container {{
            width: 100%;
            height: 300px;
            margin: 15px 0;
        }}
        
        .heads-grid {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 20px;
            margin: 30px 0;
        }}
        
        .head-box {{
            background: #f8f9fa;
            border-radius: 8px;
            padding: 20px;
            border-left: 4px solid #667eea;
        }}
        
        .head-title {{
            font-size: 18px;
            font-weight: bold;
            margin-bottom: 15px;
            color: #495057;
        }}
        
        .concatenation-visual {{
            background: #e3f2fd;
            border-radius: 8px;
            padding: 20px;
            margin: 20px 0;
        }}
        
        .controls {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 30px;
            text-align: center;
        }}
        
        button {{
            padding: 10px 20px;
            margin: 5px;
            background: #667eea;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 14px;
            transition: background 0.3s;
        }}
        
        button:hover {{
            background: #5a6fd8;
        }}
        
        button.active {{
            background: #764ba2;
        }}
        
        .hidden {{
            display: none;
        }}
        
        .arrow {{
            text-align: center;
            font-size: 24px;
            color: #667eea;
            margin: 10px 0;
        }}
        
        .data-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
            font-size: 12px;
        }}
        
        .data-table th, .data-table td {{
            border: 1px solid #dee2e6;
            padding: 8px;
            text-align: center;
        }}
        
        .data-table th {{
            background: #f8f9fa;
            font-weight: bold;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🧠 多头注意力工作流程</h1>
            <p class="subtitle">从词向量到多头注意力的完整技术流程</p>
        </header>

        <div class="main-content">
            <!-- 词向量嵌入 -->
            <div class="workflow-section">
                <h2 class="section-title">📝 词向量嵌入</h2>
                <div class="matrix-container">
                    <div class="matrix-info">
                        <span>手工设计的词向量</span>
                        <span>句子: "{' '.join(sentence_words)}"</span>
                    </div>
                    <div id="word-embeddings-plot" class="plot-container"></div>
                    <div class="step-details">
                        <strong>词向量设计说明：</strong><br>
                        • The/the: 定冠词特征 [1.0, 0.0, ...]<br>
                        • cat: 动物特征 [0.0, 1.0, ...]<br>
                        • sits: 动词特征 [0.0, 0.0, 1.0, ...]<br>
                        • on: 介词特征 [0.5, 0.0, ...]<br>
                        • mat: 物体特征 [0.0, 0.8, ...]
                    </div>
                </div>
            </div>

            <!-- 单个头详细工作流程 -->
            <div class="workflow-section">
                <h2 class="section-title">🔍 单个注意力头详细工作流程</h2>
                
                <div class="controls">
                    <button class="step-btn active" onclick="showStep(0)">步骤1: 输入</button>
                    <button class="step-btn" onclick="showStep(1)">步骤2: Q矩阵</button>
                    <button class="step-btn" onclick="showStep(2)">步骤3: K矩阵</button>
                    <button class="step-btn" onclick="showStep(3)">步骤4: V矩阵</button>
                    <button class="step-btn" onclick="showStep(4)">步骤5: 注意力分数</button>
                    <button class="step-btn" onclick="showStep(5)">步骤6: 注意力权重</button>
                    <button class="step-btn" onclick="showStep(6)">步骤7: Z矩阵</button>
                </div>

                <div id="steps-container">
                    <!-- 动态生成步骤内容 -->
                </div>
            </div>

            <!-- 多头并行工作 -->
            <div class="workflow-section">
                <h2 class="section-title">🔄 多头并行工作</h2>
                <div class="heads-grid" id="heads-grid">
                    <!-- 动态生成多头内容 -->
                </div>
            </div>

            <!-- 拼接与最终输出 -->
            <div class="workflow-section">
                <h2 class="section-title">🔗 拼接与最终输出</h2>
                
                <div class="concatenation-visual">
                    <h3>拼接过程</h3>
                    <div class="formula">
                        Concatenated = Concat(Z₁, Z₂, Z₃, Z₄)<br>
                        Final Output = Concatenated × W_O
                    </div>
                    <div id="concatenation-plot" class="plot-container"></div>
                    <div class="arrow">↓</div>
                    <div id="final-output-plot" class="plot-container"></div>
                </div>
            </div>
        </div>
    </div>

    <script>
        // 内嵌工作流数据
        const workflowData = {json.dumps(workflow_data, ensure_ascii=False)};
        let currentStep = 0;
        
        // 直接初始化
        initializeWorkflow();

        function initializeWorkflow() {{
            // 渲染词向量
            plotWordEmbeddings();
            
            // 渲染步骤
            renderSteps();
            showStep(0);
            
            // 渲染多头
            renderAllHeads();
            
            // 渲染拼接
            plotConcatenation();
            plotFinalOutput();
        }}

        function plotWordEmbeddings() {{
            const embeddings = workflowData.word_embeddings;
            const trace = {{
                z: embeddings.selected_embeddings,
                x: embeddings.selected_words,
                y: Array.from({{length: embeddings.shape[1]}}, (_, i) => `dim${{i}}`),
                type: 'heatmap',
                colorscale: 'Viridis'
            }};

            const layout = {{
                title: '输入词向量矩阵 X',
                margin: {{t: 30, b: 40, l: 60, r: 20}},
                xaxis: {{title: '词'}},
                yaxis: {{title: '维度'}}
            }};

            Plotly.newPlot('word-embeddings-plot', [trace], layout, {{displayModeBar: false}});
        }}

        function renderSteps() {{
            const container = document.getElementById('steps-container');
            const steps = workflowData.single_head_workflow.steps;
            
            steps.forEach((step, index) => {{
                const stepDiv = document.createElement('div');
                stepDiv.className = 'step-container';
                stepDiv.id = `step-${{index}}`;
                
                stepDiv.innerHTML = `
                    <div class="step-header">
                        <div class="step-number">${{step.step}}</div>
                        <div class="step-title">${{step.title}}</div>
                    </div>
                    <div class="step-description">${{step.description}}</div>
                    <div class="formula">${{step.formula}}</div>
                    <div class="step-details">${{step.details}}</div>
                    <div class="matrix-container">
                        <div class="matrix-info">
                            <span>矩阵形状: ${{step.shape[0]}} × ${{step.shape[1]}}</span>
                        </div>
                        <div id="step-plot-${{index}}" class="plot-container"></div>
                        <div id="step-table-${{index}}"></div>
                    </div>
                `;
                
                container.appendChild(stepDiv);
            }});
        }}

        function showStep(stepIndex) {{
            // 更新按钮状态
            document.querySelectorAll('.step-btn').forEach((btn, index) => {{
                btn.classList.toggle('active', index === stepIndex);
            }});
            
            // 隐藏所有步骤
            document.querySelectorAll('.step-container').forEach(step => {{
                step.classList.add('hidden');
            }});
            
            // 显示当前步骤
            document.getElementById(`step-${{stepIndex}}`).classList.remove('hidden');
            
            // 绘制当前步骤的矩阵
            plotStepMatrix(stepIndex);
        }}

        function plotStepMatrix(stepIndex) {{
            const step = workflowData.single_head_workflow.steps[stepIndex];
            const words = workflowData.words;
            
            let trace;
            let layout;
            
            if (stepIndex === 0) {{
                // 输入矩阵
                trace = {{
                    z: step.matrix,
                    x: words,
                    y: Array.from({{length: step.shape[1]}}, (_, i) => `d${{i}}`),
                    type: 'heatmap',
                    colorscale: 'Viridis'
                }};
                layout = {{
                    title: '输入词向量矩阵',
                    margin: {{t: 30, b: 40, l: 60, r: 20}},
                    height: 250
                }};
            }} else if (stepIndex <= 3) {{
                // Q, K, V矩阵
                const labels = ['q', 'k', 'v'][stepIndex - 1];
                const colors = ['Reds', 'Blues', 'Greens'][stepIndex - 1];
                trace = {{
                    z: step.matrix,
                    x: Array.from({{length: step.shape[1]}}, (_, i) => `${{labels}}${{i}}`),
                    y: words,
                    type: 'heatmap',
                    colorscale: colors
                }};
                layout = {{
                    title: `${{labels.toUpperCase()}}矩阵`,
                    margin: {{t: 30, b: 40, l: 50, r: 20}},
                    height: 250
                }};
            }} else if (stepIndex === 4 || stepIndex === 5) {{
                // 注意力分数或权重
                trace = {{
                    z: step.matrix,
                    x: words,
                    y: words,
                    type: 'heatmap',
                    colorscale: 'Reds'
                }};
                layout = {{
                    title: stepIndex === 4 ? '注意力分数矩阵' : '注意力权重矩阵',
                    margin: {{t: 30, b: 40, l: 50, r: 20}},
                    height: 250
                }};
            }} else {{
                // Z矩阵
                trace = {{
                    z: step.matrix,
                    x: words,
                    y: Array.from({{length: step.shape[1]}}, (_, i) => `z${{i}}`),
                    type: 'heatmap',
                    colorscale: 'Purples'
                }};
                layout = {{
                    title: 'Z矩阵（输出）',
                    margin: {{t: 30, b: 40, l: 50, r: 20}},
                    height: 250
                }};
            }}
            
            Plotly.newPlot(`step-plot-${{stepIndex}}`, [trace], layout, {{displayModeBar: false}});
            
            // 添加数据表格
            addDataTable(stepIndex, step);
        }}

        function addDataTable(stepIndex, step) {{
            const tableDiv = document.getElementById(`step-table-${{stepIndex}}`);
            const words = workflowData.words;
            
            let tableHtml = '<table class="data-table"><thead><tr><th>词/维度</th>';
            
            // 表头
            if (stepIndex === 0) {{
                for (let j = 0; j < step.shape[1]; j++) {{
                    tableHtml += `<th>d${{j}}</th>`;
                }}
            }} else if (stepIndex <= 3) {{
                const labels = ['q', 'k', 'v'][stepIndex - 1];
                for (let j = 0; j < step.shape[1]; j++) {{
                    tableHtml += `<th>${{labels}}${{j}}</th>`;
                }}
            }} else if (stepIndex <= 5) {{
                for (let word of words) {{
                    tableHtml += `<th>${{word}}</th>`;
                }}
            }} else {{
                for (let j = 0; j < step.shape[1]; j++) {{
                    tableHtml += `<th>z${{j}}</th>`;
                }}
            }}
            
            tableHtml += '</tr></thead><tbody>';
            
            // 表格数据
            if (stepIndex <= 3 || stepIndex === 6) {{
                for (let i = 0; i < step.matrix.length; i++) {{
                    tableHtml += `<tr><td><strong>${{words[i]}}</strong></td>`;
                    for (let j = 0; j < step.matrix[i].length; j++) {{
                        tableHtml += `<td>${{step.matrix[i][j].toFixed(3)}}</td>`;
                    }}
                    tableHtml += '</tr>';
                }}
            }} else {{
                for (let i = 0; i < step.matrix.length; i++) {{
                    tableHtml += `<tr><td><strong>${{words[i]}}</strong></td>`;
                    for (let j = 0; j < step.matrix[i].length; j++) {{
                        tableHtml += `<td>${{step.matrix[i][j].toFixed(3)}}</td>`;
                    }}
                    tableHtml += '</tr>';
                }}
            }}
            
            tableHtml += '</tbody></table>';
            tableDiv.innerHTML = tableHtml;
        }}

        function renderAllHeads() {{
            const grid = document.getElementById('heads-grid');
            const heads = workflowData.all_heads;
            const words = workflowData.words;
            
            heads.forEach((head, index) => {{
                const headDiv = document.createElement('div');
                headDiv.className = 'head-box';
                
                headDiv.innerHTML = `
                    <div class="head-title">头 ${{index + 1}} 的Z矩阵</div>
                    <div id="head-${{index}}-plot" class="plot-container"></div>
                `;
                
                grid.appendChild(headDiv);
                
                // 绘制Z矩阵
                const trace = {{
                    z: head.Z,
                    x: words,
                    y: Array.from({{length: head.Z[0].length}}, (_, i) => `z${{i}}`),
                    type: 'heatmap',
                    colorscale: ['Reds', 'Blues', 'Greens', 'Purples'][index]
                }};
                
                const layout = {{
                    title: `头${{index + 1}} Z矩阵`,
                    margin: {{t: 30, b: 40, l: 50, r: 20}},
                    height: 200
                }};
                
                Plotly.newPlot(`head-${{index}}-plot`, [trace], layout, {{displayModeBar: false}});
            }});
        }}

        function plotConcatenation() {{
            const concat = workflowData.concatenation;
            const words = workflowData.words;
            
            const trace = {{
                z: concat.concatenated_Z,
                x: words,
                y: Array.from({{length: concat.concatenated_Z[0].length}}, (_, i) => `c${{i}}`),
                type: 'heatmap',
                colorscale: 'Blues'
            }};
            
            const layout = {{
                title: '拼接后的Z矩阵',
                margin: {{t: 30, b: 40, l: 50, r: 20}},
                height: 250
            }};
            
            Plotly.newPlot('concatenation-plot', [trace], layout, {{displayModeBar: false}});
        }}

        function plotFinalOutput() {{
            const final = workflowData.concatenation;
            const words = workflowData.words;
            
            const trace = {{
                z: final.final_output,
                x: words,
                y: Array.from({{length: final.final_output[0].length}}, (_, i) => `f${{i}}`),
                type: 'heatmap',
                colorscale: 'Viridis'
            }};
            
            const layout = {{
                title: '最终输出',
                margin: {{t: 30, b: 40, l: 50, r: 20}},
                height: 250
            }};
            
            Plotly.newPlot('final-output-plot', [trace], layout, {{displayModeBar: false}});
        }}
    </script>
</body>
</html>
"""

# 保存HTML文件
with open(
    os.path.join(output_dir, "attention_workflow.html"), "w", encoding="utf-8"
) as f:
    f.write(html_content)

print(f"✅ 工作流可视化页面已保存到: {output_dir}/attention_workflow.html")
print("\n" + "=" * 60)
print("🎯 多头注意力工作流程可视化完成！")
print("=" * 60)
print(f"\n📁 生成的文件:")
print(f"  • attention_workflow.py - 计算脚本")
print(f"  • {output_dir}/attention_workflow.html - 可视化页面（内嵌数据）")
print(f"\n💡 打开HTML文件查看完整的工作流程！")
print(f"\n🔧 特点：")
print(f"  • 不使用sklearn，手工设计词向量")
print(f"  • 详细展示单个头的7个计算步骤")
print(f"  • 展示4个头的并行工作")
print(f"  • 清晰的拼接过程和最终输出")
print(f"  • 数据内嵌，无CORS问题")
