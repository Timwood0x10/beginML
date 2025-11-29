"""
多头注意力完整工作流程可视化
展示详细的计算步骤和多头并行工作
"""

import numpy as np
import json
import os

# 创建输出目录
output_dir = 'attention_mechanism'
os.makedirs(output_dir, exist_ok=True)

print("=" * 60)
print("🧠 多头注意力完整工作流程可视化")
print("=" * 60)

# ============================================
# 1. 创建词向量
# ============================================

sentence_words = ["The", "cat", "sits", "on", "the", "mat"]
n_words = len(sentence_words)
d_model = 6
num_heads = 3
d_k = d_model // num_heads

print(f"句子: {' '.join(sentence_words)}")
print(f"配置: {n_words}个词, {d_model}维, {num_heads}个头, 每头{d_k}维")

# 手工设计词向量
X = np.array([
    [1.0, 0.0, 0.0, 0.5, 0.2, 0.1],  # The
    [0.0, 1.0, 0.0, 0.8, 0.6, 0.4],  # cat
    [0.0, 0.0, 1.0, 0.3, 0.7, 0.5],  # sits
    [0.5, 0.0, 0.0, 0.0, 0.3, 0.2],  # on
    [1.0, 0.0, 0.0, 0.4, 0.1, 0.2],  # the
    [0.0, 0.8, 0.0, 0.6, 0.3, 0.7]   # mat
])

print(f"\n输入矩阵 X 形状: {X.shape}")
print(f"X =\n{np.round(X, 2)}")

# ============================================
# 2. 计算多头注意力
# ============================================

np.random.seed(42)
heads_data = []

for head_idx in range(num_heads):
    # 权重矩阵
    W_Q = np.random.randn(d_model, d_k) * 0.1
    W_K = np.random.randn(d_model, d_k) * 0.1
    W_V = np.random.randn(d_model, d_k) * 0.1
    
    # 计算Q, K, V
    Q = X @ W_Q
    K = X @ W_K
    V = X @ W_V
    
    # 计算注意力分数
    scores = Q @ K.T / np.sqrt(d_k)
    
    # Softmax归一化
    scores_stable = scores - np.max(scores, axis=1, keepdims=True)
    exp_scores = np.exp(scores_stable)
    attention = exp_scores / exp_scores.sum(axis=1, keepdims=True)
    
    # 计算Z矩阵
    Z = attention @ V
    
    # 保存详细计算步骤
    head_data = {
        'head_idx': head_idx,
        'W_Q': W_Q.tolist(),
        'W_K': W_K.tolist(),
        'W_V': W_V.tolist(),
        'Q': Q.tolist(),
        'K': K.tolist(),
        'V': V.tolist(),
        'scores': scores.tolist(),
        'attention': attention.tolist(),
        'Z': Z.tolist(),
        'calculations': {
            'Q_formula': f"Q = X @ W_Q[{head_idx}]",
            'K_formula': f"K = X @ W_K[{head_idx}]", 
            'V_formula': f"V = X @ W_V[{head_idx}]",
            'scores_formula': f"Scores = Q @ K.T / sqrt({d_k})",
            'attention_formula': "Attention = softmax(Scores)",
            'Z_formula': "Z = Attention @ V"
        }
    }
    
    heads_data.append(head_data)

# 拼接所有头的输出
concatenated = np.concatenate([np.array(head['Z']) for head in heads_data], axis=1)
W_O = np.random.randn(num_heads * d_k, d_model) * 0.1
final_output = concatenated @ W_O

print(f"\n计算完成!")
print(f"拼接后形状: {concatenated.shape}")
print(f"最终输出形状: {final_output.shape}")

# ============================================
# 3. 生成详细HTML页面
# ============================================

html_content = f"""<!DOCTYPE html>
<html lang="zh">
<head>
    <meta charset="utf-8">
    <title>多头注意力完整工作流程</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f5f5f5;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            text-align: center;
            color: #333;
            margin-bottom: 30px;
        }}
        .section {{
            margin-bottom: 30px;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 8px;
        }}
        .section h2 {{
            color: #495057;
            margin-bottom: 20px;
            border-bottom: 2px solid #007bff;
            padding-bottom: 10px;
        }}
        .workflow-nav {{
            display: flex;
            justify-content: center;
            margin-bottom: 20px;
            gap: 10px;
        }}
        .nav-btn {{
            padding: 10px 20px;
            background: #007bff;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            transition: background 0.3s;
        }}
        .nav-btn:hover {{
            background: #0056b3;
        }}
        .nav-btn.active {{
            background: #28a745;
        }}
        .step-content {{
            display: none;
        }}
        .step-content.active {{
            display: block;
        }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }}
        .plot {{
            height: 300px;
        }}
        .formula {{
            background: #fff3cd;
            border: 1px solid #ffeaa7;
            border-radius: 5px;
            padding: 15px;
            margin: 10px 0;
            text-align: center;
            font-family: monospace;
        }}
        .calculation {{
            background: #e3f2fd;
            border-left: 4px solid #2196f3;
            padding: 15px;
            margin: 10px 0;
            border-radius: 5px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 10px 0;
            font-size: 12px;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 8px;
            text-align: center;
        }}
        th {{
            background: #f2f2f2;
            font-weight: bold;
        }}
        .head-selector {{
            display: flex;
            justify-content: center;
            gap: 10px;
            margin: 20px 0;
        }}
        .head-btn {{
            padding: 8px 16px;
            background: #6c757d;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
        }}
        .head-btn.active {{
            background: #007bff;
        }}
        .heads-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
        }}
        .head-box {{
            background: white;
            border: 1px solid #dee2e6;
            border-radius: 8px;
            padding: 15px;
            text-align: center;
        }}
        .head-title {{
            font-weight: bold;
            margin-bottom: 10px;
            color: #495057;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🧠 多头注意力完整工作流程</h1>
        
        <div class="section">
            <h2>📝 输入词向量</h2>
            <div id="input-plot" class="plot"></div>
            <table id="input-table"></table>
        </div>

        <div class="section">
            <h2>🔍 单个注意力头详细流程</h2>
            <div class="head-selector">
                <button class="head-btn active" onclick="selectHead(0)">头 1</button>
                <button class="head-btn" onclick="selectHead(1)">头 2</button>
                <button class="head-btn" onclick="selectHead(2)">头 3</button>
            </div>
            
            <div class="workflow-nav">
                <button class="nav-btn active" onclick="showStep('input')">输入</button>
                <button class="nav-btn" onclick="showStep('q')">Q矩阵</button>
                <button class="nav-btn" onclick="showStep('k')">K矩阵</button>
                <button class="nav-btn" onclick="showStep('v')">V矩阵</button>
                <button class="nav-btn" onclick="showStep('scores')">注意力分数</button>
                <button class="nav-btn" onclick="showStep('attention')">注意力权重</button>
                <button class="nav-btn" onclick="showStep('z')">Z矩阵</button>
            </div>

            <div id="step-input" class="step-content active">
                <h3>输入词向量矩阵 X</h3>
                <div class="calculation">
                    <p><strong>说明：</strong>原始词向量，每个词用6维向量表示</p>
                    <p><strong>形状：</strong>6个词 × 6个维度</p>
                </div>
                <div id="input-detailed-plot" class="plot"></div>
            </div>

            <div id="step-q" class="step-content">
                <h3>查询矩阵 Q</h3>
                <div class="formula" id="q-formula"></div>
                <div class="calculation">
                    <p><strong>计算说明：</strong>将输入向量投影到查询空间</p>
                    <p><strong>形状变换：</strong>6×6 × 6×2 = 6×2 (n_words, d_model) @ (d_model, d_k) -> (n_words, d_k)·</p>
                </div>
                <div id="q-plot" class="plot"></div>
                <table id="q-table"></table>
            </div>

            <div id="step-k" class="step-content">
                <h3>键矩阵 K</h3>
                <div class="formula" id="k-formula"></div>
                <div class="calculation">
                    <p><strong>计算说明：</strong>将输入向量投影到键空间</p>
                    <p><strong>形状变换：</strong>6×6 × 6×2 = 6×2</p>
                </div>
                <div id="k-plot" class="plot"></div>
                <table id="k-table"></table>
            </div>

            <div id="step-v" class="step-content">
                <h3>值矩阵 V</h3>
                <div class="formula" id="v-formula"></div>
                <div class="calculation">
                    <p><strong>计算说明：</strong>将输入向量投影到值空间</p>
                    <p><strong>形状变换：</strong>6×6 × 6×2 = 6×2</p>
                </div>
                <div id="v-plot" class="plot"></div>
                <table id="v-table"></table>
            </div>

            <div id="step-scores" class="step-content">
                <h3>注意力分数</h3>
                <div class="formula" id="scores-formula"></div>
                <div class="calculation">
                    <p><strong>计算说明：</strong>计算查询和键的相似度</p>
                    <p><strong>形状变换：</strong>6×2 × 2×6 = 6×6</p>
                    <p><strong>缩放因子：</strong>√{d_k} = {np.sqrt(d_k):.3f}</p>
                </div>
                <div id="scores-plot" class="plot"></div>
                <table id="scores-table"></table>
            </div>

            <div id="step-attention" class="step-content">
                <h3>注意力权重</h3>
                <div class="formula">Attention = softmax(Scores)</div>
                <div class="calculation">
                    <p><strong>计算说明：</strong>将分数转换为概率分布</p>
                    <p><strong>特性：</strong>每行和为1，表示注意力分配</p>
                </div>
                <div id="attention-plot" class="plot"></div>
                <table id="attention-table"></table>
            </div>

            <div id="step-z" class="step-content">
                <h3>Z矩阵（输出）</h3>
                <div class="formula" id="z-formula"></div>
                <div class="calculation">
                    <p><strong>计算说明：</strong>用注意力权重对值矩阵加权求和</p>
                    <p><strong>形状变换：</strong>6×6 × 6×2 = 6×2</p>
                </div>
                <div id="z-plot" class="plot"></div>
                <table id="z-table"></table>
            </div>
        </div>

        <div class="section">
            <h2>🔄 多头并行工作</h2>
            <div class="heads-grid" id="heads-grid">
                <!-- 动态生成 -->
            </div>
        </div>

        <div class="section">
            <h2>🔗 拼接与最终输出</h2>
            <div class="grid">
                <div>
                    <h3>拼接输出</h3>
                    <div class="formula">Concatenated = Concat(Z₁, Z₂, Z₃)</div>
                    <div id="concat-plot" class="plot"></div>
                </div>
                <div>
                    <h3>最终输出</h3>
                    <div class="formula">Final Output = Concatenated × W_O</div>
                    <div id="final-plot" class="plot"></div>
                </div>
            </div>
        </div>
    </div>

    <script>
        // 数据
        const words = {json.dumps(sentence_words)};
        const X = {X.tolist()};
        const headsData = {json.dumps(heads_data)};
        const concatenated = {concatenated.tolist()};
        const finalOutput = {final_output.tolist()};
        
        let currentHead = 0;
        let currentStep = 'input';

        // 初始化
        function init() {{
            plotInput();
            selectHead(0);
            showStep('input');
            renderAllHeads();
            plotConcatenated();
            plotFinal();
        }}

        // 选择头
        function selectHead(headIdx) {{
            currentHead = headIdx;
            
            // 更新按钮状态
            document.querySelectorAll('.head-btn').forEach((btn, i) => {{
                btn.classList.toggle('active', i === headIdx);
            }});
            
            // 更新当前步骤显示
            if (currentStep !== 'input') {{
                updateCurrentStep();
            }}
        }}

        // 显示步骤
        function showStep(step) {{
            currentStep = step;
            
            // 更新导航按钮
            document.querySelectorAll('.nav-btn').forEach(btn => {{
                btn.classList.remove('active');
                if (btn.textContent.toLowerCase().includes(step.toLowerCase()) || 
                    (step === 'input' && btn.textContent === '输入')) {{
                    btn.classList.add('active');
                }}
            }});
            
            // 隐藏所有步骤
            document.querySelectorAll('.step-content').forEach(content => {{
                content.classList.remove('active');
            }});
            
            // 显示当前步骤
            const stepElement = document.getElementById('step-' + step);
            if (stepElement) {{
                stepElement.classList.add('active');
                updateCurrentStep();
            }}
        }}

        // 更新当前步骤数据
        function updateCurrentStep() {{
            const head = headsData[currentHead];
            
            switch(currentStep) {{
                case 'q':
                    document.getElementById('q-formula').textContent = head.calculations.Q_formula;
                    plotMatrix('q-plot', head.Q, words, ['q0', 'q1'], 'Reds');
                    createTable('q-table', head.Q, words, ['q0', 'q1']);
                    break;
                case 'k':
                    document.getElementById('k-formula').textContent = head.calculations.K_formula;
                    plotMatrix('k-plot', head.K, words, ['k0', 'k1'], 'Blues');
                    createTable('k-table', head.K, words, ['k0', 'k1']);
                    break;
                case 'v':
                    document.getElementById('v-formula').textContent = head.calculations.V_formula;
                    plotMatrix('v-plot', head.V, words, ['v0', 'v1'], 'Greens');
                    createTable('v-table', head.V, words, ['v0', 'v1']);
                    break;
                case 'scores':
                    document.getElementById('scores-formula').textContent = head.calculations.scores_formula;
                    plotMatrix('scores-plot', head.scores, words, words, 'RdYlBu');
                    createTable('scores-table', head.scores, words, words);
                    break;
                case 'attention':
                    plotMatrix('attention-plot', head.attention, words, words, 'Reds');
                    createTable('attention-table', head.attention, words, words);
                    break;
                case 'z':
                    document.getElementById('z-formula').textContent = head.calculations.Z_formula;
                    plotMatrix('z-plot', head.Z, words, ['z0', 'z1'], 'Purples');
                    createTable('z-table', head.Z, words, ['z0', 'z1']);
                    break;
            }}
        }}

        // 绘制矩阵
        function plotMatrix(elementId, matrix, xLabels, yLabels, colorscale) {{
            const trace = {{
                z: matrix,
                x: xLabels,
                y: yLabels,
                type: 'heatmap',
                colorscale: colorscale
            }};
            
            const layout = {{
                margin: {{t: 30, b: 40, l: 50, r: 20}},
                height: 250
            }};
            
            Plotly.newPlot(elementId, [trace], layout, {{displayModeBar: false}});
        }}

        // 创建表格
        function createTable(elementId, matrix, rowLabels, colLabels) {{
            let html = '<table><tr><th></th>';
            colLabels.forEach(col => html += `<th>${{col}}</th>`);
            html += '</tr>';
            
            matrix.forEach((row, i) => {{
                html += `<tr><td><strong>${{rowLabels[i]}}</strong></td>`;
                row.forEach(val => html += `<td>${{val.toFixed(3)}}</td>`);
                html += '</tr>';
            }});
            
            html += '</table>';
            document.getElementById(elementId).innerHTML = html;
        }}

        // 绘制输入
        function plotInput() {{
            plotMatrix('input-plot', X, words, ['d0', 'd1', 'd2', 'd3', 'd4', 'd5'], 'Blues');
            createTable('input-table', X, words, ['d0', 'd1', 'd2', 'd3', 'd4', 'd5']);
            
            // 详细输入图
            plotMatrix('input-detailed-plot', X, words, ['d0', 'd1', 'd2', 'd3', 'd4', 'd5'], 'Viridis');
        }}

        // 渲染所有头
        function renderAllHeads() {{
            const grid = document.getElementById('heads-grid');
            grid.innerHTML = '';
            
            headsData.forEach((head, index) => {{
                const div = document.createElement('div');
                div.className = 'head-box';
                div.innerHTML = `
                    <div class="head-title">头 ${{index + 1}}</div>
                    <div style="margin-bottom: 10px;">
                        <div>注意力权重矩阵</div>
                        <div id="head-attention-${{index}}" class="plot" style="height: 200px;"></div>
                    </div>
                    <div style="margin-bottom: 10px;">
                        <div>Z矩阵</div>
                        <div id="head-z-${{index}}" class="plot" style="height: 200px;"></div>
                    </div>
                `;
                grid.appendChild(div);
                
                // 绘制注意力矩阵
                plotMatrix(`head-attention-${{index}}`, head.attention, words, words, 
                    ['Reds', 'Blues', 'Greens'][index]);
                
                // 绘制Z矩阵
                plotMatrix(`head-z-${{index}}`, head.Z, words, ['z0', 'z1'], 
                    ['Purples', 'Oranges', 'Teal'][index]);
            }});
        }}

        // 绘制拼接
        function plotConcatenated() {{
            plotMatrix('concat-plot', concatenated, words, 
                ['c0', 'c1', 'c2', 'c3', 'c4', 'c5'], 'Blues');
        }}

        // 绘制最终输出
        function plotFinal() {{
            plotMatrix('final-plot', finalOutput, words, 
                ['f0', 'f1', 'f2', 'f3', 'f4', 'f5'], 'Viridis');
        }}

        // 页面加载完成后初始化
        window.onload = init;
    </script>
</body>
</html>
"""

# 保存HTML文件
with open(os.path.join(output_dir, 'attention_complete.html'), 'w', encoding='utf-8') as f:
    f.write(html_content)

print(f"\n✅ 完整工作流可视化页面已保存到: {output_dir}/attention_complete.html")
print("\n" + "=" * 60)
print("🎯 多头注意力完整工作流程可视化完成！")
print("=" * 60)
print(f"\n📁 文件: {output_dir}/attention_complete.html")
print("\n✨ 特点:")
print("  • 完整的7个计算步骤展示")
print("  • 可切换查看不同注意力头")
print("  • 详细的计算公式和说明")
print("  • 多头并行工作对比")
print("  • 拼接和最终输出展示")