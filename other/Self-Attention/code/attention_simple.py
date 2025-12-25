"""
多头注意力工作流程可视化 - 简洁版
清晰展示多头注意力的核心流程
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
# 1. 创建简单的词向量
# ============================================

sentence_words = ["The", "cat", "sits", "on", "the", "mat"]
n_words = len(sentence_words)
d_model = 6  # 简化维度
num_heads = 3  # 3个头
d_k = d_model // num_heads

print(f"句子: {' '.join(sentence_words)}")
print(f"配置: {n_words}个词, {d_model}维, {num_heads}个头")

# 手工设计简单的词向量
X = np.array(
    [
        [1.0, 0.0, 0.0, 0.5, 0.2, 0.1],  # The
        [0.0, 1.0, 0.0, 0.8, 0.6, 0.4],  # cat
        [0.0, 0.0, 1.0, 0.3, 0.7, 0.5],  # sits
        [0.5, 0.0, 0.0, 0.0, 0.3, 0.2],  # on
        [1.0, 0.0, 0.0, 0.4, 0.1, 0.2],  # the
        [0.0, 0.8, 0.0, 0.6, 0.3, 0.7],  # mat
    ]
)

print(f"\n输入矩阵 X 形状: {X.shape}")
print(f"X =\n{X}")

# ============================================
# 2. 计算多头注意力
# ============================================

np.random.seed(42)

# 为每个头创建权重矩阵
heads = []
for i in range(num_heads):
    W_Q = np.random.randn(d_model, d_k) * 0.1
    W_K = np.random.randn(d_model, d_k) * 0.1
    W_V = np.random.randn(d_model, d_k) * 0.1

    # 计算Q, K, V
    Q = X @ W_Q
    K = X @ W_K
    V = X @ W_V

    # 计算注意力
    scores = Q @ K.T / np.sqrt(d_k)
    attention = np.exp(scores) / np.exp(scores).sum(axis=1, keepdims=True)
    Z = attention @ V

    heads.append(
        {
            "Q": Q.tolist(),
            "K": K.tolist(),
            "V": V.tolist(),
            "attention": attention.tolist(),
            "Z": Z.tolist(),
        }
    )

# 拼接输出
concatenated = np.concatenate([np.array(head["Z"]) for head in heads], axis=1)
W_O = np.random.randn(num_heads * d_k, d_model) * 0.1
final_output = concatenated @ W_O

print(f"\n计算完成!")
print(f"拼接后形状: {concatenated.shape}")
print(f"最终输出形状: {final_output.shape}")

# ============================================
# 3. 生成简洁的HTML页面
# ============================================

html_content = f"""<!DOCTYPE html>
<html lang="zh">
<head>
    <meta charset="utf-8">
    <title>多头注意力工作流程</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
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
            margin-bottom: 40px;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 8px;
        }}
        .section h2 {{
            color: #495057;
            margin-bottom: 20px;
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
        button {{
            padding: 10px 20px;
            margin: 5px;
            background: #007bff;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
        }}
        button.active {{
            background: #0056b3;
        }}
        .hidden {{
            display: none;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 10px 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 8px;
            text-align: center;
        }}
        th {{
            background: #f2f2f2;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🧠 多头注意力工作流程</h1>
        
        <div class="section">
            <h2>📝 输入词向量</h2>
            <div id="input-plot" class="plot"></div>
            <table id="input-table"></table>
        </div>

        <div class="section">
            <h2>🔄 多头并行计算</h2>
            <div style="text-align: center; margin-bottom: 20px;">
                <button onclick="showHead(0)" class="head-btn active">头 1</button>
                <button onclick="showHead(1)" class="head-btn">头 2</button>
                <button onclick="showHead(2)" class="head-btn">头 3</button>
            </div>
            
            <div class="grid">
                <div>
                    <h3>注意力权重</h3>
                    <div id="attention-plot" class="plot"></div>
                </div>
                <div>
                    <h3>输出 Z 矩阵</h3>
                    <div id="z-plot" class="plot"></div>
                </div>
            </div>
        </div>

        <div class="section">
            <h2>🔗 拼接与最终输出</h2>
            <div class="grid">
                <div>
                    <h3>拼接输出</h3>
                    <div id="concat-plot" class="plot"></div>
                </div>
                <div>
                    <h3>最终输出</h3>
                    <div id="final-plot" class="plot"></div>
                </div>
            </div>
        </div>
    </div>

    <script>
        // 数据
        const words = {sentence_words};
        const X = {X.tolist()};
        const heads = {json.dumps(heads)};
        const concatenated = {concatenated.tolist()};
        const finalOutput = {final_output.tolist()};
        
        let currentHead = 0;

        // 绘制输入
        function plotInput() {{
            const trace = {{
                z: X,
                x: words,
                y: ['d0', 'd1', 'd2', 'd3', 'd4', 'd5'],
                type: 'heatmap',
                colorscale: 'Blues'
            }};
            
            const layout = {{
                title: '输入词向量矩阵',
                margin: {{t: 30, b: 40, l: 50, r: 20}}
            }};
            
            Plotly.newPlot('input-plot', [trace], layout, {{displayModeBar: false}});
            
            // 添加表格
            let tableHtml = '<table><tr><th>词</th>';
            for (let i = 0; i < 6; i++) {{
                tableHtml += `<th>d${{i}}</th>`;
            }}
            tableHtml += '</tr>';
            
            for (let i = 0; i < X.length; i++) {{
                tableHtml += `<tr><td>${{words[i]}}</td>`;
                for (let j = 0; j < X[i].length; j++) {{
                    tableHtml += `<td>${{X[i][j].toFixed(2)}}</td>`;
                }}
                tableHtml += '</tr>';
            }}
            tableHtml += '</table>';
            
            document.getElementById('input-table').innerHTML = tableHtml;
        }}

        // 显示特定头
        function showHead(headIndex) {{
            currentHead = headIndex;
            
            // 更新按钮状态
            document.querySelectorAll('.head-btn').forEach((btn, i) => {{
                btn.classList.toggle('active', i === headIndex);
            }});
            
            plotAttention(headIndex);
            plotZ(headIndex);
        }}

        // 绘制注意力权重
        function plotAttention(headIndex) {{
            const trace = {{
                z: heads[headIndex].attention,
                x: words,
                y: words,
                type: 'heatmap',
                colorscale: 'Reds'
            }};
            
            const layout = {{
                title: `头${{headIndex + 1}} 注意力权重`,
                margin: {{t: 30, b: 40, l: 50, r: 20}}
            }};
            
            Plotly.newPlot('attention-plot', [trace], layout, {{displayModeBar: false}});
        }}

        // 绘制Z矩阵
        function plotZ(headIndex) {{
            const trace = {{
                z: heads[headIndex].Z,
                x: words,
                y: ['z0', 'z1'],
                type: 'heatmap',
                colorscale: 'Purples'
            }};
            
            const layout = {{
                title: `头${{headIndex + 1}} Z 矩阵`,
                margin: {{t: 30, b: 40, l: 50, r: 20}}
            }};
            
            Plotly.newPlot('z-plot', [trace], layout, {{displayModeBar: false}});
        }}

        // 绘制拼接输出
        function plotConcatenated() {{
            const trace = {{
                z: concatenated,
                x: words,
                y: ['c0', 'c1', 'c2', 'c3', 'c4', 'c5'],
                type: 'heatmap',
                colorscale: 'Blues'
            }};
            
            const layout = {{
                title: '拼接输出',
                margin: {{t: 30, b: 40, l: 50, r: 20}}
            }};
            
            Plotly.newPlot('concat-plot', [trace], layout, {{displayModeBar: false}});
        }}

        // 绘制最终输出
        function plotFinal() {{
            const trace = {{
                z: finalOutput,
                x: words,
                y: ['f0', 'f1', 'f2', 'f3', 'f4', 'f5'],
                type: 'heatmap',
                colorscale: 'Viridis'
            }};
            
            const layout = {{
                title: '最终输出',
                margin: {{t: 30, b: 40, l: 50, r: 20}}
            }};
            
            Plotly.newPlot('final-plot', [trace], layout, {{displayModeBar: false}});
        }}

        // 初始化
        plotInput();
        showHead(0);
        plotConcatenated();
        plotFinal();
    </script>
</body>
</html>
"""

# 保存HTML文件
with open(
    os.path.join(output_dir, "attention_simple.html"), "w", encoding="utf-8"
) as f:
    f.write(html_content)

print(f"\n✅ 简洁版可视化页面已保存到: {output_dir}/attention_simple.html")
print("\n" + "=" * 60)
print("🎯 多头注意力可视化完成！")
print("=" * 60)
print(f"\n📁 文件: {output_dir}/attention_simple.html")
print("\n✨ 特点:")
print("  • 简洁明了的界面")
print("  • 可切换查看不同头")
print("  • 清晰的矩阵可视化")
print("  • 完整的计算流程")
