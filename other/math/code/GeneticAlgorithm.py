"""
遗传算法交互式可视化脚本
基于 GeneticAlgorithm.md 文档
使用 Plotly 创建交互式动画图表，包含数学公式
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# 创建输出目录
output_dir = "GeneticAlgorithm"
os.makedirs(output_dir, exist_ok=True)

print("=" * 60)
print("🎨 遗传算法交互式可视化")
print("=" * 60)


def add_formula_annotation(fig, formula_text, x=0.5, y=1.05):
    """添加公式注释到图表"""
    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=x,
        y=y,
        text=formula_text,
        showarrow=False,
        font=dict(size=16, family="Arial, sans-serif"),
        bgcolor="rgba(255, 250, 205, 0.9)",
        bordercolor="orange",
        borderwidth=2,
        borderpad=10,
        xanchor="center",
        align="center",
    )
    return fig


# 目标函数
def target_function(x):
    """f(x) = x * sin(10πx) + 2.0"""
    return x * np.sin(10 * np.pi * x) + 2.0


# ============================================
# 1. 遗传算法进化过程动画
# ============================================
print("\n1️⃣ 创建遗传算法进化动画...")

# 参数设置
X_BOUND = [-1, 2]
POP_SIZE = 50
N_GENERATIONS = 100
CROSS_RATE = 0.8
MUTATION_RATE = 0.05


# 简化的GA实现
def initialize_population():
    return np.random.uniform(X_BOUND[0], X_BOUND[1], POP_SIZE)


def get_fitness(pop):
    return target_function(pop)


def select(pop, fitness):
    fitness = fitness - np.min(fitness) + 1e-4
    idx = np.random.choice(
        np.arange(POP_SIZE), size=POP_SIZE, replace=True, p=fitness / np.sum(fitness)
    )
    return pop[idx]


def crossover_mutate(pop):
    new_pop = []
    for i in range(0, POP_SIZE, 2):
        p1, p2 = pop[i], pop[i + 1] if i + 1 < POP_SIZE else pop[i]
        if np.random.rand() < CROSS_RATE:
            alpha = np.random.rand()
            c1 = alpha * p1 + (1 - alpha) * p2
            c2 = alpha * p2 + (1 - alpha) * p1
        else:
            c1, c2 = p1, p2

        # Mutation
        if np.random.rand() < MUTATION_RATE:
            c1 += np.random.normal(0, 0.1)
        if np.random.rand() < MUTATION_RATE:
            c2 += np.random.normal(0, 0.1)

        c1 = np.clip(c1, X_BOUND[0], X_BOUND[1])
        c2 = np.clip(c2, X_BOUND[0], X_BOUND[1])
        new_pop.extend([c1, c2])

    return np.array(new_pop[:POP_SIZE])


# 运行GA并记录历史
pop = initialize_population()
history = [pop.copy()]
best_history = []

for gen in range(N_GENERATIONS):
    fitness = get_fitness(pop)
    best_history.append(np.max(fitness))
    pop = select(pop, fitness)
    pop = crossover_mutate(pop)
    if gen % 5 == 0:  # 每5代记录一次
        history.append(pop.copy())

# 创建动画
x_range = np.linspace(X_BOUND[0], X_BOUND[1], 200)
y_range = target_function(x_range)

frames = []
for i, pop_snapshot in enumerate(history):
    fitness_snapshot = get_fitness(pop_snapshot)
    best_idx = np.argmax(fitness_snapshot)

    frame_data = [
        go.Scatter(
            x=x_range,
            y=y_range,
            mode="lines",
            name="目标函数",
            line=dict(color="blue", width=2),
        ),
        go.Scatter(
            x=pop_snapshot,
            y=fitness_snapshot,
            mode="markers",
            name="种群个体",
            marker=dict(size=8, color="orange", opacity=0.6),
        ),
        go.Scatter(
            x=[pop_snapshot[best_idx]],
            y=[fitness_snapshot[best_idx]],
            mode="markers",
            name="最佳个体",
            marker=dict(size=15, color="red", symbol="star"),
        ),
    ]
    frames.append(
        go.Frame(
            data=frame_data,
            name=str(i),
            layout=go.Layout(
                title_text=f"第 {i*5} 代 | 最佳适应度: {fitness_snapshot[best_idx]:.3f}"
            ),
        )
    )

fig1 = go.Figure(
    data=[
        go.Scatter(
            x=x_range,
            y=y_range,
            mode="lines",
            name="目标函数",
            line=dict(color="blue", width=2),
        ),
        go.Scatter(
            x=history[0],
            y=get_fitness(history[0]),
            mode="markers",
            name="种群个体",
            marker=dict(size=8, color="orange", opacity=0.6),
        ),
    ],
    frames=frames,
)

fig1 = add_formula_annotation(
    fig1,
    r"$$f(x) = x \cdot \sin(10\pi x) + 2.0 \quad \text{for } x \in [-1, 2]$$",
    x=0.5,
    y=1.05,
)

fig1.update_layout(
    title="遗传算法进化过程",
    xaxis_title="x",
    yaxis_title="f(x)",
    height=700,
    updatemenus=[
        dict(
            type="buttons",
            showactive=False,
            buttons=[
                dict(
                    label="▶ 播放",
                    method="animate",
                    args=[
                        None,
                        dict(
                            frame=dict(duration=200, redraw=True),
                            fromcurrent=True,
                            mode="immediate",
                        ),
                    ],
                ),
                dict(
                    label="⏸ 暂停",
                    method="animate",
                    args=[
                        [None],
                        dict(frame=dict(duration=0, redraw=False), mode="immediate"),
                    ],
                ),
            ],
            direction="left",
            pad=dict(r=10, t=10),
            x=0.0,
            xanchor="left",
            y=1.0,
            yanchor="bottom",
        )
    ],
    margin=dict(t=120, b=60, l=60, r=60),
)

output_file = os.path.join(output_dir, "1_evolution_process.html")
fig1.write_html(output_file, include_mathjax="cdn")
print(f"   ✅ 保存: {output_file}")

# ============================================
# 2. 适应度收敛曲线
# ============================================
print("\n2️⃣ 创建适应度收敛曲线...")

fig2 = go.Figure()

fig2.add_trace(
    go.Scatter(
        x=list(range(len(best_history))),
        y=best_history,
        mode="lines+markers",
        name="最佳适应度",
        line=dict(color="red", width=3),
        marker=dict(size=4),
    )
)

# 添加全局最优参考线
global_optimum = 3.85
fig2.add_hline(
    y=global_optimum,
    line=dict(color="green", dash="dash", width=2),
    annotation_text=f"全局最优 ≈ {global_optimum}",
    annotation_position="right",
)

fig2 = add_formula_annotation(
    fig2,
    r"$$\text{Fitness}(x) = x \cdot \sin(10\pi x) + 2.0 \quad \Rightarrow \quad \max_{x \in [-1,2]} f(x) \approx 3.85$$",
    x=0.5,
    y=1.05,
)

fig2.update_layout(
    title="遗传算法收敛曲线",
    xaxis_title="代数 (Generation)",
    yaxis_title="最佳适应度",
    height=700,
    hovermode="x",
    margin=dict(t=120, b=60, l=60, r=60),
)

output_file = os.path.join(output_dir, "2_convergence_curve.html")
fig2.write_html(output_file, include_mathjax="cdn")
print(f"   ✅ 保存: {output_file}")

# ============================================
# 3. 轮盘赌选择机制可视化
# ============================================
print("\n3️⃣ 创建轮盘赌选择机制可视化...")

# 模拟一个小种群
sample_pop = np.array([0.5, 0.8, 1.2, 1.5, 1.8])
sample_fitness = target_function(sample_pop)
sample_fitness_positive = sample_fitness - np.min(sample_fitness) + 0.1

# 计算选择概率
total_fitness = np.sum(sample_fitness_positive)
selection_prob = sample_fitness_positive / total_fitness
cumulative_prob = np.cumsum(selection_prob)

fig3 = make_subplots(
    rows=1,
    cols=2,
    subplot_titles=("个体适应度", "选择概率 (轮盘赌)"),
    specs=[[{"type": "bar"}, {"type": "pie"}]],
)

# 左图：适应度
fig3.add_trace(
    go.Bar(
        x=[f"个体{i+1}" for i in range(len(sample_pop))],
        y=sample_fitness,
        marker=dict(color=sample_fitness, colorscale="Viridis"),
        text=[f"{f:.2f}" for f in sample_fitness],
        textposition="outside",
        hovertemplate="适应度: %{y:.3f}",
    ),
    row=1,
    col=1,
)

# 右图：选择概率饼图
fig3.add_trace(
    go.Pie(
        labels=[f"个体{i+1}" for i in range(len(sample_pop))],
        values=selection_prob,
        text=[f"{p*100:.1f}%" for p in selection_prob],
        textposition="inside",
        hovertemplate="选择概率: %{value:.3f}",
    ),
    row=1,
    col=2,
)

fig3.update_yaxes(title_text="适应度", row=1, col=1)

fig3 = add_formula_annotation(
    fig3,
    r"$$P_{\text{select}}(i) = \frac{f(i)}{\sum_{j=1}^N f(j)} \quad \text{(轮盘赌选择)}$$",
    x=0.5,
    y=1.05,
)

fig3.update_layout(
    title="轮盘赌选择机制",
    height=600,
    showlegend=True,
    margin=dict(t=120, b=60, l=60, r=60),
)

output_file = os.path.join(output_dir, "3_roulette_wheel_selection.html")
fig3.write_html(output_file, include_mathjax="cdn")
print(f"   ✅ 保存: {output_file}")

# ============================================
# 4. 交叉和变异操作可视化
# ============================================
print("\n4️⃣ 创建交叉和变异操作可视化...")

# 模拟交叉操作
parent1 = 0.5
parent2 = 1.5
alpha_values = np.linspace(0, 1, 100)
offspring = alpha_values * parent1 + (1 - alpha_values) * parent2

fig4 = make_subplots(
    rows=2,
    cols=1,
    subplot_titles=("算术交叉 (Arithmetic Crossover)", "高斯变异 (Gaussian Mutation)"),
    vertical_spacing=0.15,
)

# 上图：交叉
fig4.add_trace(
    go.Scatter(
        x=alpha_values,
        y=offspring,
        mode="lines",
        name="后代位置",
        line=dict(color="purple", width=3),
    ),
    row=1,
    col=1,
)
fig4.add_hline(
    y=parent1,
    line=dict(color="blue", dash="dash"),
    annotation_text=f"父代1: {parent1}",
    row=1,
    col=1,
)
fig4.add_hline(
    y=parent2,
    line=dict(color="red", dash="dash"),
    annotation_text=f"父代2: {parent2}",
    row=1,
    col=1,
)

# 下图：变异
mutation_base = 1.0
mutation_samples = np.random.normal(mutation_base, 0.2, 1000)
fig4.add_trace(
    go.Histogram(
        x=mutation_samples,
        nbinsx=50,
        name="变异后分布",
        marker=dict(color="green", opacity=0.7),
    ),
    row=2,
    col=1,
)
fig4.add_vline(
    x=mutation_base,
    line=dict(color="red", dash="dash", width=2),
    annotation_text=f"原始值: {mutation_base}",
    row=2,
    col=1,
)

fig4.update_xaxes(title_text="混合比例 α", row=1, col=1)
fig4.update_yaxes(title_text="后代基因值", row=1, col=1)
fig4.update_xaxes(title_text="基因值", row=2, col=1)
fig4.update_yaxes(title_text="频数", row=2, col=1)

fig4 = add_formula_annotation(
    fig4,
    r"$$\text{Crossover: } c = \alpha \cdot p_1 + (1-\alpha) \cdot p_2 \quad | \quad \text{Mutation: } c' = c + \mathcal{N}(0, \sigma^2)$$",
    x=0.5,
    y=1.03,
)

fig4.update_layout(
    title="遗传操作：交叉与变异",
    height=800,
    showlegend=True,
    margin=dict(t=120, b=60, l=60, r=60),
)

output_file = os.path.join(output_dir, "4_crossover_mutation.html")
fig4.write_html(output_file, include_mathjax="cdn")
print(f"   ✅ 保存: {output_file}")

# ============================================
# 5. 参数影响分析
# ============================================
print("\n5️⃣ 创建参数影响分析...")

# 测试不同变异率的影响
mutation_rates = [0.01, 0.05, 0.1, 0.2]
colors_mut = ["blue", "green", "orange", "red"]

fig5 = go.Figure()

for mut_rate, color in zip(mutation_rates, colors_mut):
    pop_test = initialize_population()
    best_hist_test = []

    for gen in range(50):
        fitness_test = get_fitness(pop_test)
        best_hist_test.append(np.max(fitness_test))
        pop_test = select(pop_test, fitness_test)

        # 使用不同变异率
        new_pop = []
        for i in range(0, POP_SIZE, 2):
            p1, p2 = pop_test[i], pop_test[i + 1] if i + 1 < POP_SIZE else pop_test[i]
            alpha = np.random.rand()
            c1 = alpha * p1 + (1 - alpha) * p2
            c2 = alpha * p2 + (1 - alpha) * p1

            if np.random.rand() < mut_rate:
                c1 += np.random.normal(0, 0.1)
            if np.random.rand() < mut_rate:
                c2 += np.random.normal(0, 0.1)

            c1 = np.clip(c1, X_BOUND[0], X_BOUND[1])
            c2 = np.clip(c2, X_BOUND[0], X_BOUND[1])
            new_pop.extend([c1, c2])

        pop_test = np.array(new_pop[:POP_SIZE])

    fig5.add_trace(
        go.Scatter(
            x=list(range(len(best_hist_test))),
            y=best_hist_test,
            mode="lines",
            name=f"变异率 = {mut_rate}",
            line=dict(color=color, width=2),
        )
    )

fig5 = add_formula_annotation(
    fig5, r"$$P_m \text{ (变异率) 影响探索vs利用的平衡}$$", x=0.5, y=1.05
)

fig5.update_layout(
    title="变异率对收敛的影响",
    xaxis_title="代数",
    yaxis_title="最佳适应度",
    height=700,
    legend=dict(x=0.7, y=0.3, bgcolor="rgba(255,255,255,0.8)"),
    margin=dict(t=120, b=60, l=60, r=60),
)

output_file = os.path.join(output_dir, "5_parameter_impact.html")
fig5.write_html(output_file, include_mathjax="cdn")
print(f"   ✅ 保存: {output_file}")

# ============================================
# 6. Schema定理可视化
# ============================================
print("\n6️⃣ 创建Schema定理可视化...")

# 模拟schema的增长
generations = np.arange(0, 50)
avg_fitness = 2.5
schema_fitness_values = [2.8, 3.0, 3.2]
colors_schema = ["blue", "green", "red"]

fig6 = go.Figure()

for schema_fitness, color in zip(schema_fitness_values, colors_schema):
    # E[m(H, t+1)] ≈ m(H, t) * (f(H) / f_avg)
    growth_factor = schema_fitness / avg_fitness
    schema_count = 10 * (growth_factor**generations)

    fig6.add_trace(
        go.Scatter(
            x=generations,
            y=schema_count,
            mode="lines",
            name=f"Schema f(H)={schema_fitness}",
            line=dict(color=color, width=3),
        )
    )

fig6 = add_formula_annotation(
    fig6,
    r"$$E[m(H, t+1)] \ge m(H, t) \cdot \frac{f(H, t)}{f_{\text{avg}}(t)} \cdot \left[1 - P_c \cdot \frac{\delta(H)}{l-1} - o(H) \cdot P_m\right]$$",
    x=0.5,
    y=1.05,
)

fig6.update_layout(
    title="Schema定理：高适应度模式的指数增长",
    xaxis_title="代数",
    yaxis_title="Schema实例数量",
    height=700,
    legend=dict(x=0.02, y=0.98, bgcolor="rgba(255,255,255,0.8)"),
    margin=dict(t=120, b=60, l=60, r=60),
)

output_file = os.path.join(output_dir, "6_schema_theorem.html")
fig6.write_html(output_file, include_mathjax="cdn")
print(f"   ✅ 保存: {output_file}")

# ============================================
# 7. 综合仪表板
# ============================================
print("\n7️⃣ 创建综合仪表板...")

fig7 = make_subplots(
    rows=2,
    cols=2,
    subplot_titles=("进化过程快照", "收敛曲线", "选择概率分布", "参数影响对比"),
    specs=[
        [{"type": "scatter"}, {"type": "scatter"}],
        [{"type": "pie"}, {"type": "scatter"}],
    ],
    vertical_spacing=0.15,
    horizontal_spacing=0.12,
)

# 子图1: 最后一代的种群分布
final_pop = history[-1]
final_fitness = get_fitness(final_pop)
fig7.add_trace(
    go.Scatter(
        x=x_range,
        y=y_range,
        mode="lines",
        name="目标函数",
        line=dict(color="blue", width=2),
    ),
    row=1,
    col=1,
)
fig7.add_trace(
    go.Scatter(
        x=final_pop,
        y=final_fitness,
        mode="markers",
        name="最终种群",
        marker=dict(size=8, color="orange"),
    ),
    row=1,
    col=1,
)

# 子图2: 收敛曲线
fig7.add_trace(
    go.Scatter(
        x=list(range(len(best_history))),
        y=best_history,
        mode="lines",
        name="最佳适应度",
        line=dict(color="red", width=2),
    ),
    row=1,
    col=2,
)

# 子图3: 选择概率饼图
fig7.add_trace(
    go.Pie(
        labels=[f"个体{i+1}" for i in range(len(sample_pop))],
        values=selection_prob,
        textposition="inside",
    ),
    row=2,
    col=1,
)

# 子图4: 不同变异率的对比
for mut_rate, color in zip([0.01, 0.1], ["blue", "red"]):
    pop_test = initialize_population()
    best_hist_short = []
    for gen in range(30):
        fitness_test = get_fitness(pop_test)
        best_hist_short.append(np.max(fitness_test))
        pop_test = select(pop_test, fitness_test)
        new_pop = []
        for i in range(0, POP_SIZE, 2):
            p1, p2 = pop_test[i], pop_test[i + 1] if i + 1 < POP_SIZE else pop_test[i]
            c1, c2 = (p1 + p2) / 2, (p1 + p2) / 2
            if np.random.rand() < mut_rate:
                c1 += np.random.normal(0, 0.1)
            if np.random.rand() < mut_rate:
                c2 += np.random.normal(0, 0.1)
            c1 = np.clip(c1, X_BOUND[0], X_BOUND[1])
            c2 = np.clip(c2, X_BOUND[0], X_BOUND[1])
            new_pop.extend([c1, c2])
        pop_test = np.array(new_pop[:POP_SIZE])

    fig7.add_trace(
        go.Scatter(
            x=list(range(len(best_hist_short))),
            y=best_hist_short,
            mode="lines",
            name=f"Pm={mut_rate}",
            line=dict(color=color, width=2),
        ),
        row=2,
        col=2,
    )

# 更新坐标轴
fig7.update_xaxes(title_text="x", row=1, col=1)
fig7.update_yaxes(title_text="f(x)", row=1, col=1)
fig7.update_xaxes(title_text="代数", row=1, col=2)
fig7.update_yaxes(title_text="适应度", row=1, col=2)
fig7.update_xaxes(title_text="代数", row=2, col=2)
fig7.update_yaxes(title_text="适应度", row=2, col=2)

fig7 = add_formula_annotation(
    fig7,
    r"$$\text{遗传算法：模拟自然进化，通过选择、交叉、变异寻找最优解}$$",
    x=0.5,
    y=1.02,
)

fig7.update_layout(
    title="遗传算法综合仪表板",
    height=900,
    showlegend=True,
    margin=dict(t=120, b=60, l=60, r=60),
)

output_file = os.path.join(output_dir, "7_dashboard.html")
fig7.write_html(output_file, include_mathjax="cdn")
print(f"   ✅ 保存: {output_file}")

# ============================================
# 打印总结
# ============================================
print("\n" + "=" * 60)
print("📊 遗传算法可视化总结")
print("=" * 60)

print("\n✅ 已创建的可视化:")
print("   1. 进化过程动画")
print("   2. 适应度收敛曲线")
print("   3. 轮盘赌选择机制")
print("   4. 交叉与变异操作")
print("   5. 参数影响分析")
print("   6. Schema定理")
print("   7. 综合仪表板")

print("\n💡 核心要点:")
print("   • 遗传算法模拟自然进化：选择、交叉、变异")
print("   • 适应度函数指导搜索方向")
print("   • 种群多样性是避免局部最优的关键")
print("   • Schema定理：高适应度模式指数增长")
print("   • 参数调优：种群大小、交叉率、变异率")

print("\n" + "=" * 60)
print("✨ 所有交互式可视化已创建完成！")
print("=" * 60)
print(f"\n📂 生成的文件位于: code/{output_dir}/")
print("💡 在浏览器中打开这些 HTML 文件即可查看交互式可视化！")
print("=" * 60)
