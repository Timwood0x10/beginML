"""
L1 & L2 正则化交互式可视化脚本
基于 5.L1&L2.md 文档中的公式
使用 Plotly 创建交互式动画图表，包含数学公式
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# 创建输出目录
output_dir = "L1_L2_Regularization"
os.makedirs(output_dir, exist_ok=True)

print("=" * 60)
print("🎨 L1 & L2 正则化交互式可视化")
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


# ============================================
# 1. L1 vs L2 约束几何形状对比
# ============================================
print("\n1️⃣ 创建 L1 vs L2 约束几何形状对比可视化...")


def create_constraint_shapes():
    """创建 L1 和 L2 约束形状的对比"""

    # 创建网格
    w1 = np.linspace(-3, 3, 100)
    w2 = np.linspace(-3, 3, 100)
    W1, W2 = np.meshgrid(w1, w2)

    # 创建子图
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "L2 约束 L2 Constraint: ||w||²₂ ≤ C",
            "L1 约束 L1 Constraint: ||w||₁ ≤ C",
        ),
        specs=[[{"type": "scatter"}, {"type": "scatter"}]],
    )

    # L2 约束（圆形）
    theta = np.linspace(0, 2 * np.pi, 100)
    C = 4  # 约束值
    l2_circle_w1 = np.sqrt(C) * np.cos(theta)
    l2_circle_w2 = np.sqrt(C) * np.sin(theta)

    fig.add_trace(
        go.Scatter(
            x=l2_circle_w1,
            y=l2_circle_w2,
            mode="lines",
            name="L2 约束边界",
            line=dict(color="red", width=3),
            fill="tonexty",
            fillcolor="rgba(255, 0, 0, 0.2)",
        ),
        row=1,
        col=1,
    )

    # L1 约束（菱形）
    l1_x = [np.sqrt(C), 0, -np.sqrt(C), 0, np.sqrt(C)]
    l1_y = [0, np.sqrt(C), 0, -np.sqrt(C), 0]

    fig.add_trace(
        go.Scatter(
            x=l1_x,
            y=l1_y,
            mode="lines",
            name="L1 约束边界",
            line=dict(color="blue", width=3),
            fill="tonexty",
            fillcolor="rgba(0, 0, 255, 0.2)",
        ),
        row=1,
        col=2,
    )

    # 添加损失函数等高线（椭圆形）
    loss_center = np.array([2, 1])
    for scale in [0.5, 1.0, 1.5, 2.0]:
        ellipse_w1 = loss_center[0] + scale * np.cos(theta)
        ellipse_w2 = loss_center[1] + 0.6 * scale * np.sin(theta)

        fig.add_trace(
            go.Scatter(
                x=ellipse_w1,
                y=ellipse_w2,
                mode="lines",
                name=f"损失等高线 Loss Contour {scale}",
                line=dict(color="green", width=2, dash="dash"),
                showlegend=False,
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=ellipse_w1,
                y=ellipse_w2,
                mode="lines",
                name=f"损失等高线 Loss Contour {scale}",
                line=dict(color="green", width=2, dash="dash"),
                showlegend=False,
            ),
            row=1,
            col=2,
        )

    # 标记最优解
    # L2 最优解（投影到圆上）
    l2_optimal = loss_center / np.linalg.norm(loss_center) * np.sqrt(C)
    fig.add_trace(
        go.Scatter(
            x=[l2_optimal[0]],
            y=[l2_optimal[1]],
            mode="markers",
            name="L2 最优解 L2 Optimal",
            marker=dict(color="red", size=10, symbol="star"),
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    # L1 最优解（投影到菱形上，通常在坐标轴上）
    if abs(loss_center[0]) > abs(loss_center[1]):
        l1_optimal = np.array([np.sqrt(C) * np.sign(loss_center[0]), 0])
    else:
        l1_optimal = np.array([0, np.sqrt(C) * np.sign(loss_center[1])])

    fig.add_trace(
        go.Scatter(
            x=[l1_optimal[0]],
            y=[l1_optimal[1]],
            mode="markers",
            name="L1 最优解 L1 Optimal",
            marker=dict(color="blue", size=10, symbol="star"),
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    # 添加公式
    fig = add_formula_annotation(
        fig,
        r"$$\text{L2: } \|\mathbf{w}\|_2^2 \leq C \quad \text{vs} \quad \text{L1: } \|\mathbf{w}\|_1 \leq C$$",
        x=0.5,
        y=1.05,
    )

    fig.update_xaxes(title_text="w₁", row=1, col=1)
    fig.update_xaxes(title_text="w₁", row=1, col=2)
    fig.update_yaxes(title_text="w₂", row=1, col=1)
    fig.update_yaxes(title_text="w₂", row=1, col=2)

    fig.update_layout(
        title_text="L1 vs L2 约束几何形状对比 Constraint Shapes Comparison",
        height=600,
        showlegend=True,
        margin=dict(t=120, b=60, l=60, r=60),
    )

    return fig


fig1 = create_constraint_shapes()
output_file = os.path.join(output_dir, "1_constraint_shapes.html")
fig1.write_html(output_file, include_mathjax="cdn")
print(f"   ✅ 保存: {output_file}")

# ============================================
# 2. 权重衰减过程对比
# ============================================
print("\n2️⃣ 创建权重衰减过程对比可视化...")


def create_weight_decay_animation():
    """创建 L1 和 L2 权重衰减过程的动画"""

    # 初始参数
    initial_w = 2.0
    gradient = 0.01  # 假设的固定梯度
    learning_rate = 0.1
    lambda_reg = 0.1

    # 计算衰减过程
    steps = 100
    l2_weights = [initial_w]
    l1_weights = [initial_w]

    for i in range(steps):
        # L2 衰减：w <- (1 - lr*lambda) * w - lr*grad
        l2_new = (1 - learning_rate * lambda_reg) * l2_weights[
            -1
        ] - learning_rate * gradient
        l2_weights.append(l2_new)

        # L1 衰减：w <- w - lr*grad - lr*lambda*sign(w)
        if l1_weights[-1] > 0:
            l1_new = (
                l1_weights[-1] - learning_rate * gradient - learning_rate * lambda_reg
            )
        else:
            l1_new = (
                l1_weights[-1] - learning_rate * gradient + learning_rate * lambda_reg
            )

        # L1 可以归零
        if abs(l1_new) < learning_rate * lambda_reg:
            l1_new = 0

        l1_weights.append(l1_new)

    # 创建动画帧
    frames = []
    for i in range(0, steps, 2):  # 每2步一帧
        frame_data = [
            go.Scatter(
                x=list(range(i + 1)),
                y=l2_weights[: i + 1],
                mode="lines+markers",
                name="L2 权重衰减 L2 Weight Decay",
                line=dict(color="red", width=3),
                marker=dict(size=6),
            ),
            go.Scatter(
                x=list(range(i + 1)),
                y=l1_weights[: i + 1],
                mode="lines+markers",
                name="L1 权重衰减 L1 Weight Decay",
                line=dict(color="blue", width=3),
                marker=dict(size=6),
            ),
        ]

        frames.append(
            go.Frame(
                data=frame_data,
                name=str(i),
                layout=go.Layout(
                    title_text=f"权重衰减过程 Weight Decay Process - 步骤 Step {i}<br>"
                    + f"L2: {l2_weights[i]:.4f}, L1: {l1_weights[i]:.4f}"
                ),
            )
        )

    # 创建主图形
    fig = go.Figure(
        data=[
            go.Scatter(
                x=[0],
                y=[initial_w],
                mode="lines+markers",
                name="L2 权重衰减 L2 Weight Decay",
                line=dict(color="red", width=3),
                marker=dict(size=6),
            ),
            go.Scatter(
                x=[0],
                y=[initial_w],
                mode="lines+markers",
                name="L1 权重衰减 L1 Weight Decay",
                line=dict(color="blue", width=3),
                marker=dict(size=6),
            ),
        ],
        frames=frames,
    )

    # 添加零线
    fig.add_hline(
        y=0, line_dash="dash", line_color="gray", annotation_text="零线 Zero Line"
    )

    # 添加公式
    fig = add_formula_annotation(
        fig,
        r"$$\text{L2: } w \leftarrow (1-\eta\lambda)w - \eta\nabla \quad \text{vs} \quad \text{L1: } w \leftarrow w - \eta\nabla - \eta\lambda \cdot \text{sign}(w)$$",
        x=0.5,
        y=1.05,
    )

    # 添加播放按钮
    fig.update_layout(
        title="L1 vs L2 权重衰减过程对比 Weight Decay Comparison",
        xaxis_title="训练步骤 Training Steps",
        yaxis_title="权重值 Weight Value",
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                buttons=[
                    dict(
                        label="▶ 播放 Play",
                        method="animate",
                        args=[
                            None,
                            dict(
                                frame=dict(duration=100, redraw=True),
                                fromcurrent=True,
                                mode="immediate",
                            ),
                        ],
                    ),
                    dict(
                        label="⏸ 暂停 Pause",
                        method="animate",
                        args=[
                            [None],
                            dict(
                                frame=dict(duration=0, redraw=False), mode="immediate"
                            ),
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
        height=600,
        margin=dict(t=120, b=60, l=60, r=60),
    )

    return fig


fig2 = create_weight_decay_animation()
output_file = os.path.join(output_dir, "2_weight_decay.html")
fig2.write_html(output_file, include_mathjax="cdn")
print(f"   ✅ 保存: {output_file}")

# ============================================
# 3. 正则化路径（Ridge vs Lasso）
# ============================================
print("\n3️⃣ 创建正则化路径可视化...")


def create_regularization_path():
    """创建正则化路径（不同正则化强度下的系数变化）"""

    # 模拟数据
    np.random.seed(42)
    n_features = 8
    n_alphas = 50

    # 真实系数（有些为0，模拟稀疏性）
    true_coeffs = np.array([3.0, 0.0, 0.0, 2.0, 0.0, -1.5, 0.0, 1.0])

    # 模拟不同正则化强度下的系数
    alphas = np.logspace(-4, 0, n_alphas)

    # L2 路径（Ridge）：所有系数平滑趋向0
    l2_path = np.zeros((n_alphas, n_features))
    for i, alpha in enumerate(alphas):
        # 简化的 Ridge 路径模拟
        shrinkage_factor = 1 / (1 + alpha * 10)
        l2_path[i] = true_coeffs * shrinkage_factor + np.random.normal(
            0, 0.1, n_features
        )

    # L1 路径（Lasso）：系数逐步变为0
    l1_path = np.zeros((n_alphas, n_features))
    for i, alpha in enumerate(alphas):
        # 简化的 Lasso 路径模拟
        threshold = alpha * 2
        l1_path[i] = np.where(
            np.abs(true_coeffs) > threshold, true_coeffs * (1 - alpha), 0
        ) + np.random.normal(0, 0.05, n_features)

    # 创建子图
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "L2 正则化路径 L2 Regularization Path (Ridge)",
            "L1 正则化路径 L1 Regularization Path (Lasso)",
        ),
    )

    # 绘制 L2 路径
    colors = ["red", "blue", "green", "orange", "purple", "brown", "pink", "gray"]
    for i in range(n_features):
        fig.add_trace(
            go.Scatter(
                x=alphas,
                y=l2_path[:, i],
                mode="lines",
                name=f"w{i+1}",
                line=dict(color=colors[i], width=2),
                showlegend=False,
            ),
            row=1,
            col=1,
        )

    # 绘制 L1 路径
    for i in range(n_features):
        fig.add_trace(
            go.Scatter(
                x=alphas,
                y=l1_path[:, i],
                mode="lines",
                name=f"w{i+1}",
                line=dict(color=colors[i], width=2),
                showlegend=False,
            ),
            row=1,
            col=2,
        )

    # 添加图例
    for i in range(n_features):
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="lines",
                name=f"系数 w{i+1}",
                line=dict(color=colors[i], width=3),
            )
        )

    # 添加公式
    fig = add_formula_annotation(
        fig,
        r"$$\text{正则化路径：不同 } \lambda \text{ 值下的系数变化 Regularization Path}$$",
        x=0.5,
        y=1.05,
    )

    # 更新坐标轴
    fig.update_xaxes(type="log", title_text="正则化强度 λ (log scale)", row=1, col=1)
    fig.update_xaxes(type="log", title_text="正则化强度 λ (log scale)", row=1, col=2)
    fig.update_yaxes(title_text="系数值 Coefficient Value", row=1, col=1)
    fig.update_yaxes(title_text="系数值 Coefficient Value", row=1, col=2)

    fig.update_layout(
        title_text="正则化路径对比 Regularization Path Comparison",
        height=600,
        showlegend=True,
        legend=dict(x=1.02, y=1, bgcolor="rgba(255,255,255,0.8)"),
        margin=dict(t=120, b=60, l=60, r=120),
    )

    return fig


fig3 = create_regularization_path()
output_file = os.path.join(output_dir, "3_regularization_path.html")
fig3.write_html(output_file, include_mathjax="cdn")
print(f"   ✅ 保存: {output_file}")

# ============================================
# 4. 贝叶斯先验分布对比
# ============================================
print("\n4️⃣ 创建贝叶斯先验分布对比可视化...")


def create_bayesian_priors():
    """创建高斯先验 vs 拉普拉斯先验的对比"""

    # 创建x轴
    x = np.linspace(-4, 4, 1000)

    # 高斯分布（L2先验）
    sigma = 1.0
    gaussian = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-(x**2) / (2 * sigma**2))

    # 拉普拉斯分布（L1先验）
    b = 0.5
    laplacian = (1 / (2 * b)) * np.exp(-np.abs(x) / b)

    # 标准化以便比较
    gaussian = gaussian / np.max(gaussian)
    laplacian = laplacian / np.max(laplacian)

    # 创建图形
    fig = go.Figure()

    # 添加高斯分布
    fig.add_trace(
        go.Scatter(
            x=x,
            y=gaussian,
            mode="lines",
            name="高斯先验 Gaussian Prior (L2)",
            line=dict(color="red", width=3),
            fill="tonexty",
            fillcolor="rgba(255, 0, 0, 0.2)",
        )
    )

    # 添加拉普拉斯分布
    fig.add_trace(
        go.Scatter(
            x=x,
            y=laplacian,
            mode="lines",
            name="拉普拉斯先验 Laplacian Prior (L1)",
            line=dict(color="blue", width=3),
            fill="tonexty",
            fillcolor="rgba(0, 0, 255, 0.2)",
        )
    )

    # 标记0点处的值
    fig.add_trace(
        go.Scatter(
            x=[0],
            y=[1],
            mode="markers",
            name="峰值 Peak",
            marker=dict(color="green", size=8, symbol="diamond"),
            showlegend=False,
        )
    )

    # 添加公式
    fig = add_formula_annotation(
        fig,
        r"$$\text{高斯先验: } P(w) \propto \exp\left(-\frac{w^2}{2\sigma^2}\right) \quad \text{vs} \quad \text{拉普拉斯先验: } P(w) \propto \exp\left(-\frac{|w|}{b}\right)$$",
        x=0.5,
        y=1.05,
    )

    fig.update_layout(
        title="贝叶斯先验分布对比 Bayesian Prior Comparison",
        xaxis_title="权重值 Weight Value (w)",
        yaxis_title="概率密度 Probability Density (标准化)",
        height=600,
        legend=dict(x=0.02, y=0.98, bgcolor="rgba(255,255,255,0.8)"),
        margin=dict(t=120, b=60, l=60, r=60),
        annotations=[
            dict(
                x=0,
                y=0.5,
                text="高斯：温和地相信权重应该小<br>Gaussian: Gently believe weights should be small",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor="red",
                ax=100,
                ay=-50,
                font=dict(size=10, color="red"),
            ),
            dict(
                x=0,
                y=1,
                text="拉普拉斯：强烈地相信权重应该是0<br>Laplacian: Strongly believe weights should be 0",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor="blue",
                ax=100,
                ay=50,
                font=dict(size=10, color="blue"),
            ),
        ],
    )

    return fig


fig4 = create_bayesian_priors()
output_file = os.path.join(output_dir, "4_bayesian_priors.html")
fig4.write_html(output_file, include_mathjax="cdn")
print(f"   ✅ 保存: {output_file}")

# ============================================
# 5. 拉格朗日对偶性可视化
# ============================================
print("\n5️⃣ 创建拉格朗日对偶性可视化...")


def create_lagrange_duality():
    """创建约束形式与正则化形式的等价性可视化"""

    # 创建参数空间
    w1 = np.linspace(-3, 3, 100)
    w2 = np.linspace(-3, 3, 100)
    W1, W2 = np.meshgrid(w1, w2)

    # 简单的二次损失函数
    loss = (W1 - 2) ** 2 + (W2 - 1) ** 2

    # 不同的约束值C对应的正则化强度λ
    C_values = [1, 2, 4]
    lambda_values = [1.0, 0.5, 0.25]  # λ ≈ 1/C 的关系

    # 创建子图
    fig = make_subplots(
        rows=2,
        cols=3,
        subplot_titles=(
            "约束形式 C=1 Constraint Form",
            "约束形式 C=2 Constraint Form",
            "约束形式 C=4 Constraint Form",
            "正则化形式 λ=1.0 Regularization Form",
            "正则化形式 λ=0.5 Regularization Form",
            "正则化形式 λ=0.25 Regularization Form",
        ),
        specs=[
            [{"type": "contour"}, {"type": "contour"}, {"type": "contour"}],
            [{"type": "contour"}, {"type": "contour"}, {"type": "contour"}],
        ],
    )

    colorscale = "Viridis"

    # 第一行：约束形式
    for i, (C, lam) in enumerate(zip(C_values, lambda_values)):
        # 损失函数等高线
        fig.add_trace(
            go.Contour(
                x=w1,
                y=w2,
                z=loss,
                colorscale=colorscale,
                showscale=False,
                contours=dict(start=0, end=20, size=2, showlabels=True),
                name=f"损失等高线 Loss Contour C={C}",
            ),
            row=1,
            col=i + 1,
        )

        # 约束边界
        if i == 0:  # L2 约束（圆）
            theta = np.linspace(0, 2 * np.pi, 100)
            constraint_w1 = np.sqrt(C) * np.cos(theta)
            constraint_w2 = np.sqrt(C) * np.sin(theta)
            fig.add_trace(
                go.Scatter(
                    x=constraint_w1,
                    y=constraint_w2,
                    mode="lines",
                    name=f"L2约束 L2 Constraint C={C}",
                    line=dict(color="red", width=3),
                    showlegend=False,
                ),
                row=1,
                col=i + 1,
            )
        else:  # L1 约束（菱形）
            constraint_w1 = [np.sqrt(C), 0, -np.sqrt(C), 0, np.sqrt(C)]
            constraint_w2 = [0, np.sqrt(C), 0, -np.sqrt(C), 0]
            fig.add_trace(
                go.Scatter(
                    x=constraint_w1,
                    y=constraint_w2,
                    mode="lines",
                    name=f"L1约束 L1 Constraint C={C}",
                    line=dict(color="blue", width=3),
                    showlegend=False,
                ),
                row=1,
                col=i + 1,
            )

    # 第二行：正则化形式
    for i, (C, lam) in enumerate(zip(C_values, lambda_values)):
        # 正则化后的目标函数
        if i == 0:  # L2 正则化
            regularized = loss + lam * (W1**2 + W2**2)
        else:  # L1 正则化
            regularized = loss + lam * (np.abs(W1) + np.abs(W2))

        fig.add_trace(
            go.Contour(
                x=w1,
                y=w2,
                z=regularized,
                colorscale=colorscale,
                showscale=False,
                contours=dict(start=0, end=20, size=2, showlabels=True),
                name=f"正则化目标 Regularized Objective λ={lam}",
            ),
            row=2,
            col=i + 1,
        )

        # 标记最优点
        if i == 0:  # L2 的解析解
            w_opt = np.array([2, 1]) / (1 + lam)
        else:  # L1 的数值解（简化）
            if i == 1:
                w_opt = np.array([1.5, 0.5])  # 简化的L1解
            else:
                w_opt = np.array([1.8, 0.8])

        fig.add_trace(
            go.Scatter(
                x=[w_opt[0]],
                y=[w_opt[1]],
                mode="markers",
                name=f"最优点 Optimal λ={lam}",
                marker=dict(color="green", size=8, symbol="star"),
                showlegend=False,
            ),
            row=2,
            col=i + 1,
        )

    # 添加公式
    fig = add_formula_annotation(
        fig,
        r"$$\min_{\|\mathbf{w}\| \le C} \text{Loss}(\mathbf{w}) \quad \Leftrightarrow \quad \min_{\mathbf{w}} \left[\text{Loss}(\mathbf{w}) + \lambda \|\mathbf{w}\|\right]$$",
        x=0.5,
        y=1.02,
    )

    # 更新坐标轴
    for i in range(3):
        fig.update_xaxes(title_text="w₁", row=1, col=i + 1)
        fig.update_xaxes(title_text="w₁", row=2, col=i + 1)
        fig.update_yaxes(title_text="w₂", row=1, col=i + 1)
        fig.update_yaxes(title_text="w₂", row=2, col=i + 1)

    fig.update_layout(
        title_text="拉格朗日对偶性：约束形式 ↔ 正则化形式 Lagrange Duality: Constraint ↔ Regularization",
        height=800,
        margin=dict(t=120, b=60, l=60, r=60),
    )

    return fig


fig5 = create_lagrange_duality()
output_file = os.path.join(output_dir, "5_lagrange_duality.html")
fig5.write_html(output_file, include_mathjax="cdn")
print(f"   ✅ 保存: {output_file}")

# ============================================
# 6. 综合仪表板
# ============================================
print("\n6️⃣ 创建综合仪表板...")


def create_comprehensive_dashboard():
    """创建 L1 & L2 正则化综合仪表板"""

    fig = make_subplots(
        rows=2,
        cols=3,
        subplot_titles=(
            "约束形状 Constraint Shapes",
            "权重衰减 Weight Decay",
            "正则化路径 Regularization Path",
            "贝叶斯先验 Bayesian Priors",
            "拉格朗日对偶性 Lagrange Duality",
            "稀疏性对比 Sparsity Comparison",
        ),
        specs=[
            [{"type": "scatter"}, {"type": "scatter"}, {"type": "scatter"}],
            [{"type": "scatter"}, {"type": "scatter"}, {"type": "bar"}],
        ],
    )

    # 1. 约束形状（简化版）
    theta = np.linspace(0, 2 * np.pi, 50)
    circle_x = np.cos(theta)
    circle_y = np.sin(theta)

    fig.add_trace(
        go.Scatter(
            x=circle_x,
            y=circle_y,
            mode="lines",
            name="L2约束 L2 Constraint",
            line=dict(color="red", width=2),
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    diamond_x = [1, 0, -1, 0, 1]
    diamond_y = [0, 1, 0, -1, 0]
    fig.add_trace(
        go.Scatter(
            x=diamond_x,
            y=diamond_y,
            mode="lines",
            name="L1约束 L1 Constraint",
            line=dict(color="blue", width=2),
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    # 2. 权重衰减（简化版）
    steps = np.arange(20)
    l2_decay = 2.0 * (0.9**steps)  # 指数衰减
    l1_decay = np.maximum(2.0 - 0.1 * steps, 0)  # 线性衰减到0

    fig.add_trace(
        go.Scatter(
            x=steps,
            y=l2_decay,
            mode="lines+markers",
            name="L2衰减 L2 Decay",
            line=dict(color="red", width=2),
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    fig.add_trace(
        go.Scatter(
            x=steps,
            y=l1_decay,
            mode="lines+markers",
            name="L1衰减 L1 Decay",
            line=dict(color="blue", width=2),
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    # 3. 正则化路径（简化版）
    alphas = np.logspace(-3, 0, 20)
    for i in range(3):
        l2_coeffs = (i + 1) / (1 + alphas * 10)
        l1_coeffs = np.maximum((i + 1) * (1 - alphas * 2), 0)

        fig.add_trace(
            go.Scatter(
                x=alphas,
                y=l2_coeffs,
                mode="lines",
                name=f"L2 w{i+1}",
                line=dict(color=["red", "orange", "pink"][i], width=2),
                showlegend=False,
            ),
            row=1,
            col=3,
        )

        fig.add_trace(
            go.Scatter(
                x=alphas,
                y=l1_coeffs,
                mode="lines",
                name=f"L1 w{i+1}",
                line=dict(color=["blue", "cyan", "lightblue"][i], width=2),
                showlegend=False,
            ),
            row=1,
            col=3,
        )

    # 4. 贝叶斯先验（简化版）
    x = np.linspace(-3, 3, 100)
    gaussian = np.exp(-(x**2) / 2)
    laplacian = np.exp(-np.abs(x))

    fig.add_trace(
        go.Scatter(
            x=x,
            y=gaussian,
            mode="lines",
            name="高斯先验 Gaussian",
            line=dict(color="red", width=2),
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=x,
            y=laplacian,
            mode="lines",
            name="拉普拉斯先验 Laplacian",
            line=dict(color="blue", width=2),
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    # 5. 拉格朗日对偶性（简化版）
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="markers+lines",
            name="等价性 Equivalence",
            marker=dict(size=8, color="green"),
            line=dict(width=2, color="green"),
            showlegend=False,
            text=["约束形式 Constraint Form", "正则化形式 Regularization Form"],
            hovertemplate="%{text}<extra></extra>",
        ),
        row=2,
        col=2,
    )

    # 6. 稀疏性对比（条形图）
    methods = ["无正则化 No Reg", "L2正则化 L2 Reg", "L1正则化 L1 Reg"]
    non_zero_counts = [10, 10, 4]  # 非零系数数量

    fig.add_trace(
        go.Bar(
            x=methods,
            y=non_zero_counts,
            name="非零系数 Non-zero Coefficients",
            marker=dict(color=["gray", "red", "blue"]),
            showlegend=False,
        ),
        row=2,
        col=3,
    )

    # 更新坐标轴
    for i in range(3):
        fig.update_xaxes(title_text="w₁", row=1, col=i + 1)
        fig.update_yaxes(title_text="w₂", row=1, col=1)
        fig.update_yaxes(title_text="权重 Weight", row=1, col=2)
        fig.update_xaxes(type="log", title_text="λ", row=1, col=3)

        fig.update_xaxes(title_text="权重 Weight", row=2, col=1)
        fig.update_yaxes(title_text="概率密度 Density", row=2, col=1)
        fig.update_xaxes(title_text="形式 Form", row=2, col=2)
        fig.update_yaxes(title_text="等价性 Equivalence", row=2, col=2)
        fig.update_yaxes(title_text="非零系数数量 Non-zero Count", row=2, col=3)

    fig.update_layout(
        title_text="L1 & L2 正则化综合仪表板 Comprehensive Dashboard",
        height=800,
        showlegend=False,
        margin=dict(t=100, b=60, l=60, r=60),
    )

    return fig


fig6 = create_comprehensive_dashboard()
output_file = os.path.join(output_dir, "6_dashboard.html")
fig6.write_html(output_file, include_mathjax="cdn")
print(f"   ✅ 保存: {output_file}")

# ============================================
# 打印计算示例
# ============================================
print("\n" + "=" * 60)
print("📊 L1 & L2 正则化计算示例")
print("=" * 60)

print("\n1️⃣ 权重衰减公式对比:")
print("   L2 正则化:")
print("     w_new = (1 - ηλ) * w_old - η * gradient")
print("     特点：乘法衰减，权重按比例缩小，永不归零")
print("   ")
print("   L1 正则化:")
print("     w_new = w_old - η * gradient - η * λ * sign(w_old)")
print("     特点：减法衰减，权重线性减少，可以归零")

print("\n2️⃣ 贝叶斯解释:")
print("   L2 正则化 ↔ 高斯先验:")
print("     P(w) ∝ exp(-w²/(2σ²))")
print("     温和地相信权重应该小，但不强制为0")
print("   ")
print("   L1 正则化 ↔ 拉普拉斯先验:")
print("     P(w) ∝ exp(-|w|/b)")
print("     强烈地相信权重应该是0，产生稀疏性")

print("\n3️⃣ 拉格朗日对偶性:")
print("   约束形式: min Loss(w)  s.t. ||w|| ≤ C")
print("   正则化形式: min [Loss(w) + λ||w||]")
print("   关系：λ ≈ 1/C，λ越大约束越紧")

print("\n4️⃣ 稀疏性来源:")
print("   L1 在0处有尖角 → 等高线容易先碰到坐标轴 → 部分权重为0")
print("   L2 是平滑的圆形 → 等高线通常不在坐标轴上相切 → 权重都不为0")

print("\n" + "=" * 60)
print("✨ 所有交互式可视化已创建完成！")
print("=" * 60)
print(f"\n📂 生成的文件位于: code/{output_dir}/")
print("   1. 1_constraint_shapes.html - 约束形状对比")
print("   2. 2_weight_decay.html - 权重衰减过程动画")
print("   3. 3_regularization_path.html - 正则化路径")
print("   4. 4_bayesian_priors.html - 贝叶斯先验分布")
print("   5. 5_lagrange_duality.html - 拉格朗日对偶性")
print("   6. 6_dashboard.html - 综合仪表板")
print("\n💡 在浏览器中打开这些 HTML 文件即可查看交互式可视化！")
print("=" * 60)
