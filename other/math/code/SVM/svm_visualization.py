"""
SVM（支持向量机）交互式可视化脚本
基于 6.SVM.md 文档中的公式
使用 Plotly 创建交互式动画图表，包含数学公式
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# 创建输出目录
output_dir = "SVM"
os.makedirs(output_dir, exist_ok=True)

print("=" * 60)
print("🎨 SVM（支持向量机）交互式可视化")
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
# 1. 硬间隔SVM可视化
# ============================================
print("\n1️⃣ 创建硬间隔SVM可视化...")


def create_hard_margin_svm():
    """创建硬间隔SVM的可视化"""

    # 生成示例数据
    np.random.seed(42)

    # 正类（类别1）
    X_pos = np.array(
        [
            [2, 2],
            [3, 2],
            [2.5, 3],
            [3, 3],
            [2, 3],
            [3.5, 2.5],
            [2.5, 2.5],
            [4, 2],
            [2, 4],
        ]
    )

    # 负类（类别-1）
    X_neg = np.array(
        [
            [-2, -2],
            [-3, -2],
            [-2.5, -3],
            [-3, -3],
            [-2, -3],
            [-3.5, -2.5],
            [-2.5, -2.5],
            [-4, -2],
            [-2, -4],
        ]
    )

    # 计算SVM参数（简化版本）
    # 决策边界：w1*x1 + w2*x2 + b = 0
    # 这里我们手动设置一个合理的分割线
    w = np.array([1, 1])  # 法向量
    b = 0  # 偏置

    # 计算间隔边界
    margin = 1 / np.linalg.norm(w)

    # 创建网格
    x1 = np.linspace(-5, 5, 100)
    x2 = np.linspace(-5, 5, 100)
    X1, X2 = np.meshgrid(x1, x2)

    # 计算决策函数值
    Z = w[0] * X1 + w[1] * X2 + b

    # 创建图形
    fig = go.Figure()

    # 添加决策边界等高线
    fig.add_trace(
        go.Contour(
            x=x1,
            y=x2,
            z=Z,
            colorscale="RdBu",
            contours=dict(
                start=-2,
                end=2,
                size=0.5,
                showlabels=True,
                labelfont=dict(size=10, color="white"),
            ),
            colorbar=dict(
                title="决策函数值 Decision Function Value", titleside="right"
            ),
            hoverinfo="skip",
        )
    )

    # 添加正类数据点
    fig.add_trace(
        go.Scatter(
            x=X_pos[:, 0],
            y=X_pos[:, 1],
            mode="markers",
            name="正类 Positive Class",
            marker=dict(color="red", size=10, symbol="circle"),
            text=[f"点{i+1}: ({x[0]:.1f}, {x[1]:.1f})" for i, x in enumerate(X_pos)],
            hovertemplate="%{text}<extra></extra>",
        )
    )

    # 添加负类数据点
    fig.add_trace(
        go.Scatter(
            x=X_neg[:, 0],
            y=X_neg[:, 1],
            mode="markers",
            name="负类 Negative Class",
            marker=dict(color="blue", size=10, symbol="square"),
            text=[f"点{i+1}: ({x[0]:.1f}, {x[1]:.1f})" for i, x in enumerate(X_neg)],
            hovertemplate="%{text}<extra></extra>",
        )
    )

    # 识别支持向量（距离决策边界最近的点）
    all_points = np.vstack([X_pos, X_neg])
    all_labels = np.hstack([np.ones(len(X_pos)), -np.ones(len(X_neg))])

    distances = np.abs(all_points.dot(w) + b) / np.linalg.norm(w)
    support_vector_idx = np.where(np.isclose(distances, margin, atol=0.3))[0]
    support_vectors = all_points[support_vector_idx]
    support_labels = all_labels[support_vector_idx]

    # 标记支持向量
    fig.add_trace(
        go.Scatter(
            x=support_vectors[:, 0],
            y=support_vectors[:, 1],
            mode="markers",
            name="支持向量 Support Vectors",
            marker=dict(
                color="green", size=15, symbol="star", line=dict(width=2, color="black")
            ),
            text=[
                f"支持向量{i+1}<br>({x[0]:.1f}, {x[1]:.1f})"
                for i, x in enumerate(support_vectors)
            ],
            hovertemplate="%{text}<extra></extra>",
        )
    )

    # 添加公式
    fig = add_formula_annotation(
        fig,
        r"$$\min_{\mathbf{w},b} \frac{1}{2}\|\mathbf{w}\|^2 \quad \text{s.t.} \quad y_i(\mathbf{w}^\top\mathbf{x}_i + b) \ge 1$$",
        x=0.5,
        y=1.05,
    )

    fig.update_layout(
        title="硬间隔SVM Hard Margin SVM<br>最大间隔分类器 Maximum Margin Classifier",
        xaxis_title="特征1 Feature 1",
        yaxis_title="特征2 Feature 2",
        height=700,
        legend=dict(x=0.02, y=0.98, bgcolor="rgba(255,255,255,0.8)"),
        margin=dict(t=120, b=60, l=60, r=60),
    )

    return fig, {"w": w, "b": b, "margin": margin, "support_vectors": support_vectors}


fig1, svm_info = create_hard_margin_svm()
output_file = os.path.join(output_dir, "1_hard_margin_svm.html")
fig1.write_html(output_file, include_mathjax="cdn")
print(f"   ✅ 保存: {output_file}")

# ============================================
# 2. 软间隔SVM可视化
# ============================================
print("\n2️⃣ 创建软间隔SVM可视化...")


def create_soft_margin_svm():
    """创建软间隔SVM的可视化，展示不同C值的影响和松弛变量"""

    # 生成包含明显噪声的数据
    np.random.seed(42)

    # 正类（包含异常点）
    X_pos = np.array(
        [
            [2, 2],
            [3, 2],
            [2.5, 3],
            [3, 3],
            [2, 3],
            [3.5, 2.5],
            [2.5, 2.5],
            [4, 2],
            [2, 4],
            [0.3, 0.2],
            [0.5, -0.3],
            [-0.2, 0.4],  # 明显的异常点
        ]
    )

    # 负类（包含异常点）
    X_neg = np.array(
        [
            [-2, -2],
            [-3, -2],
            [-2.5, -3],
            [-3, -3],
            [-2, -3],
            [-3.5, -2.5],
            [-2.5, -2.5],
            [-4, -2],
            [-2, -4],
            [0.2, -0.1],
            [-0.3, 0.3],
            [0.4, -0.2],  # 明显的异常点
        ]
    )

    # 创建不同C值的动画
    C_values = [0.01, 0.1, 1, 10, 100]
    frames = []

    for C in C_values:
        # 根据C值调整决策边界
        if C < 0.1:
            # C很小：非常宽容，间隔很宽，允许很多错误
            w = np.array([0.2, 0.2])
            b = 0
            margin_width = 5.0
        elif C < 1:
            # C较小：比较宽容
            w = np.array([0.4, 0.4])
            b = 0
            margin_width = 2.5
        elif C < 10:
            # C中等：平衡
            w = np.array([0.7, 0.7])
            b = 0
            margin_width = 1.4
        else:
            # C很大：很严格，间隔很窄
            w = np.array([1.0, 1.0])
            b = 0
            margin_width = 1.0

        # 创建网格
        x1 = np.linspace(-5, 5, 50)
        x2 = np.linspace(-5, 5, 50)
        X1, X2 = np.meshgrid(x1, x2)
        Z = w[0] * X1 + w[1] * X2 + b

        # 计算松弛变量
        all_points = np.vstack([X_pos, X_neg])
        all_labels = np.hstack([np.ones(len(X_pos)), -np.ones(len(X_neg))])

        # 计算每个点的违反程度和松弛变量
        margin_violations = 1 - all_labels * (all_points.dot(w) + b)
        slack_variables = np.maximum(0, margin_violations)

        # 区分不同类型的点
        correctly_classified = slack_variables < 0.01
        within_margin = (slack_variables >= 0.01) & (slack_variables < 1)
        misclassified = slack_variables >= 1

        # 创建可视化数据
        frame_data = [
            # 决策边界等高线
            go.Contour(
                x=x1,
                y=x2,
                z=Z,
                colorscale="RdBu",
                contours=dict(
                    start=-2,
                    end=2,
                    size=0.5,
                    showlabels=True,
                    labelfont=dict(size=8, color="white"),
                ),
                showscale=False,
                hoverinfo="skip",
            ),
            # 正确分类的点
            go.Scatter(
                x=all_points[correctly_classified & (all_labels == 1), 0],
                y=all_points[correctly_classified & (all_labels == 1), 1],
                mode="markers",
                name="正确分类 Correctly Classified",
                marker=dict(color="red", size=8),
                showlegend=False,
                text=[
                    f"正确分类<br>松弛变量 ξ: {slack_variables[i]:.3f}"
                    for i in np.where(correctly_classified & (all_labels == 1))[0]
                ],
                hovertemplate="%{text}<extra></extra>",
            ),
            go.Scatter(
                x=all_points[correctly_classified & (all_labels == -1), 0],
                y=all_points[correctly_classified & (all_labels == -1), 1],
                mode="markers",
                name="正确分类 Correctly Classified",
                marker=dict(color="blue", size=8),
                showlegend=False,
                text=[
                    f"正确分类<br>松弛变量 ξ: {slack_variables[i]:.3f}"
                    for i in np.where(correctly_classified & (all_labels == -1))[0]
                ],
                hovertemplate="%{text}<extra></extra>",
            ),
            # 在间隔内但分类正确的点
            go.Scatter(
                x=all_points[within_margin & (all_labels == 1), 0],
                y=all_points[within_margin & (all_labels == 1), 1],
                mode="markers",
                name="间隔内 Within Margin",
                marker=dict(color="orange", size=10, symbol="diamond"),
                showlegend=False,
                text=[
                    f"间隔内<br>松弛变量 ξ: {slack_variables[i]:.3f}"
                    for i in np.where(within_margin & (all_labels == 1))[0]
                ],
                hovertemplate="%{text}<extra></extra>",
            ),
            go.Scatter(
                x=all_points[within_margin & (all_labels == -1), 0],
                y=all_points[within_margin & (all_labels == -1), 1],
                mode="markers",
                name="间隔内 Within Margin",
                marker=dict(color="orange", size=10, symbol="diamond"),
                showlegend=False,
                text=[
                    f"间隔内<br>松弛变量 ξ: {slack_variables[i]:.3f}"
                    for i in np.where(within_margin & (all_labels == -1))[0]
                ],
                hovertemplate="%{text}<extra></extra>",
            ),
            # 错误分类的点
            go.Scatter(
                x=all_points[misclassified & (all_labels == 1), 0],
                y=all_points[misclassified & (all_labels == 1), 1],
                mode="markers",
                name="错误分类 Misclassified",
                marker=dict(color="purple", size=12, symbol="x"),
                line=dict(width=2),
                showlegend=False,
                text=[
                    f"错误分类<br>松弛变量 ξ: {slack_variables[i]:.3f}"
                    for i in np.where(misclassified & (all_labels == 1))[0]
                ],
                hovertemplate="%{text}<extra></extra>",
            ),
            go.Scatter(
                x=all_points[misclassified & (all_labels == -1), 0],
                y=all_points[misclassified & (all_labels == -1), 1],
                mode="markers",
                name="错误分类 Misclassified",
                marker=dict(color="purple", size=12, symbol="x"),
                line=dict(width=2),
                showlegend=False,
                text=[
                    f"错误分类<br>松弛变量 ξ: {slack_variables[i]:.3f}"
                    for i in np.where(misclassified & (all_labels == -1))[0]
                ],
                hovertemplate="%{text}<extra></extra>",
            ),
        ]

        frames.append(
            go.Frame(
                data=frame_data,
                name=str(C),
                layout=go.Layout(
                    title_text=f"软间隔SVM Soft Margin SVM (C={C})<br>"
                    + f"正则化强度 Regularization Strength: {C}<br>"
                    + f"间隔宽度 Margin Width: {margin_width:.2f}<br>"
                    + f"总松弛变量 Total Slack: {np.sum(slack_variables):.2f}<br>"
                    + f"错误分类 Misclassified: {np.sum(misclassified)}"
                ),
            )
        )

    # 创建主图形
    C_medium = 1
    w_medium = np.array([0.7, 0.7])
    b_medium = 0

    x1 = np.linspace(-5, 5, 50)
    x2 = np.linspace(-5, 5, 50)
    X1, X2 = np.meshgrid(x1, x2)
    Z_medium = w_medium[0] * X1 + w_medium[1] * X2 + b_medium

    # 计算中等C值下的松弛变量
    all_points = np.vstack([X_pos, X_neg])
    all_labels = np.hstack([np.ones(len(X_pos)), -np.ones(len(X_neg))])
    margin_violations_medium = 1 - all_labels * (all_points.dot(w_medium) + b_medium)
    slack_variables_medium = np.maximum(0, margin_violations_medium)

    correctly_classified_medium = slack_variables_medium < 0.01
    within_margin_medium = (slack_variables_medium >= 0.01) & (
        slack_variables_medium < 1
    )
    misclassified_medium = slack_variables_medium >= 1

    fig = go.Figure(
        data=[
            go.Contour(
                x=x1,
                y=x2,
                z=Z_medium,
                colorscale="RdBu",
                contours=dict(
                    start=-2,
                    end=2,
                    size=0.5,
                    showlabels=True,
                    labelfont=dict(size=8, color="white"),
                ),
                showscale=False,
                hoverinfo="skip",
            ),
            go.Scatter(
                x=all_points[correctly_classified_medium & (all_labels == 1), 0],
                y=all_points[correctly_classified_medium & (all_labels == 1), 1],
                mode="markers",
                name="正确分类 Correctly Classified",
                marker=dict(color="red", size=8),
            ),
            go.Scatter(
                x=all_points[correctly_classified_medium & (all_labels == -1), 0],
                y=all_points[correctly_classified_medium & (all_labels == -1), 1],
                mode="markers",
                name="正确分类 Correctly Classified",
                marker=dict(color="blue", size=8),
            ),
            go.Scatter(
                x=all_points[within_margin_medium & (all_labels == 1), 0],
                y=all_points[within_margin_medium & (all_labels == 1), 1],
                mode="markers",
                name="间隔内 Within Margin",
                marker=dict(color="orange", size=10, symbol="diamond"),
            ),
            go.Scatter(
                x=all_points[within_margin_medium & (all_labels == -1), 0],
                y=all_points[within_margin_medium & (all_labels == -1), 1],
                mode="markers",
                name="间隔内 Within Margin",
                marker=dict(color="orange", size=10, symbol="diamond"),
            ),
            go.Scatter(
                x=all_points[misclassified_medium & (all_labels == 1), 0],
                y=all_points[misclassified_medium & (all_labels == 1), 1],
                mode="markers",
                name="错误分类 Misclassified",
                marker=dict(color="purple", size=12, symbol="x"),
                line=dict(width=2),
            ),
            go.Scatter(
                x=all_points[misclassified_medium & (all_labels == -1), 0],
                y=all_points[misclassified_medium & (all_labels == -1), 1],
                mode="markers",
                name="错误分类 Misclassified",
                marker=dict(color="purple", size=12, symbol="x"),
                line=dict(width=2),
            ),
        ],
        frames=frames,
    )

    # 添加公式
    fig = add_formula_annotation(
        fig,
        r"$\min_{\mathbf{w},b,\boldsymbol{\xi}} \frac{1}{2}\|\mathbf{w}\|^2 + C\sum_{i}\xi_i \quad \text{s.t.} \quad y_i(\mathbf{w}^\top\mathbf{x}_i + b) \ge 1-\xi_i$",
        x=0.5,
        y=1.05,
    )

    # 添加图例说明
    fig.update_layout(
        annotations=[
            dict(
                x=0.02,
                y=0.98,
                xref="paper",
                yref="paper",
                text="● 红色/蓝色: 正确分类且间隔≥1<br>◆ 橙色: 间隔内 (0<ξ<1)<br>✗ 紫色: 错误分类 (ξ≥1)",
                showarrow=False,
                font=dict(size=10),
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="gray",
                borderwidth=1,
            )
        ]
    )

    # 添加播放按钮
    fig.update_layout(
        title="软间隔SVM Soft Margin SVM - 松弛变量可视化 Slack Variables Visualization",
        xaxis_title="特征1 Feature 1",
        yaxis_title="特征2 Feature 2",
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
                                frame=dict(duration=1500, redraw=True),
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
        height=700,
        legend=dict(x=0.02, y=0.85, bgcolor="rgba(255,255,255,0.8)"),
        margin=dict(t=120, b=60, l=60, r=60),
    )

    return fig


fig2 = create_soft_margin_svm()
output_file = os.path.join(output_dir, "2_soft_margin_svm.html")
fig2.write_html(output_file, include_mathjax="cdn")
print(f"   ✅ 保存: {output_file}")

# ============================================
# 3. 合页损失函数可视化
# ============================================
print("\n3️⃣ 创建合页损失函数可视化...")


def create_hinge_loss():
    """创建合页损失函数的可视化"""

    # 创建x轴（预测分数）
    y_f = np.linspace(-3, 3, 1000)

    # 计算合页损失
    hinge_loss = np.maximum(0, 1 - y_f)

    # 计算其他损失函数作为对比
    zero_one_loss = (y_f < 0).astype(float)  # 0-1损失
    logistic_loss = np.log(1 + np.exp(-y_f))  # 逻辑损失

    # 创建图形
    fig = go.Figure()

    # 添加合页损失
    fig.add_trace(
        go.Scatter(
            x=y_f,
            y=hinge_loss,
            mode="lines",
            name="合页损失 Hinge Loss",
            line=dict(color="red", width=3),
            hovertemplate="预测分数 y·f(x): %{x:.2f}<br>损失 Loss: %{y:.2f}<extra></extra>",
        )
    )

    # 添加0-1损失作为对比
    fig.add_trace(
        go.Scatter(
            x=y_f,
            y=zero_one_loss,
            mode="lines",
            name="0-1损失 0-1 Loss",
            line=dict(color="blue", width=2, dash="dash"),
            hovertemplate="预测分数 y·f(x): %{x:.2f}<br>损失 Loss: %{y:.2f}<extra></extra>",
        )
    )

    # 添加逻辑损失作为对比
    fig.add_trace(
        go.Scatter(
            x=y_f,
            y=logistic_loss,
            mode="lines",
            name="逻辑损失 Logistic Loss",
            line=dict(color="green", width=2, dash="dot"),
            hovertemplate="预测分数 y·f(x): %{x:.2f}<br>损失 Loss: %{y:.2f}<extra></extra>",
        )
    )

    # 标记关键点
    fig.add_trace(
        go.Scatter(
            x=[1, 0],
            y=[0, 1],
            mode="markers",
            name="关键点 Key Points",
            marker=dict(color="orange", size=8, symbol="circle"),
            text=[
                "正确分类边界 Correct Classification Boundary",
                "最大损失点 Maximum Loss",
            ],
            hovertemplate="%{text}<br>(%{x:.1f}, %{y:.1f})<extra></extra>",
        )
    )

    # 添加阴影区域
    fig.add_shape(
        type="rect",
        x0=-3,
        y0=0,
        x1=1,
        y1=0.1,
        fillcolor="lightgreen",
        opacity=0.3,
        layer="below",
        line_width=0,
    )

    # 添加公式
    fig = add_formula_annotation(
        fig,
        r"$$\ell_{\text{hinge}}(y, f(\mathbf{x})) = \max(0, 1 - y \cdot f(\mathbf{x}))$$",
        x=0.5,
        y=1.05,
    )

    fig.update_layout(
        title="合页损失函数 Hinge Loss Function",
        xaxis_title="预测分数 y·f(x) Prediction Score",
        yaxis_title="损失值 Loss Value",
        height=600,
        legend=dict(x=0.02, y=0.98, bgcolor="rgba(255,255,255,0.8)"),
        annotations=[
            dict(
                x=0.5,
                y=0.05,
                text="正确分类且间隔≥1的区域<br>Correctly classified with margin ≥ 1",
                showarrow=False,
                font=dict(size=10, color="green"),
            ),
            dict(
                x=-1.5,
                y=0.5,
                text="错误分类或间隔不足的区域<br>Misclassified or insufficient margin",
                showarrow=False,
                font=dict(size=10, color="red"),
            ),
        ],
        margin=dict(t=120, b=60, l=60, r=60),
    )

    return fig


fig3 = create_hinge_loss()
output_file = os.path.join(output_dir, "3_hinge_loss.html")
fig3.write_html(output_file, include_mathjax="cdn")
print(f"   ✅ 保存: {output_file}")

# ============================================
# 4. 核函数可视化
# ============================================
print("\n4️⃣ 创建核函数可视化...")


def create_kernel_visualization():
    """创建不同核函数的可视化"""

    # 生成示例数据（非线性可分）
    np.random.seed(42)

    # 内圈（类别1）
    theta_inner = np.linspace(0, 2 * np.pi, 20)
    r_inner = 2
    X_inner = np.array(
        [
            [
                r_inner * np.cos(t) + np.random.normal(0, 0.1),
                r_inner * np.sin(t) + np.random.normal(0, 0.1),
            ]
            for t in theta_inner
        ]
    )

    # 外圈（类别-1）
    theta_outer = np.linspace(0, 2 * np.pi, 20)
    r_outer = 4
    X_outer = np.array(
        [
            [
                r_outer * np.cos(t) + np.random.normal(0, 0.1),
                r_outer * np.sin(t) + np.random.normal(0, 0.1),
            ]
            for t in theta_outer
        ]
    )

    # 创建子图
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=(
            "线性核 Linear Kernel",
            "多项式核 Polynomial Kernel (d=2)",
            "RBF核 RBF Kernel (γ=0.5)",
        ),
        specs=[[{"type": "scatter"}, {"type": "scatter"}, {"type": "scatter"}]],
    )

    # 线性核（无法解决）
    fig.add_trace(
        go.Scatter(
            x=X_inner[:, 0],
            y=X_inner[:, 1],
            mode="markers",
            name="内圈 Inner Class",
            marker=dict(color="red", size=8),
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=X_outer[:, 0],
            y=X_outer[:, 1],
            mode="markers",
            name="外圈 Outer Class",
            marker=dict(color="blue", size=8),
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    # 添加线性决策边界（效果不好）
    x_line = np.linspace(-5, 5, 100)
    y_line = np.zeros_like(x_line)
    fig.add_trace(
        go.Scatter(
            x=x_line,
            y=y_line,
            mode="lines",
            name="线性决策边界 Linear Decision Boundary",
            line=dict(color="green", width=2, dash="dash"),
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    # 多项式核（映射到高维空间后的效果）
    # 在原始空间中可视化多项式特征
    xx, yy = np.meshgrid(np.linspace(-5, 5, 50), np.linspace(-5, 5, 50))
    # 简化的多项式特征：x² + y²
    poly_feature = xx**2 + yy**2

    fig.add_trace(
        go.Contour(
            x=xx[0],
            y=yy[:, 0],
            z=poly_feature,
            colorscale="RdBu",
            contours=dict(start=5, end=15, size=2, showlabels=False),
            showscale=False,
            hoverinfo="skip",
        ),
        row=1,
        col=2,
    )

    fig.add_trace(
        go.Scatter(
            x=X_inner[:, 0],
            y=X_inner[:, 1],
            mode="markers",
            marker=dict(color="red", size=8),
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    fig.add_trace(
        go.Scatter(
            x=X_outer[:, 0],
            y=X_outer[:, 1],
            mode="markers",
            marker=dict(color="blue", size=8),
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    # RBF核（高斯径向基函数）
    gamma = 0.5
    # 选择一个中心点来展示RBF核的效果
    center = np.array([0, 0])

    rbf_feature = np.exp(-gamma * ((xx - center[0]) ** 2 + (yy - center[1]) ** 2))

    fig.add_trace(
        go.Contour(
            x=xx[0],
            y=yy[:, 0],
            z=rbf_feature,
            colorscale="Viridis",
            contours=dict(start=0.1, end=1, size=0.1, showlabels=False),
            showscale=False,
            hoverinfo="skip",
        ),
        row=1,
        col=3,
    )

    fig.add_trace(
        go.Scatter(
            x=X_inner[:, 0],
            y=X_inner[:, 1],
            mode="markers",
            marker=dict(color="red", size=8),
            showlegend=False,
        ),
        row=1,
        col=3,
    )

    fig.add_trace(
        go.Scatter(
            x=X_outer[:, 0],
            y=X_outer[:, 1],
            mode="markers",
            marker=dict(color="blue", size=8),
            showlegend=False,
        ),
        row=1,
        col=3,
    )

    # 添加图例
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            name="内圈 Inner Class",
            marker=dict(color="red", size=8),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            name="外圈 Outer Class",
            marker=dict(color="blue", size=8),
        )
    )

    # 添加公式
    fig = add_formula_annotation(
        fig,
        r"$$K(\mathbf{x}_i, \mathbf{x}_j) = \begin{cases} \mathbf{x}_i^\top\mathbf{x}_j & \text{线性} \\ (\gamma\langle\mathbf{x}_i, \mathbf{x}_j\rangle + r)^d & \text{多项式} \\ \exp(-\gamma\|\mathbf{x}_i - \mathbf{x}_j\|^2) & \text{RBF} \end{cases}$$",
        x=0.5,
        y=1.05,
    )

    # 更新坐标轴
    for i in range(3):
        fig.update_xaxes(title_text="特征1 Feature 1", row=1, col=i + 1)
        fig.update_yaxes(title_text="特征2 Feature 2", row=1, col=i + 1)

    fig.update_layout(
        title_text="核函数对比 Kernel Comparison - 非线性可分数据 Non-linearly Separable Data",
        height=600,
        showlegend=True,
        legend=dict(x=1.02, y=1, bgcolor="rgba(255,255,255,0.8)"),
        margin=dict(t=120, b=60, l=60, r=120),
    )

    return fig


fig4 = create_kernel_visualization()
output_file = os.path.join(output_dir, "4_kernel_functions.html")
fig4.write_html(output_file, include_mathjax="cdn")
print(f"   ✅ 保存: {output_file}")

# ============================================
# 5. SVM vs 神经网络对比
# ============================================
print("\n5️⃣ 创建SVM vs 神经网络对比可视化...")


def create_svm_vs_nn():
    """创建SVM与神经网络的对比可视化"""

    # 创建子图
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "SVM: 找最优分界线 Find Optimal Boundary",
            "NN: 学习复杂边界 Learn Complex Boundary",
            "SVM: 只关注支持向量 Focus on Support Vectors",
            "NN: 使用所有数据点 Use All Data Points",
        ),
        specs=[
            [{"type": "scatter"}, {"type": "scatter"}],
            [{"type": "scatter"}, {"type": "scatter"}],
        ],
    )

    # 生成复杂数据
    np.random.seed(42)

    # 月形数据
    n_samples = 100
    X = np.zeros((n_samples * 2, 2))
    y = np.zeros(n_samples * 2, dtype=int)

    # 上半月
    for i in range(n_samples):
        theta = np.random.uniform(0, np.pi)
        r = np.random.uniform(0, 1)
        X[i] = [r * np.cos(theta), r * np.sin(theta)]
        y[i] = 1

    # 下半月
    for i in range(n_samples, 2 * n_samples):
        theta = np.random.uniform(np.pi, 2 * np.pi)
        r = np.random.uniform(0, 1)
        X[i] = [r * np.cos(theta), r * np.sin(theta)]
        y[i] = 0

    X_pos = X[y == 1]
    X_neg = X[y == 0]

    # SVM可视化（简化版本）
    # 这里我们展示SVM的局限性：只能找到线性分界
    fig.add_trace(
        go.Scatter(
            x=X_pos[:, 0],
            y=X_pos[:, 1],
            mode="markers",
            name="类别1 Class 1",
            marker=dict(color="red", size=6),
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=X_neg[:, 0],
            y=X_neg[:, 1],
            mode="markers",
            name="类别0 Class 0",
            marker=dict(color="blue", size=6),
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    # 添加简化的线性分界（SVM用线性核的效果）
    x_line = np.linspace(-1, 1, 100)
    y_line = np.zeros_like(x_line)
    fig.add_trace(
        go.Scatter(
            x=x_line,
            y=y_line,
            mode="lines",
            name="SVM线性分界 SVM Linear Boundary",
            line=dict(color="green", width=2),
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    # 神经网络可视化（可以学习复杂边界）
    fig.add_trace(
        go.Scatter(
            x=X_pos[:, 0],
            y=X_pos[:, 1],
            mode="markers",
            marker=dict(color="red", size=6),
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    fig.add_trace(
        go.Scatter(
            x=X_neg[:, 0],
            y=X_neg[:, 1],
            mode="markers",
            marker=dict(color="blue", size=6),
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    # 添加复杂的非线性分界（神经网络的效果）
    theta_complex = np.linspace(0, 2 * np.pi, 100)
    complex_boundary = 0.5 * np.sin(3 * theta_complex)
    x_complex = 0.7 * np.cos(theta_complex)
    y_complex = 0.7 * np.sin(theta_complex) + complex_boundary * 0.3

    fig.add_trace(
        go.Scatter(
            x=x_complex,
            y=y_complex,
            mode="lines",
            name="NN复杂分界 NN Complex Boundary",
            line=dict(color="purple", width=2),
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    # SVM支持向量可视化
    # 随机选择一些点作为支持向量的示例
    support_indices = np.random.choice(len(X), size=10, replace=False)
    support_points = X[support_indices]
    support_labels = y[support_indices]

    fig.add_trace(
        go.Scatter(
            x=support_points[support_labels == 1, 0],
            y=support_points[support_labels == 1, 1],
            mode="markers",
            name="支持向量 Support Vectors",
            marker=dict(color="green", size=10, symbol="star"),
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=support_points[support_labels == 0, 0],
            y=support_points[support_labels == 0, 1],
            mode="markers",
            marker=dict(color="green", size=10, symbol="star"),
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    # 其他点（非支持向量）
    non_support_indices = np.setdiff1d(np.arange(len(X)), support_indices)
    non_support_points = X[non_support_indices]
    non_support_labels = y[non_support_indices]

    fig.add_trace(
        go.Scatter(
            x=non_support_points[non_support_labels == 1, 0],
            y=non_support_points[non_support_labels == 1, 1],
            mode="markers",
            marker=dict(color="red", size=4, opacity=0.5),
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=non_support_points[non_support_labels == 0, 0],
            y=non_support_points[non_support_labels == 0, 1],
            mode="markers",
            marker=dict(color="blue", size=4, opacity=0.5),
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    # 神经网络使用所有数据点
    fig.add_trace(
        go.Scatter(
            x=X_pos[:, 0],
            y=X_pos[:, 1],
            mode="markers",
            marker=dict(color="red", size=6),
            showlegend=False,
        ),
        row=2,
        col=2,
    )

    fig.add_trace(
        go.Scatter(
            x=X_neg[:, 0],
            y=X_neg[:, 1],
            mode="markers",
            marker=dict(color="blue", size=6),
            showlegend=False,
        ),
        row=2,
        col=2,
    )

    # 添加连接线表示神经网络考虑所有点的关系
    for i in range(0, len(X), 10):  # 每10个点画一条连接线
        for j in range(i + 1, min(i + 5, len(X))):
            if y[i] != y[j]:  # 只连接不同类别的点
                fig.add_trace(
                    go.Scatter(
                        x=[X[i, 0], X[j, 0]],
                        y=[X[i, 1], X[j, 1]],
                        mode="lines",
                        line=dict(color="gray", width=0.5),
                        showlegend=False,
                    ),
                    row=2,
                    col=2,
                )

    # 添加图例
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            name="类别1 Class 1",
            marker=dict(color="red", size=6),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            name="类别0 Class 0",
            marker=dict(color="blue", size=6),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            name="支持向量 Support Vectors",
            marker=dict(color="green", size=10, symbol="star"),
        )
    )

    # 更新坐标轴
    for i in range(2):
        for j in range(2):
            fig.update_xaxes(title_text="特征1 Feature 1", row=i + 1, col=j + 1)
            fig.update_yaxes(title_text="特征2 Feature 2", row=i + 1, col=j + 1)

    fig.update_layout(
        title_text="SVM vs 神经网络对比 SVM vs Neural Network Comparison",
        height=800,
        showlegend=True,
        legend=dict(x=1.02, y=1, bgcolor="rgba(255,255,255,0.8)"),
        margin=dict(t=100, b=60, l=60, r=120),
    )

    return fig


fig5 = create_svm_vs_nn()
output_file = os.path.join(output_dir, "5_svm_vs_nn.html")
fig5.write_html(output_file, include_mathjax="cdn")
print(f"   ✅ 保存: {output_file}")

# ============================================
# 6. 综合仪表板
# ============================================
print("\n6️⃣ 创建综合仪表板...")


def create_comprehensive_dashboard():
    """创建SVM综合仪表板"""

    fig = make_subplots(
        rows=2,
        cols=3,
        subplot_titles=(
            "硬间隔SVM Hard Margin SVM",
            "软间隔SVM Soft Margin SVM",
            "合页损失 Hinge Loss",
            "核函数 Kernel Functions",
            "SVM vs NN",
            "多分类策略 Multi-class Strategy",
        ),
        specs=[
            [{"type": "scatter"}, {"type": "scatter"}, {"type": "scatter"}],
            [{"type": "scatter"}, {"type": "scatter"}, {"type": "scatter"}],
        ],
    )

    # 1. 硬间隔SVM（简化版）
    theta = np.linspace(0, 2 * np.pi, 20)
    inner_points = np.array([[np.cos(t), np.sin(t)] for t in theta])
    outer_points = np.array([[2 * np.cos(t), 2 * np.sin(t)] for t in theta])

    fig.add_trace(
        go.Scatter(
            x=inner_points[:, 0],
            y=inner_points[:, 1],
            mode="markers",
            name="正类 Positive",
            marker=dict(color="red", size=6),
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=outer_points[:, 0],
            y=outer_points[:, 1],
            mode="markers",
            name="负类 Negative",
            marker=dict(color="blue", size=6),
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    # 2. 软间隔SVM（简化版）
    # 添加一个异常点
    fig.add_trace(
        go.Scatter(
            x=[1.5],
            y=[1.5],
            mode="markers",
            name="异常点 Outlier",
            marker=dict(color="orange", size=8, symbol="diamond"),
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    # 3. 合页损失（简化版）
    x_loss = np.linspace(-2, 2, 50)
    y_loss = np.maximum(0, 1 - x_loss)

    fig.add_trace(
        go.Scatter(
            x=x_loss,
            y=y_loss,
            mode="lines",
            name="合页损失 Hinge Loss",
            line=dict(color="red", width=2),
            showlegend=False,
        ),
        row=1,
        col=3,
    )

    # 4. 核函数（简化版）
    fig.add_trace(
        go.Scatter(
            x=[0, 1, -1, 0, 0],
            y=[0, 0, 0, 1, -1],
            mode="markers",
            name="核映射 Kernel Mapping",
            marker=dict(color=["red", "blue", "blue", "red", "red"], size=8),
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    # 5. SVM vs NN（简化版）
    fig.add_trace(
        go.Scatter(
            x=[0, 1, 2],
            y=[0, 1, 0],
            mode="lines+markers",
            name="SVM边界 SVM Boundary",
            line=dict(color="green", width=2),
            marker=dict(size=6),
            showlegend=False,
        ),
        row=2,
        col=2,
    )

    fig.add_trace(
        go.Scatter(
            x=[0, 0.5, 1, 1.5, 2],
            y=[0.2, 0.8, 0.5, 1.2, 0.3],
            mode="lines+markers",
            name="NN边界 NN Boundary",
            line=dict(color="purple", width=2),
            marker=dict(size=6),
            showlegend=False,
        ),
        row=2,
        col=2,
    )

    # 6. 多分类策略（简化版）
    fig.add_trace(
        go.Bar(
            x=["一对多 OvR", "一对一 OvO"],
            y=[3, 6],
            name="分类器数量 Number of Classifiers",
            marker=dict(color=["orange", "cyan"]),
            showlegend=False,
        ),
        row=2,
        col=3,
    )

    # 更新坐标轴
    for i in range(2):
        for j in range(3):
            if i == 1 and j == 2:  # 条形图
                fig.update_xaxes(title_text="策略 Strategy", row=i + 1, col=j + 1)
                fig.update_yaxes(
                    title_text="分类器数量 Classifiers", row=i + 1, col=j + 1
                )
            else:
                fig.update_xaxes(title_text="特征1 Feature 1", row=i + 1, col=j + 1)
                fig.update_yaxes(title_text="特征2 Feature 2", row=i + 1, col=j + 1)

    fig.update_layout(
        title_text="SVM综合仪表板 SVM Comprehensive Dashboard",
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
print("📊 SVM计算示例")
print("=" * 60)

print("\n1️⃣ 硬间隔SVM优化问题:")
print("   目标函数: min (1/2)||w||²")
print("   约束条件: y_i(w·x_i + b) ≥ 1, ∀i")
print("   几何意义: 找到间隔最大的分类超平面")

print("\n2️⃣ 软间隔SVM优化问题:")
print("   目标函数: min (1/2)||w||² + C∑ξ_i")
print("   约束条件: y_i(w·x_i + b) ≥ 1 - ξ_i, ξ_i ≥ 0")
print("   参数C: 控制对错误的容忍程度")

print("\n3️⃣ 合页损失函数:")
print("   ℓ(y, f(x)) = max(0, 1 - y·f(x))")
print("   特点: 正确且间隔≥1时损失为0，否则线性增加")

print("\n4️⃣ 核函数类型:")
print("   线性核: K(x_i, x_j) = x_i·x_j")
print("   多项式核: K(x_i, x_j) = (γx_i·x_j + r)^d")
print("   RBF核: K(x_i, x_j) = exp(-γ||x_i - x_j||²)")

print("\n5️⃣ 多分类策略:")
print("   一对多(OvR): K个分类器，每个区分一类vs其余")
print("   一对一(OvO): K(K-1)/2个分类器，每对类别一个")

print("\n" + "=" * 60)
print("✨ 所有交互式可视化已创建完成！")
print("=" * 60)
print(f"\n📂 生成的文件位于: code/{output_dir}/")
print("   1. 1_hard_margin_svm.html - 硬间隔SVM")
print("   2. 2_soft_margin_svm.html - 软间隔SVM动画")
print("   3. 3_hinge_loss.html - 合页损失函数")
print("   4. 4_kernel_functions.html - 核函数对比")
print("   5. 5_svm_vs_nn.html - SVM vs 神经网络")
print("   6. 6_dashboard.html - 综合仪表板")
print("\n💡 在浏览器中打开这些 HTML 文件即可查看交互式可视化！")
print("=" * 60)
