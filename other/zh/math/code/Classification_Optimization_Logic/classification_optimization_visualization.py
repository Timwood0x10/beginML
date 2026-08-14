"""
分类模型优化逻辑交互式可视化脚本
基于 8.Classification_Optimization.md 文档中的公式
使用 Plotly 创建交互式动画图表，包含数学公式
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# 创建输出目录
output_dir = "Classification_Optimization_Logic"
os.makedirs(output_dir, exist_ok=True)

print("=" * 60)
print("🎨 分类模型优化逻辑交互式可视化")
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
# 1. 最小二乘法可视化
# ============================================
print("\n1️⃣ 创建最小二乘法可视化...")


def create_least_squares_visualization():
    """创建最小二乘法的可视化"""

    # 生成模拟数据
    np.random.seed(42)
    X = np.linspace(-3, 3, 20)
    # 真实关系是 y = 2x + 1 + 噪声
    true_y = 2 * X + 1
    y = true_y + np.random.normal(0, 1, len(X))

    # 创建动画帧 - 展示不同斜率的拟合
    slope_values = np.linspace(0.5, 3.5, 30)
    frames = []

    for i, slope in enumerate(slope_values):
        intercept = 1  # 固定截距
        y_pred = slope * X + intercept

        # 计算平方误差
        mse = np.mean((y - y_pred) ** 2)

        frame_data = [
            go.Scatter(
                x=X,
                y=y,
                mode="markers",
                name="数据点 Data Points",
                marker=dict(size=10, color="blue"),
            ),
            go.Scatter(
                x=X,
                y=y_pred,
                mode="lines",
                name=f"拟合线 Fit Line (slope={slope:.2f})",
                line=dict(color="red", width=3),
            ),
        ]

        # 添加误差线
        for j in range(len(X)):
            frame_data.append(
                go.Scatter(
                    x=[X[j], X[j]],
                    y=[y[j], y_pred[j]],
                    mode="lines",
                    name=f"误差 Error {j+1}",
                    line=dict(color="orange", width=1, dash="dot"),
                    showlegend=False,
                    hovertemplate=f"数据点 {j+1}<br>真实值: {y[j]:.2f}<br>预测值: {y_pred[j]:.2f}<br>误差: {abs(y[j]-y_pred[j]):.2f}<extra></extra>",
                )
            )

        frames.append(
            go.Frame(
                data=frame_data,
                name=str(i),
                layout=go.Layout(
                    title_text=f"最小二乘法 Least Squares Method<br>"
                    + f"斜率 Slope: {slope:.2f}, 截距 Intercept: {intercept}<br>"
                    + f"均方误差 MSE: {mse:.3f}"
                ),
            )
        )

    # 创建主图形
    fig = go.Figure(
        data=[
            go.Scatter(
                x=X,
                y=y,
                mode="markers",
                name="数据点 Data Points",
                marker=dict(size=10, color="blue"),
            ),
            go.Scatter(
                x=X,
                y=slope_values[0] * X + 1,
                mode="lines",
                name="拟合线 Fit Line",
                line=dict(color="red", width=3),
            ),
        ],
        frames=frames,
    )

    # 添加公式
    fig = add_formula_annotation(
        fig,
        r"$$\min_{\theta} \sum_{i=1}^{n} (y_i - f_\theta(x_i))^2 \quad \text{where} \quad f_\theta(x) = w^\top x + b$$",
        x=0.5,
        y=1.05,
    )

    # 添加播放按钮
    fig.update_layout(
        title="最小二乘法：寻找最佳拟合直线 Least Squares: Finding Best Fit Line",
        xaxis_title="输入特征 Input Feature (x)",
        yaxis_title="目标值 Target Value (y)",
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
        height=700,
        margin=dict(t=120, b=60, l=60, r=60),
    )

    return fig


fig1 = create_least_squares_visualization()
output_file = os.path.join(output_dir, "1_least_squares.html")
fig1.write_html(output_file, include_mathjax="cdn")
print(f"   ✅ 保存: {output_file}")

# ============================================
# 2. 最大似然估计可视化
# ============================================
print("\n2️⃣ 创建最大似然估计可视化...")


def create_maximum_likelihood_visualization():
    """创建最大似然估计的可视化"""

    # 生成二分类数据
    np.random.seed(42)

    # 类别1的数据（正态分布，均值=2，标准差=1）
    X_class1 = np.random.normal(2, 1, 50)

    # 类别0的数据（正态分布，均值=-2，标准差=1）
    X_class0 = np.random.normal(-2, 1, 50)

    # 合并数据
    X = np.concatenate([X_class1, X_class0])
    y = np.concatenate([np.ones(50), np.zeros(50)])

    # 创建动画帧 - 展示不同模型参数下的似然函数
    mu_values = np.linspace(-3, 3, 20)
    sigma_values = [0.5, 1.0, 1.5, 2.0]
    frames = []

    frame_count = 0
    for mu in mu_values:
        for sigma in sigma_values:
            # 计算两个类别的概率密度
            x_range = np.linspace(-6, 6, 200)

            # 类别1的概率密度
            pdf_class1 = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(
                -0.5 * ((x_range - mu) / sigma) ** 2
            )

            # 类别0的概率密度（固定）
            pdf_class0 = (1 / (1.0 * np.sqrt(2 * np.pi))) * np.exp(
                -0.5 * ((x_range + 2) / 1.0) ** 2
            )

            # 计算似然
            likelihood_class1 = np.array(
                [pdf_class1[np.argmin(np.abs(x_range - xi))] for xi in X_class1]
            )
            likelihood_class0 = np.array(
                [pdf_class0[np.argmin(np.abs(x_range - xi))] for xi in X_class0]
            )

            total_likelihood = np.prod(likelihood_class1) * np.prod(likelihood_class0)
            log_likelihood = np.sum(np.log(likelihood_class1)) + np.sum(
                np.log(likelihood_class0)
            )

            frame_data = [
                # 类别1的数据点
                go.Scatter(
                    x=X_class1,
                    y=np.zeros(len(X_class1)),
                    mode="markers",
                    name="类别1 Class 1",
                    marker=dict(color="red", size=6),
                    showlegend=False,
                ),
                # 类别0的数据点
                go.Scatter(
                    x=X_class0,
                    y=np.zeros(len(X_class0)),
                    mode="markers",
                    name="类别0 Class 0",
                    marker=dict(color="blue", size=6),
                    showlegend=False,
                ),
                # 类别1的概率密度曲线
                go.Scatter(
                    x=x_range,
                    y=pdf_class1,
                    mode="lines",
                    name="P(x|class=1)",
                    line=dict(color="red", width=3),
                    fill="tonexty",
                    fillcolor="rgba(255, 0, 0, 0.2)",
                    showlegend=False,
                ),
                # 类别0的概率密度曲线
                go.Scatter(
                    x=x_range,
                    y=pdf_class0,
                    mode="lines",
                    name="P(x|class=0)",
                    line=dict(color="blue", width=3),
                    fill="tonexty",
                    fillcolor="rgba(0, 0, 255, 0.2)",
                    showlegend=False,
                ),
            ]

            frames.append(
                go.Frame(
                    data=frame_data,
                    name=str(frame_count),
                    layout=go.Layout(
                        title_text=f"最大似然估计 Maximum Likelihood Estimation<br>"
                        + f"参数 Parameters: μ={mu:.2f}, σ={sigma:.2f}<br>"
                        + f"对数似然 Log Likelihood: {log_likelihood:.2f}"
                    ),
                )
            )
            frame_count += 1

    # 创建主图形
    x_range = np.linspace(-6, 6, 200)
    pdf_class1_init = (1 / (1.0 * np.sqrt(2 * np.pi))) * np.exp(
        -0.5 * ((x_range - 0) / 1.0) ** 2
    )
    pdf_class0_fixed = (1 / (1.0 * np.sqrt(2 * np.pi))) * np.exp(
        -0.5 * ((x_range + 2) / 1.0) ** 2
    )

    fig = go.Figure(
        data=[
            go.Scatter(x=X_class1, y=np.zeros(len(X_class1))),
            go.Scatter(x=X_class0, y=np.zeros(len(X_class0))),
            go.Scatter(
                x=x_range,
                y=pdf_class1_init,
                mode="lines",
                fill="tonexty",
                fillcolor="rgba(255, 0, 0, 0.2)",
                line=dict(color="red", width=3),
            ),
            go.Scatter(
                x=x_range,
                y=pdf_class0_fixed,
                mode="lines",
                fill="tonexty",
                fillcolor="rgba(0, 0, 255, 0.2)",
                line=dict(color="blue", width=3),
            ),
        ],
        frames=frames,
    )

    # 添加公式
    fig = add_formula_annotation(
        fig,
        r"$$\max_{\theta} \prod_{i=1}^{n} P(y_i|x_i,\theta) \quad \Leftrightarrow \quad \min_{\theta} -\sum_{i=1}^{n} \log P(y_i|x_i,\theta)$$",
        x=0.5,
        y=1.05,
    )

    # 添加播放按钮
    fig.update_layout(
        title="最大似然估计：寻找最可能的数据生成模型 MLE: Finding Most Likely Data Generation Model",
        xaxis_title="特征值 Feature Value (x)",
        yaxis_title="概率密度 Probability Density",
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
                                frame=dict(duration=200, redraw=True),
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
        margin=dict(t=120, b=60, l=60, r=60),
    )

    return fig


fig2 = create_maximum_likelihood_visualization()
output_file = os.path.join(output_dir, "2_maximum_likelihood.html")
fig2.write_html(output_file, include_mathjax="cdn")
print(f"   ✅ 保存: {output_file}")

# ============================================
# 3. SVM可视化
# ============================================
print("\n3️⃣ 创建SVM可视化...")


def create_svm_visualization():
    """创建SVM的可视化"""

    # 生成线性可分的数据
    np.random.seed(42)

    # 正类
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

    # 负类
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
    w = np.array([1, 1])  # 法向量
    b = 0  # 偏置

    # 创建网格
    x1 = np.linspace(-5, 5, 100)
    x2 = np.linspace(-5, 5, 100)
    X1, X2 = np.meshgrid(x1, x2)

    # 计算决策函数值
    Z = w[0] * X1 + w[1] * X2 + b

    # 计算间隔
    margin = 1 / np.linalg.norm(w)

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
                labelfont=dict(size=12, color="white"),
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
        )
    )

    # 识别支持向量（距离决策边界最近的点）
    all_points = np.vstack([X_pos, X_neg])
    all_labels = np.hstack([np.ones(len(X_pos)), -np.ones(len(X_neg))])

    distances = np.abs(all_points.dot(w) + b) / np.linalg.norm(w)
    support_vector_idx = np.where(np.isclose(distances, margin, atol=0.3))[0]
    support_vectors = all_points[support_vector_idx]

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
        r"$$\min_{w,b} \frac{1}{2}\|w\|^2 \quad \text{s.t.} \quad y_i(w^\top x_i + b) \ge 1$$",
        x=0.5,
        y=1.05,
    )

    fig.update_layout(
        title="SVM: 最大间隔分类器 Maximum Margin Classifier",
        xaxis_title="特征1 Feature 1",
        yaxis_title="特征2 Feature 2",
        height=700,
        legend=dict(x=0.02, y=0.98, bgcolor="rgba(255,255,255,0.8)"),
        margin=dict(t=120, b=60, l=60, r=60),
    )

    return fig


fig3 = create_svm_visualization()
output_file = os.path.join(output_dir, "3_svm.html")
fig3.write_html(output_file, include_mathjax="cdn")
print(f"   ✅ 保存: {output_file}")

# ============================================
# 4. 三种方法对比
# ============================================
print("\n4️⃣ 创建三种方法对比可视化...")


def create_three_methods_comparison():
    """创建三种分类方法的对比可视化"""

    # 生成测试数据
    np.random.seed(42)

    # 简单的线性可分数据
    X = np.array(
        [
            [1, 1],
            [2, 1],
            [3, 2],
            [2, 3],
            [3, 3],
            [4, 2],
            [-1, -1],
            [-2, -1],
            [-3, -2],
            [-2, -3],
            [-3, -3],
            [-4, -2],
        ]
    )
    y = np.array([1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1])

    # 创建子图
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=(
            "最小二乘法 Least Squares",
            "最大似然估计 Maximum Likelihood",
            "SVM 支持向量机",
        ),
        specs=[[{"type": "scatter"}, {"type": "scatter"}, {"type": "scatter"}]],
    )

    # 最小二乘法可视化
    # 拟合线性模型
    X_with_bias = np.column_stack([X, np.ones(len(X))])
    w_ls = np.linalg.lstsq(X_with_bias, y, rcond=None)[0]

    x1_range = np.linspace(-5, 5, 50)
    x2_range = np.linspace(-5, 5, 50)
    X1_grid, X2_grid = np.meshgrid(x1_range, x2_range)
    X_grid = np.column_stack(
        [X1_grid.flatten(), X2_grid.flatten(), np.ones(len(X1_grid.flatten()))]
    )

    Z_ls = X_grid.dot(w_ls)
    Z_ls = Z_ls.reshape(X1_grid.shape)

    fig.add_trace(
        go.Contour(
            x=x1_range,
            y=x2_range,
            z=Z_ls,
            colorscale="RdBu",
            contours=dict(
                start=-3,
                end=3,
                size=0.5,
                showlabels=True,
                labelfont=dict(size=8, color="white"),
            ),
            showscale=False,
            hoverinfo="skip",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=X[y == 1, 0],
            y=X[y == 1, 1],
            mode="markers",
            name="正类 Positive",
            marker=dict(color="red", size=8),
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=X[y == -1, 0],
            y=X[y == -1, 1],
            mode="markers",
            name="负类 Negative",
            marker=dict(color="blue", size=8),
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    # 最大似然估计可视化（简化版）
    # 使用简单的线性分类器近似
    x1_range = np.linspace(-5, 5, 50)
    x2_range = np.linspace(-5, 5, 50)
    X1_grid, X2_grid = np.meshgrid(x1_range, x2_range)

    # 简化的概率边界
    Z_lr = X1_grid + X2_grid  # 简化的决策边界

    fig.add_trace(
        go.Contour(
            x=x1_range,
            y=x2_range,
            z=Z_lr,
            colorscale="RdBu",
            contours=dict(
                start=-4,
                end=4,
                size=0.5,
                showlabels=True,
                labelfont=dict(size=8, color="white"),
            ),
            showscale=False,
            hoverinfo="skip",
        ),
        row=1,
        col=2,
    )

    fig.add_trace(
        go.Scatter(
            x=X[y == 1, 0],
            y=X[y == 1, 1],
            mode="markers",
            marker=dict(color="red", size=8),
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    fig.add_trace(
        go.Scatter(
            x=X[y == -1, 0],
            y=X[y == -1, 1],
            mode="markers",
            marker=dict(color="blue", size=8),
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    # SVM可视化（复用之前的函数）
    w_svm = np.array([0.7, 0.7])  # 调整以适应数据
    b_svm = 0
    Z_svm = w_svm[0] * X1_grid + w_svm[1] * X2_grid + b_svm

    fig.add_trace(
        go.Contour(
            x=x1_range,
            y=x2_range,
            z=Z_svm,
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
        row=1,
        col=3,
    )

    fig.add_trace(
        go.Scatter(
            x=X[y == 1, 0],
            y=X[y == 1, 1],
            mode="markers",
            marker=dict(color="red", size=8),
            showlegend=False,
        ),
        row=1,
        col=3,
    )

    fig.add_trace(
        go.Scatter(
            x=X[y == -1, 0],
            y=X[y == -1, 1],
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
            name="正类 Positive Class",
            marker=dict(color="red", size=8),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            name="负类 Negative Class",
            marker=dict(color="blue", size=8),
        )
    )

    # 更新坐标轴
    for i in range(3):
        fig.update_xaxes(title_text="特征1 Feature 1", row=1, col=i + 1)
        fig.update_yaxes(title_text="特征2 Feature 2", row=1, col=i + 1)

    fig.update_layout(
        title_text="三种分类方法对比 Three Classification Methods Comparison",
        height=600,
        showlegend=True,
        legend=dict(x=1.02, y=1, bgcolor="rgba(255,255,255,0.8)"),
        margin=dict(t=100, b=60, l=60, r=120),
    )

    return fig


fig4 = create_three_methods_comparison()
output_file = os.path.join(output_dir, "4_methods_comparison.html")
fig4.write_html(output_file, include_mathjax="cdn")
print(f"   ✅ 保存: {output_file}")

# ============================================
# 5. 损失函数对比
# ============================================
print("\n5️⃣ 创建损失函数对比可视化...")


def create_loss_functions_comparison():
    """创建三种损失函数的对比可视化"""

    # 创建x轴（预测分数）
    y_f = np.linspace(-3, 3, 1000)

    # 计算不同的损失函数
    # 1. 均方误差损失（回归）
    mse_loss = (y_f - 1) ** 2  # 假设真实标签为1

    # 2. 交叉熵损失（分类）
    def binary_cross_entropy(y_true, y_pred):
        return -y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred)

    ce_loss_class1 = binary_cross_entropy(1, 1 / (1 + np.exp(-y_f)))
    ce_loss_class0 = binary_cross_entropy(0, 1 / (1 + np.exp(y_f)))
    ce_loss = ce_loss_class1 + ce_loss_class0

    # 3. 合页损失（SVM）
    hinge_loss = np.maximum(0, 1 - y_f)

    # 创建图形
    fig = go.Figure()

    # 添加平方误差损失
    fig.add_trace(
        go.Scatter(
            x=y_f,
            y=mse_loss,
            mode="lines",
            name="平方误差 MSE",
            line=dict(color="red", width=3),
            hovertemplate="预测分数 y·f(x): %{x:.2f}<br>损失: %{y:.2f}<extra></extra>",
        )
    )

    # 添加交叉熵损失
    fig.add_trace(
        go.Scatter(
            x=y_f,
            y=ce_loss,
            mode="lines",
            name="交叉熵 Cross-Entropy",
            line=dict(color="blue", width=3),
            hovertemplate="预测分数 y·f(x): %{x:.2f}<br>损失: %{y:.2f}<extra></extra>",
        )
    )

    # 添加合页损失
    fig.add_trace(
        go.Scatter(
            x=y_f,
            y=hinge_loss,
            mode="lines",
            name="合页损失 Hinge",
            line=dict(color="green", width=3),
            hovertemplate="预测分数 y·f(x): %{x:.2f}<br>损失: %{y:.2f}<extra></extra>",
        )
    )

    # 标记关键点
    fig.add_trace(
        go.Scatter(
            x=[1, 0, -1],
            y=[0, 1, 4],
            mode="markers",
            name="关键点 Key Points",
            marker=dict(color="orange", size=8, symbol="circle"),
            text=["正确分类边界", "最大损失点", "错误分类"],
            hovertemplate="%{text}<br>(%{x:.1f}, %{y:.1f})<extra></extra>",
        )
    )

    # 添加公式
    fig = add_formula_annotation(
        fig,
        r"$$\begin{aligned} \text{MSE: } (y-f(x))^2 \quad \text{CE: } -\log(\sigma(y \cdot f(x))) \quad \text{Hinge: } \max(0, 1-y \cdot f(x)) \end{aligned}$$",
        x=0.5,
        y=1.05,
    )

    fig.update_layout(
        title="损失函数对比 Loss Functions Comparison",
        xaxis_title="预测分数 y·f(x) Prediction Score",
        yaxis_title="损失值 Loss Value",
        height=600,
        legend=dict(x=0.02, y=0.98, bgcolor="rgba(255,255,255,0.8)"),
        annotations=[
            dict(
                x=1,
                y=0.5,
                text="正确分类且置信度高<br>Correct & Confident",
                showarrow=False,
                font=dict(size=10, color="green"),
            ),
            dict(
                x=-1.5,
                y=2,
                text="错误分类<br>Misclassified",
                showarrow=False,
                font=dict(size=10, color="red"),
            ),
        ],
        margin=dict(t=120, b=60, l=60, r=60),
    )

    return fig


fig5 = create_loss_functions_comparison()
output_file = os.path.join(output_dir, "5_loss_functions.html")
fig5.write_html(output_file, include_mathjax="cdn")
print(f"   ✅ 保存: {output_file}")

# ============================================
# 6. 综合仪表板
# ============================================
print("\n6️⃣ 创建综合仪表板...")


def create_comprehensive_dashboard():
    """创建分类模型优化逻辑综合仪表板"""

    fig = make_subplots(
        rows=2,
        cols=3,
        subplot_titles=(
            "最小二乘法 Least Squares",
            "最大似然估计 Maximum Likelihood",
            "SVM 支持向量机",
            "损失函数对比 Loss Functions",
            "几何解释 Geometry Interpretation",
            "方法特点对比 Method Characteristics",
        ),
        specs=[
            [{"type": "scatter"}, {"type": "scatter"}, {"type": "scatter"}],
            [{"type": "scatter"}, {"type": "scatter"}, {"type": "bar"}],
        ],
    )

    # 1. 最小二乘法（简化版）
    x_ls = np.linspace(-3, 3, 20)
    y_ls = 2 * x_ls + 1 + np.random.normal(0, 0.5, 20)

    fig.add_trace(
        go.Scatter(
            x=x_ls,
            y=y_ls,
            mode="markers",
            name="数据点 Data Points",
            marker=dict(color="blue", size=6),
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    # 拟合线
    fig.add_trace(
        go.Scatter(
            x=x_ls,
            y=2 * x_ls + 1,
            mode="lines",
            name="拟合线 Fit Line",
            line=dict(color="red", width=2),
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    # 2. 最大似然估计（简化版）
    x_mle = np.random.normal(2, 1, 30)

    fig.add_trace(
        go.Histogram(
            x=x_mle,
            nbinsx=20,
            name="概率分布 Probability Distribution",
            marker=dict(color="blue", opacity=0.7),
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    # 3. SVM（简化版）
    x_svm = np.array([1, 2, 3, 4, -1, -2, -3, -4])
    y_svm = np.array([1, 1, 1, 1, -1, -1, -1, -1])

    fig.add_trace(
        go.Scatter(
            x=x_svm[y_svm == 1],
            y=np.zeros(np.sum(y_svm == 1)),
            mode="markers",
            name="正类 Positive",
            marker=dict(color="red", size=8),
            showlegend=False,
        ),
        row=1,
        col=3,
    )

    fig.add_trace(
        go.Scatter(
            x=x_svm[y_svm == -1],
            y=np.zeros(np.sum(y_svm == -1)),
            mode="markers",
            name="负类 Negative",
            marker=dict(color="blue", size=8),
            showlegend=False,
        ),
        row=1,
        col=3,
    )

    # 添加分隔线
    fig.add_trace(
        go.Scatter(
            x=[0, 0],
            y=[-1, 1],
            mode="lines",
            name="决策边界 Decision Boundary",
            line=dict(color="green", width=2, dash="dash"),
            showlegend=False,
        ),
        row=1,
        col=3,
    )

    # 4. 损失函数对比（简化版）
    loss_x = np.linspace(-2, 2, 50)

    fig.add_trace(
        go.Scatter(
            x=loss_x,
            y=(loss_x - 1) ** 2,
            mode="lines",
            name="MSE",
            line=dict(color="red", width=2),
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=loss_x,
            y=-np.log(1 / (1 + np.exp(-loss_x))),
            mode="lines",
            name="Cross-Entropy",
            line=dict(color="blue", width=2),
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=loss_x,
            y=np.maximum(0, 1 - loss_x),
            mode="lines",
            name="Hinge",
            line=dict(color="green", width=2),
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    # 5. 几何解释
    geometry_x = np.linspace(-3, 3, 50)

    fig.add_trace(
        go.Scatter(
            x=geometry_x,
            y=geometry_x,
            mode="lines",
            name="数值拟合 Numeric Fitting",
            line=dict(color="red", width=2),
            showlegend=False,
        ),
        row=2,
        col=2,
    )

    fig.add_trace(
        go.Scatter(
            x=geometry_x,
            y=1 / (1 + np.exp(-geometry_x)),
            mode="lines",
            name="概率转换 Probability Transform",
            line=dict(color="blue", width=2),
            showlegend=False,
        ),
        row=2,
        col=2,
    )

    fig.add_trace(
        go.Scatter(
            x=geometry_x,
            y=geometry_x,
            mode="lines",
            name="几何距离 Geometric Distance",
            line=dict(color="green", width=2),
            showlegend=False,
        ),
        row=2,
        col=2,
    )

    # 6. 方法特点对比（条形图）
    methods = ["最小二乘法", "最大似然", "SVM"]
    characteristics = ["数值驱动", "概率驱动", "几何驱动"]
    interpretability = ["中等", "高", "中等"]

    fig.add_trace(
        go.Bar(
            x=methods,
            y=[3, 3, 3],
            name="特点数量 Feature Count",
            marker=dict(color=["red", "blue", "green"]),
            showlegend=False,
        ),
        row=2,
        col=3,
    )

    # 更新坐标轴
    for i in range(2):
        for j in range(3):
            fig.update_xaxes(title_text="值 Value", row=i + 1, col=j + 1)
            fig.update_yaxes(title_text="值 Value", row=i + 1, col=j + 1)

    fig.update_layout(
        title_text="分类模型优化逻辑综合仪表板 Classification Optimization Logic Dashboard",
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
print("📊 分类模型优化逻辑计算示例")
print("=" * 60)

print("\n1️⃣ 最小二乘法:")
print("   目标: min Σ(y_i - f(x_i))²")
print("   优点: 简单直观，计算快速")
print("   缺点: 对异常值敏感，不适合分类问题")

print("\n2️⃣ 最大似然估计:")
print("   目标: max ∏P(y_i|x_i,θ)")
print("   优点: 概率解释，适合分类")
print("   缺点: 需要假设数据分布")

print("\n3️⃣ 支持向量机:")
print("   目标: max 间隔，s.t. y_i(w·x_i + b) ≥ 1")
print("   优点: 泛化能力强，只关心支持向量")
print("   缺点: 计算复杂，参数调优困难")

print("\n4️⃣ 三种方法的本质区别:")
print("   - 最小二乘法: 数值拟合，最小化平方误差")
print("   - 最大似然: 概率驱动，最大化数据似然")
print("   - SVM: 几何驱动，最大化分类间隔")

print("\n" + "=" * 60)
print("✨ 所有交互式可视化已创建完成！")
print("=" * 60)
print(f"\n📂 生成的文件位于: code/{output_dir}/")
print("   1. 1_least_squares.html - 最小二乘法动画")
print("   2. 2_maximum_likelihood.html - 最大似然估计动画")
print("   3. 3_svm.html - SVM支持向量机")
print("   4. 4_methods_comparison.html - 三种方法对比")
print("   5. 5_loss_functions.html - 损失函数对比")
print("   6. 6_dashboard.html - 综合仪表板")
print("\n💡 在浏览器中打开这些 HTML 文件即可查看交互式可视化！")
print("=" * 60)
