"""
损失函数交互式可视化脚本
基于 2.lossfunction.md 文档中的公式
使用 Plotly 创建交互式动画图表，包含数学公式
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# 创建输出目录
output_dir = "lossfunction"
os.makedirs(output_dir, exist_ok=True)

print("=" * 60)
print("🎨 损失函数交互式可视化")
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
# 1. 最小二乘法 - 线性回归动画
# ============================================
print("\n1️⃣ 创建最小二乘法动画...")

# 生成模拟数据
np.random.seed(42)
x_data = np.linspace(0, 10, 20)
y_true = 2 * x_data + 3 + np.random.randn(20) * 2

# 创建动画帧 - 展示不同斜率的拟合
a_values = np.linspace(0.5, 3.5, 30)
b_optimal = 3

frames = []
for i, a in enumerate(a_values):
    y_pred = a * x_data + b_optimal
    mse = np.mean((y_true - y_pred) ** 2)

    frame_data = [
        go.Scatter(
            x=x_data,
            y=y_true,
            mode="markers",
            name="数据点",
            marker=dict(size=10, color="blue"),
        ),
        go.Scatter(
            x=x_data,
            y=y_pred,
            mode="lines",
            name=f"拟合线 (a={a:.2f})",
            line=dict(color="red", width=3),
        ),
    ]

    frames.append(
        go.Frame(
            data=frame_data,
            name=str(i),
            layout=go.Layout(
                title_text=f"最小二乘法: y = {a:.2f}x + {b_optimal:.2f}<br>MSE = {mse:.2f}"
            ),
        )
    )

# 初始帧
fig1 = go.Figure(
    data=[
        go.Scatter(
            x=x_data,
            y=y_true,
            mode="markers",
            name="数据点",
            marker=dict(size=10, color="blue"),
        ),
        go.Scatter(
            x=x_data,
            y=a_values[0] * x_data + b_optimal,
            mode="lines",
            name="拟合线",
            line=dict(color="red", width=3),
        ),
    ],
    frames=frames,
)

# 添加公式
fig1 = add_formula_annotation(
    fig1,
    r"$$L(a, b) = \sum_{i=1}^{n} (y_i - (ax_i + b))^2 \quad \Rightarrow \quad \min_{a,b} L(a,b)$$",
    x=0.5,
    y=1.05,
)

# 添加播放按钮
fig1.update_layout(
    title="最小二乘法：寻找最佳拟合直线",
    xaxis_title="x",
    yaxis_title="y",
    hovermode="closest",
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
                            frame=dict(duration=100, redraw=True),
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
    height=700,
    margin=dict(t=120, b=60, l=60, r=60),
)

output_file = os.path.join(output_dir, "1_least_squares.html")
fig1.write_html(output_file, include_mathjax="cdn")
print(f"   ✅ 保存: {output_file}")

# ============================================
# 2. 交叉熵损失函数 - 双图对比
# ============================================
print("\n2️⃣ 创建交叉熵损失可视化...")

p = np.linspace(0.001, 0.999, 100)

# 创建子图
fig2 = make_subplots(
    rows=1,
    cols=2,
    subplot_titles=("真值 y=1 时的交叉熵", "真值 y=0 时的交叉熵"),
    specs=[[{"type": "scatter"}, {"type": "scatter"}]],
)

# y=1 的情况
ce_y1 = -np.log(p)
fig2.add_trace(
    go.Scatter(
        x=p,
        y=ce_y1,
        mode="lines",
        name="y=1: -log(p)",
        line=dict(color="red", width=3),
        hovertemplate="预测概率: %{x:.3f}<br>交叉熵: %{y:.3f}",
    ),
    row=1,
    col=1,
)

# y=0 的情况
ce_y0 = -np.log(1 - p)
fig2.add_trace(
    go.Scatter(
        x=p,
        y=ce_y0,
        mode="lines",
        name="y=0: -log(1-p)",
        line=dict(color="blue", width=3),
        hovertemplate="预测概率: %{x:.3f}<br>交叉熵: %{y:.3f}",
    ),
    row=1,
    col=2,
)

fig2.update_xaxes(title_text="预测概率 p", row=1, col=1)
fig2.update_xaxes(title_text="预测概率 p", row=1, col=2)
fig2.update_yaxes(title_text="交叉熵损失", row=1, col=1)
fig2.update_yaxes(title_text="交叉熵损失", row=1, col=2)

# 添加公式
fig2 = add_formula_annotation(
    fig2, r"$$L = -[y \log(p) + (1-y) \log(1-p)]$$", x=0.5, y=1.05
)

fig2.update_layout(
    title_text="二分类交叉熵损失函数",
    height=600,
    showlegend=True,
    margin=dict(t=120, b=60, l=60, r=60),
)

output_file = os.path.join(output_dir, "2_cross_entropy.html")
fig2.write_html(output_file, include_mathjax="cdn")
print(f"   ✅ 保存: {output_file}")

# ============================================
# 3. 交叉熵惩罚强度动画
# ============================================
print("\n3️⃣ 创建交叉熵惩罚强度动画...")

# 创建动画展示不同预测概率的惩罚
p_test_values = np.linspace(0.1, 0.9, 20)
frames_ce = []

for i, p_test in enumerate(p_test_values):
    ce_val = -np.log(p_test)

    # 创建条形图显示惩罚强度
    frame_data = [
        go.Bar(
            x=[
                "预测准确<br>(p=0.9)",
                f"当前预测<br>(p={p_test:.2f})",
                "预测错误<br>(p=0.1)",
            ],
            y=[-np.log(0.9), ce_val, -np.log(0.1)],
            marker=dict(color=["green", "orange", "red"]),
            text=[f"{-np.log(0.9):.2f}", f"{ce_val:.2f}", f"{-np.log(0.1):.2f}"],
            textposition="outside",
            hovertemplate="交叉熵损失: %{y:.3f}",
        )
    ]

    frames_ce.append(go.Frame(data=frame_data, name=str(i)))

fig3 = go.Figure(
    data=[
        go.Bar(
            x=["预测准确<br>(p=0.9)", "当前预测<br>(p=0.5)", "预测错误<br>(p=0.1)"],
            y=[-np.log(0.9), -np.log(0.5), -np.log(0.1)],
            marker=dict(color=["green", "orange", "red"]),
            text=[f"{-np.log(0.9):.2f}", f"{-np.log(0.5):.2f}", f"{-np.log(0.1):.2f}"],
            textposition="outside",
            hovertemplate="交叉熵损失: %{y:.3f}",
        )
    ],
    frames=frames_ce,
)

# 添加公式
fig3 = add_formula_annotation(
    fig3, r"$$L = -\log(p) \quad \text{when } y=1$$", x=0.5, y=1.05
)

fig3.update_layout(
    title="交叉熵惩罚强度对比 (真值 y=1)",
    yaxis_title="交叉熵损失",
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

output_file = os.path.join(output_dir, "3_penalty_animation.html")
fig3.write_html(output_file, include_mathjax="cdn")
print(f"   ✅ 保存: {output_file}")

# ============================================
# 4. Softmax 函数 3D 可视化
# ============================================
print("\n4️⃣ 创建 Softmax 函数 3D 可视化...")

# 创建网格
z1 = np.linspace(-3, 3, 50)
z2 = np.linspace(-3, 3, 50)
Z1, Z2 = np.meshgrid(z1, z2)

# 假设第三个 logit 固定为 0
z3 = 0
# 计算 softmax
exp_z1 = np.exp(Z1)
exp_z2 = np.exp(Z2)
exp_z3 = np.exp(z3)
softmax_1 = exp_z1 / (exp_z1 + exp_z2 + exp_z3)

fig4 = go.Figure(
    data=[
        go.Surface(
            x=Z1,
            y=Z2,
            z=softmax_1,
            colorscale="Viridis",
            hovertemplate="z₁: %{x:.2f}<br>z₂: %{y:.2f}<br>Softmax(z₁): %{z:.3f}",
            colorbar=dict(title="概率"),
        )
    ]
)

# 添加公式
fig4 = add_formula_annotation(
    fig4,
    r"$$\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}$$",
    x=0.5,
    y=0.98,
)

fig4.update_layout(
    title="Softmax 函数 3D 可视化 (z₃=0)",
    scene=dict(
        xaxis_title="logit z₁",
        yaxis_title="logit z₂",
        zaxis_title="Softmax(z₁)",
        camera=dict(eye=dict(x=1.5, y=1.5, z=1.3)),
    ),
    height=750,
    margin=dict(t=100, b=60, l=60, r=60),
)

output_file = os.path.join(output_dir, "4_softmax_3d.html")
fig4.write_html(output_file, include_mathjax="cdn")
print(f"   ✅ 保存: {output_file}")

# ============================================
# 5. 损失函数对比 - 多个损失函数的交互式对比
# ============================================
print("\n5️⃣ 创建多种损失函数对比...")

predictions = np.linspace(-3, 3, 200)
true_value = 0

# 计算不同损失函数
squared_error = (predictions - true_value) ** 2
absolute_error = np.abs(predictions - true_value)
huber_delta = 1.0
huber_loss = np.where(
    np.abs(predictions - true_value) <= huber_delta,
    0.5 * (predictions - true_value) ** 2,
    huber_delta * (np.abs(predictions - true_value) - 0.5 * huber_delta),
)

fig5 = go.Figure()

fig5.add_trace(
    go.Scatter(
        x=predictions,
        y=squared_error,
        mode="lines",
        name="平方误差 (MSE)",
        line=dict(width=3, color="red"),
        visible=True,
    )
)

fig5.add_trace(
    go.Scatter(
        x=predictions,
        y=absolute_error,
        mode="lines",
        name="绝对误差 (MAE)",
        line=dict(width=3, color="blue"),
        visible=True,
    )
)

fig5.add_trace(
    go.Scatter(
        x=predictions,
        y=huber_loss,
        mode="lines",
        name="Huber Loss",
        line=dict(width=3, color="green"),
        visible=True,
    )
)

# 添加公式
fig5 = add_formula_annotation(
    fig5,
    r"$$\text{MSE: } (y-\hat{y})^2 \quad | \quad \text{MAE: } |y-\hat{y}| \quad | \quad \text{Huber: } \begin{cases} \frac{1}{2}(y-\hat{y})^2 & |y-\hat{y}| \leq \delta \\ \delta(|y-\hat{y}| - \frac{1}{2}\delta) & \text{otherwise} \end{cases}$$",
    x=0.5,
    y=1.08,
)

fig5.update_layout(
    title="回归损失函数对比 (真值 = 0)",
    xaxis_title="预测值",
    yaxis_title="损失",
    hovermode="x unified",
    height=700,
    legend=dict(x=0.02, y=0.98, bgcolor="rgba(255,255,255,0.8)"),
    margin=dict(t=140, b=60, l=60, r=60),
)

output_file = os.path.join(output_dir, "5_comparison.html")
fig5.write_html(output_file, include_mathjax="cdn")
print(f"   ✅ 保存: {output_file}")

# ============================================
# 6. 梯度下降优化过程动画
# ============================================
print("\n6️⃣ 创建梯度下降优化动画...")


# 定义一个简单的损失函数 L(a) = (a-2)^2
def loss_func(a):
    return (a - 2) ** 2


def gradient(a):
    return 2 * (a - 2)


# 梯度下降过程
a_init = 5.0
learning_rate = 0.1
iterations = 50

a_history = [a_init]
loss_history = [loss_func(a_init)]

a_current = a_init
for _ in range(iterations):
    grad = gradient(a_current)
    a_current = a_current - learning_rate * grad
    a_history.append(a_current)
    loss_history.append(loss_func(a_current))

# 创建动画
a_range = np.linspace(0, 6, 200)
loss_range = loss_func(a_range)

frames_gd = []
for i in range(len(a_history)):
    frame_data = [
        go.Scatter(
            x=a_range,
            y=loss_range,
            mode="lines",
            name="损失函数",
            line=dict(color="blue", width=2),
        ),
        go.Scatter(
            x=a_history[: i + 1],
            y=loss_history[: i + 1],
            mode="lines+markers",
            name="优化路径",
            line=dict(color="red", width=2),
            marker=dict(size=8, color="red"),
        ),
    ]
    frames_gd.append(
        go.Frame(
            data=frame_data,
            name=str(i),
            layout=go.Layout(
                title_text=f"梯度下降优化 - 迭代 {i}<br>a = {a_history[i]:.4f}, L = {loss_history[i]:.4f}"
            ),
        )
    )

fig6 = go.Figure(
    data=[
        go.Scatter(
            x=a_range,
            y=loss_range,
            mode="lines",
            name="损失函数",
            line=dict(color="blue", width=2),
        ),
        go.Scatter(
            x=[a_init],
            y=[loss_func(a_init)],
            mode="markers",
            name="起始点",
            marker=dict(size=10, color="red"),
        ),
    ],
    frames=frames_gd,
)

# 添加公式
fig6 = add_formula_annotation(
    fig6,
    r"$$a_{t+1} = a_t - \eta \frac{\partial L}{\partial a} \quad \text{where } L(a) = (a-2)^2, \quad \frac{\partial L}{\partial a} = 2(a-2)$$",
    x=0.5,
    y=1.05,
)

fig6.update_layout(
    title="梯度下降优化过程 L(a) = (a-2)²",
    xaxis_title="参数 a",
    yaxis_title="损失 L(a)",
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
                            frame=dict(duration=100, redraw=True),
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

output_file = os.path.join(output_dir, "6_gradient_descent.html")
fig6.write_html(output_file, include_mathjax="cdn")
print(f"   ✅ 保存: {output_file}")

# ============================================
# 7. 创建综合仪表板
# ============================================
print("\n7️⃣ 创建综合仪表板...")

fig7 = make_subplots(
    rows=2,
    cols=2,
    subplot_titles=("平方误差", "交叉熵 (y=1)", "交叉熵 (y=0)", "Softmax"),
    specs=[
        [{"type": "scatter"}, {"type": "scatter"}],
        [{"type": "scatter"}, {"type": "scatter"}],
    ],
)

# 1. 平方误差
error = np.linspace(-5, 5, 100)
squared = error**2
fig7.add_trace(
    go.Scatter(
        x=error,
        y=squared,
        mode="lines",
        name="(y-ŷ)²",
        line=dict(color="purple", width=3),
    ),
    row=1,
    col=1,
)

# 2. 交叉熵 y=1
p = np.linspace(0.001, 0.999, 100)
ce_y1 = -np.log(p)
fig7.add_trace(
    go.Scatter(
        x=p, y=ce_y1, mode="lines", name="-log(p)", line=dict(color="red", width=3)
    ),
    row=1,
    col=2,
)

# 3. 交叉熵 y=0
ce_y0 = -np.log(1 - p)
fig7.add_trace(
    go.Scatter(
        x=p, y=ce_y0, mode="lines", name="-log(1-p)", line=dict(color="blue", width=3)
    ),
    row=2,
    col=1,
)

# 4. Softmax
z = np.linspace(-3, 3, 100)
softmax = np.exp(z) / (np.exp(z) + np.exp(0) + np.exp(0))
fig7.add_trace(
    go.Scatter(
        x=z,
        y=softmax,
        mode="lines",
        name="softmax(z₁)",
        line=dict(color="green", width=3),
    ),
    row=2,
    col=2,
)

# 更新坐标轴
fig7.update_xaxes(title_text="误差 e", row=1, col=1)
fig7.update_xaxes(title_text="预测概率 p", row=1, col=2)
fig7.update_xaxes(title_text="预测概率 p", row=2, col=1)
fig7.update_xaxes(title_text="logit z₁", row=2, col=2)

fig7.update_yaxes(title_text="e²", row=1, col=1)
fig7.update_yaxes(title_text="交叉熵", row=1, col=2)
fig7.update_yaxes(title_text="交叉熵", row=2, col=1)
fig7.update_yaxes(title_text="概率", row=2, col=2)

# 添加公式
fig7 = add_formula_annotation(
    fig7,
    r"$$\text{MSE: } (y-\hat{y})^2 \quad | \quad \text{CE: } -[y\log p + (1-y)\log(1-p)] \quad | \quad \text{Softmax: } \frac{e^{z_i}}{\sum_j e^{z_j}}$$",
    x=0.5,
    y=1.03,
)

fig7.update_layout(
    title_text="损失函数综合仪表板",
    height=850,
    showlegend=True,
    margin=dict(t=120, b=60, l=60, r=60),
)

output_file = os.path.join(output_dir, "7_dashboard.html")
fig7.write_html(output_file, include_mathjax="cdn")
print(f"   ✅ 保存: {output_file}")

# ============================================
# 打印统计信息
# ============================================
print("\n" + "=" * 60)
print("📊 损失函数计算示例")
print("=" * 60)

print("\n1️⃣ 最小二乘法示例:")
A = np.vstack([x_data, np.ones(len(x_data))]).T
a_opt, b_opt = np.linalg.lstsq(A, y_true, rcond=None)[0]
y_pred_opt = a_opt * x_data + b_opt
print(f"   拟合直线: y = {a_opt:.3f}x + {b_opt:.3f}")
print(f"   总平方误差: {np.sum((y_true - y_pred_opt)**2):.3f}")

print("\n2️⃣ 交叉熵示例（二分类）:")
test_cases = [
    (1, 0.9, "预测准确"),
    (1, 0.5, "不确定"),
    (1, 0.1, "预测错误"),
    (0, 0.1, "预测准确"),
    (0, 0.9, "预测错误"),
]

for y_true_val, p_val, desc in test_cases:
    if y_true_val == 1:
        ce = -np.log(p_val)
    else:
        ce = -np.log(1 - p_val)
    print(f"   真值={y_true_val}, 预测概率={p_val:.1f}: CE={ce:.3f} ({desc})")

print("\n3️⃣ 多分类交叉熵示例:")
y_true_multi = np.array([1, 0, 0])
y_pred_multi = np.array([0.7, 0.2, 0.1])
ce_multi = -np.sum(y_true_multi * np.log(y_pred_multi + 1e-10))
print(f"   真实类别: 0 (one-hot: {y_true_multi})")
print(f"   预测概率: {y_pred_multi}")
print(f"   交叉熵损失: {ce_multi:.3f}")

print("\n" + "=" * 60)
print("✨ 所有交互式可视化已创建完成！")
print("=" * 60)
print(f"\n📂 生成的文件位于: code/{output_dir}/")
print("   1. 1_least_squares.html - 最小二乘法动画")
print("   2. 2_cross_entropy.html - 交叉熵损失")
print("   3. 3_penalty_animation.html - 惩罚强度动画")
print("   4. 4_softmax_3d.html - Softmax 3D可视化")
print("   5. 5_comparison.html - 损失函数对比")
print("   6. 6_gradient_descent.html - 梯度下降动画")
print("   7. 7_dashboard.html - 综合仪表板")
print("\n💡 在浏览器中打开这些 HTML 文件即可查看交互式可视化！")
print("=" * 60)
