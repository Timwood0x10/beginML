"""
PyTorch优化器交互式可视化脚本
基于 3.grand_optimizer.md 文档中的优化算法
使用 Plotly 创建交互式动画图表，包含数学公式
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# 创建输出目录
output_dir = "grand_optimizer"
os.makedirs(output_dir, exist_ok=True)

print("=" * 60)
print("🎨 PyTorch优化器交互式可视化")
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


# 定义测试损失函数：Rosenbrock函数（经典优化测试函数）
def rosenbrock(x, y):
    """Rosenbrock函数：f(x,y) = (1-x)^2 + 100(y-x^2)^2"""
    return (1 - x) ** 2 + 100 * (y - x**2) ** 2


def rosenbrock_grad(x, y):
    """Rosenbrock函数的梯度"""
    dx = -2 * (1 - x) - 400 * x * (y - x**2)
    dy = 200 * (y - x**2)
    return np.array([dx, dy])


# ============================================
# 1. SGD vs SGD+Momentum 对比动画
# ============================================
print("\n1️⃣ 创建SGD vs Momentum对比动画...")

# 初始化
x_init, y_init = -0.5, -0.5
lr = 0.001
momentum = 0.9
iterations = 200

# SGD轨迹
sgd_path = [[x_init, y_init]]
x, y = x_init, y_init
for _ in range(iterations):
    grad = rosenbrock_grad(x, y)
    x = x - lr * grad[0]
    y = y - lr * grad[1]
    sgd_path.append([x, y])

# SGD + Momentum轨迹
momentum_path = [[x_init, y_init]]
x, y = x_init, y_init
v = np.array([0.0, 0.0])
for _ in range(iterations):
    grad = rosenbrock_grad(x, y)
    v = momentum * v + grad
    x = x - lr * v[0]
    y = y - lr * v[1]
    momentum_path.append([x, y])

sgd_path = np.array(sgd_path)
momentum_path = np.array(momentum_path)

# 创建等高线背景
x_range = np.linspace(-1, 1.5, 100)
y_range = np.linspace(-1, 1.5, 100)
X, Y = np.meshgrid(x_range, y_range)
Z = rosenbrock(X, Y)

# 创建动画帧
frames = []
for i in range(0, len(sgd_path), 2):
    frame_data = [
        go.Contour(
            x=x_range,
            y=y_range,
            z=Z,
            colorscale="Viridis",
            opacity=0.6,
            contours=dict(start=0, end=100, size=10),
            showscale=False,
            name="损失函数",
        ),
        go.Scatter(
            x=sgd_path[: i + 1, 0],
            y=sgd_path[: i + 1, 1],
            mode="lines+markers",
            name="SGD",
            line=dict(color="red", width=2),
            marker=dict(size=4),
        ),
        go.Scatter(
            x=momentum_path[: i + 1, 0],
            y=momentum_path[: i + 1, 1],
            mode="lines+markers",
            name="SGD+Momentum",
            line=dict(color="blue", width=2),
            marker=dict(size=4),
        ),
    ]
    frames.append(go.Frame(data=frame_data, name=str(i)))

# 初始帧
fig1 = go.Figure(
    data=[
        go.Contour(
            x=x_range,
            y=y_range,
            z=Z,
            colorscale="Viridis",
            opacity=0.6,
            contours=dict(start=0, end=100, size=10),
            showscale=False,
            name="损失函数",
        ),
        go.Scatter(
            x=[x_init],
            y=[y_init],
            mode="markers",
            name="SGD",
            marker=dict(size=8, color="red"),
        ),
        go.Scatter(
            x=[x_init],
            y=[y_init],
            mode="markers",
            name="SGD+Momentum",
            marker=dict(size=8, color="blue"),
        ),
    ],
    frames=frames,
)

# 添加公式
fig1 = add_formula_annotation(
    fig1,
    r"$$\text{SGD: } \theta \leftarrow \theta - \eta \cdot g \quad | \quad \text{Momentum: } v \leftarrow \mu v + g, \; \theta \leftarrow \theta - \eta v$$",
    x=0.5,
    y=1.05,
)

fig1.update_layout(
    title="SGD vs SGD+Momentum 优化路径对比",
    xaxis_title="参数 x",
    yaxis_title="参数 y",
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
                            frame=dict(duration=50, redraw=True),
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

output_file = os.path.join(output_dir, "1_sgd_vs_momentum.html")
fig1.write_html(output_file, include_mathjax="cdn")
print(f"   ✅ 保存: {output_file}")

# ============================================
# 2. Adam 优化器动画（动量+自适应步长）
# ============================================
print("\n2️⃣ 创建Adam优化器动画...")

# Adam轨迹
adam_path = [[x_init, y_init]]
x, y = x_init, y_init
m = np.array([0.0, 0.0])
v_adam = np.array([0.0, 0.0])
beta1, beta2 = 0.9, 0.999
epsilon = 1e-8
lr_adam = 0.01

for t in range(1, iterations + 1):
    grad = rosenbrock_grad(x, y)
    m = beta1 * m + (1 - beta1) * grad
    v_adam = beta2 * v_adam + (1 - beta2) * grad**2
    m_hat = m / (1 - beta1**t)
    v_hat = v_adam / (1 - beta2**t)
    x = x - lr_adam * m_hat[0] / (np.sqrt(v_hat[0]) + epsilon)
    y = y - lr_adam * m_hat[1] / (np.sqrt(v_hat[1]) + epsilon)
    adam_path.append([x, y])

adam_path = np.array(adam_path)

# 创建动画帧
frames_adam = []
for i in range(0, len(adam_path), 2):
    frame_data = [
        go.Contour(
            x=x_range,
            y=y_range,
            z=Z,
            colorscale="Viridis",
            opacity=0.6,
            contours=dict(start=0, end=100, size=10),
            showscale=False,
        ),
        go.Scatter(
            x=adam_path[: i + 1, 0],
            y=adam_path[: i + 1, 1],
            mode="lines+markers",
            name="Adam",
            line=dict(color="green", width=3),
            marker=dict(size=5, color="green"),
        ),
    ]
    frames_adam.append(go.Frame(data=frame_data, name=str(i)))

fig2 = go.Figure(
    data=[
        go.Contour(
            x=x_range,
            y=y_range,
            z=Z,
            colorscale="Viridis",
            opacity=0.6,
            contours=dict(start=0, end=100, size=10),
            showscale=False,
        ),
        go.Scatter(
            x=[x_init],
            y=[y_init],
            mode="markers",
            name="Adam",
            marker=dict(size=10, color="green"),
        ),
    ],
    frames=frames_adam,
)

# 添加公式
fig2 = add_formula_annotation(
    fig2,
    r"$$m \leftarrow \beta_1 m + (1-\beta_1)g, \; v \leftarrow \beta_2 v + (1-\beta_2)g^2, \; \theta \leftarrow \theta - \eta \frac{\hat{m}}{\sqrt{\hat{v}} + \epsilon}$$",
    x=0.5,
    y=1.05,
)

fig2.update_layout(
    title="Adam优化器（动量+自适应步长）",
    xaxis_title="参数 x",
    yaxis_title="参数 y",
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
                            frame=dict(duration=50, redraw=True),
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

output_file = os.path.join(output_dir, "2_adam_optimizer.html")
fig2.write_html(output_file, include_mathjax="cdn")
print(f"   ✅ 保存: {output_file}")

# ============================================
# 3. 多优化器对比（SGD, Momentum, RMSprop, Adam, AdamW）
# ============================================
print("\n3️⃣ 创建多优化器对比...")

optimizers = {
    "SGD": {"color": "red", "path": sgd_path},
    "Momentum": {"color": "blue", "path": momentum_path},
    "Adam": {"color": "green", "path": adam_path},
}

# RMSprop轨迹
rmsprop_path = [[x_init, y_init]]
x, y = x_init, y_init
E_g2 = np.array([0.0, 0.0])
rho = 0.9
lr_rms = 0.01

for _ in range(iterations):
    grad = rosenbrock_grad(x, y)
    E_g2 = rho * E_g2 + (1 - rho) * grad**2
    x = x - lr_rms * grad[0] / (np.sqrt(E_g2[0]) + epsilon)
    y = y - lr_rms * grad[1] / (np.sqrt(E_g2[1]) + epsilon)
    rmsprop_path.append([x, y])

rmsprop_path = np.array(rmsprop_path)
optimizers["RMSprop"] = {"color": "purple", "path": rmsprop_path}

# Adagrad轨迹
adagrad_path = [[x_init, y_init]]
x, y = x_init, y_init
G = np.array([0.0, 0.0])
lr_ada = 0.1

for _ in range(iterations):
    grad = rosenbrock_grad(x, y)
    G = G + grad**2
    x = x - lr_ada * grad[0] / (np.sqrt(G[0]) + epsilon)
    y = y - lr_ada * grad[1] / (np.sqrt(G[1]) + epsilon)
    adagrad_path.append([x, y])

adagrad_path = np.array(adagrad_path)
optimizers["Adagrad"] = {"color": "orange", "path": adagrad_path}

# 创建对比图
fig3 = go.Figure()

# 添加等高线
fig3.add_trace(
    go.Contour(
        x=x_range,
        y=y_range,
        z=Z,
        colorscale="Viridis",
        opacity=0.4,
        contours=dict(start=0, end=100, size=10),
        showscale=False,
        name="损失函数",
        hoverinfo="skip",
    )
)

# 添加所有优化器的路径
for name, opt in optimizers.items():
    fig3.add_trace(
        go.Scatter(
            x=opt["path"][:, 0],
            y=opt["path"][:, 1],
            mode="lines",
            name=name,
            line=dict(color=opt["color"], width=2),
            hovertemplate=f"{name}<br>x: %{{x:.3f}}<br>y: %{{y:.3f}}",
        )
    )

# 添加起点
fig3.add_trace(
    go.Scatter(
        x=[x_init],
        y=[y_init],
        mode="markers",
        name="起点",
        marker=dict(size=15, color="black", symbol="star"),
    )
)

# 添加最优点
fig3.add_trace(
    go.Scatter(
        x=[1],
        y=[1],
        mode="markers",
        name="最优点(1,1)",
        marker=dict(size=15, color="gold", symbol="star"),
    )
)

# 添加公式
fig3 = add_formula_annotation(
    fig3,
    r"$$\text{SGD, Momentum, RMSprop, Adam, Adagrad 在 Rosenbrock 函数上的优化路径对比}$$",
    x=0.5,
    y=1.05,
)

fig3.update_layout(
    title="五种优化器性能对比",
    xaxis_title="参数 x",
    yaxis_title="参数 y",
    height=700,
    hovermode="closest",
    legend=dict(x=0.02, y=0.98, bgcolor="rgba(255,255,255,0.8)"),
    margin=dict(t=120, b=60, l=60, r=60),
)

output_file = os.path.join(output_dir, "3_optimizer_comparison.html")
fig3.write_html(output_file, include_mathjax="cdn")
print(f"   ✅ 保存: {output_file}")

# ============================================
# 4. 学习率对优化的影响
# ============================================
print("\n4️⃣ 创建学习率影响可视化...")

learning_rates = [0.0001, 0.001, 0.01, 0.05]
colors = ["blue", "green", "orange", "red"]

fig4 = go.Figure()

# 添加等高线
fig4.add_trace(
    go.Contour(
        x=x_range,
        y=y_range,
        z=Z,
        colorscale="Viridis",
        opacity=0.4,
        contours=dict(start=0, end=100, size=10),
        showscale=False,
        hoverinfo="skip",
    )
)

for lr_test, color in zip(learning_rates, colors):
    path = [[x_init, y_init]]
    x, y = x_init, y_init
    v = np.array([0.0, 0.0])

    for _ in range(iterations):
        grad = rosenbrock_grad(x, y)
        v = momentum * v + grad
        x = x - lr_test * v[0]
        y = y - lr_test * v[1]
        path.append([x, y])

    path = np.array(path)
    fig4.add_trace(
        go.Scatter(
            x=path[:, 0],
            y=path[:, 1],
            mode="lines",
            name=f"lr={lr_test}",
            line=dict(color=color, width=2),
        )
    )

# 添加公式
fig4 = add_formula_annotation(
    fig4,
    r"$$\theta \leftarrow \theta - \eta \cdot g \quad \text{学习率 } \eta \text{ 的选择至关重要}$$",
    x=0.5,
    y=1.05,
)

fig4.update_layout(
    title="学习率对SGD+Momentum的影响",
    xaxis_title="参数 x",
    yaxis_title="参数 y",
    height=700,
    legend=dict(x=0.02, y=0.98, bgcolor="rgba(255,255,255,0.8)"),
    margin=dict(t=120, b=60, l=60, r=60),
)

output_file = os.path.join(output_dir, "4_learning_rate_impact.html")
fig4.write_html(output_file, include_mathjax="cdn")
print(f"   ✅ 保存: {output_file}")

# ============================================
# 5. 动量系数的影响
# ============================================
print("\n5️⃣ 创建动量系数影响可视化...")

momentum_values = [0.0, 0.5, 0.9, 0.99]
colors_mom = ["red", "blue", "green", "purple"]

fig5 = go.Figure()

# 添加等高线
fig5.add_trace(
    go.Contour(
        x=x_range,
        y=y_range,
        z=Z,
        colorscale="Viridis",
        opacity=0.4,
        contours=dict(start=0, end=100, size=10),
        showscale=False,
        hoverinfo="skip",
    )
)

for mu, color in zip(momentum_values, colors_mom):
    path = [[x_init, y_init]]
    x, y = x_init, y_init
    v = np.array([0.0, 0.0])

    for _ in range(iterations):
        grad = rosenbrock_grad(x, y)
        v = mu * v + grad
        x = x - lr * v[0]
        y = y - lr * v[1]
        path.append([x, y])

    path = np.array(path)
    label = "SGD(无动量)" if mu == 0.0 else f"Momentum μ={mu}"
    fig5.add_trace(
        go.Scatter(
            x=path[:, 0],
            y=path[:, 1],
            mode="lines",
            name=label,
            line=dict(color=color, width=2),
        )
    )

# 添加公式
fig5 = add_formula_annotation(
    fig5,
    r"$$v \leftarrow \mu \cdot v + g, \; \theta \leftarrow \theta - \eta v \quad \text{动量系数 } \mu \in [0, 1)$$",
    x=0.5,
    y=1.05,
)

fig5.update_layout(
    title="动量系数对优化的影响",
    xaxis_title="参数 x",
    yaxis_title="参数 y",
    height=700,
    legend=dict(x=0.02, y=0.98, bgcolor="rgba(255,255,255,0.8)"),
    margin=dict(t=120, b=60, l=60, r=60),
)

output_file = os.path.join(output_dir, "5_momentum_impact.html")
fig5.write_html(output_file, include_mathjax="cdn")
print(f"   ✅ 保存: {output_file}")

# ============================================
# 6. Adam的自适应步长可视化
# ============================================
print("\n6️⃣ 创建Adam自适应步长可视化...")

# 跟踪每一步的步长
adam_steps = [[x_init, y_init]]
step_sizes = []
x, y = x_init, y_init
m = np.array([0.0, 0.0])
v_adam = np.array([0.0, 0.0])

for t in range(1, 100):
    grad = rosenbrock_grad(x, y)
    m = beta1 * m + (1 - beta1) * grad
    v_adam = beta2 * v_adam + (1 - beta2) * grad**2
    m_hat = m / (1 - beta1**t)
    v_hat = v_adam / (1 - beta2**t)

    step_x = lr_adam * m_hat[0] / (np.sqrt(v_hat[0]) + epsilon)
    step_y = lr_adam * m_hat[1] / (np.sqrt(v_hat[1]) + epsilon)
    step_size = np.sqrt(step_x**2 + step_y**2)
    step_sizes.append(step_size)

    x = x - step_x
    y = y - step_y
    adam_steps.append([x, y])

fig6 = make_subplots(
    rows=1,
    cols=2,
    subplot_titles=("Adam优化路径", "每步的步长变化"),
    specs=[[{"type": "scatter"}, {"type": "scatter"}]],
)

adam_steps = np.array(adam_steps)

# 左图：优化路径
fig6.add_trace(
    go.Contour(
        x=x_range,
        y=y_range,
        z=Z,
        colorscale="Viridis",
        opacity=0.4,
        contours=dict(start=0, end=100, size=10),
        showscale=False,
        hoverinfo="skip",
    ),
    row=1,
    col=1,
)
fig6.add_trace(
    go.Scatter(
        x=adam_steps[:, 0],
        y=adam_steps[:, 1],
        mode="lines+markers",
        name="Adam路径",
        line=dict(color="green", width=2),
        marker=dict(size=4),
    ),
    row=1,
    col=1,
)

# 右图：步长变化
fig6.add_trace(
    go.Scatter(
        x=list(range(len(step_sizes))),
        y=step_sizes,
        mode="lines",
        name="步长",
        line=dict(color="red", width=2),
    ),
    row=1,
    col=2,
)

fig6.update_xaxes(title_text="参数 x", row=1, col=1)
fig6.update_yaxes(title_text="参数 y", row=1, col=1)
fig6.update_xaxes(title_text="迭代次数", row=1, col=2)
fig6.update_yaxes(title_text="步长大小", row=1, col=2)

# 添加公式
fig6 = add_formula_annotation(
    fig6,
    r"$$\text{Adam自适应调整每个参数的步长：} \Delta\theta_i = \frac{\eta \cdot \hat{m}_i}{\sqrt{\hat{v}_i} + \epsilon}$$",
    x=0.5,
    y=1.05,
)

fig6.update_layout(
    title="Adam优化器的自适应步长机制",
    height=600,
    showlegend=True,
    margin=dict(t=120, b=60, l=60, r=60),
)

output_file = os.path.join(output_dir, "6_adam_adaptive_stepsize.html")
fig6.write_html(output_file, include_mathjax="cdn")
print(f"   ✅ 保存: {output_file}")

# ============================================
# 7. 综合仪表板
# ============================================
print("\n7️⃣ 创建综合仪表板...")

fig7 = make_subplots(
    rows=2,
    cols=2,
    subplot_titles=(
        "SGD vs Momentum",
        "Adam (动量+自适应)",
        "学习率影响",
        "优化器对比",
    ),
    specs=[
        [{"type": "scatter"}, {"type": "scatter"}],
        [{"type": "scatter"}, {"type": "scatter"}],
    ],
)

# 子图1: SGD vs Momentum
fig7.add_trace(
    go.Scatter(
        x=sgd_path[:, 0],
        y=sgd_path[:, 1],
        mode="lines",
        name="SGD",
        line=dict(color="red", width=1.5),
    ),
    row=1,
    col=1,
)
fig7.add_trace(
    go.Scatter(
        x=momentum_path[:, 0],
        y=momentum_path[:, 1],
        mode="lines",
        name="Momentum",
        line=dict(color="blue", width=1.5),
    ),
    row=1,
    col=1,
)

# 子图2: Adam
fig7.add_trace(
    go.Scatter(
        x=adam_path[:, 0],
        y=adam_path[:, 1],
        mode="lines",
        name="Adam",
        line=dict(color="green", width=2),
    ),
    row=1,
    col=2,
)

# 子图3: 学习率影响
for lr_test, color in zip([0.001, 0.01], ["blue", "red"]):
    path = [[x_init, y_init]]
    x, y = x_init, y_init
    for _ in range(100):
        grad = rosenbrock_grad(x, y)
        x = x - lr_test * grad[0]
        y = y - lr_test * grad[1]
        path.append([x, y])
    path = np.array(path)
    fig7.add_trace(
        go.Scatter(
            x=path[:, 0],
            y=path[:, 1],
            mode="lines",
            name=f"lr={lr_test}",
            line=dict(color=color, width=1.5),
        ),
        row=2,
        col=1,
    )

# 子图4: 多优化器
for name, opt in list(optimizers.items())[:3]:
    fig7.add_trace(
        go.Scatter(
            x=opt["path"][:, 0],
            y=opt["path"][:, 1],
            mode="lines",
            name=name,
            line=dict(color=opt["color"], width=1.5),
        ),
        row=2,
        col=2,
    )

# 更新坐标轴
for i, j in [(1, 1), (1, 2), (2, 1), (2, 2)]:
    fig7.update_xaxes(title_text="x", row=i, col=j)
    fig7.update_yaxes(title_text="y", row=i, col=j)

# 添加公式
fig7 = add_formula_annotation(
    fig7,
    r"$$\text{PyTorch优化器核心：动量、自适应步长、偏差校正、权重衰减}$$",
    x=0.5,
    y=1.03,
)

fig7.update_layout(
    title="PyTorch优化器综合对比仪表板",
    height=850,
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
print("📊 优化器性能总结")
print("=" * 60)

print("\n✅ 已创建的可视化:")
print("   1. SGD vs Momentum对比")
print("   2. Adam优化器演示")
print("   3. 五种优化器性能对比")
print("   4. 学习率影响分析")
print("   5. 动量系数影响分析")
print("   6. Adam自适应步长机制")
print("   7. 综合仪表板")

print("\n📚 优化器选择建议:")
print("   • 默认首选: AdamW")
print("   • 大规模图像分类: SGD+Momentum")
print("   • 稀疏特征/大词表: SparseAdam/Adagrad")
print("   • 训练不稳: RAdam 或 AdamW+warmup")
print("   • 小模型快速收敛: LBFGS")

print("\n" + "=" * 60)
print("✨ 所有交互式可视化已创建完成！")
print("=" * 60)
print(f"\n📂 生成的文件位于: code/{output_dir}/")
print("💡 在浏览器中打开这些 HTML 文件即可查看交互式可视化！")
print("=" * 60)
