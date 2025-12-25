"""
交互式梯度下降可视化工具
可以实时调整学习率、动量、起始点等参数
包含多种测试函数和优化算法
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import warnings

# 抑制数值计算警告
warnings.filterwarnings("ignore", category=RuntimeWarning)

# 设置数值稳定性参数
np.seterr(divide="ignore", invalid="ignore", over="ignore")

# 创建输出目录
output_dir = "interactive_gradient_descent"
os.makedirs(output_dir, exist_ok=True)

print("=" * 60)
print("🎨 创建交互式梯度下降可视化工具")
print("=" * 60)

# ============================================
# 定义测试函数
# ============================================


def sphere_function(x, y):
    """简单的球形函数"""
    return x**2 + y**2


def sphere_grad(x, y):
    return np.array([2 * x, 2 * y])


def rosenbrock(x, y):
    """Rosenbrock函数 - 经典优化测试"""
    return (1 - x) ** 2 + 100 * (y - x**2) ** 2


def rosenbrock_grad(x, y):
    dx = -2 * (1 - x) - 400 * x * (y - x**2)
    dy = 200 * (y - x**2)
    return np.array([dx, dy])


def rastrigin(x, y):
    """Rastrigin函数 - 多局部极小值"""
    A = 10
    # 限制输入范围以避免数值溢出
    x = np.clip(x, -5.12, 5.12)
    y = np.clip(y, -5.12, 5.12)
    return (
        A * 2 + (x**2 - A * np.cos(2 * np.pi * x)) + (y**2 - A * np.cos(2 * np.pi * y))
    )


def rastrigin_grad(x, y):
    A = 10
    # 限制输入范围以避免数值溢出
    x = np.clip(x, -5.12, 5.12)
    y = np.clip(y, -5.12, 5.12)
    dx = 2 * x + 2 * np.pi * A * np.sin(2 * np.pi * x)
    dy = 2 * y + 2 * np.pi * A * np.sin(2 * np.pi * y)
    return np.array([dx, dy])


def beale(x, y):
    """Beale函数"""
    # 限制输入范围以避免数值溢出
    x = np.clip(x, -4.5, 4.5)
    y = np.clip(y, -4.5, 4.5)
    return (
        (1.5 - x + x * y) ** 2
        + (2.25 - x + x * y**2) ** 2
        + (2.625 - x + x * y**3) ** 2
    )


def beale_grad(x, y):
    # 限制输入范围以避免数值溢出
    x = np.clip(x, -4.5, 4.5)
    y = np.clip(y, -4.5, 4.5)
    dx = (
        2 * (1.5 - x + x * y) * (y - 1)
        + 2 * (2.25 - x + x * y**2) * (y**2 - 1)
        + 2 * (2.625 - x + x * y**3) * (y**3 - 1)
    )
    dy = (
        2 * (1.5 - x + x * y) * x
        + 2 * (2.25 - x + x * y**2) * 2 * x * y
        + 2 * (2.625 - x + x * y**3) * 3 * x * y**2
    )
    return np.array([dx, dy])


# ============================================
# 优化算法实现
# ============================================


def optimize_sgd(grad_func, x_init, y_init, lr, iterations):
    """标准SGD"""
    path = [[x_init, y_init]]
    x, y = x_init, y_init

    for _ in range(iterations):
        grad = grad_func(x, y)
        # 梯度裁剪以避免数值不稳定
        grad = np.clip(grad, -100, 100)
        x = x - lr * grad[0]
        y = y - lr * grad[1]
        path.append([x, y])

    return np.array(path)


def optimize_momentum(grad_func, x_init, y_init, lr, momentum, iterations):
    """SGD + Momentum"""
    path = [[x_init, y_init]]
    x, y = x_init, y_init
    v = np.array([0.0, 0.0])

    for _ in range(iterations):
        grad = grad_func(x, y)
        v = momentum * v + grad
        x = x - lr * v[0]
        y = y - lr * v[1]
        path.append([x, y])

    return np.array(path)


def optimize_adam(grad_func, x_init, y_init, lr, iterations, beta1=0.9, beta2=0.999):
    """Adam优化器"""
    path = [[x_init, y_init]]
    x, y = x_init, y_init
    m = np.array([0.0, 0.0])
    v = np.array([0.0, 0.0])
    epsilon = 1e-8

    for t in range(1, iterations + 1):
        grad = grad_func(x, y)
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad**2
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        x = x - lr * m_hat[0] / (np.sqrt(v_hat[0]) + epsilon)
        y = y - lr * m_hat[1] / (np.sqrt(v_hat[1]) + epsilon)
        path.append([x, y])

    return np.array(path)


def optimize_rmsprop(grad_func, x_init, y_init, lr, iterations, beta=0.9, epsilon=1e-8):
    """RMSprop优化器"""
    path = [[x_init, y_init]]
    x, y = x_init, y_init
    v = np.array([0.0, 0.0])

    for _ in range(iterations):
        grad = grad_func(x, y)
        v = beta * v + (1 - beta) * grad**2
        x = x - lr * grad[0] / (np.sqrt(v[0]) + epsilon)
        y = y - lr * grad[1] / (np.sqrt(v[1]) + epsilon)
        path.append([x, y])

    return np.array(path)


def optimize_adagrad(grad_func, x_init, y_init, lr, iterations, epsilon=1e-8):
    """AdaGrad优化器"""
    path = [[x_init, y_init]]
    x, y = x_init, y_init
    G = np.array([0.0, 0.0])

    for _ in range(iterations):
        grad = grad_func(x, y)
        G = G + grad**2
        x = x - lr * grad[0] / (np.sqrt(G[0]) + epsilon)
        y = y - lr * grad[1] / (np.sqrt(G[1]) + epsilon)
        path.append([x, y])

    return np.array(path)


def optimize_nesterov(grad_func, x_init, y_init, lr, momentum, iterations):
    """Nesterov动量优化器"""
    path = [[x_init, y_init]]
    x, y = x_init, y_init
    v = np.array([0.0, 0.0])

    for _ in range(iterations):
        # Nesterov前瞻
        x_lookahead = x - momentum * v[0]
        y_lookahead = y - momentum * v[1]
        grad = grad_func(x_lookahead, y_lookahead)
        v = momentum * v + grad
        x = x - lr * v[0]
        y = y - lr * v[1]
        path.append([x, y])

    return np.array(path)


# ============================================
# 创建完全交互式可视化
# ============================================

print("\n🎯 创建交互式梯度下降工具...")

# 函数选择
functions = {
    "Sphere_Easy": {
        "func": sphere_function,
        "grad": sphere_grad,
        "bounds": [-2, 2],
        "optimal": [0, 0],
        "display": "Sphere (Easy)",
    },
    "Rosenbrock_Hard": {
        "func": rosenbrock,
        "grad": rosenbrock_grad,
        "bounds": [-1, 2],
        "optimal": [1, 1],
        "display": "Rosenbrock (Hard)",
    },
    "Rastrigin_MultiModal": {
        "func": rastrigin,
        "grad": rastrigin_grad,
        "bounds": [-5, 5],
        "optimal": [0, 0],
        "display": "Rastrigin (Multi-Modal)",
    },
    "Beale": {
        "func": beale,
        "grad": beale_grad,
        "bounds": [-4.5, 4.5],
        "optimal": [3, 0.5],
        "display": "Beale",
    },
}

# 参数配置 - 添加更多优化算法
configs = {
    "SGD (lr=0.01)": {"method": "sgd", "lr": 0.01, "momentum": 0, "color": "red"},
    "SGD (lr=0.1)": {"method": "sgd", "lr": 0.1, "momentum": 0, "color": "orange"},
    "Momentum (lr=0.01, μ=0.9)": {
        "method": "momentum",
        "lr": 0.01,
        "momentum": 0.9,
        "color": "blue",
    },
    "Nesterov (lr=0.01, μ=0.9)": {
        "method": "nesterov",
        "lr": 0.01,
        "momentum": 0.9,
        "color": "purple",
    },
    "Adam (lr=0.1)": {"method": "adam", "lr": 0.1, "momentum": 0, "color": "green"},
    "RMSprop (lr=0.1)": {
        "method": "rmsprop",
        "lr": 0.1,
        "momentum": 0,
        "color": "cyan",
    },
    "AdaGrad (lr=0.1)": {
        "method": "adagrad",
        "lr": 0.1,
        "momentum": 0,
        "color": "magenta",
    },
}

# 为每个函数创建独立的可视化
for func_name, func_info in functions.items():
    display_name = func_info.get("display", func_name)
    print(f"\n   创建 {display_name} 的可视化...")

    func = func_info["func"]
    grad_func = func_info["grad"]
    bounds = func_info["bounds"]
    optimal = func_info["optimal"]

    # 创建网格
    x_range = np.linspace(bounds[0], bounds[1], 100)
    y_range = np.linspace(bounds[0], bounds[1], 100)
    X, Y = np.meshgrid(x_range, y_range)
    Z = func(X, Y)

    # 起始点
    x_init, y_init = bounds[0] * 0.7, bounds[0] * 0.7
    iterations = 100

    # 创建图表
    fig = go.Figure()

    # 添加等高线
    fig.add_trace(
        go.Contour(
            x=x_range,
            y=y_range,
            z=Z,
            colorscale="Viridis",
            opacity=0.6,
            showscale=True,
            contours=dict(
                start=np.min(Z),
                end=np.min(Z) + (np.max(Z) - np.min(Z)) * 0.3,
                size=(np.max(Z) - np.min(Z)) * 0.05,
            ),
            name="目标函数",
            hovertemplate="x: %{x:.3f}<br>y: %{y:.3f}<br>f(x,y): %{z:.3f}",
        )
    )

    # 计算所有优化路径
    all_paths = {}
    for config_name, config in configs.items():
        if config["method"] == "sgd":
            path = optimize_sgd(grad_func, x_init, y_init, config["lr"], iterations)
        elif config["method"] == "momentum":
            path = optimize_momentum(
                grad_func, x_init, y_init, config["lr"], config["momentum"], iterations
            )
        elif config["method"] == "nesterov":
            path = optimize_nesterov(
                grad_func, x_init, y_init, config["lr"], config["momentum"], iterations
            )
        elif config["method"] == "adam":
            path = optimize_adam(grad_func, x_init, y_init, config["lr"], iterations)
        elif config["method"] == "rmsprop":
            path = optimize_rmsprop(grad_func, x_init, y_init, config["lr"], iterations)
        elif config["method"] == "adagrad":
            path = optimize_adagrad(grad_func, x_init, y_init, config["lr"], iterations)

        all_paths[config_name] = path

        # 添加路径（默认显示）
        fig.add_trace(
            go.Scatter(
                x=path[:, 0],
                y=path[:, 1],
                mode="lines+markers",
                name=config_name,
                line=dict(color=config["color"], width=2),
                marker=dict(size=4),
                visible=True,
                hovertemplate=config_name + "<br>x: %{x:.3f}<br>y: %{y:.3f}",
            )
        )

    # 添加起始点
    fig.add_trace(
        go.Scatter(
            x=[x_init],
            y=[y_init],
            mode="markers",
            name="起始点",
            marker=dict(size=15, color="black", symbol="star"),
            hovertemplate="起始点<br>x: %{x:.3f}<br>y: %{y:.3f}",
        )
    )

    # 添加最优点
    fig.add_trace(
        go.Scatter(
            x=[optimal[0]],
            y=[optimal[1]],
            mode="markers",
            name="最优点",
            marker=dict(size=15, color="gold", symbol="star"),
            hovertemplate="最优点<br>x: %{x:.3f}<br>y: %{y:.3f}",
        )
    )

    # 添加公式注释
    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0.5,
        y=1.08,
        text=f"$$\\text{{Target Function: {display_name}}} \\quad \\nabla f = [\\frac{{\\partial f}}{{\\partial x}}, \\frac{{\\partial f}}{{\\partial y}}]$$",
        showarrow=False,
        font=dict(size=16),
        bgcolor="rgba(255, 250, 205, 0.9)",
        bordercolor="orange",
        borderwidth=2,
        borderpad=10,
        xanchor="center",
    )

    # 添加说明文本
    instructions = (
        "💡 使用说明：<br>"
        "• 点击图例可以显示/隐藏不同的优化算法<br>"
        "• 悬停在路径上查看详细信息<br>"
        "• 使用鼠标滚轮缩放，拖动平移<br>"
        "• 不同颜色代表不同的优化算法和参数"
    )

    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0.02,
        y=0.02,
        text=instructions,
        showarrow=False,
        font=dict(size=11),
        bgcolor="rgba(255, 255, 255, 0.9)",
        bordercolor="gray",
        borderwidth=1,
        borderpad=8,
        xanchor="left",
        yanchor="bottom",
        align="left",
    )

    # 更新布局
    fig.update_layout(
        title=f"Interactive Gradient Descent - {display_name}",
        xaxis_title="参数 x",
        yaxis_title="参数 y",
        height=800,
        hovermode="closest",
        legend=dict(
            x=1.02,
            y=1,
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="gray",
            borderwidth=1,
        ),
        margin=dict(t=140, b=80, l=60, r=200),
    )

    # 保存文件
    output_file = os.path.join(output_dir, f"{func_name.lower()}.html")
    fig.write_html(output_file, include_mathjax="cdn")
    print(f"   ✅ 保存: {output_file}")

# ============================================
# 创建参数对比大屏
# ============================================

print("\n🎯 创建参数对比大屏...")

# 使用Rosenbrock函数
func = rosenbrock
grad_func = rosenbrock_grad
bounds = [-1, 2]

# 创建网格
x_range = np.linspace(bounds[0], bounds[1], 100)
y_range = np.linspace(bounds[0], bounds[1], 100)
X, Y = np.meshgrid(x_range, y_range)
Z = rosenbrock(X, Y)

# 不同学习率
learning_rates = [0.001, 0.005, 0.01, 0.05]
x_init, y_init = -0.5, -0.5
iterations = 200

fig_compare = make_subplots(
    rows=2,
    cols=2,
    subplot_titles=[f"学习率 = {lr}" for lr in learning_rates],
    specs=[
        [{"type": "scatter"}, {"type": "scatter"}],
        [{"type": "scatter"}, {"type": "scatter"}],
    ],
)

for idx, lr in enumerate(learning_rates):
    row = idx // 2 + 1
    col = idx % 2 + 1

    # SGD路径
    path_sgd = optimize_sgd(grad_func, x_init, y_init, lr, iterations)

    # Momentum路径
    path_momentum = optimize_momentum(grad_func, x_init, y_init, lr, 0.9, iterations)

    # 添加等高线
    fig_compare.add_trace(
        go.Contour(
            x=x_range,
            y=y_range,
            z=Z,
            colorscale="Viridis",
            opacity=0.3,
            showscale=False,
            hoverinfo="skip",
        ),
        row=row,
        col=col,
    )

    # 添加SGD路径
    fig_compare.add_trace(
        go.Scatter(
            x=path_sgd[:, 0],
            y=path_sgd[:, 1],
            mode="lines",
            name=f"SGD (lr={lr})",
            line=dict(color="red", width=2),
            showlegend=(idx == 0),
        ),
        row=row,
        col=col,
    )

    # 添加Momentum路径
    fig_compare.add_trace(
        go.Scatter(
            x=path_momentum[:, 0],
            y=path_momentum[:, 1],
            mode="lines",
            name=f"Momentum (lr={lr})",
            line=dict(color="blue", width=2),
            showlegend=(idx == 0),
        ),
        row=row,
        col=col,
    )

    # 添加最优点
    fig_compare.add_trace(
        go.Scatter(
            x=[1],
            y=[1],
            mode="markers",
            marker=dict(size=10, color="gold", symbol="star"),
            showlegend=False,
            hoverinfo="skip",
        ),
        row=row,
        col=col,
    )

    fig_compare.update_xaxes(title_text="x", row=row, col=col)
    fig_compare.update_yaxes(title_text="y", row=row, col=col)

# 添加公式
fig_compare.add_annotation(
    xref="paper",
    yref="paper",
    x=0.5,
    y=1.05,
    text=r"$$\theta_{t+1} = \theta_t - \eta \nabla f(\theta_t) \quad \text{学习率 } \eta \text{ 的选择至关重要}$$",
    showarrow=False,
    font=dict(size=16),
    bgcolor="rgba(255, 250, 205, 0.9)",
    bordercolor="orange",
    borderwidth=2,
    borderpad=10,
    xanchor="center",
)

fig_compare.update_layout(
    title="学习率对比：SGD vs Momentum",
    height=900,
    margin=dict(t=120, b=60, l=60, r=60),
)

output_file = os.path.join(output_dir, "learning_rate_comparison.html")
fig_compare.write_html(output_file, include_mathjax="cdn")
print(f"   ✅ 保存: {output_file}")

# ============================================
# 创建动量系数对比
# ============================================

print("\n🎯 创建动量系数对比...")

momentum_values = [0.0, 0.5, 0.9, 0.99]
lr_fixed = 0.01

fig_momentum = make_subplots(
    rows=2,
    cols=2,
    subplot_titles=[f"动量 μ = {mu}" for mu in momentum_values],
    specs=[
        [{"type": "scatter"}, {"type": "scatter"}],
        [{"type": "scatter"}, {"type": "scatter"}],
    ],
)

for idx, mu in enumerate(momentum_values):
    row = idx // 2 + 1
    col = idx % 2 + 1

    if mu == 0:
        path = optimize_sgd(grad_func, x_init, y_init, lr_fixed, iterations)
        label = "SGD (无动量)"
    else:
        path = optimize_momentum(grad_func, x_init, y_init, lr_fixed, mu, iterations)
        label = f"Momentum μ={mu}"

    # 添加等高线
    fig_momentum.add_trace(
        go.Contour(
            x=x_range,
            y=y_range,
            z=Z,
            colorscale="Viridis",
            opacity=0.3,
            showscale=False,
            hoverinfo="skip",
        ),
        row=row,
        col=col,
    )

    # 添加路径
    fig_momentum.add_trace(
        go.Scatter(
            x=path[:, 0],
            y=path[:, 1],
            mode="lines+markers",
            name=label,
            line=dict(width=2),
            marker=dict(size=3),
            showlegend=(idx == 0),
        ),
        row=row,
        col=col,
    )

    # 添加最优点
    fig_momentum.add_trace(
        go.Scatter(
            x=[1],
            y=[1],
            mode="markers",
            marker=dict(size=10, color="gold", symbol="star"),
            showlegend=False,
            hoverinfo="skip",
        ),
        row=row,
        col=col,
    )

    fig_momentum.update_xaxes(title_text="x", row=row, col=col)
    fig_momentum.update_yaxes(title_text="y", row=row, col=col)

# 添加公式
fig_momentum.add_annotation(
    xref="paper",
    yref="paper",
    x=0.5,
    y=1.05,
    text=r"$$v_t = \mu \cdot v_{t-1} + \nabla f(\theta_t), \quad \theta_{t+1} = \theta_t - \eta v_t$$",
    showarrow=False,
    font=dict(size=16),
    bgcolor="rgba(255, 250, 205, 0.9)",
    bordercolor="orange",
    borderwidth=2,
    borderpad=10,
    xanchor="center",
)

fig_momentum.update_layout(
    title="动量系数对比 (固定学习率 lr=0.01)",
    height=900,
    margin=dict(t=120, b=60, l=60, r=60),
)

output_file = os.path.join(output_dir, "momentum_comparison.html")
fig_momentum.write_html(output_file, include_mathjax="cdn")
print(f"   ✅ 保存: {output_file}")

# ============================================
# 创建3D可视化 - 完全静态版本
# ============================================

print("\n🎯 创建3D可视化（静态稳定版）...")

# 参数设置 - 使用Sphere函数避免数值问题
x_init_3d, y_init_3d = 1.5, 1.5
iterations_3d = 30  # 减少迭代次数

# 创建网格
x_3d = np.linspace(-2, 2, 40)
y_3d = np.linspace(-2, 2, 40)
X_3d, Y_3d = np.meshgrid(x_3d, y_3d)
Z_3d = sphere_function(X_3d, Y_3d)  # 使用简单的Sphere函数

# 计算优化路径
print("   计算优化路径...")
paths_3d = {
    "SGD": optimize_sgd(sphere_grad, x_init_3d, y_init_3d, 0.1, iterations_3d),
    "Momentum": optimize_momentum(
        sphere_grad, x_init_3d, y_init_3d, 0.1, 0.9, iterations_3d
    ),
    "Adam": optimize_adam(sphere_grad, x_init_3d, y_init_3d, 0.1, iterations_3d),
}

print("   创建3D图表...")

# 创建图表 - 完全静态版本
fig_3d = go.Figure()

# 添加曲面（静态背景）
fig_3d.add_trace(
    go.Surface(
        x=X_3d,
        y=Y_3d,
        z=Z_3d,
        colorscale="Viridis",
        opacity=0.8,
        name="Sphere Function",
        showscale=True,
        colorbar=dict(title="f(x,y)", x=1.1),
        hovertemplate="x: %{x:.3f}<br>y: %{y:.3f}<br>f(x,y): %{z:.3f}<extra></extra>",
    )
)

# 添加所有优化路径（静态显示完整路径）
colors = {"SGD": "red", "Momentum": "blue", "Adam": "green"}

for name, path in paths_3d.items():
    z_values = sphere_function(path[:, 0], path[:, 1])

    # 添加完整路径
    fig_3d.add_trace(
        go.Scatter3d(
            x=path[:, 0],
            y=path[:, 1],
            z=z_values,
            mode="lines+markers",
            name=name,
            line=dict(color=colors[name], width=4),
            marker=dict(size=3, color=colors[name]),
            hovertemplate=f"{name}<br>x: %{{x:.3f}}<br>y: %{{y:.3f}}<br>f: %{{z:.3f}}<extra></extra>",
        )
    )

# 添加起始点和最优点
start_z = sphere_function(x_init_3d, y_init_3d)
fig_3d.add_trace(
    go.Scatter3d(
        x=[x_init_3d],
        y=[y_init_3d],
        z=[start_z],
        mode="markers",
        name="起始点",
        marker=dict(size=10, color="black"),
        hovertemplate=f"起始点<br>x: {x_init_3d:.3f}<br>y: {y_init_3d:.3f}<extra></extra>",
    )
)

fig_3d.add_trace(
    go.Scatter3d(
        x=[0],
        y=[0],
        z=[0],
        mode="markers",
        name="最优点",
        marker=dict(size=12, color="gold"),
        hovertemplate="最优点<br>x: 0.0<br>y: 0.0<br>f: 0.0<extra></extra>",
    )
)


# 添加公式
fig_3d.add_annotation(
    xref="paper",
    yref="paper",
    x=0.5,
    y=0.98,
    text=r"$f(x,y) = x^2 + y^2 \quad \text{(Sphere Function)}$",
    showarrow=False,
    font=dict(size=14),
    bgcolor="rgba(255, 250, 205, 0.9)",
    bordercolor="orange",
    borderwidth=2,
    borderpad=10,
    xanchor="center",
)

# 更新布局 - 静态版本
fig_3d.update_layout(
    title="3D 梯度下降可视化 - Sphere函数",
    scene=dict(
        xaxis_title="参数 x",
        yaxis_title="参数 y",
        zaxis_title="函数值 f(x,y)",
        camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
        aspectmode="cube",
    ),
    height=700,
    margin=dict(t=100, b=60, l=60, r=60),
    legend=dict(x=0.02, y=0.98),
    hovermode="closest",
)

output_file = os.path.join(output_dir, "3d_visualization.html")
print(f"   保存3D可视化...")
fig_3d.write_html(output_file, include_mathjax="cdn")
print(f"   ✅ 保存: {output_file}")

# ============================================
# 创建Rosenbrock 3D可视化 - 额外版本
# ============================================

print("\n🎯 创建Rosenbrock 3D可视化...")

# 参数设置
x_init_ros, y_init_ros = -0.5, -0.5
iterations_ros = 30

# 创建网格 - 限制范围避免数值问题
x_ros = np.linspace(-0.8, 1.5, 30)
y_ros = np.linspace(-0.5, 2, 30)
X_ros, Y_ros = np.meshgrid(x_ros, y_ros)
Z_ros = rosenbrock(X_ros, Y_ros)

# 限制Z值范围
Z_ros = np.clip(Z_ros, 0, 100)

# 计算优化路径
paths_ros = {
    "SGD": optimize_sgd(rosenbrock_grad, x_init_ros, y_init_ros, 0.001, iterations_ros),
    "Momentum": optimize_momentum(
        rosenbrock_grad, x_init_ros, y_init_ros, 0.001, 0.9, iterations_ros
    ),
    "Adam": optimize_adam(
        rosenbrock_grad, x_init_ros, y_init_ros, 0.01, iterations_ros
    ),
}

# 创建图表 - 动画版本
fig_ros = go.Figure()

# 添加曲面（静态背景）
fig_ros.add_trace(
    go.Surface(
        x=X_ros,
        y=Y_ros,
        z=Z_ros,
        colorscale="Hot",
        opacity=0.7,
        name="Rosenbrock",
        showscale=True,
        colorbar=dict(title="f(x,y)", x=1.1),
        hovertemplate="x: %{x:.3f}<br>y: %{y:.3f}<br>f: %{z:.3f}<extra></extra>",
    )
)

# 静态Rosenbrock实现
colors_ros = {"SGD": "red", "Momentum": "blue", "Adam": "green"}

# 创建基础图表
fig_ros = go.Figure()

# 添加曲面
fig_ros.add_trace(
    go.Surface(
        x=X_ros,
        y=Y_ros,
        z=Z_ros,
        colorscale="Hot",
        opacity=0.7,
        name="Rosenbrock",
        showscale=True,
        colorbar=dict(title="f(x,y)", x=1.1),
    )
)

# 添加所有优化路径
for name, path in paths_ros.items():
    z_values = np.clip(rosenbrock(path[:, 0], path[:, 1]), 0, 100)

    fig_ros.add_trace(
        go.Scatter3d(
            x=path[:, 0],
            y=path[:, 1],
            z=z_values,
            mode="lines+markers",
            name=name,
            line=dict(color=colors_ros[name], width=3),
            marker=dict(size=2, color=colors_ros[name]),
            hovertemplate=f"{name}<br>x: %{{x:.3f}}<br>y: %{{y:.3f}}<br>f: %{{z:.3f}}<extra></extra>",
        )
    )

# 添加关键点
fig_ros.add_trace(
    go.Scatter3d(
        x=[x_init_ros],
        y=[y_init_ros],
        z=[np.clip(rosenbrock(x_init_ros, y_init_ros), 0, 100)],
        mode="markers",
        name="起始点",
        marker=dict(size=8, color="black"),
        hovertemplate=f"起始点<br>x: {x_init_ros:.3f}<br>y: {y_init_ros:.3f}<extra></extra>",
    )
)

fig_ros.add_trace(
    go.Scatter3d(
        x=[1],
        y=[1],
        z=[0],
        mode="markers",
        name="最优点",
        marker=dict(size=10, color="gold"),
        hovertemplate="最优点<br>x: 1.0<br>y: 1.0<br>f: 0.0<extra></extra>",
    )
)

# 更新布局 - 静态版本
fig_ros.update_layout(
    title="Rosenbrock函数 3D可视化",
    scene=dict(
        xaxis_title="x",
        yaxis_title="y",
        zaxis_title="f(x,y)",
        camera=dict(eye=dict(x=1.2, y=1.2, z=0.8)),
    ),
    height=600,
    margin=dict(t=100, b=60, l=60, r=60),
    legend=dict(x=0.02, y=0.98),
)

output_file2 = os.path.join(output_dir, "rosenbrock_3d.html")
fig_ros.write_html(output_file2, include_mathjax="cdn")
print(f"   ✅ 保存: {output_file2}")

# ============================================
# 创建损失值收敛图表
# ============================================

print("\n🎯 创建损失值收敛图表...")

# 使用Sphere函数展示收敛过程
func_loss = sphere_function
grad_loss = sphere_grad
x_init_loss, y_init_loss = 1.5, 1.5
iterations_loss = 50

# 计算所有算法的损失值历史
loss_history = {}
algorithms_loss = {
    "SGD": {"method": "sgd", "lr": 0.1, "momentum": 0, "color": "red"},
    "Momentum": {"method": "momentum", "lr": 0.1, "momentum": 0.9, "color": "blue"},
    "Adam": {"method": "adam", "lr": 0.1, "momentum": 0, "color": "green"},
    "RMSprop": {"method": "rmsprop", "lr": 0.1, "momentum": 0, "color": "cyan"},
    "AdaGrad": {"method": "adagrad", "lr": 0.1, "momentum": 0, "color": "magenta"},
    "Nesterov": {"method": "nesterov", "lr": 0.1, "momentum": 0.9, "color": "purple"},
}


# 修改优化算法以记录损失值
def optimize_sgd_with_loss(grad_func, x_init, y_init, lr, iterations, func):
    path = [[x_init, y_init]]
    losses = [func(x_init, y_init)]
    x, y = x_init, y_init

    for _ in range(iterations):
        grad = grad_func(x, y)
        grad = np.clip(grad, -100, 100)
        x = x - lr * grad[0]
        y = y - lr * grad[1]
        path.append([x, y])
        losses.append(func(x, y))

    return np.array(path), np.array(losses)


def optimize_momentum_with_loss(
    grad_func, x_init, y_init, lr, momentum, iterations, func
):
    path = [[x_init, y_init]]
    losses = [func(x_init, y_init)]
    x, y = x_init, y_init
    v = np.array([0.0, 0.0])

    for _ in range(iterations):
        grad = grad_func(x, y)
        v = momentum * v + grad
        x = x - lr * v[0]
        y = y - lr * v[1]
        path.append([x, y])
        losses.append(func(x, y))

    return np.array(path), np.array(losses)


def optimize_adam_with_loss(
    grad_func, x_init, y_init, lr, iterations, func, beta1=0.9, beta2=0.999
):
    path = [[x_init, y_init]]
    losses = [func(x_init, y_init)]
    x, y = x_init, y_init
    m = np.array([0.0, 0.0])
    v = np.array([0.0, 0.0])
    epsilon = 1e-8

    for t in range(1, iterations + 1):
        grad = grad_func(x, y)
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad**2
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        x = x - lr * m_hat[0] / (np.sqrt(v_hat[0]) + epsilon)
        y = y - lr * m_hat[1] / (np.sqrt(v_hat[1]) + epsilon)
        path.append([x, y])
        losses.append(func(x, y))

    return np.array(path), np.array(losses)


def optimize_rmsprop_with_loss(
    grad_func, x_init, y_init, lr, iterations, func, beta=0.9, epsilon=1e-8
):
    path = [[x_init, y_init]]
    losses = [func(x_init, y_init)]
    x, y = x_init, y_init
    v = np.array([0.0, 0.0])

    for _ in range(iterations):
        grad = grad_func(x, y)
        v = beta * v + (1 - beta) * grad**2
        x = x - lr * grad[0] / (np.sqrt(v[0]) + epsilon)
        y = y - lr * grad[1] / (np.sqrt(v[1]) + epsilon)
        path.append([x, y])
        losses.append(func(x, y))

    return np.array(path), np.array(losses)


def optimize_adagrad_with_loss(
    grad_func, x_init, y_init, lr, iterations, func, epsilon=1e-8
):
    path = [[x_init, y_init]]
    losses = [func(x_init, y_init)]
    x, y = x_init, y_init
    G = np.array([0.0, 0.0])

    for _ in range(iterations):
        grad = grad_func(x, y)
        G = G + grad**2
        x = x - lr * grad[0] / (np.sqrt(G[0]) + epsilon)
        y = y - lr * grad[1] / (np.sqrt(G[1]) + epsilon)
        path.append([x, y])
        losses.append(func(x, y))

    return np.array(path), np.array(losses)


def optimize_nesterov_with_loss(
    grad_func, x_init, y_init, lr, momentum, iterations, func
):
    path = [[x_init, y_init]]
    losses = [func(x_init, y_init)]
    x, y = x_init, y_init
    v = np.array([0.0, 0.0])

    for _ in range(iterations):
        x_lookahead = x - momentum * v[0]
        y_lookahead = y - momentum * v[1]
        grad = grad_func(x_lookahead, y_lookahead)
        v = momentum * v + grad
        x = x - lr * v[0]
        y = y - lr * v[1]
        path.append([x, y])
        losses.append(func(x, y))

    return np.array(path), np.array(losses)


# 计算所有算法的损失历史
for name, config in algorithms_loss.items():
    if config["method"] == "sgd":
        _, losses = optimize_sgd_with_loss(
            grad_loss,
            x_init_loss,
            y_init_loss,
            config["lr"],
            iterations_loss,
            func_loss,
        )
    elif config["method"] == "momentum":
        _, losses = optimize_momentum_with_loss(
            grad_loss,
            x_init_loss,
            y_init_loss,
            config["lr"],
            config["momentum"],
            iterations_loss,
            func_loss,
        )
    elif config["method"] == "nesterov":
        _, losses = optimize_nesterov_with_loss(
            grad_loss,
            x_init_loss,
            y_init_loss,
            config["lr"],
            config["momentum"],
            iterations_loss,
            func_loss,
        )
    elif config["method"] == "adam":
        _, losses = optimize_adam_with_loss(
            grad_loss,
            x_init_loss,
            y_init_loss,
            config["lr"],
            iterations_loss,
            func_loss,
        )
    elif config["method"] == "rmsprop":
        _, losses = optimize_rmsprop_with_loss(
            grad_loss,
            x_init_loss,
            y_init_loss,
            config["lr"],
            iterations_loss,
            func_loss,
        )
    elif config["method"] == "adagrad":
        _, losses = optimize_adagrad_with_loss(
            grad_loss,
            x_init_loss,
            y_init_loss,
            config["lr"],
            iterations_loss,
            func_loss,
        )

    loss_history[name] = losses

# 创建损失收敛图表
fig_loss = go.Figure()

# 添加每个算法的损失曲线
for name, losses in loss_history.items():
    fig_loss.add_trace(
        go.Scatter(
            x=list(range(len(losses))),
            y=losses,
            mode="lines",
            name=name,
            line=dict(color=algorithms_loss[name]["color"], width=2),
            hovertemplate=f"{name}<br>迭代: %{{x}}<br>损失值: %{{y:.4f}}<extra></extra>",
        )
    )

# 添加公式
fig_loss.add_annotation(
    xref="paper",
    yref="paper",
    x=0.5,
    y=1.05,
    text=r"$\text{损失函数收敛曲线} \quad f(x,y) = x^2 + y^2$",
    showarrow=False,
    font=dict(size=16),
    bgcolor="rgba(255, 250, 205, 0.9)",
    bordercolor="orange",
    borderwidth=2,
    borderpad=10,
    xanchor="center",
)

# 添加说明文本
loss_instructions = (
    "💡 观察要点：<br>"
    "• 收敛速度：算法达到最小值的快慢<br>"
    "• 稳定性：损失曲线的平滑程度<br>"
    "• 最终性能：算法达到的最小损失值<br>"
    "• Adam和RMSprop通常收敛最快"
)

fig_loss.add_annotation(
    xref="paper",
    yref="paper",
    x=0.02,
    y=0.02,
    text=loss_instructions,
    showarrow=False,
    font=dict(size=11),
    bgcolor="rgba(255, 255, 255, 0.9)",
    bordercolor="gray",
    borderwidth=1,
    borderpad=8,
    xanchor="left",
    yanchor="bottom",
    align="left",
)

# 更新布局
fig_loss.update_layout(
    title="梯度下降算法损失值收敛对比",
    xaxis_title="迭代次数",
    yaxis_title="损失值 f(x,y)",
    height=600,
    hovermode="closest",
    legend=dict(
        x=1.02, y=1, bgcolor="rgba(255,255,255,0.9)", bordercolor="gray", borderwidth=1
    ),
    margin=dict(t=120, b=80, l=80, r=150),
    yaxis_type="log",  # 使用对数坐标更好展示收敛过程
)

output_file = os.path.join(output_dir, "loss_convergence.html")
fig_loss.write_html(output_file, include_mathjax="cdn")
print(f"   ✅ 保存: {output_file}")

# ============================================
# 创建算法对比大屏 - 新增
# ============================================

print("\n🎯 创建算法对比大屏...")

# 使用Sphere函数作为简单示例
func_sphere = sphere_function
grad_sphere = sphere_grad
bounds_sphere = [-2, 2]

# 创建网格
x_range = np.linspace(bounds_sphere[0], bounds_sphere[1], 100)
y_range = np.linspace(bounds_sphere[0], bounds_sphere[1], 100)
X_sphere, Y_sphere = np.meshgrid(x_range, y_range)
Z_sphere = sphere_function(X_sphere, Y_sphere)

# 算法配置
algorithms = {
    "SGD": {"method": "sgd", "lr": 0.1, "momentum": 0, "color": "red"},
    "Momentum": {"method": "momentum", "lr": 0.1, "momentum": 0.9, "color": "blue"},
    "Nesterov": {"method": "nesterov", "lr": 0.1, "momentum": 0.9, "color": "purple"},
    "Adam": {"method": "adam", "lr": 0.1, "momentum": 0, "color": "green"},
    "RMSprop": {"method": "rmsprop", "lr": 0.1, "momentum": 0, "color": "cyan"},
    "AdaGrad": {"method": "adagrad", "lr": 0.1, "momentum": 0, "color": "magenta"},
}

x_init_comp, y_init_comp = 1.5, 1.5
iterations_comp = 50

fig_algorithms = make_subplots(
    rows=2,
    cols=3,
    subplot_titles=list(algorithms.keys()),
    specs=[[{"type": "scatter"} for _ in range(3)] for _ in range(2)],
)

for idx, (name, config) in enumerate(algorithms.items()):
    row = idx // 3 + 1
    col = idx % 3 + 1

    # 计算路径
    if config["method"] == "sgd":
        path = optimize_sgd(
            grad_sphere, x_init_comp, y_init_comp, config["lr"], iterations_comp
        )
    elif config["method"] == "momentum":
        path = optimize_momentum(
            grad_sphere,
            x_init_comp,
            y_init_comp,
            config["lr"],
            config["momentum"],
            iterations_comp,
        )
    elif config["method"] == "nesterov":
        path = optimize_nesterov(
            grad_sphere,
            x_init_comp,
            y_init_comp,
            config["lr"],
            config["momentum"],
            iterations_comp,
        )
    elif config["method"] == "adam":
        path = optimize_adam(
            grad_sphere, x_init_comp, y_init_comp, config["lr"], iterations_comp
        )
    elif config["method"] == "rmsprop":
        path = optimize_rmsprop(
            grad_sphere, x_init_comp, y_init_comp, config["lr"], iterations_comp
        )
    elif config["method"] == "adagrad":
        path = optimize_adagrad(
            grad_sphere, x_init_comp, y_init_comp, config["lr"], iterations_comp
        )

    # 添加等高线
    fig_algorithms.add_trace(
        go.Contour(
            x=x_range,
            y=y_range,
            z=Z_sphere,
            colorscale="Viridis",
            opacity=0.3,
            showscale=False,
            hoverinfo="skip",
        ),
        row=row,
        col=col,
    )

    # 添加路径
    fig_algorithms.add_trace(
        go.Scatter(
            x=path[:, 0],
            y=path[:, 1],
            mode="lines+markers",
            name=name,
            line=dict(color=config["color"], width=2),
            marker=dict(size=3, color=config["color"]),
            showlegend=False,
            hovertemplate=f"{name}<br>x: %{{x:.3f}}<br>y: %{{y:.3f}}<extra></extra>",
        ),
        row=row,
        col=col,
    )

    # 添加最优点
    fig_algorithms.add_trace(
        go.Scatter(
            x=[0],
            y=[0],
            mode="markers",
            marker=dict(size=8, color="gold", symbol="circle"),
            showlegend=False,
            hoverinfo="skip",
        ),
        row=row,
        col=col,
    )

    # 添加起始点
    fig_algorithms.add_trace(
        go.Scatter(
            x=[x_init_comp],
            y=[y_init_comp],
            mode="markers",
            marker=dict(size=8, color="black", symbol="circle"),
            showlegend=False,
            hoverinfo="skip",
        ),
        row=row,
        col=col,
    )

    fig_algorithms.update_xaxes(title_text="x", row=row, col=col)
    fig_algorithms.update_yaxes(title_text="y", row=row, col=col)

# 添加算法说明
algorithm_info = (
    "算法特点：<br>"
    + "• SGD: 基础梯度下降<br>"
    + "• Momentum: 加入动量加速<br>"
    + "• Nesterov: 前瞻动量<br>"
    + "• Adam: 自适应学习率<br>"
    + "• RMSprop: 均方根传播<br>"
    + "• AdaGrad: 累积梯度平方"
)

fig_algorithms.add_annotation(
    xref="paper",
    yref="paper",
    x=0.5,
    y=1.05,
    text=r"$\text{梯度下降算法对比} \quad f(x,y) = x^2 + y^2$",
    showarrow=False,
    font=dict(size=16),
    bgcolor="rgba(255, 250, 205, 0.9)",
    bordercolor="orange",
    borderwidth=2,
    borderpad=10,
    xanchor="center",
)

fig_algorithms.add_annotation(
    xref="paper",
    yref="paper",
    x=0.98,
    y=0.02,
    text=algorithm_info,
    showarrow=False,
    font=dict(size=10),
    bgcolor="rgba(255, 255, 255, 0.9)",
    bordercolor="gray",
    borderwidth=1,
    borderpad=8,
    xanchor="right",
    yanchor="bottom",
    align="left",
)

fig_algorithms.update_layout(
    title="梯度下降算法全面对比", height=800, margin=dict(t=120, b=80, l=60, r=60)
)

output_file = os.path.join(output_dir, "algorithms_comparison.html")
fig_algorithms.write_html(output_file, include_mathjax="cdn")
print(f"   ✅ 保存: {output_file}")

# ============================================
# 打印总结
# ============================================

print("\n" + "=" * 60)
print("✨ 交互式梯度下降工具创建完成！")
print("=" * 60)

print(f"\n📂 生成的文件位于: code/{output_dir}/")
print("   1. sphere_easy.html")
print("   2. rosenbrock_hard.html")
print("   3. rastrigin_multimodal.html")
print("   4. beale.html")
print("   5. learning_rate_comparison.html")
print("   6. momentum_comparison.html")
print("   7. 3d_visualization.html (静态版)")
print("   8. rosenbrock_3d.html (静态版)")
print("   9. loss_convergence.html (新增!)")
print("   10. algorithms_comparison.html")

print("\n💡 使用说明:")
print("   • 点击图例显示/隐藏不同的优化算法")
print("   • 悬停查看详细的x, y坐标和函数值")
print("   • 鼠标滚轮缩放，拖动平移")
print("   • 3D视图支持旋转查看不同角度")
print("   • 📊 损失收敛图表：")
print("     - 对数坐标展示收敛过程")
print("     - 比较不同算法的收敛速度")
print("     - 观察算法的稳定性")

print("\n🎯 包含的优化算法:")
print("   • SGD (标准梯度下降)")
print("   • Momentum (动量加速)")
print("   • Nesterov (前瞻动量)")
print("   • Adam (自适应学习率)")
print("   • RMSprop (均方根传播)")
print("   • AdaGrad (累积梯度平方)")

print("\n📊 包含的测试函数:")
print("   • Sphere - 简单凸函数")
print("   • Rosenbrock - 经典困难函数")
print("   • Rastrigin - 多局部极小值")
print("   • Beale - 非凸函数")

print("\n🎨 3D可视化特点:")
print("   • 静态显示完整优化路径")
print("   • 多角度旋转查看")
print("   • 清晰的曲面渲染")
print("   • 优化的数值稳定性")
print("\n🔧 3D可视化改进:")
print("   • 修复了曲面渲染问题")
print("   • 使用稳定的数值范围")
print("   • 改进了交互性和视角")
print("   • 确保所有路径清晰可见")

print("\n" + "=" * 60)
print("🎉 可以在浏览器中打开HTML文件开始实验！")
print("=" * 60)
