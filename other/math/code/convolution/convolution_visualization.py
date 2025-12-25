"""
卷积神经网络交互式可视化脚本
基于 1.convolution.md 文档中的公式
使用 Plotly 创建交互式动画图表，包含数学公式
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# 创建输出目录
output_dir = "convolution"
os.makedirs(output_dir, exist_ok=True)

print("=" * 60)
print("🎨 卷积神经网络交互式可视化")
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
# 1. 卷积的数学定义可视化
# ============================================
print("\n1️⃣ 创建卷积的数学定义可视化...")


def create_convolution_math_definition():
    """创建卷积的数学定义可视化"""

    # 创建1D信号和卷积核
    n_points = 50
    x = np.linspace(-5, 5, n_points)

    # 输入信号 f(k) - 高斯脉冲
    f = np.exp(-(x**2) / 2)

    # 卷积核 g(n-k) - 另一个高斯
    kernel_center = 0
    kernel_width = 1.0
    g = np.exp(-((x - kernel_center) ** 2) / (2 * kernel_width**2))

    # 创建动画帧 - 展示卷积过程
    frames = []
    n_frames = 30
    positions = np.linspace(-3, 3, n_frames)

    for i, pos in enumerate(positions):
        # 翻转的卷积核 g(n-k)
        g_flipped = np.exp(-((x - pos) ** 2) / (2 * kernel_width**2))

        # 计算卷积结果
        conv_result = np.sum(f * g_flipped) * (x[1] - x[0])

        frame_data = [
            # 输入信号
            go.Scatter(
                x=x,
                y=f,
                mode="lines",
                name="输入信号 f(k)",
                line=dict(color="blue", width=3),
            ),
            # 翻转的卷积核
            go.Scatter(
                x=x,
                y=g_flipped,
                mode="lines",
                name="卷积核 g(n-k)",
                line=dict(color="red", width=3),
                fill="tonexty",
                fillcolor="rgba(255, 0, 0, 0.2)",
            ),
            # 乘积
            go.Scatter(
                x=x,
                y=f * g_flipped,
                mode="lines",
                name="乘积 f(k)·g(n-k)",
                line=dict(color="green", width=2, dash="dot"),
            ),
            # 卷积结果点
            go.Scatter(
                x=[pos],
                y=[conv_result],
                mode="markers",
                name=f"卷积结果 (f*g)[{pos:.1f}]",
                marker=dict(color="purple", size=10, symbol="star"),
            ),
        ]

        frames.append(
            go.Frame(
                data=frame_data,
                name=str(i),
                layout=go.Layout(
                    title_text=f"卷积过程 Convolution Process<br>"
                    + f"位置 Position: {pos:.2f}, 结果 Result: {conv_result:.3f}<br>"
                    + f"(f*g)[n] = Σ f[k]·g[n-k]"
                ),
            )
        )

    # 创建主图形
    fig = go.Figure(
        data=[
            go.Scatter(
                x=x,
                y=f,
                mode="lines",
                name="输入信号 f(k)",
                line=dict(color="blue", width=3),
            ),
            go.Scatter(
                x=x,
                y=g,
                mode="lines",
                name="卷积核 g(n-k)",
                line=dict(color="red", width=3),
                fill="tonexty",
                fillcolor="rgba(255, 0, 0, 0.2)",
            ),
            go.Scatter(
                x=x,
                y=f * g,
                mode="lines",
                name="乘积 f(k)·g(n-k)",
                line=dict(color="green", width=2, dash="dot"),
            ),
            go.Scatter(
                x=[0],
                y=[np.sum(f * g) * (x[1] - x[0])],
                mode="markers",
                name="卷积结果",
                marker=dict(color="purple", size=10, symbol="star"),
            ),
        ],
        frames=frames,
    )

    # 添加公式
    fig = add_formula_annotation(
        fig,
        r"$$(f * g)[n] = \sum_{k=-\infty}^{\infty} f[k] \cdot g[n - k]$$",
        x=0.5,
        y=1.08,
    )

    # 添加播放按钮
    fig.update_layout(
        title="卷积的数学定义 Mathematical Definition of Convolution",
        xaxis_title="位置 Position (k/n)",
        yaxis_title="值 Value",
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
        margin=dict(t=130, b=60, l=60, r=60),
    )

    return fig


fig1 = create_convolution_math_definition()
output_file = os.path.join(output_dir, "1_convolution_math.html")
fig1.write_html(output_file, include_mathjax="cdn")
print(f"   ✅ 保存: {output_file}")

# ============================================
# 2. CNN中的卷积操作可视化
# ============================================
print("\n2️⃣ 创建CNN中的卷积操作可视化...")


def create_cnn_convolution():
    """创建CNN中的卷积操作可视化"""

    # 创建一个简单的图像（5x5）
    image = np.array(
        [
            [10, 10, 10, 100, 100],
            [10, 10, 10, 100, 100],
            [10, 10, 10, 100, 100],
            [50, 50, 50, 200, 200],
            [50, 50, 50, 200, 200],
        ]
    )

    # 垂直边缘检测卷积核
    kernel = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])

    # 计算卷积结果
    output_size = image.shape[0] - kernel.shape[0] + 1
    output = np.zeros((output_size, output_size))

    # 创建动画帧 - 展示卷积核在图像上滑动
    frames = []
    positions = []
    for i in range(output_size):
        for j in range(output_size):
            positions.append((i, j))

    for idx, (i, j) in enumerate(positions):
        # 提取当前图像块
        image_patch = image[i : i + kernel.shape[0], j : j + kernel.shape[1]]

        # 计算卷积
        conv_value = np.sum(image_patch * kernel)
        output[i, j] = conv_value

        # 创建热力图数据
        frame_data = [
            # 原始图像
            go.Heatmap(
                z=image,
                colorscale="Viridis",
                name="输入图像 Input Image",
                showscale=False,
                hovertemplate="值: %{z}<extra></extra>",
            ),
            # 卷积核位置
            go.Heatmap(
                x=np.arange(j, j + kernel.shape[1]),
                y=np.arange(i, i + kernel.shape[0]),
                z=kernel,
                colorscale="RdBu",
                name="卷积核 Kernel",
                showscale=False,
                opacity=0.8,
                hovertemplate="核权重: %{z}<extra></extra>",
            ),
            # 输出特征图
            go.Heatmap(
                z=output,
                colorscale="Plasma",
                name="输出特征图 Output Feature Map",
                showscale=True,
                hovertemplate="位置: (%{x}, %{y})<br>值: %{z}<extra></extra>",
            ),
        ]

        frames.append(
            go.Frame(
                data=frame_data,
                name=str(idx),
                layout=go.Layout(
                    title_text=f"CNN卷积操作 CNN Convolution Operation<br>"
                    + f"卷积核位置 Kernel Position: ({i}, {j})<br>"
                    + f"卷积结果 Convolution Result: {conv_value}"
                ),
            )
        )

    # 创建主图形
    fig = go.Figure(
        data=[
            go.Heatmap(
                z=image,
                colorscale="Viridis",
                name="输入图像 Input Image",
                hovertemplate="值: %{z}<extra></extra>",
            ),
            go.Heatmap(
                x=np.arange(0, kernel.shape[1]),
                y=np.arange(0, kernel.shape[0]),
                z=kernel,
                colorscale="RdBu",
                name="卷积核 Kernel",
                opacity=0.8,
                hovertemplate="核权重: %{z}<extra></extra>",
            ),
            go.Heatmap(
                z=np.zeros((output_size, output_size)),
                colorscale="Plasma",
                name="输出特征图 Output Feature Map",
                showscale=True,
                hovertemplate="位置: (%{x}, %{y})<br>值: %{z}<extra></extra>",
            ),
        ],
        frames=frames,
    )

    # 添加公式
    fig = add_formula_annotation(
        fig,
        r"$$\text{Output}[i,j] = \sum_{m=0}^{2}\sum_{n=0}^{2} \text{Input}[i+m,j+n] \times \text{Kernel}[m,n]$$",
        x=0.5,
        y=1.08,
    )

    # 添加播放按钮
    fig.update_layout(
        title="CNN中的卷积操作 Convolution Operation in CNN",
        xaxis_title="列 Column (j)",
        yaxis_title="行 Row (i)",
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
                                frame=dict(duration=500, redraw=True),
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
        margin=dict(t=130, b=60, l=60, r=60),
    )

    return fig


fig2 = create_cnn_convolution()
output_file = os.path.join(output_dir, "2_cnn_convolution.html")
fig2.write_html(output_file, include_mathjax="cdn")
print(f"   ✅ 保存: {output_file}")

# ============================================
# 3. 不同类型卷积核对比
# ============================================
print("\n3️⃣ 创建不同类型卷积核对比可视化...")


def create_kernel_types_comparison():
    """创建不同类型卷积核的对比"""

    # 定义不同类型的卷积核
    kernels = {
        "垂直边缘检测 Vertical Edge": np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]),
        "水平边缘检测 Horizontal Edge": np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]]),
        "拉普拉斯算子 Laplacian": np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]]),
        "高斯模糊 Gaussian Blur": np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16,
        "锐化 Sharpen": np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]),
        "自定义 Custom": np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]]),
    }

    # 创建测试图像
    test_image = np.array(
        [
            [50, 50, 50, 200, 200, 200],
            [50, 50, 50, 200, 200, 200],
            [50, 50, 50, 200, 200, 200],
            [100, 100, 100, 150, 150, 150],
            [100, 100, 100, 150, 150, 150],
            [100, 100, 100, 150, 150, 150],
        ]
    )

    # 创建子图
    fig = make_subplots(
        rows=2,
        cols=3,
        subplot_titles=list(kernels.keys()),
        specs=[
            [{"type": "heatmap"}, {"type": "heatmap"}, {"type": "heatmap"}],
            [{"type": "heatmap"}, {"type": "heatmap"}, {"type": "heatmap"}],
        ],
    )

    # 对每个卷积核进行处理
    for idx, (name, kernel) in enumerate(kernels.items()):
        row = idx // 3 + 1
        col = idx % 3 + 1

        # 计算卷积结果
        output_size = test_image.shape[0] - kernel.shape[0] + 1
        output = np.zeros((output_size, output_size))

        for i in range(output_size):
            for j in range(output_size):
                image_patch = test_image[
                    i : i + kernel.shape[0], j : j + kernel.shape[1]
                ]
                output[i, j] = np.sum(image_patch * kernel)

        # 添加到子图
        fig.add_trace(
            go.Heatmap(
                z=output,
                colorscale="RdBu",
                showscale=False,
                hovertemplate=f"{name}<br>位置: (%{{x}}, %{{y}})<br>值: %{{z}}<extra></extra>",
            ),
            row=row,
            col=col,
        )

    # 添加公式
    fig = add_formula_annotation(
        fig,
        r"$$\text{不同卷积核提取不同特征 Different Kernels Extract Different Features}$$",
        x=0.5,
        y=1.02,
    )

    # 更新坐标轴
    for i in range(2):
        for j in range(3):
            fig.update_xaxes(title_text="列 Column", row=i + 1, col=j + 1)
            fig.update_yaxes(title_text="行 Row", row=i + 1, col=j + 1)

    fig.update_layout(
        title_text="不同类型卷积核对比 Comparison of Different Kernel Types",
        height=600,
        margin=dict(t=120, b=60, l=60, r=60),
    )

    return fig


fig3 = create_kernel_types_comparison()
output_file = os.path.join(output_dir, "3_kernel_types.html")
fig3.write_html(output_file, include_mathjax="cdn")
print(f"   ✅ 保存: {output_file}")

# ============================================
# 4. 权重共享可视化
# ============================================
print("\n4️⃣ 创建权重共享可视化...")


def create_weight_sharing():
    """创建权重共享概念的可视化"""

    # 创建一个较大的输入图像
    input_image = np.random.randint(0, 100, (8, 8))

    # 3x3卷积核
    kernel = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])

    # 计算所有位置的卷积
    output_size = input_image.shape[0] - kernel.shape[0] + 1
    output = np.zeros((output_size, output_size))

    for i in range(output_size):
        for j in range(output_size):
            image_patch = input_image[i : i + kernel.shape[0], j : j + kernel.shape[1]]
            output[i, j] = np.sum(image_patch * kernel)

    # 创建子图
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=(
            "输入图像 Input Image",
            "权重共享的卷积核 Weight-Shared Kernel",
            "输出特征图 Output Feature Map",
        ),
        specs=[[{"type": "heatmap"}, {"type": "heatmap"}, {"type": "heatmap"}]],
    )

    # 输入图像
    fig.add_trace(
        go.Heatmap(
            z=input_image,
            colorscale="Viridis",
            name="输入图像",
            showscale=False,
            hovertemplate="值: %{z}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # 卷积核（显示多个位置但权重相同）
    fig.add_trace(
        go.Heatmap(
            z=kernel,
            colorscale="RdBu",
            name="卷积核",
            showscale=False,
            hovertemplate="核权重: %{z}<extra></extra>",
        ),
        row=1,
        col=2,
    )

    # 输出特征图
    fig.add_trace(
        go.Heatmap(
            z=output,
            colorscale="Plasma",
            name="输出",
            showscale=True,
            hovertemplate="位置: (%{x}, %{y})<br>值: %{z}<extra></extra>",
        ),
        row=1,
        col=3,
    )

    # 添加连接线示意权重共享
    annotations = []
    for i in range(min(3, output_size)):
        for j in range(min(3, output_size)):
            annotations.append(
                dict(
                    x=0.5 + j * 0.15,
                    y=0.5 + i * 0.15,
                    xref="x2 domain",
                    yref="y2 domain",
                    text=f"({i},{j})",
                    showarrow=False,
                    font=dict(size=10, color="black"),
                )
            )

    # 添加公式
    fig = add_formula_annotation(
        fig,
        r"$$\text{权重共享 Weight Sharing: 同一个核在所有位置使用相同权重}$$",
        x=0.5,
        y=1.05,
    )

    fig.update_layout(
        title_text="权重共享概念 Weight Sharing Concept",
        height=500,
        annotations=annotations,
        margin=dict(t=120, b=60, l=60, r=60),
    )

    # 更新坐标轴
    fig.update_xaxes(title_text="列 Column", row=1, col=1)
    fig.update_yaxes(title_text="行 Row", row=1, col=1)
    fig.update_xaxes(title_text="核列 Kernel Col", row=1, col=2)
    fig.update_yaxes(title_text="核行 Kernel Row", row=1, col=2)
    fig.update_xaxes(title_text="列 Column", row=1, col=3)
    fig.update_yaxes(title_text="行 Row", row=1, col=3)

    return fig


fig4 = create_weight_sharing()
output_file = os.path.join(output_dir, "4_weight_sharing.html")
fig4.write_html(output_file, include_mathjax="cdn")
print(f"   ✅ 保存: {output_file}")

# ============================================
# 5. 偏置项的作用可视化
# ============================================
print("\n5️⃣ 创建偏置项的作用可视化...")


def create_bias_effect():
    """创建偏置项作用的可视化"""

    # 创建输入数据
    x = np.linspace(-5, 5, 100)

    # 卷积结果（无偏置）
    conv_no_bias = np.maximum(0, x)  # 简化的ReLU激活

    # 不同偏置值的影响
    bias_values = [-2, 0, 2]
    colors = ["blue", "red", "green"]

    # 创建图形
    fig = go.Figure()

    # 添加无偏置的曲线
    fig.add_trace(
        go.Scatter(
            x=x,
            y=conv_no_bias,
            mode="lines",
            name="无偏置 No Bias (b=0)",
            line=dict(color="gray", width=3, dash="dash"),
            hovertemplate="输入: %{x:.2f}<br>输出: %{y:.2f}<extra></extra>",
        )
    )

    # 添加不同偏置的曲线
    for bias, color in zip(bias_values, colors):
        with_bias = np.maximum(0, x + bias)
        fig.add_trace(
            go.Scatter(
                x=x,
                y=with_bias,
                mode="lines",
                name=f"偏置 Bias = {bias}",
                line=dict(color=color, width=3),
                hovertemplate=f"偏置={bias}<br>输入: %{{x:.2f}}<br>输出: %{{y:.2f}}<extra></extra>",
            )
        )

    # 标记激活点
    for bias, color in zip(bias_values, colors):
        activation_point = -bias
        if -5 <= activation_point <= 5:
            fig.add_trace(
                go.Scatter(
                    x=[activation_point],
                    y=[0],
                    mode="markers",
                    name=f"激活点 Activation (b={bias})",
                    marker=dict(color=color, size=10, symbol="circle"),
                    showlegend=False,
                    hovertemplate=f"偏置={bias}<br>激活点: {activation_point:.2f}<extra></extra>",
                )
            )

    # 添加公式
    fig = add_formula_annotation(
        fig,
        r"$$\text{输出} = \max(0, \text{输入} + \text{偏置}) \quad \text{偏置控制激活阈值}$$",
        x=0.5,
        y=1.05,
    )

    fig.update_layout(
        title="偏置项的作用 The Role of Bias in CNN",
        xaxis_title="输入值 Input Value",
        yaxis_title="输出值 Output Value (after ReLU)",
        height=600,
        legend=dict(x=0.02, y=0.98, bgcolor="rgba(255,255,255,0.8)"),
        margin=dict(t=120, b=60, l=60, r=60),
    )

    return fig


fig5 = create_bias_effect()
output_file = os.path.join(output_dir, "5_bias_effect.html")
fig5.write_html(output_file, include_mathjax="cdn")
print(f"   ✅ 保存: {output_file}")

# ============================================
# 6. 互相关 vs 卷积对比
# ============================================
print("\n6️⃣ 创建互相关 vs 卷积对比可视化...")


def create_correlation_vs_convolution():
    """创建互相关与卷积的对比"""

    # 输入信号
    x = np.linspace(-4, 4, 50)
    signal = np.exp(-(x**2) / 2)  # 高斯信号

    # 卷积核
    kernel = np.array([1, 2, 3, 2, 1]) / 9  # 简单的平滑核

    # 位置演示
    position = 1.0
    kernel_indices = np.arange(-2, 3)
    kernel_values = kernel

    # 互相关（不翻转）
    correlation_values = []
    for i, xi in enumerate(x):
        if abs(xi - position) < 2.5:
            # 简化：根据距离计算权重
            correlation_values.append(np.exp(-((xi - position) ** 2) / 2))
        else:
            correlation_values.append(0)

    # 卷积（翻转）
    convolution_values = []
    for i, xi in enumerate(x):
        if abs(xi - position) < 2.5:
            # 翻转后的权重
            convolution_values.append(np.exp(-((xi + position) ** 2) / 2))
        else:
            convolution_values.append(0)

    # 创建子图
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("互相关 Cross-Correlation (不翻转)", "卷积 Convolution (翻转)"),
        specs=[[{"type": "scatter"}, {"type": "scatter"}]],
    )

    # 互相关图
    fig.add_trace(
        go.Scatter(
            x=x,
            y=signal,
            mode="lines",
            name="输入信号",
            line=dict(color="blue", width=3),
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=[position + kernel_indices],
            y=[kernel_values],
            mode="markers+lines",
            name="卷积核（不翻转）",
            line=dict(color="red", width=2),
            marker=dict(size=8, color="red"),
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    # 卷积图
    fig.add_trace(
        go.Scatter(
            x=x,
            y=signal,
            mode="lines",
            name="输入信号",
            line=dict(color="blue", width=3),
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    fig.add_trace(
        go.Scatter(
            x=[position + kernel_indices[::-1]],
            y=[kernel_values],
            mode="markers+lines",
            name="卷积核（翻转）",
            line=dict(color="red", width=2),
            marker=dict(size=8, color="red"),
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    # 添加公式
    fig = add_formula_annotation(
        fig,
        r"$$\text{互相关: } (f ⋆ g)[n] = \sum f[k] \cdot g[n+k] \quad \text{vs} \quad \text{卷积: } (f * g)[n] = \sum f[k] \cdot g[n-k]$$",
        x=0.5,
        y=1.05,
    )

    fig.update_layout(
        title_text="互相关 vs 卷积 Cross-Correlation vs Convolution",
        height=500,
        margin=dict(t=120, b=60, l=60, r=60),
    )

    # 更新坐标轴
    fig.update_xaxes(title_text="位置 Position", row=1, col=1)
    fig.update_yaxes(title_text="值 Value", row=1, col=1)
    fig.update_xaxes(title_text="位置 Position", row=1, col=2)
    fig.update_yaxes(title_text="值 Value", row=1, col=2)

    return fig


fig6 = create_correlation_vs_convolution()
output_file = os.path.join(output_dir, "6_correlation_vs_convolution.html")
fig6.write_html(output_file, include_mathjax="cdn")
print(f"   ✅ 保存: {output_file}")

# ============================================
# 打印计算示例
# ============================================
print("\n" + "=" * 60)
print("📊 卷积神经网络计算示例")
print("=" * 60)

print("\n1️⃣ 卷积的数学定义:")
print("   (f * g)[n] = Σ f[k] · g[n-k]")
print("   本质：当前输出 = 历史输入的加权记忆汇总")
print("   应用：信号处理、图像滤波、特征提取")

print("\n2️⃣ CNN中的卷积特点:")
print("   - 互相关操作（不翻转核）")
print("   - 权重共享（同一核扫描全图）")
print("   - 偏置项（调整激活阈值）")
print("   - 多通道处理（RGB图像）")

print("\n3️⃣ 卷积核类型:")
print("   - 边缘检测：Sobel、Prewitt、Laplacian")
print("   - 模糊平滑：均值、高斯滤波")
print("   - 锐化增强：增强细节")
print("   - 可学习核：通过训练自动优化")

print("\n4️⃣ 现代卷积变体:")
print("   - 深度可分离卷积：减少参数量")
print("   - 空洞卷积：扩大感受野")
print("   - 可变形卷积：自适应采样")
print("   - 组卷积：提高计算效率")

print("\n" + "=" * 60)
print("✨ 所有交互式可视化已创建完成！")
print("=" * 60)
print(f"\n📂 生成的文件位于: code/{output_dir}/")
print("   1. 1_convolution_math.html - 卷积的数学定义")
print("   2. 2_cnn_convolution.html - CNN中的卷积操作")
print("   3. 3_kernel_types.html - 不同类型卷积核对比")
print("   4. 4_weight_sharing.html - 权重共享概念")
print("   5. 5_bias_effect.html - 偏置项的作用")
print("   6. 6_correlation_vs_convolution.html - 互相关 vs 卷积")
print("\n💡 在浏览器中打开这些 HTML 文件即可查看交互式可视化！")
print("=" * 60)
