"""
希尔伯特空间与傅里叶变换交互式可视化脚本
基于 12.Hilbert_space.md 文档中的公式
使用 Plotly 和 sklearn 创建交互式动画图表，包含数学公式
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.preprocessing import StandardScaler
import os

# 创建输出目录
output_dir = "hilbert_space"
os.makedirs(output_dir, exist_ok=True)

print("=" * 60)
print("🎨 希尔伯特空间与傅里叶变换交互式可视化")
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
# 1. 希尔伯特空间的内积可视化
# ============================================
print("\n1️⃣ 创建希尔伯特空间的内积可视化...")


def create_hilbert_inner_product():
    """创建希尔伯特空间内积的直观理解"""

    # 创建2D向量（有限维希尔伯特空间）
    theta1 = np.pi / 4  # 45度
    theta2 = np.pi / 3  # 60度

    v1 = np.array([np.cos(theta1), np.sin(theta1)]) * 2
    v2 = np.array([np.cos(theta2), np.sin(theta2)]) * 1.5

    # 计算内积
    inner_product = np.dot(v1, v2)
    angle = np.arccos(inner_product / (np.linalg.norm(v1) * np.linalg.norm(v2)))

    # 创建图形
    fig = go.Figure()

    # 添加坐标轴
    fig.add_trace(
        go.Scatter(
            x=[0, 3],
            y=[0, 0],
            mode="lines",
            line=dict(color="black", width=1),
            name="x轴",
            showlegend=False,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[0, 0],
            y=[0, 3],
            mode="lines",
            line=dict(color="black", width=1),
            name="y轴",
            showlegend=False,
        )
    )

    # 添加向量
    fig.add_trace(
        go.Scatter(
            x=[0, v1[0]],
            y=[0, v1[1]],
            mode="lines+markers",
            line=dict(color="blue", width=4),
            marker=dict(size=10),
            name=f"向量 v₁ = ({v1[0]:.2f}, {v1[1]:.2f})",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[0, v2[0]],
            y=[0, v2[1]],
            mode="lines+markers",
            line=dict(color="red", width=4),
            marker=dict(size=10),
            name=f"向量 v₂ = ({v2[0]:.2f}, {v2[1]:.2f})",
        )
    )

    # 添加投影（内积的几何意义）
    projection_length = inner_product / np.linalg.norm(v2)
    projection = projection_length * v2 / np.linalg.norm(v2)

    fig.add_trace(
        go.Scatter(
            x=[v1[0], projection[0]],
            y=[v1[1], projection[1]],
            mode="lines",
            line=dict(color="green", width=2, dash="dot"),
            name="投影",
            showlegend=False,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[0, projection[0]],
            y=[0, projection[1]],
            mode="lines",
            line=dict(color="green", width=3),
            name=f"v₁在v₂上的投影",
        )
    )

    # 添加角度弧
    theta_range = np.linspace(0, angle, 20)
    arc_x = 0.5 * np.cos(theta_range)
    arc_y = 0.5 * np.sin(theta_range)

    fig.add_trace(
        go.Scatter(
            x=arc_x,
            y=arc_y,
            mode="lines",
            line=dict(color="purple", width=2),
            name=f"角度 = {np.degrees(angle):.1f}°",
            showlegend=False,
        )
    )

    # 添加公式
    fig = add_formula_annotation(
        fig,
        r"$\langle v_1, v_2 \rangle = \|v_1\| \cdot \|v_2\| \cdot \cos(\theta)$",
        x=0.5,
        y=1.08,
    )

    fig.update_layout(
        title="希尔伯特空间的内积几何意义<br>Geometric Meaning of Inner Product in Hilbert Space",
        xaxis_title="x",
        yaxis_title="y",
        xaxis=dict(scaleanchor="y", scaleratio=1),
        yaxis=dict(scaleanchor="x", scaleratio=1),
        height=600,
        legend=dict(x=0.02, y=0.98, bgcolor="rgba(255,255,255,0.8)"),
        margin=dict(t=130, b=60, l=60, r=60),
    )

    return fig, inner_product, angle


fig1, inner_prod, angle_val = create_hilbert_inner_product()
output_file = os.path.join(output_dir, "1_hilbert_inner_product.html")
fig1.write_html(output_file, include_mathjax="cdn")
print(f"   ✅ 保存: {output_file}")
print(f"   📊 内积值: {inner_prod:.3f}, 夹角: {np.degrees(angle_val):.1f}°")

# ============================================
# 2. 傅里叶变换作为酉算子可视化
# ============================================
print("\n2️⃣ 创建傅里叶变换作为酉算子可视化...")


def create_fourier_unitary_operator():
    """展示傅里叶变换保持内积不变的性质"""

    # 创建信号
    t = np.linspace(0, 2 * np.pi, 100)
    f1 = np.sin(t) + 0.5 * np.sin(3 * t)
    f2 = np.cos(t) + 0.3 * np.cos(2 * t)

    # 计算傅里叶变换（简化版）
    def simple_fft(signal):
        N = len(signal)
        n = np.arange(N)
        k = n.reshape((N, 1))
        M = np.exp(-2j * np.pi * k * n / N)
        return np.dot(M, signal)

    f1_fft = simple_fft(f1)
    f2_fft = simple_fft(f2)

    # 计算内积
    inner_product_time = np.dot(f1, f2)
    inner_product_freq = np.dot(f1_fft, np.conj(f2_fft))

    # 创建子图
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=("时域信号 f₁(t)", "时域信号 f₂(t)", "频域 F₁(ω)", "频域 F₂(ω)"),
        specs=[
            [{"type": "scatter"}, {"type": "scatter"}],
            [{"type": "scatter"}, {"type": "scatter"}],
        ],
    )

    # 时域信号
    fig.add_trace(
        go.Scatter(
            x=t,
            y=f1,
            mode="lines",
            name="f₁(t) = sin(t) + 0.5sin(3t)",
            line=dict(color="blue", width=2),
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=t,
            y=f2,
            mode="lines",
            name="f₂(t) = cos(t) + 0.3cos(2t)",
            line=dict(color="red", width=2),
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    # 频域信号
    freqs = np.arange(len(f1_fft))
    fig.add_trace(
        go.Scatter(
            x=freqs[:50],
            y=np.abs(f1_fft[:50]),
            mode="lines+markers",
            name="|F₁(ω)|",
            line=dict(color="blue", width=2),
            marker=dict(size=4),
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=freqs[:50],
            y=np.abs(f2_fft[:50]),
            mode="lines+markers",
            name="|F₂(ω)|",
            line=dict(color="red", width=2),
            marker=dict(size=4),
            showlegend=False,
        ),
        row=2,
        col=2,
    )

    # 添加内积信息
    fig = add_formula_annotation(
        fig,
        f"内积时域: {inner_product_time:.2f}<br>内积频域: {inner_product_freq.real:.2f}<br>"
        + r"Parseval恒等式: $\langle f_1, f_2 \rangle = \langle \hat{f}_1, \hat{f}_2 \rangle$",
        x=0.5,
        y=1.05,
    )

    fig.update_layout(
        title_text="傅里叶变换作为酉算子（保持内积不变）<br>Fourier Transform as Unitary Operator",
        height=600,
        margin=dict(t=120, b=60, l=60, r=60),
    )

    return fig, inner_product_time, inner_product_freq.real


fig2, inner_time, inner_freq = create_fourier_unitary_operator()
output_file = os.path.join(output_dir, "2_fourier_unitary.html")
fig2.write_html(output_file, include_mathjax="cdn")
print(f"   ✅ 保存: {output_file}")
print(f"   📊 时域内积: {inner_time:.2f}, 频域内积: {inner_freq:.2f}")

# ============================================
# 3. 卷积定理的动态演示
# ============================================
print("\n3️⃣ 创建卷积定理的动态演示...")


def create_convolution_theorem():
    """动态展示卷积定理：时域卷积 = 频域相乘"""

    # 创建信号
    x = np.linspace(-5, 5, 200)
    signal = np.exp(-(x**2) / 2)  # 高斯信号
    kernel = np.sinc(x)  # sinc函数作为卷积核

    # 计算卷积
    convolution = np.convolve(signal, kernel, mode="same") * (x[1] - x[0])

    # 计算傅里叶变换
    signal_fft = np.fft.fftshift(np.fft.fft(signal))
    kernel_fft = np.fft.fftshift(np.fft.fft(kernel))
    product_fft = signal_fft * kernel_fft
    convolution_from_fft = np.real(np.fft.ifft(np.fft.ifftshift(product_fft)))

    # 创建动画帧
    frames = []
    n_frames = 30
    frequencies = np.linspace(0, len(x) // 2, n_frames, dtype=int)

    for i, freq_idx in enumerate(frequencies):
        frame_data = [
            # 时域信号
            go.Scatter(
                x=x,
                y=signal,
                mode="lines",
                name="信号 f(t)",
                line=dict(color="blue", width=2),
                showlegend=False,
            ),
            go.Scatter(
                x=x,
                y=kernel,
                mode="lines",
                name="核 g(t)",
                line=dict(color="red", width=2),
                showlegend=False,
            ),
            go.Scatter(
                x=x,
                y=convolution,
                mode="lines",
                name="卷积 (f*g)(t)",
                line=dict(color="green", width=3),
                showlegend=False,
            ),
            # 频域表示
            go.Scatter(
                x=np.fft.fftshift(np.fft.fftfreq(len(x))),
                y=np.abs(signal_fft),
                mode="lines",
                name="|F(ω)|",
                line=dict(color="blue", width=2),
                showlegend=False,
            ),
            go.Scatter(
                x=np.fft.fftshift(np.fft.fftfreq(len(x))),
                y=np.abs(kernel_fft),
                mode="lines",
                name="|G(ω)|",
                line=dict(color="red", width=2),
                showlegend=False,
            ),
            go.Scatter(
                x=np.fft.fftshift(np.fft.fftfreq(len(x))),
                y=np.abs(product_fft),
                mode="lines",
                name="|F(ω)·G(ω)|",
                line=dict(color="green", width=3),
                showlegend=False,
            ),
            # 频率指示器
            go.Scatter(
                x=[np.fft.fftshift(np.fft.fftfreq(len(x)))[freq_idx]],
                y=[0],
                mode="markers",
                name="当前频率",
                marker=dict(color="purple", size=10, symbol="star"),
                showlegend=False,
            ),
        ]

        frames.append(
            go.Frame(
                data=frame_data,
                name=str(i),
                layout=go.Layout(
                    title_text=f"卷积定理演示 Convolution Theorem<br>"
                    + f"频率索引 Frequency Index: {freq_idx}<br>"
                    + f"时域卷积 Time Convolution = 频域相乘 Frequency Multiplication"
                ),
            )
        )

    # 创建主图形
    fig = go.Figure(
        data=[
            go.Scatter(
                x=x,
                y=signal,
                mode="lines",
                name="信号 f(t)",
                line=dict(color="blue", width=2),
            ),
            go.Scatter(
                x=x,
                y=kernel,
                mode="lines",
                name="核 g(t)",
                line=dict(color="red", width=2),
            ),
            go.Scatter(
                x=x,
                y=convolution,
                mode="lines",
                name="卷积 (f*g)(t)",
                line=dict(color="green", width=3),
            ),
        ],
        frames=frames,
    )

    # 添加公式
    fig = add_formula_annotation(
        fig,
        r"$\mathcal{F}\{f * g\} = \mathcal{F}\{f\} \cdot \mathcal{F}\{g\}$",
        x=0.5,
        y=1.05,
    )

    # 添加播放按钮
    fig.update_layout(
        title_text="卷积定理动态演示 Dynamic Convolution Theorem",
        xaxis_title="时间/位置 Time/Position",
        yaxis_title="幅度 Amplitude",
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
        height=500,
        margin=dict(t=130, b=60, l=60, r=60),
    )

    return fig


fig3 = create_convolution_theorem()
output_file = os.path.join(output_dir, "3_convolution_theorem.html")
fig3.write_html(output_file, include_mathjax="cdn")
print(f"   ✅ 保存: {output_file}")

# ============================================
# 4. CNN滤波器的频域特性分析
# ============================================
print("\n4️⃣ 创建CNN滤波器的频域特性分析...")


def create_cnn_frequency_analysis():
    """分析CNN滤波器的频域特性"""

    # 创建不同的卷积核
    kernels = {
        "低通滤波器 Low-pass": np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16,
        "高通滤波器 High-pass": np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]) / 8,
        "边缘检测 Edge": np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]) / 3,
        "拉普拉斯 Laplacian": np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]]),
    }

    # 创建测试图像
    test_image = np.zeros((64, 64))
    # 添加不同频率的成分
    y, x = np.mgrid[0:64, 0:64]
    test_image += np.sin(2 * np.pi * x / 32)  # 低频
    test_image += 0.5 * np.sin(2 * np.pi * x / 8)  # 中频
    test_image += 0.3 * np.sin(2 * np.pi * x / 4)  # 高频

    # 创建子图
    fig = make_subplots(
        rows=2,
        cols=4,
        subplot_titles=list(kernels.keys()) + ["频谱 Spectrum"],
        specs=[
            [
                {"type": "heatmap"},
                {"type": "heatmap"},
                {"type": "heatmap"},
                {"type": "heatmap"},
            ],
            [
                {"type": "scatter"},
                {"type": "scatter"},
                {"type": "scatter"},
                {"type": "scatter"},
            ],
        ],
    )

    # 分析每个核的频域特性
    for idx, (name, kernel) in enumerate(kernels.items()):
        # 显示核
        fig.add_trace(
            go.Heatmap(
                z=kernel,
                colorscale="RdBu",
                showscale=False,
                hovertemplate=f"{name}<br>值: %{{z}}<extra></extra>",
            ),
            row=1,
            col=idx + 1,
        )

        # 计算并显示频谱
        kernel_padded = np.zeros((64, 64))
        h, w = kernel.shape
        kernel_padded[:h, :w] = kernel

        kernel_fft = np.fft.fftshift(np.fft.fft2(kernel_padded))
        freqs = np.fft.fftshift(np.fft.fftfreq(64))

        # 取中心剖面
        center_slice = np.abs(kernel_fft[32, :])

        fig.add_trace(
            go.Scatter(
                x=freqs,
                y=center_slice,
                mode="lines",
                name=name,
                line=dict(width=2),
                showlegend=False,
            ),
            row=2,
            col=idx + 1,
        )

    # 添加测试图像的频谱
    image_fft = np.fft.fftshift(np.fft.fft2(test_image))
    fig.add_trace(
        go.Heatmap(
            z=np.log10(np.abs(image_fft) + 1e-10),
            colorscale="Viridis",
            showscale=True,
            hovertemplate="频率: (%{x}, %{y})<br>幅度: %{z}<extra></extra>",
        ),
        row=1,
        col=4,
    )

    # 添加测试图像频谱的剖面
    image_spectrum = np.abs(image_fft[32, :])
    fig.add_trace(
        go.Scatter(
            x=freqs,
            y=image_spectrum,
            mode="lines",
            name="测试图像频谱",
            line=dict(color="black", width=3),
            showlegend=False,
        ),
        row=2,
        col=4,
    )

    # 添加公式
    fig = add_formula_annotation(
        fig, r"CNN滤波器频域特性: 低通滤波器保留低频，高通滤波器增强高频", x=0.5, y=1.05
    )

    fig.update_layout(
        title_text="CNN滤波器的频域特性分析 Frequency Analysis of CNN Filters",
        height=600,
        margin=dict(t=120, b=60, l=60, r=60),
    )

    return fig


fig4 = create_cnn_frequency_analysis()
output_file = os.path.join(output_dir, "4_cnn_frequency_analysis.html")
fig4.write_html(output_file, include_mathjax="cdn")
print(f"   ✅ 保存: {output_file}")

# ============================================
# 5. 图傅里叶变换可视化
# ============================================
print("\n5️⃣ 创建图傅里叶变换可视化...")


def create_graph_fourier_transform():
    """可视化图傅里叶变换和图频率概念"""

    # 创建一个简单的图结构
    # 邻接矩阵
    A = np.array(
        [
            [0, 1, 1, 0, 0, 0],
            [1, 0, 1, 1, 0, 0],
            [1, 1, 0, 1, 1, 0],
            [0, 1, 1, 0, 1, 1],
            [0, 0, 1, 1, 0, 1],
            [0, 0, 0, 1, 1, 0],
        ]
    )

    # 计算度矩阵
    D = np.diag(np.sum(A, axis=1))

    # 计算拉普拉斯矩阵
    L = D - A

    # 特征分解
    eigenvalues, eigenvectors = np.linalg.eigh(L)

    # 创建图信号
    signal = np.array([1, 2, 3, 2, 1, 0])

    # 图傅里叶变换
    signal_fft = eigenvectors.T @ signal

    # 创建子图
    fig = make_subplots(
        rows=2,
        cols=3,
        subplot_titles=(
            "图结构 Graph Structure",
            "图信号 Graph Signal",
            "特征值 Eigenvalues",
            "特征向量 Eigenvectors",
            "图傅里叶变换 Graph FFT",
            "重构信号 Reconstructed",
        ),
        specs=[
            [{"type": "scatter"}, {"type": "scatter"}, {"type": "scatter"}],
            [{"type": "scatter"}, {"type": "scatter"}, {"type": "scatter"}],
        ],
    )

    # 绘制图结构
    pos = np.array([[0, 0], [1, 1], [2, 0], [3, 1], [4, 0], [5, 1]])

    # 添加边
    edge_x = []
    edge_y = []
    for i in range(len(A)):
        for j in range(i + 1, len(A)):
            if A[i, j] > 0:
                edge_x.extend([pos[i, 0], pos[j, 0], None])
                edge_y.extend([pos[i, 1], pos[j, 1], None])

    fig.add_trace(
        go.Scatter(
            x=edge_x,
            y=edge_y,
            mode="lines",
            line=dict(width=2, color="gray"),
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    # 添加节点
    fig.add_trace(
        go.Scatter(
            x=pos[:, 0],
            y=pos[:, 1],
            mode="markers+text",
            marker=dict(size=20, color="lightblue"),
            text=[f"{i}" for i in range(len(A))],
            textposition="middle center",
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    # 图信号
    fig.add_trace(
        go.Scatter(
            x=np.arange(len(signal)),
            y=signal,
            mode="lines+markers",
            line=dict(color="blue", width=3),
            marker=dict(size=10),
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    # 特征值
    fig.add_trace(
        go.Scatter(
            x=np.arange(len(eigenvalues)),
            y=eigenvalues,
            mode="lines+markers",
            line=dict(color="red", width=3),
            marker=dict(size=10),
            showlegend=False,
        ),
        row=1,
        col=3,
    )

    # 特征向量（前3个）
    colors = ["green", "orange", "purple"]
    for i in range(min(3, len(eigenvectors[0]))):
        fig.add_trace(
            go.Scatter(
                x=np.arange(len(eigenvectors)),
                y=eigenvectors[:, i],
                mode="lines+markers",
                line=dict(color=colors[i], width=2),
                marker=dict(size=6),
                name=f"特征向量 {i+1}",
                showlegend=False if i > 0 else True,
            ),
            row=2,
            col=1,
        )

    # 图傅里叶变换
    fig.add_trace(
        go.Scatter(
            x=np.arange(len(signal_fft)),
            y=np.abs(signal_fft),
            mode="lines+markers",
            line=dict(color="red", width=3),
            marker=dict(size=10),
            showlegend=False,
        ),
        row=2,
        col=2,
    )

    # 重构信号
    reconstructed = eigenvectors @ signal_fft
    fig.add_trace(
        go.Scatter(
            x=np.arange(len(reconstructed)),
            y=reconstructed,
            mode="lines+markers",
            line=dict(color="green", width=3, dash="dash"),
            marker=dict(size=10),
            showlegend=False,
        ),
        row=2,
        col=3,
    )

    # 添加原始信号对比
    fig.add_trace(
        go.Scatter(
            x=np.arange(len(signal)),
            y=signal,
            mode="lines+markers",
            line=dict(color="blue", width=2),
            marker=dict(size=8),
            name="原始信号",
            showlegend=False,
        ),
        row=2,
        col=3,
    )

    # 添加公式
    fig = add_formula_annotation(
        fig, r"图傅里叶变换: $\hat{x} = U^T x$, 其中 $L = U\Lambda U^T$", x=0.5, y=1.05
    )

    fig.update_layout(
        title_text="图傅里叶变换与图频率 Graph Fourier Transform & Graph Frequencies",
        height=600,
        margin=dict(t=120, b=60, l=60, r=60),
    )

    return fig


fig5 = create_graph_fourier_transform()
output_file = os.path.join(output_dir, "5_graph_fourier.html")
fig5.write_html(output_file, include_mathjax="cdn")
print(f"   ✅ 保存: {output_file}")

# ============================================
# 6. Transformer位置编码的频率分析
# ============================================
print("\n6️⃣ 创建Transformer位置编码的频率分析...")


def create_transformer_positional_encoding():
    """分析Transformer位置编码的频率特性"""

    # 生成位置编码
    max_len = 100
    d_model = 64

    def get_positional_encoding(max_len, d_model):
        position = np.arange(max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))

        pe = np.zeros((max_len, d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)

        return pe

    pe = get_positional_encoding(max_len, d_model)

    # 分析不同频率分量
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "位置编码矩阵 Positional Encoding Matrix",
            "低频分量 Low Frequency",
            "中频分量 Medium Frequency",
            "高频分量 High Frequency",
        ),
        specs=[
            [{"type": "heatmap"}, {"type": "scatter"}],
            [{"type": "scatter"}, {"type": "scatter"}],
        ],
    )

    # 位置编码热力图
    fig.add_trace(
        go.Heatmap(
            z=pe.T,
            colorscale="RdBu",
            showscale=True,
            hovertemplate="位置: %{x}<br>维度: %{y}<br>值: %{z}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # 不同频率分量
    frequencies = [
        (0, "低频", "blue"),
        (d_model // 4, "中频", "green"),
        (d_model // 2, "高频", "red"),
    ]

    for idx, (dim, freq_name, color) in enumerate(frequencies):
        row = (idx // 2) + 1
        col = (idx % 2) + 1

        if idx == 0:  # 跳过热力图位置
            continue

        fig.add_trace(
            go.Scatter(
                x=np.arange(max_len),
                y=pe[:, dim],
                mode="lines",
                line=dict(color=color, width=2),
                name=freq_name,
                showlegend=False,
            ),
            row=row,
            col=col,
        )

    # 计算频率谱
    freq_spectrum = np.abs(np.fft.fft(pe, axis=0))

    # 添加频率谱分析
    fig2 = go.Figure()

    for dim, freq_name, color in frequencies:
        fig2.add_trace(
            go.Scatter(
                x=np.arange(len(freq_spectrum) // 2),
                y=freq_spectrum[: len(freq_spectrum) // 2, dim],
                mode="lines",
                line=dict(color=color, width=2),
                name=freq_name,
            )
        )

    # 添加正交性分析
    fig3 = go.Figure()

    # 计算不同位置编码之间的内积
    positions = [0, 10, 20, 30, 40, 50]
    for i, pos1 in enumerate(positions):
        for j, pos2 in enumerate(positions):
            if i < j:
                inner_prod = np.dot(pe[pos1], pe[pos2])
                fig3.add_trace(
                    go.Scatter(
                        x=[pos1],
                        y=[inner_prod],
                        mode="markers",
                        marker=dict(size=8, color=f"rgb({i*40}, {j*40}, 100)"),
                        showlegend=False,
                        hovertemplate=f"位置{pos1}与位置{pos2}<br>内积: {inner_prod:.3f}<extra></extra>",
                    )
                )

    # 保存主图
    fig = add_formula_annotation(
        fig,
        r"$PE_{(pos,2i)} = \sin(pos/10000^{2i/d})$, $PE_{(pos,2i+1)} = \cos(pos/10000^{2i/d})$",
        x=0.5,
        y=1.05,
    )

    fig.update_layout(
        title_text="Transformer位置编码的频率特性 Frequency Properties of Positional Encoding",
        height=600,
        margin=dict(t=120, b=60, l=60, r=60),
    )

    # 保存频率谱图
    fig2 = add_formula_annotation(
        fig2,
        "位置编码的频谱分析 Spectrum Analysis of Positional Encoding",
        x=0.5,
        y=1.05,
    )

    fig2.update_layout(
        title="位置编码频谱分析 Positional Encoding Spectrum Analysis",
        xaxis_title="频率 Frequency",
        yaxis_title="幅度 Magnitude",
        height=400,
        margin=dict(t=120, b=60, l=60, r=60),
    )

    # 保存正交性图
    fig3 = add_formula_annotation(
        fig3, "位置编码的近似正交性 Approximate Orthogonality", x=0.5, y=1.05
    )

    fig3.update_layout(
        title="位置编码正交性分析 Positional Encoding Orthogonality",
        xaxis_title="位置 Position",
        yaxis_title="内积 Inner Product",
        height=400,
        margin=dict(t=120, b=60, l=60, r=60),
    )

    return fig, fig2, fig3


fig6, fig6_2, fig6_3 = create_transformer_positional_encoding()
output_file = os.path.join(output_dir, "6_transformer_positional.html")
fig6.write_html(output_file, include_mathjax="cdn")
print(f"   ✅ 保存: {output_file}")

output_file2 = os.path.join(output_dir, "6_positional_spectrum.html")
fig6_2.write_html(output_file2, include_mathjax="cdn")
print(f"   ✅ 保存: {output_file2}")

output_file3 = os.path.join(output_dir, "6_positional_orthogonality.html")
fig6_3.write_html(output_file3, include_mathjax="cdn")
print(f"   ✅ 保存: {output_file3}")

# ============================================
# 打印计算示例和总结
# ============================================
print("\n" + "=" * 60)
print("📊 希尔伯特空间与傅里叶变换计算示例")
print("=" * 60)

print("\n1️⃣ 希尔伯特空间的核心概念:")
print("   - 内积: ⟨f, g⟩ = ∫ f(x)·g(x)dx")
print("   - 范数: ||f|| = √⟨f, f⟩")
print("   - 完备性: 柯西序列收敛")
print("   - 应用: 神经网络的函数逼近理论基础")

print("\n2️⃣ 傅里叶变换的酉算子性质:")
print("   - 能量守恒: Parseval恒等式")
print("   - 可逆性: 酉变换保证完全重构")
print("   - 对角化: 卷积算子在频域变为对角矩阵")
print("   - 应用: CNN的数学理论基础")

print("\n3️⃣ 卷积定理的工程意义:")
print("   - 时域卷积 = 频域相乘")
print("   - FFT算法: O(n²) → O(n log n)")
print("   - 滤波器设计: 频域更直观")
print("   - 实际CNN: 时域实现，频域理解")

print("\n4️⃣ 现代架构的频域视角:")
print("   - CNN: 空间频率，平移不变性")
print("   - GNN: 图频率，结构依赖性")
print("   - Transformer: 位置编码，近似正交基")
print("   - 共同点: 都在希尔伯特空间中学习最优表示")

print("\n" + "=" * 60)
print("✨ 所有交互式可视化已创建完成！")
print("=" * 60)
print(f"\n📂 生成的文件位于: code/{output_dir}/")
print("   1. 1_hilbert_inner_product.html - 希尔伯特空间内积几何意义")
print("   2. 2_fourier_unitary.html - 傅里叶变换作为酉算子")
print("   3. 3_convolution_theorem.html - 卷积定理动态演示")
print("   4. 4_cnn_frequency_analysis.html - CNN滤波器频域特性")
print("   5. 5_graph_fourier.html - 图傅里叶变换可视化")
print("   6. 6_transformer_positional.html - Transformer位置编码频率分析")
print("   6_positional_spectrum.html - 位置编码频谱分析")
print("   6_positional_orthogonality.html - 位置编码正交性分析")
print("\n💡 在浏览器中打开这些 HTML 文件即可查看交互式可视化！")
print("=" * 60)
