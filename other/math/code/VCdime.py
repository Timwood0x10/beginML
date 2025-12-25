"""
VC维理论交互式可视化脚本
基于 7.VCdime.md 文档中的理论
使用 Plotly 创建交互式动画图表，包含数学公式
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# 创建输出目录
output_dir = "VCdime"
os.makedirs(output_dir, exist_ok=True)

print("=" * 60)
print("🎨 VC维理论交互式可视化")
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
# 1. VC维上界公式可视化
# ============================================
print("\n1️⃣ 创建VC维上界公式可视化...")

# 不同VC维下的复杂度惩罚
n_samples = np.linspace(10, 1000, 100)
vc_dims = [1, 5, 10, 50, 100]
colors = ["green", "blue", "orange", "red", "purple"]

fig1 = go.Figure()

for h, color in zip(vc_dims, colors):
    # 简化的复杂度惩罚项：Φ(h/n) ≈ sqrt(h/n)
    complexity_penalty = np.sqrt(h / n_samples)

    fig1.add_trace(
        go.Scatter(
            x=n_samples,
            y=complexity_penalty,
            mode="lines",
            name=f"VC维 h={h}",
            line=dict(color=color, width=3),
            hovertemplate="样本数: %{x:.0f}<br>复杂度惩罚: %{y:.4f}",
        )
    )

# 添加公式
fig1 = add_formula_annotation(
    fig1,
    r"$$R(f) \le R_{\text{emp}}(f) + \Phi\left(\frac{h}{n}\right) \quad \text{where } \Phi\left(\frac{h}{n}\right) \approx \sqrt{\frac{h}{n}}$$",
    x=0.5,
    y=1.05,
)

fig1.update_layout(
    title="VC维复杂度惩罚项 Φ(h/n) vs 样本数量",
    xaxis_title="训练样本数 n",
    yaxis_title="复杂度惩罚 Φ(h/n)",
    height=700,
    hovermode="x unified",
    legend=dict(x=0.7, y=0.95, bgcolor="rgba(255,255,255,0.8)"),
    margin=dict(t=120, b=60, l=60, r=60),
    annotations=[
        dict(
            x=500,
            y=0.3,
            text="VC维越大，需要的样本数越多",
            showarrow=True,
            arrowhead=2,
            ax=-100,
            ay=-50,
            font=dict(size=14, color="red"),
        )
    ],
)

output_file = os.path.join(output_dir, "1_vc_bound.html")
fig1.write_html(output_file, include_mathjax="cdn")
print(f"   ✅ 保存: {output_file}")

# ============================================
# 2. 真实风险上界分解
# ============================================
print("\n2️⃣ 创建真实风险上界分解可视化...")

n = 200
h = 20
n_range = np.linspace(50, 500, 50)

# 模拟经验风险（随样本增加而减少）
empirical_risk = 0.3 / np.sqrt(n_range / 50)

# 复杂度惩罚
complexity_penalty = np.sqrt(h / n_range)

# 真实风险上界
true_risk_bound = empirical_risk + complexity_penalty

fig2 = go.Figure()

fig2.add_trace(
    go.Scatter(
        x=n_range,
        y=empirical_risk,
        mode="lines",
        name="经验风险 R_emp(f)",
        line=dict(color="blue", width=3, dash="dash"),
        fill=None,
    )
)

fig2.add_trace(
    go.Scatter(
        x=n_range,
        y=complexity_penalty,
        mode="lines",
        name="复杂度惩罚 Φ(h/n)",
        line=dict(color="orange", width=3, dash="dot"),
        fill=None,
    )
)

fig2.add_trace(
    go.Scatter(
        x=n_range,
        y=true_risk_bound,
        mode="lines",
        name="真实风险上界 R(f)",
        line=dict(color="red", width=4),
        fill="tonexty",
        fillcolor="rgba(255,0,0,0.1)",
    )
)

# 找到最优点（真实风险最小）
optimal_idx = np.argmin(true_risk_bound)
optimal_n = n_range[optimal_idx]

fig2.add_trace(
    go.Scatter(
        x=[optimal_n],
        y=[true_risk_bound[optimal_idx]],
        mode="markers",
        name="最优样本数",
        marker=dict(size=15, color="green", symbol="star"),
    )
)

# 添加公式
fig2 = add_formula_annotation(
    fig2,
    r"$$R(f) = \underbrace{R_{\text{emp}}(f)}_{\text{训练误差}} + \underbrace{\Phi(h/n)}_{\text{复杂度惩罚}}$$",
    x=0.5,
    y=1.05,
)

fig2.update_layout(
    title=f"VC维上界的两个组成部分 (h={h})",
    xaxis_title="训练样本数 n",
    yaxis_title="风险/误差",
    height=700,
    hovermode="x unified",
    legend=dict(x=0.6, y=0.95, bgcolor="rgba(255,255,255,0.8)"),
    margin=dict(t=120, b=60, l=60, r=60),
)

output_file = os.path.join(output_dir, "2_risk_decomposition.html")
fig2.write_html(output_file, include_mathjax="cdn")
print(f"   ✅ 保存: {output_file}")

# ============================================
# 3. SVM的间隔与VC维关系
# ============================================
print("\n3️⃣ 创建SVM间隔与VC维关系可视化...")

# 权重范数与间隔的关系
w_norm = np.linspace(0.1, 5, 100)
margin = 2 / w_norm  # 间隔 ρ = 2/||w||
vc_dim_upper = w_norm**2  # VC维上界 ∝ ||w||^2

fig3 = make_subplots(
    rows=1,
    cols=2,
    subplot_titles=("权重范数 vs 间隔", "权重范数 vs VC维上界"),
    specs=[[{"type": "scatter"}, {"type": "scatter"}]],
)

# 左图：间隔
fig3.add_trace(
    go.Scatter(
        x=w_norm,
        y=margin,
        mode="lines",
        name="间隔 ρ = 2/||w||",
        line=dict(color="blue", width=3),
        hovertemplate="||w||: %{x:.2f}<br>间隔: %{y:.2f}",
    ),
    row=1,
    col=1,
)

# 右图：VC维
fig3.add_trace(
    go.Scatter(
        x=w_norm,
        y=vc_dim_upper,
        mode="lines",
        name="VC维上界 ∝ ||w||²",
        line=dict(color="red", width=3),
        hovertemplate="||w||: %{x:.2f}<br>VC维: %{y:.2f}",
    ),
    row=1,
    col=2,
)

fig3.update_xaxes(title_text="权重范数 ||w||", row=1, col=1)
fig3.update_xaxes(title_text="权重范数 ||w||", row=1, col=2)
fig3.update_yaxes(title_text="间隔 ρ", row=1, col=1)
fig3.update_yaxes(title_text="VC维上界 h", row=1, col=2)

# 添加公式
fig3 = add_formula_annotation(
    fig3,
    r"$$\rho = \frac{2}{\|w\|} \quad \Rightarrow \quad h \propto \frac{1}{\rho^2} \propto \|w\|^2 \quad \text{(最小化||w||² = 最小化VC维)}$$",
    x=0.5,
    y=1.05,
)

fig3.update_layout(
    title="SVM的关键洞察：最大化间隔 = 最小化VC维",
    height=600,
    showlegend=True,
    margin=dict(t=120, b=60, l=60, r=60),
)

output_file = os.path.join(output_dir, "3_svm_margin_vc.html")
fig3.write_html(output_file, include_mathjax="cdn")
print(f"   ✅ 保存: {output_file}")

# ============================================
# 4. 正则化参数λ的影响动画
# ============================================
print("\n4️⃣ 创建正则化参数影响动画...")

# 不同λ值下的权重范数和总损失
lambda_values = np.logspace(-3, 2, 30)  # λ从0.001到100
empirical_loss_base = 0.5  # 基础经验风险

frames_reg = []
for i, lam in enumerate(lambda_values):
    # 模拟：当λ增大时，权重被压缩
    w_norm_opt = 1 / np.sqrt(lam + 0.01)  # 最优权重范数
    empirical_loss = empirical_loss_base + 0.1 * (lam / 10)  # λ大时欠拟合
    regularization_term = lam * w_norm_opt**2
    total_loss = empirical_loss + regularization_term

    frame_data = [
        go.Bar(
            x=["经验风险", "正则化项", "总损失"],
            y=[empirical_loss, regularization_term, total_loss],
            marker=dict(color=["blue", "orange", "red"]),
            text=[
                f"{empirical_loss:.3f}",
                f"{regularization_term:.3f}",
                f"{total_loss:.3f}",
            ],
            textposition="outside",
            hovertemplate="%{x}: %{y:.3f}",
        )
    ]

    frames_reg.append(
        go.Frame(
            data=frame_data,
            name=str(i),
            layout=go.Layout(title_text=f"λ = {lam:.4f}, ||w|| = {w_norm_opt:.3f}"),
        )
    )

fig4 = go.Figure(
    data=[
        go.Bar(
            x=["经验风险", "正则化项", "总损失"],
            y=[0.5, 0.2, 0.7],
            marker=dict(color=["blue", "orange", "red"]),
            text=["0.500", "0.200", "0.700"],
            textposition="outside",
        )
    ],
    frames=frames_reg,
)

# 添加公式
fig4 = add_formula_annotation(
    fig4,
    r"$$\min \left[ \underbrace{\frac{1}{n}\sum \ell(y_i, f(x_i))}_{\text{经验风险}} + \underbrace{\lambda \|w\|^2}_{\text{正则化项/VC维惩罚}} \right]$$",
    x=0.5,
    y=1.05,
)

fig4.update_layout(
    title="正则化参数 λ 的权衡",
    yaxis_title="损失值",
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

output_file = os.path.join(output_dir, "4_regularization_tradeoff.html")
fig4.write_html(output_file, include_mathjax="cdn")
print(f"   ✅ 保存: {output_file}")

# ============================================
# 5. 过拟合vs欠拟合的VC维视角
# ============================================
print("\n5️⃣ 创建过拟合/欠拟合可视化...")

n_samples_range = np.linspace(20, 500, 50)

# 三种模型复杂度
scenarios = {
    "简单模型 (h=5)": {"h": 5, "color": "blue"},
    "适中模型 (h=20)": {"h": 20, "color": "green"},
    "复杂模型 (h=100)": {"h": 100, "color": "red"},
}

fig5 = go.Figure()

for name, params in scenarios.items():
    h = params["h"]
    color = params["color"]

    # 经验风险（复杂模型在训练集上表现更好）
    emp_risk = 0.4 / (h**0.3)

    # 复杂度惩罚
    comp_penalty = np.sqrt(h / n_samples_range)

    # 真实风险
    true_risk = emp_risk + comp_penalty

    fig5.add_trace(
        go.Scatter(
            x=n_samples_range,
            y=true_risk,
            mode="lines",
            name=name,
            line=dict(color=color, width=3),
            hovertemplate="样本数: %{x:.0f}<br>真实风险: %{y:.3f}",
        )
    )

# 添加区域标注
fig5.add_vrect(
    x0=20,
    x1=100,
    fillcolor="red",
    opacity=0.1,
    annotation_text="小样本区<br>(复杂模型过拟合)",
    annotation_position="top left",
)
fig5.add_vrect(
    x0=300,
    x1=500,
    fillcolor="green",
    opacity=0.1,
    annotation_text="大样本区<br>(复杂模型可行)",
    annotation_position="top right",
)

# 添加公式
fig5 = add_formula_annotation(
    fig5,
    r"$$\text{智商(h)必须匹配数据量(n)：} \frac{h}{n} \text{ 比值是关键}$$",
    x=0.5,
    y=1.05,
)

fig5.update_layout(
    title="VC维视角：模型复杂度 vs 样本数量的权衡",
    xaxis_title="训练样本数 n",
    yaxis_title="真实风险 R(f)",
    height=700,
    hovermode="x unified",
    legend=dict(x=0.6, y=0.95, bgcolor="rgba(255,255,255,0.8)"),
    margin=dict(t=120, b=60, l=60, r=60),
)

output_file = os.path.join(output_dir, "5_overfitting_underfitting.html")
fig5.write_html(output_file, include_mathjax="cdn")
print(f"   ✅ 保存: {output_file}")

# ============================================
# 6. SVM优化目标的VC维解读
# ============================================
print("\n6️⃣ 创建SVM优化目标可视化...")

# C参数对权重的影响
C_values = np.logspace(-2, 2, 100)  # C从0.01到100

# 模拟最优权重（C越大，越关注经验风险，权重可能更大）
w_norm_values = 1 + np.log(C_values + 0.1)

# 间隔
margin_values = 2 / w_norm_values

# VC维上界
vc_values = w_norm_values**2

fig6 = make_subplots(
    rows=2,
    cols=1,
    subplot_titles=("SVM参数C vs 权重范数", "SVM参数C vs VC维上界"),
    vertical_spacing=0.15,
)

# 上图：C vs 权重
fig6.add_trace(
    go.Scatter(
        x=C_values,
        y=w_norm_values,
        mode="lines",
        name="||w||",
        line=dict(color="blue", width=3),
    ),
    row=1,
    col=1,
)

# 下图：C vs VC维
fig6.add_trace(
    go.Scatter(
        x=C_values,
        y=vc_values,
        mode="lines",
        name="VC维上界 ∝ ||w||²",
        line=dict(color="red", width=3),
    ),
    row=2,
    col=1,
)

fig6.update_xaxes(title_text="SVM参数 C", type="log", row=1, col=1)
fig6.update_xaxes(title_text="SVM参数 C", type="log", row=2, col=1)
fig6.update_yaxes(title_text="权重范数 ||w||", row=1, col=1)
fig6.update_yaxes(title_text="VC维上界", row=2, col=1)

# 添加公式
fig6 = add_formula_annotation(
    fig6,
    r"$$\min \left[ \underbrace{\frac{1}{2}\|w\|^2}_{\text{VC维控制}} + C \sum \underbrace{\max(0, 1-y_i f(x_i))}_{\text{经验风险}} \right]$$",
    x=0.5,
    y=1.03,
)

fig6.update_layout(
    title="SVM的C参数：平衡VC维控制和经验风险",
    height=800,
    showlegend=True,
    margin=dict(t=120, b=60, l=60, r=60),
)

output_file = os.path.join(output_dir, "6_svm_c_parameter.html")
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
        "VC维复杂度惩罚",
        "真实风险上界分解",
        "SVM：间隔与VC维",
        "模型选择：样本数vs复杂度",
    ),
    specs=[
        [{"type": "scatter"}, {"type": "scatter"}],
        [{"type": "scatter"}, {"type": "scatter"}],
    ],
    vertical_spacing=0.15,
    horizontal_spacing=0.12,
)

# 子图1: VC维惩罚
for h, color in zip([10, 50, 100], ["blue", "orange", "red"]):
    penalty = np.sqrt(h / n_samples)
    fig7.add_trace(
        go.Scatter(
            x=n_samples,
            y=penalty,
            mode="lines",
            name=f"h={h}",
            line=dict(color=color, width=2),
        ),
        row=1,
        col=1,
    )

# 子图2: 风险分解
n_range_short = np.linspace(50, 300, 30)
emp_risk_short = 0.3 / np.sqrt(n_range_short / 50)
comp_pen_short = np.sqrt(20 / n_range_short)
fig7.add_trace(
    go.Scatter(
        x=n_range_short,
        y=emp_risk_short,
        mode="lines",
        name="经验风险",
        line=dict(color="blue", width=2, dash="dash"),
    ),
    row=1,
    col=2,
)
fig7.add_trace(
    go.Scatter(
        x=n_range_short,
        y=comp_pen_short,
        mode="lines",
        name="复杂度惩罚",
        line=dict(color="orange", width=2, dash="dot"),
    ),
    row=1,
    col=2,
)
fig7.add_trace(
    go.Scatter(
        x=n_range_short,
        y=emp_risk_short + comp_pen_short,
        mode="lines",
        name="总风险",
        line=dict(color="red", width=3),
    ),
    row=1,
    col=2,
)

# 子图3: 间隔与VC维
w_short = np.linspace(0.5, 3, 30)
margin_short = 2 / w_short
fig7.add_trace(
    go.Scatter(
        x=w_short,
        y=margin_short,
        mode="lines",
        name="间隔 ρ",
        line=dict(color="green", width=2),
    ),
    row=2,
    col=1,
)

# 子图4: 过拟合/欠拟合
for name, params in scenarios.items():
    h = params["h"]
    color = params["color"]
    emp_risk = 0.4 / (h**0.3)
    comp_penalty = np.sqrt(h / n_samples_range)
    true_risk = emp_risk + comp_penalty
    fig7.add_trace(
        go.Scatter(
            x=n_samples_range,
            y=true_risk,
            mode="lines",
            name=name.split("(")[0].strip(),
            line=dict(color=color, width=2),
        ),
        row=2,
        col=2,
    )

# 更新坐标轴
fig7.update_xaxes(title_text="样本数 n", row=1, col=1)
fig7.update_xaxes(title_text="样本数 n", row=1, col=2)
fig7.update_xaxes(title_text="||w||", row=2, col=1)
fig7.update_xaxes(title_text="样本数 n", row=2, col=2)

fig7.update_yaxes(title_text="Φ(h/n)", row=1, col=1)
fig7.update_yaxes(title_text="风险", row=1, col=2)
fig7.update_yaxes(title_text="间隔 ρ", row=2, col=1)
fig7.update_yaxes(title_text="R(f)", row=2, col=2)

# 添加公式
fig7 = add_formula_annotation(
    fig7,
    r"$$\text{VC维理论：} R(f) \le R_{\text{emp}}(f) + \Phi(h/n) \quad \text{智商必须匹配数据量}$$",
    x=0.5,
    y=1.02,
)

fig7.update_layout(
    title="VC维理论综合仪表板",
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
print("📊 VC维理论可视化总结")
print("=" * 60)

print("\n✅ 已创建的可视化:")
print("   1. VC维上界公式")
print("   2. 真实风险上界分解")
print("   3. SVM间隔与VC维关系")
print("   4. 正则化参数权衡")
print("   5. 过拟合/欠拟合分析")
print("   6. SVM的C参数影响")
print("   7. 综合仪表板")

print("\n💡 核心要点:")
print("   • VC维 = 模型的'智商'或'容量'")
print("   • R(f) ≤ R_emp(f) + Φ(h/n)")
print("   • SVM通过最大化间隔来最小化VC维")
print("   • 正则化是通用的VC维控制手段")
print("   • 智商(h)必须匹配数据量(n)")

print("\n" + "=" * 60)
print("✨ 所有交互式可视化已创建完成！")
print("=" * 60)
print(f"\n📂 生成的文件位于: code/{output_dir}/")
print("💡 在浏览器中打开这些 HTML 文件即可查看交互式可视化！")
print("=" * 60)

# ============================================
# 补充可视化
# ============================================

print("\n" + "=" * 60)
print("🔧 补充核心概念可视化...")
print("=" * 60)

# ============================================
# 8. Shattering概念演示（核心定义）
# ============================================
print("\n8️⃣ 创建Shattering概念演示...")

# 3个点的所有标注组合
np.random.seed(42)
points_3 = np.array([[0.2, 0.3], [0.8, 0.3], [0.5, 0.7]])

# 生成所有2^3=8种标注
all_labels_3 = []
for i in range(8):
    labels = [(i >> j) & 1 for j in range(3)]
    all_labels_3.append(labels)

# 创建动画展示所有分类
frames_shatter = []
for idx, labels in enumerate(all_labels_3):
    # 为每个标注找一条分隔线
    if sum(labels) == 0:  # 全0
        x_line = np.array([0, 1])
        y_line = np.array([0.1, 0.1])
    elif sum(labels) == 3:  # 全1
        x_line = np.array([0, 1])
        y_line = np.array([0.9, 0.9])
    elif labels == [0, 0, 1]:
        x_line = np.array([0, 1])
        y_line = np.array([0.5, 0.5])
    elif labels == [0, 1, 0]:
        x_line = np.array([0.5, 0.5])
        y_line = np.array([0, 1])
    elif labels == [1, 0, 0]:
        x_line = np.array([0.5, 0.5])
        y_line = np.array([0, 1])
    elif labels == [0, 1, 1]:
        x_line = np.array([0, 1])
        y_line = np.array([0.45, 0.45])
    elif labels == [1, 0, 1]:
        x_line = np.array([0, 1])
        y_line = np.array([0.5, 0.5])
    else:  # [1, 1, 0]
        x_line = np.array([0, 1])
        y_line = np.array([0.55, 0.55])

    colors = ["red" if l == 0 else "blue" for l in labels]
    symbols = ["x" if l == 0 else "circle" for l in labels]

    frame_data = [
        go.Scatter(
            x=points_3[:, 0],
            y=points_3[:, 1],
            mode="markers",
            marker=dict(
                size=20, color=colors, symbol=symbols, line=dict(width=2, color="black")
            ),
            name="数据点",
            hovertemplate="点 %{pointNumber}<br>标签: %{marker.color}",
        ),
        go.Scatter(
            x=x_line,
            y=y_line,
            mode="lines",
            name="分隔线",
            line=dict(color="green", width=3, dash="dash"),
        ),
    ]

    label_str = "".join(map(str, labels))
    frames_shatter.append(
        go.Frame(
            data=frame_data,
            name=str(idx),
            layout=go.Layout(
                title_text=f"标注 {idx+1}/8: [{label_str}] - 线性分类器可以分隔"
            ),
        )
    )

fig8 = go.Figure(
    data=[
        go.Scatter(
            x=points_3[:, 0],
            y=points_3[:, 1],
            mode="markers",
            marker=dict(
                size=20, color=["red", "red", "red"], line=dict(width=2, color="black")
            ),
            name="数据点",
        )
    ],
    frames=frames_shatter,
)

fig8 = add_formula_annotation(
    fig8,
    r"$$\text{Shattering: 假设类 } \mathcal{H} \text{ 能实现 } 2^n \text{ 种标注} \quad \Rightarrow \quad \text{VC}(\mathcal{H}) \ge n$$",
    x=0.5,
    y=1.05,
)

fig8.update_layout(
    title="Shattering演示：3个点可以被线性分类器打散（VC维≥3）",
    xaxis=dict(range=[-0.1, 1.1], showgrid=True),
    yaxis=dict(range=[-0.1, 1.1], showgrid=True),
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
                            frame=dict(duration=1000, redraw=True),
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

output_file = os.path.join(output_dir, "8_shattering_demo.html")
fig8.write_html(output_file, include_mathjax="cdn")
print(f"   ✅ 保存: {output_file}")

# ============================================
# 9. XOR问题：4个点无法被线性分类器打散
# ============================================
print("\n9️⃣ 创建XOR问题演示（无法打散）...")

# XOR配置的4个点
points_4 = np.array([[0.2, 0.2], [0.8, 0.2], [0.2, 0.8], [0.8, 0.8]])
xor_labels = [0, 1, 1, 0]  # XOR标注

fig9 = go.Figure()

# 显示XOR配置
colors_xor = ["red" if l == 0 else "blue" for l in xor_labels]
fig9.add_trace(
    go.Scatter(
        x=points_4[:, 0],
        y=points_4[:, 1],
        mode="markers+text",
        marker=dict(size=25, color=colors_xor, line=dict(width=2, color="black")),
        text=["类0", "类1", "类1", "类0"],
        textposition="top center",
        textfont=dict(size=14, color="black"),
        name="XOR配置",
    )
)

# 尝试多条分隔线都无法正确分类
trial_lines = [
    ([0, 1], [0.5, 0.5], "水平线"),
    ([0.5, 0.5], [0, 1], "垂直线"),
    ([0, 1], [0, 1], "对角线"),
    ([0, 1], [1, 0], "反对角线"),
]

for x_l, y_l, desc in trial_lines:
    fig9.add_trace(
        go.Scatter(
            x=x_l,
            y=y_l,
            mode="lines",
            name=desc,
            line=dict(width=2, dash="dash"),
            visible="legendonly",
        )
    )

fig9 = add_formula_annotation(
    fig9,
    r"$$\text{XOR问题：4个点无法被线性分类器打散} \quad \Rightarrow \quad \text{VC}(\text{线性分类器}_{\mathbb{R}^2}) = 3$$",
    x=0.5,
    y=1.05,
)

fig9.update_layout(
    title="XOR问题：线性分类器的局限性（VC维=3，不是4）",
    xaxis=dict(range=[-0.1, 1.1], showgrid=True, title="x₁"),
    yaxis=dict(range=[-0.1, 1.1], showgrid=True, title="x₂"),
    height=700,
    legend=dict(x=0.02, y=0.98, bgcolor="rgba(255,255,255,0.8)"),
    margin=dict(t=120, b=60, l=60, r=60),
    annotations=[
        dict(
            x=0.5,
            y=-0.05,
            xref="paper",
            yref="paper",
            text="💡 尝试点击图例中的不同分隔线，看看哪条能正确分类（答案：都不行！）",
            showarrow=False,
            font=dict(size=12, color="red"),
        )
    ],
)

output_file = os.path.join(output_dir, "9_xor_problem.html")
fig9.write_html(output_file, include_mathjax="cdn")
print(f"   ✅ 保存: {output_file}")

# ============================================
# 10. 增长函数可视化
# ============================================
print("\n🔟 创建增长函数可视化...")

n_range = np.arange(1, 20)

# 不同假设类的增长函数
growth_functions = {
    "线性分类器 (VC=3)": lambda n: np.minimum(
        2**n, (n**3) / 6 + (n**2) / 2 + n / 3 + 1
    ),
    "指数增长 2^n": lambda n: 2**n,
    "多项式 O(n^3)": lambda n: (n**3) / 6 + (n**2) / 2 + n / 3 + 1,
}

fig10 = go.Figure()

for name, func in growth_functions.items():
    m_h = func(n_range)
    fig10.add_trace(
        go.Scatter(
            x=n_range,
            y=m_h,
            mode="lines+markers",
            name=name,
            line=dict(width=3),
            marker=dict(size=6),
        )
    )

# 添加VC维=3的分界点
fig10.add_vline(
    x=3,
    line=dict(color="red", dash="dash", width=2),
    annotation_text="VC维=3",
    annotation_position="top right",
)

fig10 = add_formula_annotation(
    fig10,
    r"$$m_{\mathcal{H}}(n) \le \sum_{i=0}^{h} \binom{n}{i} \quad \text{(Sauer引理)} \quad \Rightarrow \quad m_{\mathcal{H}}(n) = O(n^h)$$",
    x=0.5,
    y=1.05,
)

fig10.update_layout(
    title="增长函数：从指数到多项式（VC维有限时）",
    xaxis_title="样本数 n",
    yaxis_title="可实现的标注数 m_H(n)",
    yaxis_type="log",
    height=700,
    legend=dict(x=0.02, y=0.98, bgcolor="rgba(255,255,255,0.8)"),
    margin=dict(t=120, b=60, l=60, r=60),
    annotations=[
        dict(
            x=10,
            y=3,
            xref="x",
            yref="y",
            text="VC维有限 → 多项式增长<br>VC维无限 → 指数增长",
            showarrow=True,
            arrowhead=2,
            ax=-80,
            ay=-50,
            font=dict(size=12, color="blue"),
            bgcolor="rgba(255,255,255,0.8)",
        )
    ],
)

output_file = os.path.join(output_dir, "10_growth_function.html")
fig10.write_html(output_file, include_mathjax="cdn")
print(f"   ✅ 保存: {output_file}")

# ============================================
# 11. 不同模型的VC维对比
# ============================================
print("\n1️⃣1️⃣ 创建不同模型VC维对比...")

models = {
    "线性分类器 (2D)": 3,
    "线性分类器 (d维)": "d+1",
    "2次多项式 (2D)": 6,
    "RBF核 (γ→∞)": "无限",
    "RBF核 (有正则化)": "有效VC维↓",
    "1层神经网络 (k个神经元)": "O(k·d)",
    "深度神经网络": "O(W·L)",
}

model_names = list(models.keys())
vc_values_display = list(models.values())

# 为了绘图，把字符串转换为数值
vc_values_numeric = []
for v in vc_values_display:
    if v == "无限":
        vc_values_numeric.append(100)
    elif v == "d+1":
        vc_values_numeric.append(4)  # 假设d=3
    elif v == "有效VC维↓":
        vc_values_numeric.append(20)
    elif "O(k·d)" in str(v):
        vc_values_numeric.append(30)
    elif "O(W·L)" in str(v):
        vc_values_numeric.append(50)
    else:
        vc_values_numeric.append(int(v))

fig11 = go.Figure()

colors_model = ["blue", "green", "orange", "red", "purple", "brown", "pink"]
fig11.add_trace(
    go.Bar(
        x=model_names,
        y=vc_values_numeric,
        marker=dict(color=colors_model),
        text=vc_values_display,
        textposition="outside",
        hovertemplate="%{x}<br>VC维: %{text}",
    )
)

fig11 = add_formula_annotation(
    fig11,
    r"$$\text{VC}(\mathcal{H}) = \max\{n : m_{\mathcal{H}}(n) = 2^n\} \quad \text{(最大可打散点数)}$$",
    x=0.5,
    y=1.05,
)

fig11.update_layout(
    title="不同模型的VC维对比",
    xaxis_title="模型类型",
    yaxis_title="VC维（数值化表示）",
    yaxis_type="log",
    height=700,
    margin=dict(t=120, b=100, l=60, r=60),
    xaxis=dict(tickangle=-45),
)

output_file = os.path.join(output_dir, "11_model_vc_comparison.html")
fig11.write_html(output_file, include_mathjax="cdn")
print(f"   ✅ 保存: {output_file}")

# ============================================
# 12. PAC学习框架可视化
# ============================================
print("\n1️⃣2️⃣ 创建PAC学习框架可视化...")

# PAC界：n ≥ (1/ε²) * (h*log(n/h) + log(1/δ))
epsilon_values = np.linspace(0.01, 0.2, 50)
delta = 0.05
h = 20

# 简化的样本复杂度公式
sample_complexity = (1 / epsilon_values**2) * (h * np.log(100 / h) + np.log(1 / delta))

fig12 = make_subplots(
    rows=1,
    cols=2,
    subplot_titles=("误差 ε vs 所需样本数", "置信度 1-δ vs 所需样本数"),
    specs=[[{"type": "scatter"}, {"type": "scatter"}]],
)

# 左图：ε vs n
fig12.add_trace(
    go.Scatter(
        x=epsilon_values,
        y=sample_complexity,
        mode="lines",
        name="n(ε)",
        line=dict(color="blue", width=3),
        fill="tozeroy",
        fillcolor="rgba(0,0,255,0.1)",
    ),
    row=1,
    col=1,
)

# 右图：δ vs n
delta_values = np.linspace(0.01, 0.5, 50)
epsilon_fixed = 0.1
sample_complexity_delta = (1 / epsilon_fixed**2) * (
    h * np.log(100 / h) + np.log(1 / delta_values)
)

fig12.add_trace(
    go.Scatter(
        x=1 - delta_values,
        y=sample_complexity_delta,
        mode="lines",
        name="n(δ)",
        line=dict(color="red", width=3),
        fill="tozeroy",
        fillcolor="rgba(255,0,0,0.1)",
    ),
    row=1,
    col=2,
)

fig12.update_xaxes(title_text="误差 ε", row=1, col=1)
fig12.update_xaxes(title_text="置信度 1-δ", row=1, col=2)
fig12.update_yaxes(title_text="所需样本数 n", row=1, col=1)
fig12.update_yaxes(title_text="所需样本数 n", row=1, col=2)

fig12 = add_formula_annotation(
    fig12,
    r"$$P\left[ R(f) - R_{\text{emp}}(f) \le \epsilon \right] \ge 1 - \delta \quad \text{if } n \ge \frac{1}{\epsilon^2}\left(h\log\frac{n}{h} + \log\frac{1}{\delta}\right)$$",
    x=0.5,
    y=1.05,
)

fig12.update_layout(
    title="PAC学习框架：样本复杂度 (h=20)",
    height=600,
    showlegend=True,
    margin=dict(t=120, b=60, l=60, r=60),
)

output_file = os.path.join(output_dir, "12_pac_framework.html")
fig12.write_html(output_file, include_mathjax="cdn")
print(f"   ✅ 保存: {output_file}")

# ============================================
# 13. 深度学习VC维悖论
# ============================================
print("\n1️⃣3️⃣ 创建深度学习VC维悖论可视化...")

# 模拟深度学习中的双下降现象
model_complexity = np.linspace(0.1, 100, 200)

# 经典理论：误差随复杂度单调递增
classical_error = 0.1 + 0.05 * np.log(model_complexity + 1)

# 实际观察：双下降现象
interpolation_point = 20  # 插值阈值
double_descent = np.zeros_like(model_complexity)

# 第一段：经典过拟合
mask1 = model_complexity < interpolation_point
double_descent[mask1] = 0.1 + 0.05 * np.log(model_complexity[mask1] + 1)

# 第二段：插值区域（误差达到峰值）
mask2 = (model_complexity >= interpolation_point) & (model_complexity < 40)
double_descent[mask2] = 0.3 - 0.01 * (model_complexity[mask2] - interpolation_point)

# 第三段：过参数化区域（误差再次下降）
mask3 = model_complexity >= 40
double_descent[mask3] = 0.1 + 0.5 / np.sqrt(model_complexity[mask3] - 35)

fig13 = go.Figure()

# 经典理论曲线
fig13.add_trace(
    go.Scatter(
        x=model_complexity,
        y=classical_error,
        mode="lines",
        name="经典VC理论",
        line=dict(color="red", width=3, dash="dash"),
        hovertemplate="复杂度: %{x:.2f}<br>误差: %{y:.3f}",
    )
)

# 实际双下降曲线
fig13.add_trace(
    go.Scatter(
        x=model_complexity,
        y=double_descent,
        mode="lines",
        name="实际观察（双下降）",
        line=dict(color="blue", width=3),
        fill="tonexty",
        fillcolor="rgba(0,0,255,0.1)",
        hovertemplate="复杂度: %{x:.2f}<br>误差: %{y:.3f}",
    )
)

# 标记关键点
fig13.add_trace(
    go.Scatter(
        x=[interpolation_point],
        y=[double_descent[np.where(model_complexity >= interpolation_point)[0][0]]],
        mode="markers",
        name="插值阈值",
        marker=dict(size=15, color="green", symbol="star"),
    )
)

fig13 = add_formula_annotation(
    fig13,
    r"$\text{深度学习悖论：} \text{VC维} \to \infty \quad \text{但} \quad \text{泛化误差} \downarrow$",
    x=0.5,
    y=1.05,
)

fig13.update_layout(
    title="深度学习的VC维悖论：双下降现象",
    xaxis_title="模型复杂度（参数量/VC维）",
    xaxis_type="log",
    yaxis_title="泛化误差",
    height=700,
    legend=dict(x=0.02, y=0.98, bgcolor="rgba(255,255,255,0.8)"),
    margin=dict(t=120, b=60, l=60, r=60),
    annotations=[
        dict(
            x=5,
            y=0.2,
            text="欠拟合区",
            showarrow=False,
            font=dict(size=12, color="blue"),
        ),
        dict(
            x=30,
            y=0.3,
            text="插值区<br>（误差峰值）",
            showarrow=False,
            font=dict(size=12, color="red"),
        ),
        dict(
            x=80,
            y=0.15,
            text="过参数化区<br>（误差下降）",
            showarrow=False,
            font=dict(size=12, color="green"),
        ),
    ],
)

output_file = os.path.join(output_dir, "13_deep_learning_paradox.html")
fig13.write_html(output_file, include_mathjax="cdn")
print(f"   ✅ 保存: {output_file}")

# ============================================
# 14. 正则化策略对比
# ============================================
print("\n1️⃣4️⃣ 创建正则化策略对比可视化...")

# 模拟不同正则化策略的效果
epochs = np.arange(0, 100)
np.random.seed(42)

# 无正则化（严重过拟合）
train_loss_no_reg = 0.5 * np.exp(-epochs / 20) + 0.05 * np.random.normal(
    0, 0.02, len(epochs)
)
val_loss_no_reg = (
    0.5 * np.exp(-epochs / 20)
    + 0.1 * (1 - np.exp(-epochs / 30))
    + 0.05 * np.random.normal(0, 0.02, len(epochs))
)

# L2正则化
train_loss_l2 = (
    0.5 * np.exp(-epochs / 25) + 0.1 + 0.05 * np.random.normal(0, 0.02, len(epochs))
)
val_loss_l2 = (
    0.5 * np.exp(-epochs / 25) + 0.12 + 0.05 * np.random.normal(0, 0.02, len(epochs))
)

# Dropout
train_loss_dropout = (
    0.5 * np.exp(-epochs / 30) + 0.15 + 0.05 * np.random.normal(0, 0.02, len(epochs))
)
val_loss_dropout = (
    0.5 * np.exp(-epochs / 30) + 0.13 + 0.05 * np.random.normal(0, 0.02, len(epochs))
)

# 早停
early_stop_epoch = 40
train_loss_early = train_loss_l2.copy()
val_loss_early = val_loss_l2.copy()
val_loss_early[early_stop_epoch:] = val_loss_early[early_stop_epoch]

fig14 = make_subplots(
    rows=2,
    cols=2,
    subplot_titles=("无正则化", "L2正则化", "Dropout", "早停"),
    specs=[
        [{"type": "scatter"}, {"type": "scatter"}],
        [{"type": "scatter"}, {"type": "scatter"}],
    ],
)

# 无正则化
fig14.add_trace(
    go.Scatter(
        x=epochs,
        y=train_loss_no_reg,
        mode="lines",
        name="训练损失",
        line=dict(color="blue"),
    ),
    row=1,
    col=1,
)
fig14.add_trace(
    go.Scatter(
        x=epochs,
        y=val_loss_no_reg,
        mode="lines",
        name="验证损失",
        line=dict(color="red"),
    ),
    row=1,
    col=1,
)

# L2正则化
fig14.add_trace(
    go.Scatter(
        x=epochs,
        y=train_loss_l2,
        mode="lines",
        name="训练损失",
        line=dict(color="blue"),
        showlegend=False,
    ),
    row=1,
    col=2,
)
fig14.add_trace(
    go.Scatter(
        x=epochs,
        y=val_loss_l2,
        mode="lines",
        name="验证损失",
        line=dict(color="red"),
        showlegend=False,
    ),
    row=1,
    col=2,
)

# Dropout
fig14.add_trace(
    go.Scatter(
        x=epochs,
        y=train_loss_dropout,
        mode="lines",
        name="训练损失",
        line=dict(color="blue"),
        showlegend=False,
    ),
    row=2,
    col=1,
)
fig14.add_trace(
    go.Scatter(
        x=epochs,
        y=val_loss_dropout,
        mode="lines",
        name="验证损失",
        line=dict(color="red"),
        showlegend=False,
    ),
    row=2,
    col=1,
)

# 早停
fig14.add_trace(
    go.Scatter(
        x=epochs[: early_stop_epoch + 1],
        y=train_loss_early[: early_stop_epoch + 1],
        mode="lines",
        name="训练损失",
        line=dict(color="blue"),
        showlegend=False,
    ),
    row=2,
    col=2,
)
fig14.add_trace(
    go.Scatter(
        x=epochs[: early_stop_epoch + 1],
        y=val_loss_early[: early_stop_epoch + 1],
        mode="lines",
        name="验证损失",
        line=dict(color="red"),
        showlegend=False,
    ),
    row=2,
    col=2,
)

# 更新坐标轴
for i in range(1, 3):
    for j in range(1, 3):
        fig14.update_xaxes(title_text="训练轮次", row=i, col=j)
        fig14.update_yaxes(title_text="损失值", row=i, col=j)

fig14 = add_formula_annotation(
    fig14,
    r"$\text{正则化策略：控制有效VC维} \quad \text{无正则化} \to \text{高VC维} \to \text{过拟合}$",
    x=0.5,
    y=1.03,
)

fig14.update_layout(
    title="正则化策略对比：如何控制有效VC维",
    height=800,
    showlegend=True,
    margin=dict(t=120, b=60, l=60, r=60),
)

output_file = os.path.join(output_dir, "14_regularization_strategies.html")
fig14.write_html(output_file, include_mathjax="cdn")
print(f"   ✅ 保存: {output_file}")

# ============================================
# 15. Rademacher复杂度 vs VC维
# ============================================
print("\n1️⃣5️⃣ 创建Rademacher复杂度对比可视化...")

# 比较VC维界和Rademacher复杂度界
n_samples = np.linspace(50, 1000, 100)
vc_dim = 20

# VC维界（较松）
vc_bound = np.sqrt((vc_dim * np.log(2 * n_samples / vc_dim) + np.log(4)) / n_samples)

# Rademacher复杂度界（更紧）
rademacher_bound = (
    np.sqrt(2 * np.log(2 * n_samples) / n_samples)
    + np.sqrt(2 * vc_dim * np.log(n_samples) / n_samples) / n_samples
)

# 实际泛化误差（模拟）
actual_error = 0.1 + 0.05 / np.sqrt(n_samples)

fig15 = go.Figure()

fig15.add_trace(
    go.Scatter(
        x=n_samples,
        y=vc_bound,
        mode="lines",
        name="VC维界（较松）",
        line=dict(color="red", width=3, dash="dash"),
        hovertemplate="样本数: %{x:.0f}<br>VC界: %{y:.4f}",
    )
)

fig15.add_trace(
    go.Scatter(
        x=n_samples,
        y=rademacher_bound,
        mode="lines",
        name="Rademacher界（更紧）",
        line=dict(color="orange", width=3),
        hovertemplate="样本数: %{x:.0f}<br>Rademacher界: %{y:.4f}",
    )
)

fig15.add_trace(
    go.Scatter(
        x=n_samples,
        y=actual_error,
        mode="lines",
        name="实际泛化误差",
        line=dict(color="blue", width=3),
        hovertemplate="样本数: %{x:.0f}<br>实际误差: %{y:.4f}",
    )
)

fig15 = add_formula_annotation(
    fig15,
    r"$\mathcal{R}_n(\mathcal{H}) = \mathbb{E}_\sigma \left[ \sup_{h \in \mathcal{H}} \frac{1}{n}\sum_{i=1}^n \sigma_i h(x_i) \right]$",
    x=0.5,
    y=1.05,
)

fig15.update_layout(
    title="VC维 vs Rademacher复杂度：界的紧度对比",
    xaxis_title="训练样本数 n",
    yaxis_title="泛化误差上界",
    height=700,
    legend=dict(x=0.02, y=0.98, bgcolor="rgba(255,255,255,0.8)"),
    margin=dict(t=120, b=60, l=60, r=60),
    annotations=[
        dict(
            x=200,
            y=0.4,
            text="VC维界过于保守",
            showarrow=True,
            arrowhead=2,
            ax=-50,
            ay=-30,
            font=dict(size=12, color="red"),
        ),
        dict(
            x=500,
            y=0.25,
            text="Rademacher界更接近实际",
            showarrow=True,
            arrowhead=2,
            ax=-50,
            ay=-30,
            font=dict(size=12, color="orange"),
        ),
    ],
)

output_file = os.path.join(output_dir, "15_rademacher_vs_vc.html")
fig15.write_html(output_file, include_mathjax="cdn")
print(f"   ✅ 保存: {output_file}")

# ============================================
# 16. 数据分布对VC维的影响
# ============================================
print("\n1️⃣6️⃣ 创建数据分布影响可视化...")

# 生成不同分布的数据
np.random.seed(42)

# 简单分布（线性可分）
simple_x = np.random.randn(50, 2)
simple_y = (simple_x[:, 0] + simple_x[:, 1] > 0).astype(int)

# 复杂分布（需要高VC维）
complex_x = np.random.randn(50, 2)
complex_y = (
    (complex_x[:, 0] ** 2 + complex_x[:, 1] ** 2 > 1)
    & (complex_x[:, 0] - complex_x[:, 1] > 0)
).astype(int)

fig16 = make_subplots(
    rows=1,
    cols=2,
    subplot_titles=("简单分布（线性可分）", "复杂分布（需要高VC维）"),
    specs=[[{"type": "scatter"}, {"type": "scatter"}]],
)

# 简单分布
fig16.add_trace(
    go.Scatter(
        x=simple_x[simple_y == 0, 0],
        y=simple_x[simple_y == 0, 1],
        mode="markers",
        name="类0",
        marker=dict(color="red", size=8),
        showlegend=False,
    ),
    row=1,
    col=1,
)
fig16.add_trace(
    go.Scatter(
        x=simple_x[simple_y == 1, 0],
        y=simple_x[simple_y == 1, 1],
        mode="markers",
        name="类1",
        marker=dict(color="blue", size=8),
        showlegend=False,
    ),
    row=1,
    col=1,
)

# 添加线性分隔线
x_line = np.linspace(-3, 3, 100)
y_line = -x_line
fig16.add_trace(
    go.Scatter(
        x=x_line,
        y=y_line,
        mode="lines",
        name="线性边界",
        line=dict(color="green", width=3, dash="dash"),
        showlegend=False,
    ),
    row=1,
    col=1,
)

# 复杂分布
fig16.add_trace(
    go.Scatter(
        x=complex_x[complex_y == 0, 0],
        y=complex_x[complex_y == 0, 1],
        mode="markers",
        name="类0",
        marker=dict(color="red", size=8),
        showlegend=False,
    ),
    row=1,
    col=2,
)
fig16.add_trace(
    go.Scatter(
        x=complex_x[complex_y == 1, 0],
        y=complex_x[complex_y == 1, 1],
        mode="markers",
        name="类1",
        marker=dict(color="blue", size=8),
        showlegend=True,
    ),
    row=1,
    col=2,
)

# 添加非线性边界（圆形）
theta = np.linspace(0, 2 * np.pi, 100)
circle_x = np.cos(theta)
circle_y = np.sin(theta)
fig16.add_trace(
    go.Scatter(
        x=circle_x,
        y=circle_y,
        mode="lines",
        name="非线性边界",
        line=dict(color="green", width=3, dash="dash"),
        showlegend=False,
    ),
    row=1,
    col=2,
)

fig16.update_xaxes(title_text="x₁", row=1, col=1)
fig16.update_xaxes(title_text="x₁", row=1, col=2)
fig16.update_yaxes(title_text="x₂", row=1, col=1)
fig16.update_yaxes(title_text="x₂", row=1, col=2)

fig16 = add_formula_annotation(
    fig16,
    r"$\text{VC维假设最坏分布，实际数据有结构} \quad \Rightarrow \quad \text{理论界过于保守}$",
    x=0.5,
    y=1.05,
)

fig16.update_layout(
    title="数据分布对VC维实际需求的影响",
    height=600,
    legend=dict(x=0.98, y=0.02, bgcolor="rgba(255,255,255,0.8)"),
    margin=dict(t=120, b=60, l=60, r=60),
)

output_file = os.path.join(output_dir, "16_data_distribution_impact.html")
fig16.write_html(output_file, include_mathjax="cdn")
print(f"   ✅ 保存: {output_file}")

print("\n" + "=" * 60)
print("✨ 补充可视化创建完成！")
print("=" * 60)
print("\n新增:")
print("   8. Shattering演示（3个点）")
print("   9. XOR问题（4个点无法打散）")
print("   10. 增长函数可视化")
print("   11. 不同模型VC维对比")
print("   12. PAC学习框架")
print("   13. 深度学习VC维悖论")
print("   14. 正则化策略对比")
print("   15. Rademacher复杂度 vs VC维")
print("   16. 数据分布对VC维的影响")
print("\n总计: 16个交互式HTML文件")
print("=" * 60)
