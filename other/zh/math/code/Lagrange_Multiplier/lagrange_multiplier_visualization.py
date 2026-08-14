"""
Lagrange Multiplier Interactive Visualization
Based on 4.Lagrange_Multiplier.md document formulas
Using Plotly to create interactive animated charts with mathematical formulas
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# Create output directory
output_dir = "Lagrange_Multiplier"
os.makedirs(output_dir, exist_ok=True)

print("=" * 60)
print("🎨 Lagrange Multiplier Interactive Visualization")
print("=" * 60)


def add_formula_annotation(fig, formula_text, x=0.5, y=1.05):
    """Add formula annotation to chart"""
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
# 1. Example 1: Linear function on circle constraint
# ============================================
print("\n1️⃣ Creating circle constraint linear function visualization...")


def create_circle_example(a=1, b=1):
    """Create visualization of linear function on circle constraint"""

    # Calculate extreme points
    norm = np.sqrt(a * a + b * b)
    max_point = np.array([a / norm, b / norm])
    min_point = np.array([-a / norm, -b / norm])
    max_value = a * max_point[0] + b * max_point[1]
    min_value = a * min_point[0] + b * min_point[1]
    lambda_max = -norm
    lambda_min = norm

    # Create grid data
    x_range = np.linspace(-2, 2, 100)
    y_range = np.linspace(-2, 2, 100)
    X, Y = np.meshgrid(x_range, y_range)
    Z = a * X + b * Y

    # Unit circle data
    theta = np.linspace(0, 2 * np.pi, 100)
    circle_x = np.cos(theta)
    circle_y = np.sin(theta)

    # Create figure
    fig = go.Figure()

    # Add contour lines
    fig.add_trace(
        go.Contour(
            x=x_range,
            y=y_range,
            z=Z,
            colorscale="Viridis",
            contours=dict(showlabels=True, labelfont=dict(size=12, color="white")),
            colorbar=dict(
                title="目标函数 f(x,y) = ax + by<br>Target Function", titleside="right"
            ),
            hoverinfo="skip",
        )
    )

    # Add constraint circle
    fig.add_trace(
        go.Scatter(
            x=circle_x,
            y=circle_y,
            mode="lines",
            name="约束条件 Constraint: x² + y² = 1",
            line=dict(color="blue", width=4),
        )
    )

    # Add extreme points
    fig.add_trace(
        go.Scatter(
            x=[max_point[0], min_point[0]],
            y=[max_point[1], min_point[1]],
            mode="markers",
            name="极值点 Extreme Points",
            marker=dict(
                color=["red", "green"], size=12, symbol=["triangle-up", "triangle-down"]
            ),
            text=[
                f"最大值 Max: ({max_point[0]:.3f}, {max_point[1]:.3f})<br>值 Value: {max_value:.3f}",
                f"最小值 Min: ({min_point[0]:.3f}, {min_point[1]:.3f})<br>值 Value: {min_value:.3f}",
            ],
            hovertemplate="%{text}<extra></extra>",
        )
    )

    # Add gradient arrows
    arrow_scale = 0.3
    fig.add_annotation(
        x=max_point[0],
        y=max_point[1],
        ax=max_point[0] + arrow_scale * a / norm,
        ay=max_point[1] + arrow_scale * b / norm,
        axref="x",
        ayref="y",
        xref="x",
        yref="y",
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor="red",
    )

    fig.add_annotation(
        x=min_point[0],
        y=min_point[1],
        ax=min_point[0] - arrow_scale * a / norm,
        ay=min_point[1] - arrow_scale * b / norm,
        axref="x",
        ayref="y",
        xref="x",
        yref="y",
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor="green",
    )

    # Add formula
    fig = add_formula_annotation(
        fig,
        r"$$\mathcal{L}(x,y,\lambda) = ax + by + \lambda(x^2 + y^2 - 1)$$",
        x=0.5,
        y=1.05,
    )

    fig.update_layout(
        title=f"在圆 x² + y² = 1 上求 f(x,y) = {a}x + {b}y 的极值<br>Extrema of f(x,y) = {a}x + {b}y on circle x² + y² = 1<br>"
        + f"Max: {max_value:.3f}, Min: {min_value:.3f}<br>"
        + f"λ_max = {lambda_max:.3f}, λ_min = {lambda_min:.3f}",
        xaxis_title="x",
        yaxis_title="y",
        xaxis=dict(range=[-2, 2], scaleanchor="y", scaleratio=1),
        yaxis=dict(range=[-2, 2]),
        height=700,
        margin=dict(t=120, b=60, l=60, r=60),
    )

    return fig, {
        "max_point": max_point,
        "min_point": min_point,
        "max_value": max_value,
        "min_value": min_value,
        "lambda_max": lambda_max,
        "lambda_min": lambda_min,
    }


# Create animation with different parameters
a_values = np.linspace(-2, 2, 15)
b_values = np.linspace(-2, 2, 15)
frames_circle = []

for i in range(len(a_values)):
    a = a_values[i]
    b = b_values[i]
    fig, results = create_circle_example(a, b)

    frame_data = fig.data
    frames_circle.append(
        go.Frame(
            data=frame_data,
            name=str(i),
            layout=go.Layout(
                title_text=f"在圆 x² + y² = 1 上求 f(x,y) = {a:.1f}x + {b:.1f}y 的极值<br>Extrema of f(x,y) = {a:.1f}x + {b:.1f}y on circle x² + y² = 1<br>"
                + f'Max: {results["max_value"]:.3f}, Min: {results["min_value"]:.3f}'
            ),
        )
    )

# Create main figure
fig1, _ = create_circle_example(1, 1)
fig1.frames = frames_circle

# Add play button
fig1.update_layout(
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
    ]
)

output_file = os.path.join(output_dir, "1_circle_linear.html")
fig1.write_html(output_file, include_mathjax="cdn")
print(f"   ✅ Saved: {output_file}")

# ============================================
# 2. Example 2: Quadratic function on ellipse constraint
# ============================================
print("\n2️⃣ Creating ellipse constraint quadratic function visualization...")


def create_ellipse_example(a=2, b=3):
    """Create visualization of quadratic function on ellipse constraint"""

    # Calculate extreme points
    max_point = np.array([a, 0])  # On major axis
    min_point = np.array([0, b])  # On minor axis
    max_value = max_point[0] ** 2 + max_point[1] ** 2
    min_value = min_point[0] ** 2 + min_point[1] ** 2
    lambda_max = -a * a / 2
    lambda_min = -b * b / 2

    # Create grid data
    max_range = max(a, b) + 1
    x_range = np.linspace(-max_range, max_range, 100)
    y_range = np.linspace(-max_range, max_range, 100)
    X, Y = np.meshgrid(x_range, y_range)
    Z = X**2 + Y**2

    # Ellipse data
    theta = np.linspace(0, 2 * np.pi, 100)
    ellipse_x = a * np.cos(theta)
    ellipse_y = b * np.sin(theta)

    # Create figure
    fig = go.Figure()

    # Add contour lines
    fig.add_trace(
        go.Contour(
            x=x_range,
            y=y_range,
            z=Z,
            colorscale="Viridis",
            contours=dict(showlabels=True, labelfont=dict(size=12, color="white")),
            colorbar=dict(
                title="目标函数 f(x,y) = x² + y²<br>Target Function", titleside="right"
            ),
            hoverinfo="skip",
        )
    )

    # Add constraint ellipse
    fig.add_trace(
        go.Scatter(
            x=ellipse_x,
            y=ellipse_y,
            mode="lines",
            name=f"Constraint: x²/{a}² + y²/{b}² = 1",
            line=dict(color="blue", width=4),
        )
    )

    # Add extreme points
    fig.add_trace(
        go.Scatter(
            x=[max_point[0], min_point[0]],
            y=[max_point[1], min_point[1]],
            mode="markers",
            name="极值点 Extreme Points",
            marker=dict(
                color=["red", "green"], size=12, symbol=["triangle-up", "triangle-down"]
            ),
            text=[
                f"最大值 Max: ({max_point[0]:.3f}, {max_point[1]:.3f})<br>值 Value: {max_value:.3f}",
                f"最小值 Min: ({min_point[0]:.3f}, {min_point[1]:.3f})<br>值 Value: {min_value:.3f}",
            ],
            hovertemplate="%{text}<extra></extra>",
        )
    )

    # Add formula
    fig = add_formula_annotation(
        fig,
        r"$$\mathcal{L}(x,y,\lambda) = x^2 + y^2 + \lambda\left(\frac{x^2}{a^2} + \frac{y^2}{b^2} - 1\right)$$",
        x=0.5,
        y=1.05,
    )

    fig.update_layout(
        title=f"在椭圆 x²/{a}² + y²/{b}² = 1 上求 f(x,y) = x² + y² 的极值<br>Extrema of f(x,y) = x² + y² on ellipse x²/{a}² + y²/{b}² = 1<br>"
        + f"Max: {max_value:.3f}, Min: {min_value:.3f}<br>"
        + f"λ_max = {lambda_max:.3f}, λ_min = {lambda_min:.3f}",
        xaxis_title="x",
        yaxis_title="y",
        xaxis=dict(range=[-max_range, max_range], scaleanchor="y", scaleratio=1),
        yaxis=dict(range=[-max_range, max_range]),
        height=700,
        margin=dict(t=120, b=60, l=60, r=60),
    )

    return fig


# Create ellipse animation
a_values = np.linspace(1, 4, 10)
b_values = np.linspace(1, 4, 10)
frames_ellipse = []

for i in range(len(a_values)):
    a = a_values[i]
    b = b_values[i]
    fig = create_ellipse_example(a, b)

    frame_data = fig.data
    frames_ellipse.append(go.Frame(data=frame_data, name=str(i)))

# Create main figure
fig2 = create_ellipse_example(2, 3)
fig2.frames = frames_ellipse

# Add play button
fig2.update_layout(
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
                            frame=dict(duration=300, redraw=True),
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
    ]
)

output_file = os.path.join(output_dir, "2_ellipse_quadratic.html")
fig2.write_html(output_file, include_mathjax="cdn")
print(f"   ✅ Saved: {output_file}")

# ============================================
# 3. Gradient Parallel Visualization
# ============================================
print("\n3️⃣ Creating gradient parallel visualization...")


def create_gradient_parallel_demo():
    """Create visualization of gradient parallelism"""

    # Create grid
    x = np.linspace(-3, 3, 30)
    y = np.linspace(-3, 3, 30)
    X, Y = np.meshgrid(x, y)

    # Target function f(x,y) = x + y
    F = X + Y

    # Constraint function g(x,y) = x² + y² - 1
    G = X**2 + Y**2 - 1

    # Calculate gradients
    grad_f_x = np.ones_like(X)  # ∂f/∂x = 1
    grad_f_y = np.ones_like(Y)  # ∂f/∂y = 1

    grad_g_x = 2 * X  # ∂g/∂x = 2x
    grad_g_y = 2 * Y  # ∂g/∂y = 2y

    # Create subplots
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "目标函数梯度场 Target Function Gradient Field",
            "约束函数梯度场 Constraint Function Gradient Field",
        ),
        specs=[[{"type": "scatter"}, {"type": "scatter"}]],
    )

    # Target function gradient field
    fig.add_trace(
        go.Scatter(
            x=X.flatten(),
            y=Y.flatten(),
            mode="markers",
            marker=dict(
                size=3, color=F.flatten(), colorscale="Viridis", showscale=False
            ),
            showlegend=False,
            hoverinfo="skip",
        ),
        row=1,
        col=1,
    )

    # Add target function gradient arrows (sampled)
    skip = 3
    fig.add_trace(
        go.Scatter(
            x=X[::skip, ::skip].flatten(),
            y=Y[::skip, ::skip].flatten(),
            mode="markers",
            marker=dict(size=8, color="red"),
            name="目标函数梯度 ∇f",
            text=[f"∇f = (1, 1)"] * len(X[::skip, ::skip].flatten()),
            hovertemplate="(%{x:.1f}, %{y:.1f})<br>%{text}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # Constraint function gradient field
    fig.add_trace(
        go.Scatter(
            x=X.flatten(),
            y=Y.flatten(),
            mode="markers",
            marker=dict(size=3, color=G.flatten(), colorscale="RdBu", showscale=False),
            showlegend=False,
            hoverinfo="skip",
        ),
        row=1,
        col=2,
    )

    # Add constraint function gradient arrows (sampled)
    fig.add_trace(
        go.Scatter(
            x=X[::skip, ::skip].flatten(),
            y=Y[::skip, ::skip].flatten(),
            mode="markers",
            marker=dict(size=8, color="blue"),
            name="约束函数梯度 ∇g",
            text=[
                f"∇g = ({2*X[i,j]:.1f}, {2*Y[i,j]:.1f})"
                for i in range(0, len(x), skip)
                for j in range(0, len(y), skip)
            ],
            hovertemplate="(%{x:.1f}, %{y:.1f})<br>%{text}<extra></extra>",
        ),
        row=1,
        col=2,
    )

    # Add unit circle
    theta = np.linspace(0, 2 * np.pi, 100)
    circle_x = np.cos(theta)
    circle_y = np.sin(theta)

    fig.add_trace(
        go.Scatter(
            x=circle_x,
            y=circle_y,
            mode="lines",
            name="约束条件 Constraint: x² + y² = 1",
            line=dict(color="green", width=3),
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    # Mark extreme points
    max_point = np.array([1 / np.sqrt(2), 1 / np.sqrt(2)])
    min_point = np.array([-1 / np.sqrt(2), -1 / np.sqrt(2)])

    fig.add_trace(
        go.Scatter(
            x=[max_point[0], min_point[0]],
            y=[max_point[1], min_point[1]],
            mode="markers",
            marker=dict(
                color=["red", "green"], size=12, symbol=["triangle-up", "triangle-down"]
            ),
            name="极值点 Extreme Points",
            text=["最大值点 Max Point", "最小值点 Min Point"],
            hovertemplate="%{text}<br>(%{x:.3f}, %{y:.3f})<extra></extra>",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=[max_point[0], min_point[0]],
            y=[max_point[1], min_point[1]],
            mode="markers",
            marker=dict(
                color=["red", "green"], size=12, symbol=["triangle-up", "triangle-down"]
            ),
            name="极值点 Extreme Points",
            showlegend=False,
            text=["最大值点 Max Point", "最小值点 Min Point"],
            hovertemplate="%{text}<br>(%{x:.3f}, %{y:.3f})<extra></extra>",
        ),
        row=1,
        col=2,
    )

    # Add formula
    fig = add_formula_annotation(
        fig,
        r"$$\nabla f = \lambda \nabla g \quad \text{at extreme points}$$",
        x=0.5,
        y=1.05,
    )

    fig.update_xaxes(title_text="x", row=1, col=1)
    fig.update_xaxes(title_text="x", row=1, col=2)
    fig.update_yaxes(title_text="y", row=1, col=1)
    fig.update_yaxes(title_text="y", row=1, col=2)

    fig.update_layout(
        title_text="梯度平行性演示 Gradient Parallelism Demo: ∇f = λ∇g",
        height=600,
        showlegend=True,
        margin=dict(t=120, b=60, l=60, r=60),
    )

    return fig


fig3 = create_gradient_parallel_demo()
output_file = os.path.join(output_dir, "3_gradient_parallel.html")
fig3.write_html(output_file, include_mathjax="cdn")
print(f"   ✅ Saved: {output_file}")

# ============================================
# 4. 3D Visualization: Constraint Surface and Target Function
# ============================================
print("\n4️⃣ Creating 3D constraint surface visualization...")


def create_3d_constraint_visualization():
    """Create 3D visualization of constraint surface and target function"""

    # Create grid
    x = np.linspace(-2, 2, 50)
    y = np.linspace(-2, 2, 50)
    X, Y = np.meshgrid(x, y)

    # Target function f(x,y) = x + y
    Z = X + Y

    # Constraint x² + y² = 1 appears as a cylinder in 3D
    theta = np.linspace(0, 2 * np.pi, 50)
    z_line = np.linspace(-4, 4, 30)
    THETA, Z_LINE = np.meshgrid(theta, z_line)
    CYLINDER_X = np.cos(THETA)
    CYLINDER_Y = np.sin(THETA)
    CYLINDER_Z = Z_LINE

    # Create 3D figure
    fig = go.Figure()

    # Add target function surface
    fig.add_trace(
        go.Surface(
            x=X,
            y=Y,
            z=Z,
            colorscale="Viridis",
            opacity=0.7,
            name="目标函数 f(x,y) = x + y",
            colorbar=dict(
                title="目标函数 f(x,y)<br>Target Function", titleside="right"
            ),
            hovertemplate="x: %{x:.2f}<br>y: %{y:.2f}<br>f(x,y): %{z:.2f}<extra></extra>",
        )
    )

    # Add constraint cylinder
    fig.add_trace(
        go.Surface(
            x=CYLINDER_X,
            y=CYLINDER_Y,
            z=CYLINDER_Z,
            colorscale="Reds",
            opacity=0.3,
            name="约束条件 Constraint: x² + y² = 1",
            showscale=False,
            hovertemplate="Constraint Surface<br>x: %{x:.2f}<br>y: %{y:.2f}<br>z: %{z:.2f}<extra></extra>",
        )
    )

    # Add constraint curve (intersection of target function and constraint)
    t = np.linspace(0, 2 * np.pi, 100)
    constraint_x = np.cos(t)
    constraint_y = np.sin(t)
    constraint_z = constraint_x + constraint_y

    fig.add_trace(
        go.Scatter3d(
            x=constraint_x,
            y=constraint_y,
            z=constraint_z,
            mode="lines+markers",
            name="约束曲线 Constraint Curve",
            line=dict(color="red", width=6),
            marker=dict(size=4, color="red"),
        )
    )

    # Mark extreme points
    max_point = np.array([1 / np.sqrt(2), 1 / np.sqrt(2), np.sqrt(2)])
    min_point = np.array([-1 / np.sqrt(2), -1 / np.sqrt(2), -np.sqrt(2)])

    fig.add_trace(
        go.Scatter3d(
            x=[max_point[0], min_point[0]],
            y=[max_point[1], min_point[1]],
            z=[max_point[2], min_point[2]],
            mode="markers",
            name="极值点 Extreme Points",
            marker=dict(color=["red", "green"], size=10, symbol=["diamond", "diamond"]),
            text=["最大值点 Max Point", "最小值点 Min Point"],
            hovertemplate="%{text}<br>(%{x:.3f}, %{y:.3f}, %{z:.3f})<extra></extra>",
        )
    )

    # Add formula
    fig = add_formula_annotation(
        fig,
        r"$$\mathcal{L}(x,y,\lambda) = x + y + \lambda(x^2 + y^2 - 1)$$",
        x=0.5,
        y=0.98,
    )

    fig.update_layout(
        title="3D可视化：目标函数曲面与约束圆柱面<br>3D Visualization: Target Function Surface and Constraint Cylinder",
        scene=dict(
            xaxis_title="x",
            yaxis_title="y",
            zaxis_title="f(x,y) = x + y",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.3)),
        ),
        height=750,
        margin=dict(t=120, b=60, l=60, r=60),
    )

    return fig


fig4 = create_3d_constraint_visualization()
output_file = os.path.join(output_dir, "4_3d_constraint.html")
fig4.write_html(output_file, include_mathjax="cdn")
print(f"   ✅ Saved: {output_file}")

# ============================================
# 5. Geometric Meaning of Lagrange Multiplier
# ============================================
print("\n5️⃣ Creating geometric meaning of Lagrange multiplier visualization...")


def create_lambda_geometric_meaning():
    """Create visualization of geometric meaning of Lagrange multiplier"""

    # Create Lagrangian function contour lines for different lambda values
    x = np.linspace(-2, 2, 100)
    y = np.linspace(-2, 2, 100)
    X, Y = np.meshgrid(x, y)

    fig = go.Figure()

    # Add constraint circle
    theta = np.linspace(0, 2 * np.pi, 100)
    circle_x = np.cos(theta)
    circle_y = np.sin(theta)

    fig.add_trace(
        go.Scatter(
            x=circle_x,
            y=circle_y,
            mode="lines",
            name="约束条件 Constraint: x² + y² = 1",
            line=dict(color="blue", width=4),
        )
    )

    # Add Lagrangian function contour lines for different lambda values
    lambda_values = [-2, -1, -0.5, 0, 0.5, 1, 2]
    colors = ["red", "orange", "yellow", "green", "cyan", "purple", "pink"]

    for i, lam in enumerate(lambda_values):
        # Lagrangian function: L(x,y,λ) = x + y + λ(x² + y² - 1)
        L = X + Y + lam * (X**2 + Y**2 - 1)

        # Only show contour lines near 0
        fig.add_trace(
            go.Contour(
                x=x,
                y=y,
                z=L,
                contours=dict(start=-0.5, end=0.5, size=0.1, showlabels=False),
                colorscale=[[0, colors[i]], [1, colors[i]]],
                showscale=False,
                name=f"λ = {lam}",
                hoverinfo="skip",
                opacity=0.6,
                line=dict(width=2),
            )
        )

    # Mark extreme points
    max_point = np.array([1 / np.sqrt(2), 1 / np.sqrt(2)])
    min_point = np.array([-1 / np.sqrt(2), -1 / np.sqrt(2)])

    fig.add_trace(
        go.Scatter(
            x=[max_point[0], min_point[0]],
            y=[max_point[1], min_point[1]],
            mode="markers",
            name="极值点 Extreme Points",
            marker=dict(
                color=["red", "green"], size=12, symbol=["triangle-up", "triangle-down"]
            ),
            text=[f"最大值点 Max Point (λ = -√2)", f"最小值点 Min Point (λ = √2)"],
            hovertemplate="%{text}<br>(%{x:.3f}, %{y:.3f})<extra></extra>",
        )
    )

    # Add formula
    fig = add_formula_annotation(
        fig,
        r"$$\mathcal{L}(x,y,\lambda) = x + y + \lambda(x^2 + y^2 - 1)$$",
        x=0.5,
        y=1.05,
    )

    fig.update_layout(
        title="拉格朗日乘数的几何意义：不同λ值的拉格朗日函数等高线<br>Geometric Meaning of Lagrange Multiplier: Contour Lines for Different λ Values",
        xaxis_title="x",
        yaxis_title="y",
        xaxis=dict(range=[-2, 2], scaleanchor="y", scaleratio=1),
        yaxis=dict(range=[-2, 2]),
        height=700,
        margin=dict(t=120, b=60, l=60, r=60),
    )

    return fig


fig5 = create_lambda_geometric_meaning()
output_file = os.path.join(output_dir, "5_lambda_geometry.html")
fig5.write_html(output_file, include_mathjax="cdn")
print(f"   ✅ Saved: {output_file}")

# ============================================
# 6. Dual Problem Visualization
# ============================================
print("\n6️⃣ Creating dual problem visualization...")


def create_dual_problem_visualization():
    """Create visualization of dual problem"""

    # Dual function g(λ) = inf_x,y L(x,y,λ)
    # For f(x,y) = x + y, g(x,y) = x² + y² - 1
    # g(λ) = -1/(4λ) (when λ < 0)

    lambda_range = np.linspace(-3, -0.1, 100)
    dual_values = -1 / (4 * lambda_range)

    # Optimal value of primal problem
    primal_optimal = np.sqrt(2)

    fig = go.Figure()

    # Add dual function curve
    fig.add_trace(
        go.Scatter(
            x=lambda_range,
            y=dual_values,
            mode="lines",
            name="对偶函数 Dual Function g(λ)",
            line=dict(color="blue", width=3),
            hovertemplate="λ: %{x:.3f}<br>g(λ): %{y:.3f}<extra></extra>",
        )
    )

    # Add primal optimal value line
    fig.add_hline(
        y=primal_optimal,
        line=dict(color="red", width=2, dash="dash"),
        annotation_text=f"原始最优值 Primal Optimal: {primal_optimal:.3f}",
        annotation_position="top right",
    )

    # Mark strong duality point
    optimal_lambda = -1 / np.sqrt(2)
    fig.add_trace(
        go.Scatter(
            x=[optimal_lambda],
            y=[primal_optimal],
            mode="markers",
            name="强对偶点 Strong Duality Point",
            marker=dict(color="green", size=12, symbol="diamond"),
            text=[f"最优λ Optimal λ: {optimal_lambda:.3f}"],
            hovertemplate="%{text}<br>(%{x:.3f}, %{y:.3f})<extra></extra>",
        )
    )

    # Add shaded region for weak duality
    fig.add_shape(
        type="rect",
        x0=-3,
        y0=-3,
        x1=0,
        y1=primal_optimal,
        fillcolor="lightblue",
        opacity=0.3,
        layer="below",
        line_width=0,
    )

    # Add formula
    fig = add_formula_annotation(
        fig,
        r"$$g(\lambda) = \inf_{x,y} \mathcal{L}(x,y,\lambda) = -\frac{1}{4\lambda}$$",
        x=0.5,
        y=1.05,
    )

    fig.update_layout(
        title="对偶问题可视化：强对偶与弱对偶<br>Dual Problem Visualization: Strong and Weak Duality",
        xaxis_title="拉格朗日乘数 Lagrange Multiplier λ",
        yaxis_title="对偶函数值 Dual Function Value g(λ)",
        height=600,
        margin=dict(t=120, b=60, l=60, r=60),
        annotations=[
            dict(
                x=-2,
                y=-2,
                text="弱对偶区域 Weak Duality Region:<br>g(λ) ≤ f*",
                showarrow=False,
                font=dict(size=12, color="blue"),
            )
        ],
    )

    return fig


fig6 = create_dual_problem_visualization()
output_file = os.path.join(output_dir, "6_dual_problem.html")
fig6.write_html(output_file, include_mathjax="cdn")
print(f"   ✅ Saved: {output_file}")

# ============================================
# 7. Comprehensive Dashboard
# ============================================
print("\n7️⃣ Creating comprehensive dashboard...")


def create_comprehensive_dashboard():
    """Create comprehensive dashboard for Lagrange multiplier method"""

    fig = make_subplots(
        rows=2,
        cols=3,
        subplot_titles=(
            "圆约束线性函数 Circle Constraint Linear Function",
            "椭圆约束二次函数 Ellipse Constraint Quadratic Function",
            "梯度平行性 Gradient Parallelism",
            "3D约束曲面 3D Constraint Surface",
            "拉格朗日乘数几何意义 Lagrange Multiplier Geometry",
            "对偶问题 Dual Problem",
        ),
        specs=[
            [{"type": "scatter"}, {"type": "scatter"}, {"type": "scatter"}],
            [{"type": "scatter3d"}, {"type": "scatter"}, {"type": "scatter"}],
        ],
    )

    # 1. Circle constraint example (simplified)
    theta = np.linspace(0, 2 * np.pi, 50)
    circle_x = np.cos(theta)
    circle_y = np.sin(theta)

    fig.add_trace(
        go.Scatter(
            x=circle_x,
            y=circle_y,
            mode="lines",
            name="圆约束 Circle Constraint",
            line=dict(color="blue", width=2),
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    # 2. Ellipse constraint example (simplified)
    ellipse_x = 2 * np.cos(theta)
    ellipse_y = 3 * np.sin(theta)

    fig.add_trace(
        go.Scatter(
            x=ellipse_x,
            y=ellipse_y,
            mode="lines",
            name="椭圆约束 Ellipse Constraint",
            line=dict(color="red", width=2),
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    # 3. Gradient arrow example
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="markers+lines",
            name="梯度 Gradient",
            marker=dict(size=8, color="green"),
            line=dict(width=2, color="green"),
            showlegend=False,
        ),
        row=1,
        col=3,
    )

    # 4. 3D surface example (simplified)
    x_3d = np.linspace(-1, 1, 20)
    y_3d = np.linspace(-1, 1, 20)
    X_3d, Y_3d = np.meshgrid(x_3d, y_3d)
    Z_3d = X_3d + Y_3d

    fig.add_trace(
        go.Scatter3d(
            x=X_3d.flatten(),
            y=Y_3d.flatten(),
            z=Z_3d.flatten(),
            mode="markers",
            marker=dict(
                size=2, color=Z_3d.flatten(), colorscale="Viridis", showscale=False
            ),
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    # 5. Dual function example
    lambda_range = np.linspace(-3, -0.1, 50)
    dual_values = -1 / (4 * lambda_range)

    fig.add_trace(
        go.Scatter(
            x=lambda_range,
            y=dual_values,
            mode="lines",
            name="对偶函数 Dual Function",
            line=dict(color="purple", width=2),
            showlegend=False,
        ),
        row=2,
        col=2,
    )

    # 6. Add some example points
    fig.add_trace(
        go.Scatter(
            x=[0.5, -0.5, 0, 0],
            y=[0.5, -0.5, 0.7, -0.7],
            mode="markers",
            marker=dict(color=["red", "green", "blue", "orange"], size=8),
            showlegend=False,
        ),
        row=2,
        col=3,
    )

    # Update axes
    fig.update_xaxes(title_text="x", row=1, col=1)
    fig.update_xaxes(title_text="x", row=1, col=2)
    fig.update_xaxes(title_text="x", row=1, col=3)
    fig.update_xaxes(title_text="λ", row=2, col=2)

    fig.update_yaxes(title_text="y", row=1, col=1)
    fig.update_yaxes(title_text="y", row=1, col=2)
    fig.update_yaxes(title_text="y", row=1, col=3)
    fig.update_yaxes(title_text="g(λ)", row=2, col=2)

    fig.update_layout(
        scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="f(x,y)"),
        height=800,
        showlegend=False,
        margin=dict(t=100, b=60, l=60, r=60),
    )

    return fig


fig7 = create_comprehensive_dashboard()
output_file = os.path.join(output_dir, "7_dashboard.html")
fig7.write_html(output_file, include_mathjax="cdn")
print(f"   ✅ Saved: {output_file}")

# ============================================
# Print Calculation Examples
# ============================================
print("\n" + "=" * 60)
print("📊 Lagrange Multiplier Calculation Examples")
print("=" * 60)

print("\n1️⃣ Example 1: Extrema of f(x,y) = x + y on circle x² + y² = 1")
print("   Constraint: g(x,y) = x² + y² - 1 = 0")
print("   Lagrangian: L(x,y,λ) = x + y + λ(x² + y² - 1)")
print("   Partial derivatives:")
print("     ∂L/∂x = 1 + 2λx = 0")
print("     ∂L/∂y = 1 + 2λy = 0")
print("     ∂L/∂λ = x² + y² - 1 = 0")
print("   Solution: x = y = ±1/√2, λ = ∓1/√2")
print("   Maximum: √2 (at point (1/√2, 1/√2))")
print("   Minimum: -√2 (at point (-1/√2, -1/√2))")

print("\n2️⃣ Example 2: Extrema of f(x,y) = x² + y² on ellipse x²/4 + y²/9 = 1")
print("   Constraint: g(x,y) = x²/4 + y²/9 - 1 = 0")
print("   Lagrangian: L(x,y,λ) = x² + y² + λ(x²/4 + y²/9 - 1)")
print("   Partial derivatives:")
print("     ∂L/∂x = 2x + λx/2 = 0")
print("     ∂L/∂y = 2y + 2λy/9 = 0")
print("     ∂L/∂λ = x²/4 + y²/9 - 1 = 0")
print("   Solution: Extreme points on ellipse axes")
print("   Maximum: 4 (at points (±2, 0))")
print("   Minimum: 9 (at points (0, ±3))")

print("\n3️⃣ Gradient Parallelism Verification:")
max_point = np.array([1 / np.sqrt(2), 1 / np.sqrt(2)])
grad_f = np.array([1, 1])  # ∇f = (1, 1)
grad_g = np.array([2 * max_point[0], 2 * max_point[1]])  # ∇g = (2x, 2y)
lambda_val = grad_f[0] / grad_g[0]  # λ = 1/(2x) = 1/√2
print(f"   At max point ({max_point[0]:.3f}, {max_point[1]:.3f}):")
print(f"     ∇f = {grad_f}")
print(f"     ∇g = {grad_g}")
print(f"     λ = {lambda_val:.3f}")
print(f"     Verify ∇f = λ∇g? {np.allclose(grad_f, lambda_val * grad_g)}")

print("\n" + "=" * 60)
print("✨ All interactive visualizations created successfully!")
print("=" * 60)
print(f"\n📂 Generated files located at: code/{output_dir}/")
print("   1. 1_circle_linear.html - Circle constraint linear function animation")
print(
    "   2. 2_ellipse_quadratic.html - Ellipse constraint quadratic function animation"
)
print("   3. 3_gradient_parallel.html - Gradient parallelism demonstration")
print("   4. 4_3d_constraint.html - 3D constraint surface visualization")
print("   5. 5_lambda_geometry.html - Geometric meaning of Lagrange multiplier")
print("   6. 6_dual_problem.html - Dual problem visualization")
print("   7. 7_dashboard.html - Comprehensive dashboard")
print("\n💡 Open these HTML files in your browser to view interactive visualizations!")
print("=" * 60)
