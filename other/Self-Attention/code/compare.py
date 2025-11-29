import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
import math

st.set_page_config(page_title="LLM Math Visualization", layout="wide")

st.title("🧠 LLM 数学原理可视化：Transformer vs Mamba")

# ==========================================
# 模块 1: 复杂度与显存对比 (The Battle of Complexity)
# ==========================================
st.header("1. 复杂度爆炸：O(L²) vs O(L)")
st.markdown("直观感受为什么 Transformer 处理不了百万长文档，而 Mamba 可以。")

col1, col2 = st.columns([1, 3])

with col1:
    max_seq_len = st.slider("序列长度 (Sequence Length)", 1000, 20000, 10000, step=1000)
    d_model = st.selectbox("模型维度 (d_model)", [512, 768, 1024, 2048, 4096])
    dtype_size = 2 # FP16 = 2 bytes

with col2:
    # 生成数据
    seq_lens = np.arange(100, max_seq_len, 100)
    
    # 1. Transformer KV Cache 显存 (Bytes)
    # L * d_model * 2(K+V) * Layers(假设32) * Batch(1) * 2 bytes
    # 注意：Attention 计算矩阵是 L^2，这里简化展示 KV Cache 线性增长，但强调计算量
    
    # Transformer 计算量 (FLOPs) ~ L^2
    tf_compute = seq_lens ** 2 
    
    # Mamba 计算量 ~ L
    mamba_compute = seq_lens * 1000 # 乘以常数系数
    
    fig = go.Figure()
    
    # Transformer 曲线
    fig.add_trace(go.Scatter(
        x=seq_lens, y=tf_compute,
        mode='lines', name='Transformer Compute (Attention)',
        line=dict(color='red', width=3)
    ))
    
    # Mamba 曲线
    fig.add_trace(go.Scatter(
        x=seq_lens, y=mamba_compute,
        mode='lines', name='Mamba Compute (Scan)',
        line=dict(color='green', width=3)
    ))
    
    fig.update_layout(
        title="计算量增长趋势对比 (Compute Cost)",
        xaxis_title="Sequence Length (L)",
        yaxis_title="Theoretical FLOPs (Arbitrary Scale)",
        template="plotly_dark"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    if max_seq_len > 10000:
        st.error(f"⚠️ 警告：当序列长度达到 {max_seq_len} 时，Transformer 的 Attention 矩阵需要计算 {max_seq_len**2:,} 次点积！")

# ==========================================
# 模块 2: Attention 矩阵 vs SSM 扫描
# ==========================================
st.divider()
st.header("2. 内部机制：全图连接 vs 递归状态")

col3, col4 = st.columns(2)

with col3:
    st.subheader("Transformer: Attention Matrix")
    st.caption("每个 Token 都要看其他所有 Token (全局视野)")
    
    # 模拟一个 Causal Attention Mask
    seq_len_viz = st.slider("可视化序列长度", 10, 50, 20)
    
    # 生成下三角矩阵
    mask = torch.tril(torch.ones(seq_len_viz, seq_len_viz))
    # 模拟一些随机权重
    scores = torch.randn(seq_len_viz, seq_len_viz)
    scores = scores.masked_fill(mask == 0, float('-inf'))
    attn_weights = torch.softmax(scores, dim=-1).numpy()
    
    fig_attn = go.Figure(data=go.Heatmap(
        z=attn_weights,
        colorscale='Viridis',
        showscale=False
    ))
    fig_attn.update_layout(
        title="Causal Attention Matrix (Lower Triangular)",
        xaxis_title="Key Position",
        yaxis_title="Query Position",
        width=400, height=400,
        yaxis=dict(autorange="reversed") # 0在上面
    )
    st.plotly_chart(fig_attn)

with col4:
    st.subheader("Mamba: Hidden State Evolution")
    st.caption("信息被压缩进一个固定的隐状态 h (隧道视野)")
    
    # 模拟 SSM 状态演变
    # h_t = A * h_{t-1} + B * x_t
    t_steps = np.arange(seq_len_viz)
    hidden_state_dim = 16
    
    # 随机生成隐状态演变数据用于展示
    # 模拟随着时间推移，某些状态被遗忘，某些被激活
    hidden_states = np.zeros((hidden_state_dim, seq_len_viz))
    
    current_h = np.random.randn(hidden_state_dim)
    for t in range(seq_len_viz):
        # 简单的衰减+输入模拟
        current_h = 0.9 * current_h + 0.1 * np.random.randn(hidden_state_dim)
        hidden_states[:, t] = current_h
        
    fig_ssm = go.Figure(data=go.Heatmap(
        z=hidden_states,
        colorscale='Magma',
        showscale=False
    ))
    fig_ssm.update_layout(
        title=f"SSM Hidden State (Fixed Size: {hidden_state_dim})",
        xaxis_title="Time Step (t)",
        yaxis_title="State Dimension (h)",
        width=400, height=400
    )
    st.plotly_chart(fig_ssm)

st.info("💡 对比观察：左边的矩阵大小随序列长度平方增长 (N*N)，而右边的隐状态高度永远固定 (d_state)，只随时间向右延伸。")

# ==========================================
# 模块 3: RoPE 旋转演示
# ==========================================
st.divider()
st.header("3. RoPE 旋转位置编码直观演示")

col5, col6 = st.columns([1, 2])

with col5:
    st.markdown("调整位置索引 $m$ 和 $n$，观察向量旋转角度的变化。")
    pos_m = st.slider("Token m 位置", 0, 20, 0)
    pos_n = st.slider("Token n 位置", 0, 20, 5)
    theta = st.slider("Base Theta", 0.1, 2.0, 1.0)

with col6:
    # 简单的 2D 向量旋转演示
    vec_len = 1.0
    
    # 初始向量 (1, 0)
    # RoPE: rotate by angle = pos * theta
    angle_m = pos_m * theta
    angle_n = pos_n * theta
    
    vec_m = np.array([np.cos(angle_m), np.sin(angle_m)])
    vec_n = np.array([np.cos(angle_n), np.sin(angle_n)])
    
    fig_rope = go.Figure()
    
    # 向量 m
    fig_rope.add_trace(go.Scatter(
        x=[0, vec_m[0]], y=[0, vec_m[1]],
        mode='lines+markers', name=f'Token m (pos={pos_m})',
        line=dict(color='cyan', width=4)
    ))
    
    # 向量 n
    fig_rope.add_trace(go.Scatter(
        x=[0, vec_n[0]], y=[0, vec_n[1]],
        mode='lines+markers', name=f'Token n (pos={pos_n})',
        line=dict(color='magenta', width=4)
    ))
    
    # 相对距离
    rel_dist = abs(pos_m - pos_n)
    fig_rope.update_layout(
        title=f"向量旋转示意图 (Relative Distance = {rel_dist})",
        xaxis=dict(range=[-1.5, 1.5], scaleanchor="y"),
        yaxis=dict(range=[-1.5, 1.5]),
        width=500, height=500,
        template="plotly_dark"
    )
    
    st.plotly_chart(fig_rope)