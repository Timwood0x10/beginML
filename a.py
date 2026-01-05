import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle

# ==========================================
# ⚙️ 1. 设置：你想看实时播放还是保存？
# ==========================================
SHOW_WINDOW = False      # True: 弹出窗口实时播放 (推荐)
SAVE_GIF    = True    # True: 保存为 rope_demo.gif (需要几秒钟生成)

# ==========================================
# 📊 2. 核心参数 (RoPE Physics)
# ==========================================
DIM = 32           # 模型总维度
BASE = 10000       # RoPE 基数 (10000)
SEQ_LEN = 128      # 演示的序列长度
FRAMES = 150       # 动画总帧数
FPS = 30           # 流畅度

# 选取 4 个代表性维度 (从快到慢)
target_dims_idx = [0, 4, 8, 14] 
num_plots = len(target_dims_idx)

# 计算 Theta (角速度)
d_range = np.arange(0, DIM, 2)
thetas = 1.0 / (BASE ** (d_range / DIM))
selected_thetas = [thetas[i] for i in target_dims_idx]

# ==========================================
# 🎨 3. 绘图初始化 (Dark Mode Style)
# ==========================================
plt.style.use('dark_background')
fig = plt.figure(figsize=(14, 8))
fig.suptitle(f'RoPE Animation: Rotary Positional Embedding', fontsize=20, color='white', fontweight='bold')

axes_clocks = []
axes_waves = []
# 霓虹配色方案
colors = ['#FF0055', '#00CCFF', '#00FFAA', '#FFFF00'] 

for i in range(num_plots):
    # --- 上排：旋转向量 (时钟) ---
    ax = fig.add_subplot(2, num_plots, i + 1)
    ax.set_aspect('equal')
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.axis('off')
    
    # 轨道圈
    circle = Circle((0, 0), 1.0, color='white', fill=False, alpha=0.15, linestyle='--')
    ax.add_patch(circle)
    
    dim_real = target_dims_idx[i] * 2
    freq_label = "High Freq (Local)" if i == 0 else "Low Freq (Long-term)" if i == num_plots-1 else "Mid Freq"
    ax.set_title(f'Dim {dim_real}-{dim_real+1}\n{freq_label}', fontsize=10, color=colors[i])
    axes_clocks.append(ax)

    # --- 下排：相位波形 (Cos曲线) ---
    ax2 = fig.add_subplot(2, num_plots, i + 1 + num_plots)
    ax2.set_xlim(0, SEQ_LEN)
    ax2.set_ylim(-1.2, 1.2)
    ax2.set_xlabel('Position (m)')
    
    # 极简风格坐标轴
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['bottom'].set_color('#444')
    ax2.spines['left'].set_color('#444')
    ax2.tick_params(colors='#888', labelsize=8)
    if i > 0: ax2.set_yticks([]) 
    
    axes_waves.append(ax2)

# ==========================================
# 🎬 4. 动画对象初始化
# ==========================================
vectors = []   
trails = []    
waves = []     
points = []    
verticals = [] 

# 历史轨迹数据容器
history_x = [[] for _ in range(num_plots)]
history_y = [[] for _ in range(num_plots)]
wave_x = [[] for _ in range(num_plots)]
wave_y = [[] for _ in range(num_plots)]

for i in range(num_plots):
    c = colors[i]
    # 时钟元素
    vec, = axes_clocks[i].plot([], [], color=c, lw=3, marker='o', markersize=6, markevery=[-1])
    trail, = axes_clocks[i].plot([], [], color=c, lw=1, alpha=0.5)
    vectors.append(vec)
    trails.append(trail)
    
    # 波形元素
    wav, = axes_waves[i].plot([], [], color=c, lw=2)
    pt, = axes_waves[i].plot([], [], color='white', marker='o', markersize=5)
    line = axes_waves[i].axvline(x=0, color='white', linestyle=':', alpha=0.3)
    waves.append(wav)
    points.append(pt)
    verticals.append(line)

# ==========================================
# 🔄 5. 帧更新函数 (每帧调用)
# ==========================================
def update(frame):
    # 计算当前位置 m (往复运动效果: 0->128->0)
    # 这样动画看起来是循环的，不会突然跳断
    if frame <= FRAMES // 2:
        m = (frame / (FRAMES // 2)) * SEQ_LEN
    else:
        m = SEQ_LEN - ((frame - FRAMES // 2) / (FRAMES // 2)) * SEQ_LEN
        
    for i in range(num_plots):
        # RoPE 核心公式
        theta = selected_thetas[i]
        angle = m * theta 
        
        x, y = np.cos(angle), np.sin(angle)
        
        # 更新时钟向量
        vectors[i].set_data([0, x], [0, y])
        
        # 更新拖尾
        history_x[i].append(x)
        history_y[i].append(y)
        if len(history_x[i]) > 40: # 限制拖尾长度
            history_x[i].pop(0)
            history_y[i].pop(0)
        trails[i].set_data(history_x[i], history_y[i])
        
        # 更新波形图
        # 注意：为了性能，这里我们动态重绘波形
        # 实际 RoPE 是静态计算好的，这里为了演示画出"扫描"的感觉
        current_wave_x = np.linspace(0, m, int(m)+1)
        current_wave_y = np.cos(current_wave_x * theta)
        waves[i].set_data(current_wave_x, current_wave_y)
        
        points[i].set_data([m], [x])
        verticals[i].set_xdata([m])

    return vectors + trails + waves + points + verticals

# ==========================================
# ▶️ 6. 启动引擎
# ==========================================
ani = animation.FuncAnimation(
    fig, update, frames=FRAMES, interval=1000/FPS, blit=True
)

if SAVE_GIF:
    print("🚀 正在渲染 GIF，请稍候...")
    ani.save('rope_demo.gif', writer='pillow', fps=FPS)
    print("✅ 保存成功: rope_demo.gif")

if SHOW_WINDOW:
    print("🖥️ 正在弹出窗口播放...")
    plt.tight_layout()
    plt.show() # 这句代码会让窗口弹出来