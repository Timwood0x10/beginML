# 📚 可视化脚本使用指南

## 🎯 项目概述

本项目为9个机器学习和数学理论文档创建了完整的交互式可视化，共包含 **73个HTML文件**。

---

## 📂 目录结构

```
code/
├── convolution/                          # 卷积运算可视化
│   ├── convolution_visualization.py
│   └── convolution/*.html (6个)
│
├── lossfunction/                         # 损失函数可视化
│   ├── lossfunction.py
│   └── lossfunction/*.html (7个)
│
├── grand_optimizer/                      # PyTorch优化器可视化
│   ├── grand_optimizer.py
│   └── *.html (7个)
│
├── Lagrange_Multiplier/                  # 拉格朗日乘数法可视化
│   ├── lagrange_multiplier_visualization.py
│   └── Lagrange_Multiplier/*.html (7个)
│
├── L1_L2_Regularization/                 # L1/L2正则化可视化
│   ├── l1_l2_regularization_visualization.py
│   └── L1_L2_Regularization/*.html (6个)
│
├── SVM/                                  # 支持向量机可视化
│   ├── svm_visualization.py
│   └── SVM/*.html (6个)
│
├── VCdime/                               # VC维理论可视化
│   ├── VCdime.py
│   └── *.html (12个)
│
├── GeneticAlgorithm/                     # 遗传算法可视化
│   ├── GeneticAlgorithm.py
│   └── *.html (7个)
│
├── Classification_Optimization_Logic/    # 分类模型优化可视化
│   ├── classification_optimization_logic_visualization.py
│   └── Classification_Optimization_Logic/*.html (6个)
│
└── interactive_gradient_descent/         # 交互式梯度下降工具
    ├── interactive_gradient_descent.py
    └── *.html (7个)
```

---

## 🚀 快速开始

### 1. 查看可视化

直接在浏览器中打开任意HTML文件：

```bash
# macOS
open code/lossfunction/lossfunction/1_least_squares.html

# Linux
xdg-open code/lossfunction/lossfunction/1_least_squares.html

# Windows
start code\lossfunction\lossfunction\1_least_squares.html
```

### 2. 重新生成可视化

如果需要修改参数或重新生成：

```bash
cd code

# 生成损失函数可视化
python lossfunction.py

# 生成优化器可视化
python grand_optimizer.py

# 生成VC维可视化
python VCdime.py

# 生成遗传算法可视化
python GeneticAlgorithm.py

# 生成交互式梯度下降
python interactive_gradient_descent.py
```

---

## 📊 各模块详细说明

### 1. 卷积运算 (convolution)

**对应文档**: `1.convolution.md`

**脚本**: `code/convolution/convolution_visualization.py`

**生成文件** (6个):
```
convolution/
├── 1_basic_2d_convolution.html        - 基础2D卷积动画
├── 2_padding_effect.html              - 填充效果
├── 3_stride_effect.html               - 步长影响
├── 4_multiple_filters.html            - 多卷积核
├── 5_feature_map_evolution.html       - 特征图演化
└── 6_dashboard.html                   - 综合仪表板
```

**核心公式**:
```
Output[i,j] = Σ Σ Input[i+m, j+n] × Kernel[m,n]
```

**用途**: 理解卷积神经网络的基础操作

---

### 2. 损失函数 (lossfunction)

**对应文档**: `2.lossfunction.md`

**脚本**: `code/lossfunction/lossfunction.py`

**生成文件** (7个):
```
lossfunction/lossfunction/
├── 1_least_squares.html               - 最小二乘法动画
├── 2_cross_entropy.html               - 交叉熵对比
├── 3_penalty_animation.html           - 惩罚强度动画
├── 4_softmax_3d.html                  - Softmax 3D可视化
├── 5_comparison.html                  - 损失函数对比
├── 6_gradient_descent.html            - 梯度下降过程
└── 7_dashboard.html                   - 综合仪表板
```

**核心公式**:
```
MSE: L = (y - ŷ)²
Cross Entropy: L = -[y·log(p) + (1-y)·log(1-p)]
Softmax: p_i = exp(z_i) / Σ exp(z_j)
```

**用途**: 理解不同任务应该使用什么损失函数

---

### 3. PyTorch优化器 (grand_optimizer)

**对应文档**: `3.grand_optimizer.md`

**脚本**: `code/grand_optimizer.py`

**生成文件** (7个):
```
grand_optimizer/
├── 1_sgd_vs_momentum.html             - SGD vs Momentum对比
├── 2_adam_optimizer.html              - Adam优化器演示
├── 3_optimizer_comparison.html        - 五种优化器对比
├── 4_learning_rate_impact.html        - 学习率影响
├── 5_momentum_impact.html             - 动量系数影响
├── 6_adam_adaptive_stepsize.html      - Adam自适应步长
└── 7_dashboard.html                   - 综合仪表板
```

**核心公式**:
```
SGD: θ ← θ - η·g
Momentum: v ← μ·v + g, θ ← θ - η·v
Adam: m ← β₁m + (1-β₁)g, v ← β₂v + (1-β₂)g²
      θ ← θ - η·m̂/(√v̂ + ε)
```

**用途**: 选择合适的优化算法和调参

---

### 4. 拉格朗日乘数法 (Lagrange_Multiplier)

**对应文档**: `4.Lagrange_Multiplier.md`

**脚本**: `code/Lagrange_Multiplier/lagrange_multiplier_visualization.py`

**生成文件** (7个):
```
Lagrange_Multiplier/Lagrange_Multiplier/
├── 1_circle_linear.html               - 圆约束+线性目标
├── 2_ellipse_quadratic.html           - 椭圆约束+二次目标
├── 3_gradient_parallel.html           - 梯度平行可视化
├── 4_3d_constraint.html               - 3D约束表面
├── 5_lambda_geometry.html             - λ的几何意义
├── 6_dual_problem.html                - 对偶问题
└── 7_dashboard.html                   - 综合仪表板
```

**核心公式**:
```
L(x,y,λ) = f(x,y) + λ·g(x,y)
∇f = λ·∇g  (梯度平行条件)
```

**用途**: 理解约束优化问题和对偶理论

---

### 5. L1/L2正则化 (L1_L2_Regularization)

**对应文档**: `5.L1&L2.md`

**脚本**: `code/L1_L2_Regularization/l1_l2_regularization_visualization.py`

**生成文件** (6个):
```
L1_L2_Regularization/L1_L2_Regularization/
├── 1_l1_sparsity.html                 - L1稀疏性演示
├── 2_l2_weight_decay.html             - L2权重衰减
├── 3_regularization_path.html         - 正则化路径
├── 4_feature_selection.html           - 特征选择
├── 5_comparison.html                  - L1 vs L2对比
└── 6_dashboard.html                   - 综合仪表板
```

**核心公式**:
```
L1: λ·Σ|w_i|     (稀疏性)
L2: λ·Σw_i²      (权重衰减)
```

**用途**: 防止过拟合，特征选择

---

### 6. 支持向量机 (SVM)

**对应文档**: `6.SVM.md`

**脚本**: `code/SVM/svm_visualization.py`

**生成文件** (6个):
```
SVM/SVM/
├── 1_maximum_margin.html              - 最大间隔分类器
├── 2_soft_margin.html                 - 软间隔SVM
├── 3_kernel_trick.html                - 核函数技巧
├── 4_nonlinear_boundary.html          - 非线性决策边界
├── 5_support_vectors.html             - 支持向量可视化
└── 6_dashboard.html                   - 综合仪表板
```

**核心公式**:
```
min ½||w||² + C·Σξ_i
s.t. y_i(w·x_i + b) ≥ 1 - ξ_i
```

**用途**: 理解最大间隔原则和核方法

---

### 7. VC维理论 (VCdime) ⭐ 最完整

**对应文档**: `7.VCdime.md`

**脚本**: `code/VCdime.py`

**生成文件** (12个):
```
VCdime/
├── 1_vc_bound.html                    - VC维上界公式
├── 2_risk_decomposition.html          - 真实风险分解
├── 3_svm_margin_vc.html               - SVM间隔与VC维
├── 4_regularization_tradeoff.html     - 正则化权衡
├── 5_overfitting_underfitting.html    - 过拟合/欠拟合
├── 6_svm_c_parameter.html             - SVM的C参数
├── 7_dashboard.html                   - 综合仪表板
├── 8_shattering_demo.html             - ⭐ Shattering演示
├── 9_xor_problem.html                 - ⭐ XOR问题
├── 10_growth_function.html            - ⭐ 增长函数
├── 11_model_vc_comparison.html        - ⭐ 不同模型VC维对比
└── 12_pac_framework.html              - ⭐ PAC学习框架
```

**核心公式**:
```
R(f) ≤ R_emp(f) + Φ(h/n)
Φ(h/n) ≈ √(h/n)
VC维 = max{n : m_H(n) = 2^n}
```

**用途**: 理解模型复杂度和泛化能力

---

### 8. 分类模型优化 (Classification_Optimization_Logic)

**对应文档**: `8.Classification_Optimization.md`

**脚本**: `code/Classification_Optimization_Logic/classification_optimization_logic_visualization.py`

**生成文件** (6个):
```
Classification_Optimization_Logic/Classification_Optimization_Logic/
├── 1_logistic_regression.html         - 逻辑回归
├── 2_decision_boundary.html           - 决策边界
├── 3_probability_output.html          - 概率输出
├── 4_gradient_descent_path.html       - 梯度下降路径
├── 5_multiclass.html                  - 多分类
└── 6_dashboard.html                   - 综合仪表板
```

**用途**: 理解分类模型的优化过程

---

### 9. 遗传算法 (GeneticAlgorithm)

**对应文档**: `GeneticAlgorithm.md`

**脚本**: `code/GeneticAlgorithm.py`

**生成文件** (7个):
```
GeneticAlgorithm/
├── 1_evolution_process.html           - 进化过程动画
├── 2_convergence_curve.html           - 适应度收敛曲线
├── 3_roulette_wheel_selection.html    - 轮盘赌选择
├── 4_crossover_mutation.html          - 交叉与变异
├── 5_parameter_impact.html            - 参数影响分析
├── 6_schema_theorem.html              - Schema定理
└── 7_dashboard.html                   - 综合仪表板
```

**核心公式**:
```
选择: P(i) = f(i) / Σf(j)
交叉: c = α·p₁ + (1-α)·p₂
变异: c' = c + N(0, σ²)
```

**用途**: 理解进化算法和全局优化

---

### 10. 交互式梯度下降工具 (interactive_gradient_descent) ⭐ 高度交互

**脚本**: `code/interactive_gradient_descent.py`

**生成文件** (7个):
```
interactive_gradient_descent/
├── sphere_简单.html                    - Sphere函数优化
├── rosenbrock_困难.html                - Rosenbrock函数优化
├── rastrigin_多峰.html                 - Rastrigin函数优化
├── beale.html                         - Beale函数优化
├── learning_rate_comparison.html      - 学习率对比
├── momentum_comparison.html           - 动量系数对比
└── 3d_visualization.html              - 3D可视化
```

**特点**:
- ✅ 点击图例显示/隐藏算法
- ✅ 悬停查看精确坐标
- ✅ 对比不同参数设置
- ✅ 3D旋转查看优化路径
- ✅ 4种测试函数，4种优化算法

**用途**: 实验和理解梯度下降算法

---

## 🎨 可视化特点

### 通用特性
- ✅ **数学公式**: 每个图表都包含LaTeX公式（MathJax渲染）
- ✅ **交互式**: 缩放、平移、悬停查看详细信息
- ✅ **动画**: 播放/暂停按钮控制
- ✅ **3D视图**: 部分可视化支持3D旋转
- ✅ **图例控制**: 点击显示/隐藏特定元素

### 按钮位置
- 所有播放/暂停按钮位于左下角，不遮挡内容
- 使用 `▶ 播放` 和 `⏸ 暂停` 图标

### 颜色方案
- 统一的颜色编码
- 清晰的视觉对比
- 支持色盲友好模式

---

## 📖 使用场景

### 教学
- 课堂演示机器学习概念
- 学生自主学习和实验
- 理解复杂数学公式的直观含义

### 研究
- 对比不同算法的性能
- 分析参数对结果的影响
- 验证理论推导

### 工程实践
- 选择合适的优化器
- 调整超参数
- 理解模型行为

---

## 🛠️ 技术栈

- **Python 3.x**
- **Plotly** - 交互式可视化库
- **NumPy** - 数值计算
- **MathJax** - 数学公式渲染（CDN）

---

## 📝 文件命名规范

### Python脚本
- 与markdown文档对应（小写，下划线）
- 例如: `lossfunction.py`, `VCdime.py`

### HTML文件
- 按序号命名: `1_xxx.html`, `2_xxx.html`
- 描述性名称，易于理解
- 最后一个通常是 `dashboard.html`

### 目录结构
- 脚本和HTML在同一目录或子目录
- 格式: `code/主题名/主题名/*.html` 或 `code/主题名/*.html`

---

## 🎯 推荐学习路径

### 入门路径
1. **损失函数** → 理解训练目标
2. **梯度下降** → 理解优化基础
3. **正则化** → 防止过拟合

### 进阶路径
1. **优化器对比** → 选择合适算法
2. **SVM** → 经典机器学习
3. **VC维理论** → 理论基础

### 高级路径
1. **拉格朗日乘数法** → 约束优化
2. **遗传算法** → 全局优化
3. **交互式实验** → 深入理解

---

## 💡 常见问题

### Q1: 如何修改参数？
**A**: 编辑对应的Python脚本，修改参数（如学习率、迭代次数），然后重新运行。

### Q2: 可视化显示不正常？
**A**: 
- 确保浏览器支持JavaScript
- 检查是否启用了MathJax CDN
- 尝试刷新页面

### Q3: 如何导出图像？
**A**: Plotly图表右上角有相机图标，可以导出为PNG。

### Q4: 能否批量查看所有HTML？
**A**:
```bash
# macOS - 打开前20个
find code -name "*.html" | head -20 | xargs open

# Linux
find code -name "*.html" | head -20 | xargs xdg-open
```

### Q5: 如何添加新的可视化？
**A**: 
1. 参考现有脚本的结构
2. 使用 `add_formula_annotation()` 添加公式
3. 设置 `updatemenus` 添加动画控制
4. 使用 `include_mathjax='cdn'` 保存HTML

---

## 📊 统计信息

- **Markdown文档**: 9个
- **Python脚本**: 10个
- **HTML文件**: 73个
- **代码行数**: ~6000+ 行
- **公式数量**: 120+ 个
- **动画数量**: 25+ 个

---

## 🌟 项目亮点

1. **完整性**: 覆盖所有9个理论文档
2. **交互性**: 完全可交互的可视化
3. **教育性**: 适合教学和学习
4. **专业性**: 包含数学公式和理论基础
5. **可扩展**: 易于添加新的可视化

---

## 🎉 开始使用

```bash
# 1. 查看总览
open code/README.md

# 2. 选择感兴趣的主题
cd code/lossfunction

# 3. 在浏览器中打开HTML
open lossfunction/1_least_squares.html

# 4. 实验交互功能
# - 点击图例
# - 缩放平移
# - 悬停查看
# - 播放动画

# 5. 如需修改，编辑Python脚本
vim lossfunction.py

# 6. 重新生成
python lossfunction.py
```

---
