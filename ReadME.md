# 机器学习深度学习知识库

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.x-red.svg)](https://pytorch.org/)

> 🚀 **综合性机器学习知识库，包含100+交互式可视化、实践代码示例和数学基础，涵盖从基础理论到前沿研究的完整学习路径**

---

## 📚 目录

- [🎯 快速开始](#-快速开始)
- [📖 学习路径](#-学习路径)
- [🏗️ 项目结构](#️-项目结构)
- [🛠️ 技术与工具](#️-技术与工具)
- [📊 核心特性](#-核心特性)
- [🚀 入门指南](#-入门指南)
- [📝 贡献指南](#-贡献指南)

---

## 🎯 快速开始

**机器学习新手？** 从我们的[学习路径](./LEARNING_PATH.md)开始 →

**寻找特定主题？** 按类别浏览：
- [📐 数学基础](#数学基础)
- [🧠 神经网络基础](#神经网络基础) 
- [🔥 框架实现](#框架实现)
- [🎨 交互式可视化](#交互式可视化)
- [🧬 大模型训练](#大模型训练与架构)
- [🔍 高级主题](#高级主题)

---

## 📖 学习路径

我们将内容构建为三个渐进式学习路径：

### 🌱 **初级路径** (0-6个月)
```
数学基础 → 基础神经网络 → 第一个深度学习模型
```
**关键主题:**
- 机器学习数学基础
- 基本神经网络概念  
- 第一个CNN/RNN
- TensorFlow/PyTorch基础

### 🌿 **中级路径** (6-18个月)  
```
高级架构 → 优化技术 → 实际应用
```
**关键主题:**
- 注意力机制与Transformer
- 高级优化技术
- 计算机视觉与自然语言处理
- 生产环境部署

### 🌳 **高级路径** (18+个月)
```
研究主题 → 专业应用 → 前沿技术
```
**关键主题:**
- GANs与扩散模型
- 图神经网络
- 强化学习
- 研究论文实现

📖 **详细学习路径**: [LEARNING_PATH.md](./LEARNING_PATH.md)

---

## 🏗️ 项目结构

```
ml/
├── 📁 other/                    # 高级概念与数学基础
│   ├── 📁 math/                # 数学基础 (26个核心文档 + 4个附录)
│   │   ├── 📁 code/            # 交互式可视化 (13个主题目录)
│   │   │   ├── 📄 interactive_gradient_descent.py (9个HTML可视化)
│   │   │   ├── 📄 grand_optimizer.py
│   │   │   ├── 📄 ml_curves_visualization/ (交互式仪表板)
│   │   │   ├── 📄 SVM/ (SVM可视化)
│   │   │   ├── 📄 lossfunction/ (损失函数可视化)
│   │   │   ├── 📄 L1_L2_Regularization/ (正则化可视化)
│   │   │   ├── 📄 VCdime.py (VC维可视化)
│   │   │   ├── 📄 GeneticAlgorithm.py (遗传算法)
│   │   │   └── 📄 ... (更多可视化脚本)
│   │   ├── 📄 0.1-0.4 基础数学 (微积分、矩阵、概率、几何)
│   │   ├── 📄 1-10 核心概念 (卷积、损失、优化器、正则化等)
│   │   └── 📄 11-26 高级主题 (扩散模型、GCN、因果关系等)
│   ├── 📁 Self-Attention/      # 自注意力机制详解 (7个文档)
│   │   ├── 📄 1.math.md - 6.CausalMask.md
│   │   ├── 📁 code/            # 注意力机制实现
│   │   └── 📁 images/          # 注意力机制图表
│   ├── 📁 images/              # 图表与可视化 (8个图片)
│   ├── 📄 Mamba_vs_Transformer_Architecture_Guide.md
│   ├── 📄 Neural_Network_Dimensions_and_Parameters.md
│   ├── 📄 RAG_en.md
│   ├── 📄 Train_models.md
│   ├── 📄 Computation_graphs.md
│   └── 📄 ga.py                # 遗传算法实现
├── 📁 pytorch/                  # PyTorch实现与教程
│   ├── 📁 basic/               # 基础PyTorch操作 (tensor、audio)
│   ├── 📁 computer_vision/     # 计算机视觉实现
│   ├── 📁 nlp/                 # 自然语言处理实现 (中英文)
│   ├── 📁 transformers/        # Transformer架构
│   ├── 📁 rl/                  # 强化学习
│   ├── 📁 probabilistic/       # 概率模型
│   ├── 📁 audio/               # 音频处理
│   ├── 📁 notes/               # PyTorch笔记
│   └── 📄 requirements.txt     # 依赖包列表
├── 📁 tensorflow/               # TensorFlow实现与教程
│   ├── 📁 linear/              # 线性模型 (回归、分类)
│   ├── 📁 cnn/                 # CNN实现 (5个文件 + 测试图像)
│   ├── 📁 RNN/                 # RNN实现 (文本生成、序列模型)
│   ├── 📁 NLP/                 # NLP实现 (4个文件 + 数据集)
│   ├── 📁 GANs/                # GAN实现
│   ├── 📁 GNN/                 # 图神经网络实现
│   ├── 📁 probabilistic_tf/    # 概率机器学习 (贝叶斯、HMM、高斯过程)
│   ├── 📁 notes/               # TensorFlow笔记和图片
│   ├── 📄 001_basic_tensor.py  # 基础张量操作
│   ├── 📄 dpl_tensorflow.py    # 深度学习项目
│   └── 📄 requirement.txt      # 依赖包列表
├── 📁 images/                   # 视觉辅助与图表 (8个图片)
├── 📄 neural-network.md         # 神经网络基础指南
├── 📄 LEARNING_PATH.md          # 结构化学习路径
└── 📄 DOCUMENTATION_CODE_INDEX.md  # 文档代码索引
```

---

## 🛠️ 技术与工具

### **核心框架**
- ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg) [TensorFlow 2.x](https://tensorflow.org/)
- ![PyTorch](https://img.shields.io/badge/PyTorch-1.x-red.svg) [PyTorch 1.x](https://pytorch.org/)

### **可视化库**
- ![Plotly](https://img.shields.io/badge/Plotly-Interactive-blue.svg) [Plotly](https://plotly.com/)
- ![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-green.svg) [Matplotlib](https://matplotlib.org/)
- ![NumPy](https://img.shields.io/badge/NumPy-Computational-yellow.svg) [NumPy](https://numpy.org/)

### **开发工具**
- ![Python](https://img.shields.io/badge/python-3.8+-blue.svg) [Python 3.8+](https://www.python.org/downloads/)
- ![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg) [Jupyter Notebook](https://jupyter.org/)

---

## 📊 核心特性

### 🎨 **交互式可视化**
- **15+ 交互式可视化**: 梯度下降(9个HTML)、优化器比较、学习曲线仪表板
- **实时参数调整**: 滑块和控制器的动态学习，支持多种优化算法对比
- **数学概念可视化**: SVM分类边界、VC维演示、正则化效果、卷积操作
- **高级数学工具**: 拉格朗日乘数、遗传算法、矩阵运算、分类优化逻辑

### 📐 **数学基础**
- **全面理论**: 26个核心数学文档 + 4个附录，涵盖从基础到AGI数学的完整体系
- **逐步推导**: 详细的数学证明和解释，从微积分到信息几何
- **可视化学习**: 13个交互式可视化主题，包含梯度下降、优化器、SVM等
- **实际应用**: 理论与实际实现相结合，每个概念都有对应的代码示例

### 🔧 **实践实现**
- **100+ 代码示例**: 所有主要概念的工作实现
- **框架覆盖**: 同时支持TensorFlow和PyTorch实现
- **最佳实践**: 行业标准编码模式和优化
- **生产就绪**: 适用于实际应用的代码

### 🧠 **高级主题**
- **前沿架构**: Transformer、GANs、扩散模型
- **研究论文**: 最新突破性论文的实现
- **专业应用**: 计算机视觉、自然语言处理、强化学习
- **优化技术**: 高级优化器和训练策略

### 🧬 **大模型训练与架构**
- **完整训练流程**: [大语言模型训练指南](./other/Train_models.md)
  - 预训练、监督微调、人类反馈强化学习
  - 数据处理、模型对齐、部署优化
- **架构对比分析**: [Mamba vs Transformer](./other/Mamba_vs_Transformer_Architecture_Guide.md)
  - 注意力机制 vs 状态空间模型
  - O(L²) vs O(L) 复杂度对比
  - 适用场景与性能分析
- **维度与参数**: [神经网络维度与参数详解](./other/Neural_Network_Dimensions_and_Parameters.md)
  - 全连接层、卷积层、注意力层的维度计算
  - 参数数量与模型容量的关系
  - 内存与计算复杂度分析

### 🔍 **高级应用技术**
- **检索增强生成**: [RAG技术详解](./other/RAG_en.md)
  - 检索与生成的结合
  - 向量数据库与嵌入技术
  - 知识库构建与优化
- **自注意力机制**: [Self-Attention详解](./other/Self-Attention/)
  - 数学原理与代码实现
  - 多头注意力机制
  - Transformer架构组件
- **计算图**: [计算图理论](./other/Computation_graphs.md)
  - 自动微分与反向传播
  - 计算图优化与并行化

---

## 🚀 入门指南

### **先决条件**
- 基础Python编程知识
- 微积分和线性代数基础理解
- 机器学习概念熟悉度（推荐）

### **快速浏览**
1. **从数学开始**: [数学基础](./other/math/)
2. **选择框架**: [TensorFlow](./tensorflow/) 或 [PyTorch](./pytorch/)
3. **探索可视化**: [交互式演示](./other/math/code/)
4. **遵循学习路径**: [结构化课程](./LEARNING_PATH.md)

### **大模型学习路径**
1. **基础架构**: [自注意力机制](./other/Self-Attention/)
2. **训练流程**: [大模型训练指南](./other/Train_models.md)
3. **架构对比**: [Mamba vs Transformer](./other/Mamba_vs_Transformer_Architecture_Guide.md)
4. **实际应用**: [RAG技术](./other/RAG_en.md)

---

## 📝 贡献指南

我们欢迎贡献！详情请查看[贡献指南](./CONTRIBUTING.md)。

### **如何贡献**
1. Fork 本仓库
2. 创建功能分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 打开 Pull Request

### **贡献领域**
- 📝 **文档**: 改进解释和添加示例
- 🎨 **可视化**: 创建新的交互式演示
- 🔧 **代码**: 添加实现或优化现有代码
- 🐛 **错误报告**: 帮助我们识别和修复问题
- 💡 **想法**: 建议新主题或改进

---

## 📈 项目统计

- 📚 **30个核心数学文档**: 涵盖从基础到AGI数学的完整理论体系(26+4附录)
- 📖 **40+个Markdown文档**: 包括数学理论、框架教程、高级主题
- 🧬 **7个大模型专题**: 训练、架构对比、RAG、Self-Attention等
- 💻 **15+个Python代码示例**: 实践实现和框架教程
- 🎨 **13个交互式可视化主题**: 包含梯度下降、优化器、SVM、VC维等
- 🌍 **双语支持**: 中英文文档
- 🔄 **持续更新**: 定期添加新内容和前沿技术

---

## 📄 许可证

本项目采用MIT许可证 - 详情请参阅[LICENSE](LICENSE)文件。

---

## 🙏 致谢

- 感谢所有帮助构建这个知识库的贡献者
- 受开源深度学习社区启发
- 特别感谢TensorFlow和PyTorch团队提供的优秀框架

---

<div align="center">

**⭐ 如果这个仓库帮助您学习深度学习，请给它一个星标！⭐**

</div>