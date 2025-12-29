# Transformer vs PHOTON 深度解析

核心命题：
随着上下文长度$L$迈向百万级，Transformer 的 $O(L^2)$算力消耗与 $O(L)$ 显存占用已撞上`内存墙`。
PHOTON (Parallel Hierarchical Operation for Top-down Networks, Fujitsu 2025.12) 的出现，标志着语言建模从`平坦序列检索`向`层次化特征重建`的范式转移

---

## 一、数学底层：全局点积寻址 vs 层次化潜在空间重建

### 1. Transformer: 基于各向同性注意力（Isotropic Attention）的水平扫描

Transformer 的核心是水平全局扫描。它假设所有历史 token 对当前预测都有潜在贡献，是一种非参数化的内容寻址机制。

$$ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V $$

* 数学局限：

  * 全局依赖：Softmax 强制对$0$到$t-1$的所有历史时刻分配概率密度，导致$L×L$的计算矩阵。

  * 信息冗余：Softmax 强制对所有历史 Token 分配权重，但在长文本中，信息密度并非均匀分布。

  * 计算图退化：在推理（Decode）阶段，算子退化为向量-矩阵乘法（GEMV），极度依赖内存带宽而非算力。

### 2. PHOTON: 多分辨率潜在状态的自顶向下重建

PHOTON 引入了信息率失真理论(Rate-Distortion Theory)的影子，通过分层压缩实现垂直访问。

* 自底向上与自顶向下架构：

  * 将序列$x$ 逐步压缩为多分辨率潜在状态$z$ :

      $$ Z_t=\{z^1_t,z^2_t,...,z^k_t\} $$

  * $z^1$ : 细粒度 Token 级表示 (高频信息)

  * $z^k$ : 粗粒度语义摘要 (低频全局信息)，$|z^k| \ll |x|$

  * 自顶向下解码 (Top-down Decoder)：

    * 利用上下文转换器（Context Converter）$C$ 从粗粒度状态重建细粒度特征，并限制在局部 Chunk 内进行自回归：

      $$ \hat{x}_{local}=LocalAttention(C(z_{high}),Chunk_{kv}) $$

  * 核心差异：

    Transformer 是水平的（时间轴$t$延伸）；PHOTON 是垂直的（特征层级$h$延伸）。推理时，PHOTON 主要更新粗粒度变量，细粒度 Token 在 Chunk 内并行生成。

* 核心创新： 垂直多分辨率上下文访问 (Vertical Multi-resolution Context Access) 这是 PHOTON 打破 $O(L)$ 显存墙的关键机制，它将 Transformer 的`水平全局扫描`范式转变为`垂直层次访问`。

  * 水平扫描 (Transformer)： 生成第$t$个 token 需访问$t-1$之前的所有 KV：
    $$
    Memory Access \propto \sum_{i=0}^{t-1} Size(KV_i) \approx O(t \cdot d) \text{线性增长}
    $$
  
  * 垂直访问 (PHOTON)： 生成过程被重构为利用当前层次状态进行局部重建。
  $$P(x_{t:t+C}|Z_t) \propto LocalDec(P_\downarrow(Z_t(K),…,Z_t(1)),KV_{chunk}) $$
    * 其中 $p_\downarrow$ 代表自顶向下的投影操作，$C$ 是 Chunk 大小。
    * 推理动力学：

      * 层次化 Prefill：一次性构建多级状态$Z$。

      * 生成时 (Coarse Update)：主要更新高层的粗粒度潜在变量$Z^{k}$ 由于高层状态变化率低，更新频率远低于 token 生成频率。

      * 并行 Chunk 解码：利用稳定的高层状态，在多个 Chunk 内并行解码细粒度 Token。

  * 内存流量结论：

    * $$\text{Memory Access} \propto O(∣\text{Z}∣+∣\text{Local Chunk}∣) \propto O(1)_{w.r.t L}$$

    > PS: $O(1)$ 意为内存访问量不随序列长度 $L$ 增加而增加，仅与层级深度 $K$ 相关（常数）。

通过将检索维度从时间轴（Time axis,$L$）转移到深度轴（Depth axis,$K$），PHOTON 实现了近似常数级的 Decode 内存流量，大幅降低了 KV cache 的更新频率和读写带宽需求

---

## 二、 工程实现与训练动力学 (Engineering Dynamics)

### 训练范式对比

|维度 |Transformer |PHOTON
|---|---|---|
|算子核心 |GEMM (矩阵乘) |Hierarchical Scan / Reduction
|并行策略 |序列并行 (Sequence Parallelism) |1. 压缩阶段：并行前缀扫描 (Parallel Scan)<br>2. 重建阶段：Chunk 并行解码
|硬件亲和性 |极高 (Tensor Cores 满载) |中高 (需针对层次操作优化 Kernel)

### 核心工程挑战

* Transformer：生态极其成熟，FlashAttention-3、cuDNN 均有原生支持。

* PHOTON：需要编写自定义 CUDA/Triton Kernel 来高效处理“自底向上”的聚合操作，否则 Python 循环带来的 overhead 会抵消算法优势。

---

## 三、推理效率：KV Cache 增长 vs 近常数内存流量

这是两者差异最剧烈的地方，PHOTON 的创新在此体现得淋漓尽致。

### 痛点：Transformer 的“显存墙”

* 显存占用：$1M$tokens,$d=4096$ , float16 下 KV Cache 约为 16GB (仅单层单 Batch)。多卡推理时通信开销巨大。

* 带宽瓶颈：每生成 1 个 Token，需将几十 GB 的 KV Cache 从 HBM 搬运到 SRAM，导致计算单元空转

### PHOTON 推理：层次化状态更新

实验在 NVIDIA A100 GPU 上进行，评估了 prefill-heavy（长 prompt + 短生成）和 decode-heavy（短 prompt + 长生成）两种典型场景。

* 实测数据 (600M 模型)：

  * Throughput-per-Memory (TPM)：提升高达 416x。

  * 显存流量：Decode 阶段减少$10^3$倍。

  * 长文表现：在 Prefill-heavy (长 Prompt) 场景下，首 Token 延迟 (TTFT) 略增，但后续生成速度呈数量级提升

PHOTON 的优势随上下文长度和生成长度进一步拉大，在百万级 token 场景下，传统优化（如量化、PagedAttention）已接近物理极限，而 PHOTON 提供了结构级的突破。

关键结论：在内存受限（Memory-Bound）的边缘设备或高并发服务器上，PHOTON 提供了物理层面无法通过 Transformer 优化（如 quantization）达到的吞吐量上限

---

## 代码实现对比

### Transformer (Self-Attention)

依赖全局矩阵运算，随着 L 增长，计算量呈平方级，显存呈线性级。

```python
# PyTorch 原生实现（简化）
class TransformerBlock(nn.Module):
    def forward(self, x):
        # x shape: [Batch, Length, Dim]
        # 1. 投影
        q, k, v = self.proj_qkv(x).chunk(3, dim=-1)
        
        # 2. 全局注意力 (瓶颈所在)
        # 必须计算当前 Q 与所有历史 K 的点积
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = torch.softmax(scores, dim=-1)
        
        # 3. 聚合
        out = torch.matmul(attn, v)
        return self.ffn(out)

```

### 2. PHOTON (Hierarchical Autoregressive)

依赖多级状态更新，Attention 被限制在局部 Chunk，全局信息由 Latents 传递。

```python
class PHOTONBlock(nn.Module):
    def forward(self, x, previous_hierarchy_states):
        # 1. 自底向上压缩 (Bottom-up)
        # 将输入逐步聚合，更新粗粒度状态，不依赖全历史
        coarse_latents = self.bottom_up_encoder(x, previous_hierarchy_states)
        
        # 2. 自顶向下重建 (Top-down)
        # 从最粗层摘要开始，逐层还原上下文信息
        fine_recon = self.context_converter(coarse_latents[-1])
        
        # 3. 局部自回归 (Local Autoregressive)
        # 注意力严格限制在当前 Chunk (如 64 tokens) 内，计算量恒定
        chunk_size = 64
        local_out = self.local_autoregressive_decoder(fine_recon, window=chunk_size)
        
        # 4. 状态更新
        # 仅传递更新后的压缩状态给下一步，而非所有 KV
        new_states = self.update_hierarchy(coarse_latents)
        
        return local_out, new_states
```

## 五、最终决策指南：如何选？

### 必须选择 PHOTON 的场景

* 超长文本生成 (100k - 1M+)：写小说、生成长代码库、法律文书分析。Transformer 在此时会 OOM 或慢如蜗牛。
端侧/边缘计算：在手机、车载芯片等显存带宽受限（<100GB/s）的设备上运行大模型。
高并发推理服务：需要极高的 Token/s/$ 经济效益。

### 坚持使用 Transformer 的场景

* `大海捞针` (Needle In A Haystack)：

  * Transformer：由于保留了所有 KV，理论上能召回任意位置的细节（只要注意力头足够强）。

  * PHOTON：本质是有损压缩。如果“针”的信息极其细微且在粗粒度压缩中丢失，可能无法召回。对于司法取证、精准数据提取任务，Transformer 仍是由于冗余而带来的安全选择。

* 工程即时性：需要今天就上线，且依赖 vLLM, TensorRT-LLM 等现成生态。

### 总结对比表格

|维度|Transformer (The Standard)|PHOTON (The Challenger)|
|---|---|---|
|核心范式|检索 (Retrieval)|重建 (Reconstruction)|
|推理复杂度|O(L) 线性|O(1) 常数|
|显存瓶颈|KV Cache 容量与带宽|模型权重加载|
|NIAH (大海捞针)| 完美 (无损)|  优秀 (有损压缩风险)|
|吞吐量 (TPM)| 基准线 (1x)| 416x (600M 模型)|
|部署难度| 🟢 低 (开箱即用)| 🔴 高 (需定制 Kernel)|
| 当前状态 | 🔴 工业标准（生态完备） | 🟡 论文刚发布（无开源代码，需自研 Kernel） |
