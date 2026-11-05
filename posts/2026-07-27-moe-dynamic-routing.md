---
title: MoE 动态路由与稀疏计算
date: 2026-07-27
tags: [LLM, AI-Infra, MoE, 分布式推理]
categories: [AI Infrastructure]
katex: true
---

[DeepSeek-V3](https://arxiv.org/abs/2412.19437) 包含约 671B 参数，每个 token 激活约 37B。激活参数少，降低了单 token 的部分计算量，但全部专家权重仍需安排存储。专家分布在多张 GPU 时，输入激活还要按路由结果跨卡传输。

MoE 因此需要分别计算权重容量、专家计算量和通信成本。仅看激活参数约占总参数的 5.5%，无法预测显存、吞吐或延迟；激活参数相近的 Dense 与 MoE 模型，也不一定有相近的执行时间。

## 一、参数与计算解耦

Transformer Block 的主要参数集中在 Attention 和 FFN。Dense 模型让每个 Token 经过同一组 FFN 权重，稀疏 MoE 则将这组 FFN 扩展成专家池，并在入口增加 Router。Router 根据当前 Token 的隐藏状态计算专家分数，从中选择 Top-K 专家参与前向。

设输入为 $x$，专家数量为 $N$，Router 为每个专家产生分数 $s_i(x)$。选出集合 $T(x)$ 后，MoE 层可以写成：

$$
y = \sum_{i \in T(x)} g_i(x)E_i(x), \qquad |T(x)| = K
$$

式中各项对应 MoE 的三个基本组件：

- $E_i$：第 $i$ 个专家 FFN；
- $g_i(x)$：被选中专家的组合权重；
- $T(x)$：Router 为输入 $x$ 选择的 Top-K 专家集合。

单 Token 的计算量主要由 $K$ 和专家规模决定，总参数量则会随专家总数 $N$ 增长。当 $K$ 远小于 $N$ 时，新增专家可以扩大参数空间，而无需进入每个 Token 的计算路径。

```mermaid
flowchart LR
    X1["Token 隐藏状态"] --> D["Dense FFN"]
    D --> Y1["输出"]

    X2["Token 隐藏状态"] --> R["Router"]
    R --> E1["Expert 1"]
    R --> E2["Expert 2"]
    R -. 未激活 .-> EN["其他 Experts"]
    E1 --> C["加权合并"]
    E2 --> C
    C --> Y2["输出"]

    classDef input fill:#fff8e1,stroke:#ff9800,color:#333;
    classDef process fill:#f0f4ff,stroke:#5b8def,color:#333;
    classDef inactive fill:#f5f5f5,stroke:#9e9e9e,color:#333;
    classDef output fill:#e8f5e9,stroke:#4caf50,color:#333;
    class X1,X2 input;
    class D,R,E1,E2,C process;
    class EN inactive;
    class Y1,Y2 output;
```

继续以 DeepSeek-V3 为例，它的 MoE 层包含 256 个路由专家，每个 token 选择 8 个，另有一个共享专家。总参数与激活参数还包含 Attention 等公共部分，因此不能把 8/256 直接当作整模型激活比例。上式描述路由专家分支；若存在共享专家，还需要把共享分支输出加进去。

专家池减少了每个 token 必经的 FFN 参数，接下来还要保证路由既能学习有效分工，也能形成可高效执行的负载。

## 二、路由的双重目标

Router 通常是一个从隐藏维度投影到专家数量的线性层。它先为一批 Token 生成路由分数，Top-K 再据此决定每个 Token 进入哪些专家。选择完成后，系统把原本按序列排列的 Token 按专家重新分组，形成各自的输入批次；FFN 计算结束后，再将结果还原到原位置并加权合并。

```mermaid
flowchart LR
    A["Token Batch"] --> B["Router Scores"]
    B --> C["Top-K 选择"]
    C --> D["Dispatch 重排"]
    D --> E1["Expert 1 Batch"]
    D --> E2["Expert 2 Batch"]
    D --> E3["Expert N Batch"]
    E1 --> F["Combine 还原"]
    E2 --> F
    E3 --> F
    F --> G["MoE 输出"]

    classDef input fill:#fff8e1,stroke:#ff9800,color:#333;
    classDef process fill:#f0f4ff,stroke:#5b8def,color:#333;
    classDef data fill:#f5f5f5,stroke:#9e9e9e,color:#333;
    classDef key fill:#fce4ec,stroke:#ef5350,color:#333;
    classDef output fill:#e8f5e9,stroke:#4caf50,color:#333;
    class A input;
    class B,C,F process;
    class D key;
    class E1,E2,E3 data;
    class G output;
```

分发路径由路由分数决定，也因此同时承载两项目标：

- **模型目标**：将 Token 交给亲和度较高的专家，让不同专家逐渐形成分工；
- **系统目标**：让各专家收到相近数量的 Token，避免计算和通信集中在少数设备上。

若 Router 只按亲和度独立选择 Top-K，这两个目标很容易偏离。

假设某层有 8 个专家，4096 个 Token 各选择 2 个专家。完全均衡时，每个专家得到 1024 个 Token；实际路由却可能让一个专家收到 1800 个，另一个只收到 300 个。热门专家的输入矩阵更大，所在设备需要更长时间；冷门专家获得的训练样本更少，参数更新也更有限。整层计算必须等待所有专家完成，最慢的专家最终进入关键路径。

一种实现方式为专家设置容量上限；也有 dropless 等不按固定容量丢弃 token 的实现。采用容量限制时，可以写成：

$$
C = \left\lceil \frac{T \times K}{N} \times \text{CapacityFactor} \right\rceil
$$

其中，$T$ 是本批 Token 数，$K$ 是单 Token 激活的专家数，$N$ 是专家总数。`CapacityFactor` 大于 1，意味着系统为负载波动预留了缓冲：

- 容量不足时，超出的 Token 只能被丢弃、改道或进入额外路径；
- 预留过多时，中间 Buffer 随之扩大，设备需要长期承担峰值空间。

容量限制处理的是一次前向能否完成，训练过程还会放大长期失衡。某个专家若在早期偶然得到更多 Token，便会获得更多梯度更新，随后更容易继续被 Router 选中；长期缺少 Token 的专家则逐渐失去参与训练的机会。路由坍缩会让名义上的大专家池退化成少数活跃专家，新增参数也就无法转化为有效容量。

专家分工不一定对应人工划分的领域。路由以层和 token 为粒度，同一句子的不同 token 可以进入不同专家，同一位置在不同层也可以改变路径。若要命名“数学专家”或“代码专家”，还需要路由统计和干预实验支持。

Router 因而同时参与表示学习和运行时调度：亲和度影响模型质量，Token 分布影响执行效率。MoE 需要主动约束两者的偏离，但均衡控制本身也会介入专家的学习过程。

## 三、均衡与专门化

最直接的控制方式，是在语言模型损失之外加入负载均衡辅助损失。以 Token Choice 路由为例，可以统计专家被实际选中的比例 $f_i$，以及 Router 分配给该专家的平均概率 $P_i$，再通过两者的乘积惩罚负载集中：

$$
L_{balance} = \alpha N \sum_{i=1}^{N} f_iP_i
$$

当少数专家同时获得较高概率和较多 Token 时，辅助损失会推动 Router 分散流量。系数 $\alpha$ 太小，约束不足；系数太大，Router 又可能为了接近平均分配，放弃更有价值的专家选择。模型质量和设备利用率由此进入同一个训练目标，系数也需要随模型规模、路由方式和数据分布调整。

DeepSeekMoE 同时调整了专家结构和均衡方式：

- **细粒度专家**：拆分单个专家，在相近的激活计算量下组合更多路由专家；
- **共享专家**：让所有 Token 都经过一部分公共参数，减轻路由专家重复学习通用知识的压力。

在此基础上，DeepSeek-V3 为每个路由专家维护一个偏置项。Top-K 根据“亲和度加偏置”的结果做选择：专家过载时降低对应偏置，负载不足时提高偏置。专家输出的组合权重仍由原始亲和度计算，因此偏置主要影响谁能进入 Top-K。系统可以据此修正实际负载，同时减少均衡信号对主损失梯度的直接干预。

“Auxiliary-Loss-Free”描述的是主要负载均衡策略，并不意味着训练目标完全移除了均衡辅助项。DeepSeek-V3 仍保留较弱的序列级辅助损失，用来防止单条序列内部出现极端失衡。工程上也无需追求每个局部批次绝对平均。只要批次级负载足够稳定、设备没有持续热点，专家仍能形成有效分工，均衡控制就达到了目的。

## 四、稀疏计算的通信代价

均衡策略让各专家获得了相对稳定的输入，但专家池继续扩大后，单张 GPU 很快无法容纳全部权重。Expert Parallel 因而把专家分散到不同设备：每张 GPU 保存一部分专家，Token 再根据路由结果前往对应设备。一个 MoE 层由此增加两次数据交换，先将激活分发给专家，再把专家输出送回原来的序列位置。

```mermaid
flowchart LR
    subgraph G0["GPU 0"]
        A["原始 Token"]
        E0["Expert 0"]
        O["恢复顺序"]
    end
    subgraph G1["GPU 1"]
        E1["Expert 1"]
    end
    subgraph G2["GPU 2"]
        E2["Expert 2"]
    end

    A --> R["Router 与 Dispatch"]
    R -->|"All-to-All"| E0
    R -->|"All-to-All"| E1
    R -->|"All-to-All"| E2
    E0 -->|"All-to-All"| O
    E1 -->|"All-to-All"| O
    E2 -->|"All-to-All"| O

    classDef input fill:#fff8e1,stroke:#ff9800,color:#333;
    classDef process fill:#f0f4ff,stroke:#5b8def,color:#333;
    classDef key fill:#fce4ec,stroke:#ef5350,color:#333;
    classDef output fill:#e8f5e9,stroke:#4caf50,color:#333;
    class A input;
    class R key;
    class E0,E1,E2 process;
    class O output;
```

**权重驻留。** 专家分散之后，首先要处理的仍是权重。稀疏激活允许未选中的专家跳过本次计算，却不会自动减少存储需求。671B 参数若采用 BF16，仅权重的理论体积就约为 1.34 TB。量化可以降低占用，但系统仍需为全部专家安排设备空间。若将冷专家临时卸载到 CPU 或磁盘，权重加载延迟又会进入请求路径，是否可行取决于专家命中分布和延迟目标。

**跨卡通信。** 权重就位之后，Token 还要到达对应专家。Dense FFN 的激活通常留在当前并行组中完成矩阵计算，Expert Parallel 则需要依据每轮路由结果执行 All-to-All。通信量由 Token 数、隐藏维度、激活精度和 Top-K 共同决定，具体路由还会改变各设备实际收发的数据量。网络带宽不足或跨节点链路较慢时，数据搬运可能抵消节省的计算时间。

**专家组批。** Token 到达专家后，计算形态本身也发生了变化。MoE 会把一个大批次拆成多个 Expert Batch。若 128 个 Token 在一层中各选择 8 个专家，并分给 64 个专家，即使完全均衡，每个专家平均也只有 16 个 Token。此时 Expert FFN 面对的矩阵高度很小，GPU 难以通过大 GEMM 充分利用计算单元。FLOPs 虽然下降，Kernel Launch、数据重排和访存的占比却会上升。

这种差异在 Prefill 和 Decode 之间尤其明显：

- **Prefill** 一次处理 Prompt 中的大量 Token，专家往往能获得较大的输入批次，通信与 Kernel 开销更容易被摊薄；
- **Decode** 中每条活跃序列每轮只增加少量 Token，并发不足时，专家输入会迅速变得零散。

Continuous Batching 能够汇集不同请求的 Token，但最终批量仍受实时并发、请求完成时间和调度策略限制。

路由统计还要映射到设备：专家数量、每卡放置、链路和同一轮负载波动会影响每卡完成时间。即使长期平均的专家流量接近，各轮设备也可能出现热点，因此需要同时观察专家与 GPU 级别的负载。

这些环节共同解释了 MoE 对高速互联的敏感性。峰值算力决定 Expert FFN 完成矩阵乘法的速度，互联带宽和拓扑则制约激活的传输效率。训练时的大 Token 批次通常更容易摊薄额外成本；在线推理，尤其是低并发 Decode，还需要依靠专家组批、通信重叠、融合 Kernel 和负载观测来缓解稀疏执行的碎片化。

## 五、MoE 的收益边界

前面的代价最终落在三组彼此关联的指标上：

- **容量与计算**：总参数决定权重存储和模型容量，激活参数近似描述单 Token 的主要计算路径；
- **负载与算子**：专家负载与 Expert Batch Size 影响设备长尾和 GEMM 利用率；
- **通信与请求**：互联带宽与请求并发决定 All-to-All 和专家组批的实际效率。

训练场景通常拥有大量 Token 和稳定的并行拓扑，更容易用大批次摊薄路由与通信成本。推理收益则更依赖实际负载：高并发 Prefill 可以形成较大的专家批次，低并发 Decode 可能同时受到权重访存、跨卡通信和小 GEMM 限制。因此，两个激活参数相近的模型，也会因为专家数量、Top-K、专家放置和硬件互联不同，表现出截然不同的延迟。

实际部署可分别记录专家 token 分布、Dispatch/Combine 时间、专家 GEMM 时间和权重占用。若低并发 Decode 主要耗时在通信或小矩阵执行，继续增加名义专家容量并不会自动提高吞吐。
