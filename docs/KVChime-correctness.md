# RoPE、替换视图与 MQ 的等价性

这里证明执行机制保持**给定软件重算集合所定义的 attention**。不声称部分重算与整模型 full recompute 的隐藏状态、精度或最终输出完全相同。查询侧旋转使用标准 RoPE 的位移恒等式，是已有等式在 PIM 驻留共享 K 上的应用；下列证明自包含，不依赖外部论文文件。

## 1. 参考坐标系中的共享 K

固定频率 RoPE 的旋转矩阵满足 $R(a)^T R(b)=R(b-a)$。chunk $c$ 的原始 key 为 $k_{c,t}$，PIM 保存参考起点 $r_c$ 下的 $\bar k_{c,t}=R(r_c+t)k_{c,t}$，以及不施加 RoPE 的 $v_{c,t}$。

请求 $i$ 把 chunk 放在起点 $b_{i,c}$，查询的逻辑位置为 $p$。令 $\Delta=b_{i,c}-r_c$，GPU 发送 $q'_{i,c}=R(p-\Delta)q_i$。则

$$
(q'_{i,c})^T\bar k_{c,t}
=q_i^T R(r_c+t-p+\Delta)k_{c,t}
=q_i^T R(b_{i,c}+t-p)k_{c,t}
=(R(p)q_i)^T R(b_{i,c}+t)k_{c,t}.
$$

所以，对齐 Q 后在 PIM 直接读取参考 K，与把 K 读回 GPU、移到消费位置再计算，产生相同的实数 score。若 GPU 已生成 $\hat q=R(p)q_i$，只需 $q'=R(-\Delta)\hat q$。同一 query 对不同偏移的 chunk 可以需要不同 Q；本次仿真按不同偏移数计算 Q 传输量，没有把它一律算成一份 Q。

例：$r_c=0,b_{i,c}=96,p=200,t=5$。发送 $R(104)q$ 与 $R(5)k$ 相乘，相对旋转是 $R(-99)$；消费视图中的 $R(200)q$ 与 $R(101)k$ 相乘也得到 $R(-99)$。无需搬动整个 K。

条件：同一层、同一 head、相同 RoPE 频率与缩放配置，chunk 内位移一致。若频率随请求长度改变，单一偏移恒等式不能直接套用。该等式只处理位置坐标，不消除不同上下文引起的隐藏状态变化。

## 2. 替换必须形成唯一逻辑视图

设请求有效逻辑位置为 $J=\{0,\ldots,N-1\}$，重算集合为 $D$。定义版本选择函数

$$
\phi_i(j)=\begin{cases}\text{private}(i,j),&j\in D,\\\text{shared}(c,t),&j\notin D.\end{cases}
$$

新增 token 也各占一个新的逻辑位置。要求每个位置恰好映射到一个有效 K/V 版本，且同一位置在 QK 与 PV 中使用同一版本。旧共享块中的被替换项可继续被物理读取，但它的 score 必须失效；私有项按原逻辑位置参加 causal mask，不能作为额外重复 token 加到结尾。

由第一节，每个位置的 score 等于物化后视图的 score。对全部有效位置进行一次 query-global softmax，得到相同的 $w_j=\exp(s_j)/\sum_{u\in J}\exp(s_u)$；PV 输出 $o=\sum_{j\in J}w_j v_{\phi_i(j)}$ 因此相同。若分块 softmax，必须使用全局归一化或数学等价的在线合并；不能把各块局部 softmax 输出直接相加。

例：共享位置 `[0,1,2,3]`，位置 `1` 被私有 KV 替换，新 token 位于 `4`。有效视图为 `[S0,P1,S2,S3,P4]`，长度为 5。物理扫描可以读到 `S1`，但该旧 score 不参与归一化。代码输出的 `logical-view.json` 保存这种位置/版本分区，并检查每个逻辑位置恰好出现一次。

## 3. 多查询只复用操作数，不混合结果

同一列 K/V 被 $r$ 个 query 消费时，把 $r$ 次列读改为一次列读和 $r$ 次独立 MAC，不改变任一 query 的乘积集合。每个 query 必须保留独立的累加器、causal mask、softmax 状态和输出版本。因此 MQ 与逐 query 扫描在实数运算下等价；query 可以来自一个 prefill，也可以来自多个已就绪 agent。

这是代数等价性，不是 FP16 逐位一致性声明：改变归约顺序可能改变浮点舍入。当前 Ramulator 是命令时序仿真器，不执行数值 QK/PV；时序记录、位置分区检查和数学证明各自说明不同层面的性质。
