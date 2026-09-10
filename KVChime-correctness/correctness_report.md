# KVChime 最小数值正确性实验

结论：通过。给定本实验同一构造缓存状态，共享视图与显式物化 attention 在 FP64/FP32 下均通过既定容差；关闭位置对齐的负对照被检出。

模型为 `Qwen/Qwen2.5-0.5B`（revision `060db6499f32faf8b98477b0a26969ef7d8b9987`）。只进行一次真实模型前向，在最后一层（从 0 计为 23）的 q_proj/k_proj/v_proj 上分别触发一次 hook，提取 RoPE 前 Q/K/V；随后不再运行模型或生成。
提取设备 `cpu`，模型/张量 dtype 均为 `torch.float32`，eval、no_grad、eager attention；seed=42，CPU 线程数 8。TF32 和 autocast 关闭，attention 显式计算，不调用 FlashAttention。
固定原文：Shared documents can be reused by multiple language model agents. Each agent keeps its own updates and generated tokens. 原文用一个空格连接重复 16 次，不添加特殊 token；分词后 336 个 token，截取前 320 个。
Q/K/V 形状分别为 `[320, 14, 64]`、`[320, 2, 64]`、`[320, 2, 64]`。GQA=14/2，head_dim=64。

A/B 共享源 token 0–127、128–255 两块 KV。每块 reference start=0、reference positions=0–127；A 的 consumer starts=0/128，B=64/192。A 使用 tail 256–271、replacement 288/289→共享下标17/143、Q271@271；B 使用 tail 272–287@320–335、replacement 290/291→共享下标33/159、Q287@335。replacement 保留目标逻辑位置，私有数据只由所属请求读取；每请求恰有272个有效逻辑位置。

物化参考从未旋转 K 独立构造完整 KV 并按实际位置旋转。KVChime 使用独立的块内 replacement 描述符，保留一份共享 post-RoPE K/V，屏蔽旧版本，以 R(-Δ)Q 访问共享 K；收集有效 logits 后只做一次全局 softmax。PV 分别访问共享 V 和私有 V，不构造完整私有共享-KV副本。
MQ 按 KV head 合并两个请求的 query heads，容量8；每个 KV head 的 14 条 Q 分为 [8, 6]，共 4 组，按请求和原 query head 恢复输出顺序。

RoPE 使用模型实际 inv_freq、theta=1e+06、default scaling 和 Transformers 的前后半维 `rotate_half` 配对。基础函数对原生 FP32 RoPE 的最大误差为 0.000e+00，FP64 旋转组合检查误差为 3.553e-15。FP64 从同一实际 inv_freq 转为 double 后重算角度及三角函数；没有将已舍入的 FP32 cos/sin 冒充 FP64。

工程检查阈值固定：FP64 atol=rtol=1e-8；FP32 atol=rtol=1e-4。相对 L2 = norm(test-ref)/max(norm(ref),1e-12)，比较对象是输出 projection 之前的 attention 输出。

| 精度 | 请求 | Q 位置对齐 | 最大绝对误差 | 相对 L2 误差 | torch.allclose |
| --- | --- | --- | --- | --- | --- |
| FP64 | A | 开启 | 2.22044605e-15 | 4.23448476e-16 | True |
| FP64 | B | 开启 | 2.88657986e-15 | 5.92703831e-16 | True |
| FP64 | A | 关闭（负对照） | 9.88474566e-01 | 1.60728498e-01 | False |
| FP64 | B | 关闭（负对照） | 5.12403549e-01 | 1.01756907e-01 | False |
| FP32 | A | 开启 | 1.90734863e-06 | 2.60159482e-07 | True |
| FP32 | B | 开启 | 1.78813934e-06 | 3.20693601e-07 | True |
| FP32 | A | 关闭（负对照） | 9.88475561e-01 | 1.60728469e-01 | False |
| FP32 | B | 关闭（负对照） | 5.12404203e-01 | 1.01756915e-01 | False |

负对照检查：通过。要求关闭对齐后的每个样本 allclose=False，且最大误差至少为 max(对齐版本误差, 该 dtype 的 epsilon) 的100倍；实测相对对齐版本的最小误差倍数为 2.866e+05。
有效位置和 replacement 源 token 的规范顺序核对：全部一致。

对齐路径诊断：FP64/A logits最大差=7.994e-15、softmax最大差=4.580e-16；FP64/B logits最大差=9.770e-15、softmax最大差=2.498e-16；FP32/A logits最大差=3.886e-05、softmax最大差=5.066e-07；FP32/B logits最大差=3.099e-05、softmax最大差=2.384e-07。

本实验只验证给定缓存状态的共享执行功能与数值一致性，不声称 KV 复用等价于完整重算；不验证 PIM 硬件算术、舍入、性能或复用后的任务质量，不训练、不做任务准确率。

依赖：Python 3.10.13；torch 2.6.0+cpu；transformers 4.57.6；tokenizers 0.22.2；huggingface_hub 0.36.2；safetensors 0.8.0；numpy 2.2.6。
本次调用新执行模型前向 1 次；提取记录中的前向总数为1。输入 token ID SHA-256：`db576a91d3a1725ff558ab8d916cf2c1b7dfd960625e9b11a3babbdc7ff48d7f`。
模型配置与原生 RoPE 配对实现：[Qwen 配置](https://huggingface.co/Qwen/Qwen2.5-0.5B/blob/main/config.json)、[Transformers Qwen2 实现](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/models/qwen2/modeling_qwen2.py)。

复现（装好上述依赖后，在本目录执行；固定 CPU）：

```bash
python kvchime_correctness.py
```

模型下载缓存可通过 `--cache-dir` 指定；已下载时可加 `--local-files-only`。可选 `--tensor-cache` 只保存/重用该一次前向的原始张量，不缓存实验结果；FP64、FP32 和唯一的关闭对齐负对照始终使用同一批提取张量。
