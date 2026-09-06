# Fugue：按 Evaluation 阶梯组织的四项贡献

## 一句话概括

**Fugue 通过单个 KV head 内的 diff 紧凑布局、软件表分散共读块、MQ 配合 PIM prefill attention，以及逐请求动态选边，降低共享 KV 服务的行激活、通道串行、重复读取和历史 KV 搬运成本。**

## 四项贡献：每项一句话

1. **diff 紧凑布局（A3b → A4c）：** 把同一 agent 在一个 KV head 下跨 chunk、跨轮产生的 diff 紧凑放到该 head 自己一条 channel 的 diff 行里，master 仍使用该 head 的全部 channel，以减少分散修正带来的额外行激活。
2. **软件放置表（A4c → A4e）：** 在共享块写入时，用软件表把同一个 KV head 内会同时访问的 master 块尽量分散到该 head 的不同 channel，并记录固定位置供后续扫描使用，以减少同通道串行。
3. **MQ 与 PIM prefill attention（A4e → A5）：** 将 prefill attention 放到 PIM，让多个 query 用 MQ 复用同一次共享列读取、在换行前用完当前行，同时保留 GPU 上的线性计算，以减少重复读数和历史 KV 回读。
4. **动态选边（A5 → A6）：** 对每个 prefill，按实际计算 token 数、上下文长度和链路成本比较两侧估价，选择 GPU 或 PIM 执行注意力，使不同计算量的请求采用各自更合适的执行侧。

## 小例子：情况设定、怎么做、预计收益

以下例子都固定在**一层中的一个 KV head H**，其他 KV head 独立处理；字节数、行数和读取次数是设定条件下的推导，选边耗时是假设值。

### 贡献一的例子：diff 紧凑布局

<!-- example-layout:start -->

**情况设定：** H 占 4 个 channel（ch0、ch1、ch2、ch3），只看 agent A 在 H 下的 diff；A 连续 8 轮，每轮复用一个 256-token chunk，并产生 8 个 diff token，轮间穿插 A 自己写入的 KV；假设朴素布局每轮的 diff 单占一行，并按轮次轮转到这些通道。

**怎么做：** 保持 H 的 master 分布不变，把 A 在 H 下的 D0–D7 紧凑追加到 ch3 的一个 diff 行中；ch3 仍保存 master，其他 KV head 的 diff 各自处理。

| H 的 channel | master 行（两种布局相同） | 朴素 diff 行 | 紧凑 diff 行 |
| --- | --- | --- | --- |
| ch0 | M0、M4 | D0 单占一行（8 token）；D4 单占一行（8 token） | 无 |
| ch1 | M1、M5 | D1 单占一行（8 token）；D5 单占一行（8 token） | 无 |
| ch2 | M2、M6 | D2 单占一行（8 token）；D6 单占一行（8 token） | 无 |
| ch3 | M3、M7 | D3 单占一行（8 token）；D7 单占一行（8 token） | D0–D7 紧凑放在同一行（64 token） |

**预计收益：** 在这些轮次结束后的一次 K 扫描中，diff 共 `8 × 8 = 64` token，小于行容量 256 token，所需行激活由 **8 次降为 1 次**；连同 8 个 master 行，合计由 **16 次降为 9 次**，减少 **43.75%**，V 扫描同理；这里仅统计该 head 的这些 master 和 diff 行，未把轮间自写 KV 计入这个子扫描，ACT 减少比例不等于延迟减少比例。

<!-- example-layout:end -->

### 贡献二的例子：软件表将同时访问的块分散到不同 channel

<!-- example-placement:start -->

**情况设定：** H 仍占 4 个 channel，另取按 M0 到 M4 顺序写入的共享块，每块占一行；一次扫描只读 M0 和 M4，写入 M4 时已知这两个块会一起读，且有其他通道可选。

**怎么做：** 写入序轮转会把 M0 和 M4 都放到 ch0；软件表根据共同读取关系，让 M0 留在 ch0，在写入 M4 时把它放到 ch1，记录两块各自的 channel 和 row，后续扫描按表访问。

**预计收益：** 若每块扫描耗时为 `t`，没有其他竞争且通道可并行，两块由同通道串行的 **2t** 变为跨通道并行的 **t**，这两个块的局部扫描预计加速 **2×**，总读取量不变。

<!-- example-placement:end -->

### 贡献三的例子：MQ 与 PIM prefill attention

<!-- example-mq:start -->

**情况设定：** H 占 4 个 channel，历史 KV 已驻留 PIM，由 32 个 256-token chunk 组成，共 8,192 token；agent A 的一次短 prefill 只计算 8 个新增 token，产生 8 个 query，均需读取这些历史行；head 维度为 128、每元素 2 B，每个 bank 的 query 缓冲为 512 B、每个 query 切片占 64 B，恰好驻留这批 query。

**怎么做：** GPU 生成新增 token 的 Q/K/V，将新 K/V 写入 PIM，并传入对齐后的 query；prefill attention 在 PIM 执行，扫描到 H 的一个共享行时，一次 MQ 列读取将同一份 K 数据交给这 8 个 query 使用，新增 token 间的注意力按因果关系处理，最后返回 context。

**预计收益：** PIM prefill 避免本次将 `8,192 × 2 × 128 × 2 B = 4 MiB` 历史 KV 回读 GPU，链路仍需传新 KV、query 并返回 context；MQ 相对逐 query 重复列读取，将同一共享列服务整批的读取次数由 **8 次降为 1 次**，减少 **87.5%**，但仍需完成各 query 的 MAC 和 softmax，不能把列读取减少倍数直接当作 A4e → A5 的注意力加速比。

<!-- example-mq:end -->

### 贡献四的例子：动态选边

<!-- example-adaptive:start -->

**情况设定：** H 占 4 个 channel，两个请求都扫描 8,192 token 的历史上下文，先后就绪且不重叠，但计算的 query 数不同；假设两侧估价已包含各自传输与注意力成本，共同的线性层成本省略。

**怎么做：** 对每个请求选择估价较小的一侧。

| 本次计算量 | GPU 估价（假设） | PIM 估价（假设） | 选择 |
| --- | --- | --- | --- |
| 8 query | 100 µs | 20 µs | PIM |
| 512 query | 200 µs | 900 µs | GPU |

**预计收益：** 按 A5 的固定全 PIM 策略，本例注意力总耗时为 **920 µs**；按 A6 动态选边，短请求留在 PIM、长请求转到 GPU，总耗时为 **220 µs**，相对全 PIM 预计加速 **4.18×**；全 GPU 则为 300 µs，这些是假设耗时下的局部比较，实际 A6 的结果取决于估价和资源竞争，不能推成全局最优保证。

<!-- example-adaptive:end -->

贡献顺序对应论文 [Evaluation 正文](../../KVPIM-1Fugue-ASPLOS2027/sections/07-evaluation.tex)的 placement ladder，机制依据[架构正文](../../KVPIM-1Fugue-ASPLOS2027/sections/04-design.tex)和[执行模型正文](../../KVPIM-1Fugue-ASPLOS2027/sections/05-execution.tex)，并核对仓库的[档位定义](../src/ablation.py)、[布局与执行](../src/workload_runner.py)、[MQ 定价](../src/ramulator_wrapper.py)；未采用图片或图注作为依据。

<details>
<summary>示例输入与复算脚本</summary>

场景参数如下；chunk 行容量和 query 缓冲参数从仓库源码读取。在仓库根目录执行本节 Python 代码，会重新生成本文例子，不启动仿真。

```json
{"chunks":32,"repair_tokens":8,"head_dim":128,"element_bytes":2,"rounds":8,"stripe":4,"co_read_write_indices":[0,4],"prefills":[{"queries":8,"gpu_us":100,"pim_us":20},{"queries":512,"gpu_us":200,"pim_us":900}]}
```

```python
from pathlib import Path
import ast
import json
import math
import re

path = Path('docs/README_contributions.md')
text = path.read_text()
p = json.loads(re.search(r'```json\n(.*?)\n```', text, re.S).group(1))

def constant(file, name):
    for node in ast.parse(Path(file).read_text()).body:
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError(name)

def put(name, content):
    global text
    start, end = f'<!-- example-{name}:start -->', f'<!-- example-{name}:end -->'
    assert text.count(start) == text.count(end) == 1
    before, rest = text.split(start)
    _, after = rest.split(end)
    text = before + start + '\n\n' + content.strip() + '\n\n' + end + after

def table(headers, rows):
    return '\n'.join('| ' + ' | '.join(map(str, row)) + ' |'
                     for row in [headers, ['---'] * len(headers), *rows])

chunk = constant('src/workload_runner.py', '_STRIPE_UNIT_ROWS')
buffer = constant('src/ramulator_wrapper.py', 'MQ_DEFAULT_GEMV_BUFFER_BYTES')
q_slice = constant('src/ramulator_wrapper.py', 'MQ_QUERY_SLICE_BYTES')
c, k, d, b, r, channels = (p[key] for key in (
    'chunks', 'repair_tokens', 'head_dim', 'element_bytes', 'rounds', 'stripe'))
channel_names = '、'.join(f'ch{i}' for i in range(channels))
length, per_token, mib = c * chunk, 2 * d * b, 2 ** 20
history_bytes = length * per_token

packed = math.ceil(r * k / chunk)
assert r % channels == 0 and packed == 1
rows = []
for ch in range(channels):
    ids = list(range(ch, r, channels))
    master_rows = '、'.join(f'M{i}' for i in ids)
    naive_rows = '；'.join(f'D{i} 单占一行（{k} token）' for i in ids)
    compact = f'D0–D{r - 1} 紧凑放在同一行（{r * k} token）' if ch == channels - 1 else '无'
    rows.append([f'ch{ch}', master_rows, naive_rows, compact])
layout = table(['H 的 channel', 'master 行（两种布局相同）', '朴素 diff 行', '紧凑 diff 行'], rows)
put('layout', f"""**情况设定：** H 占 {channels} 个 channel（{channel_names}），只看 agent A 在 H 下的 diff；A 连续 {r} 轮，每轮复用一个 {chunk}-token chunk，并产生 {k} 个 diff token，轮间穿插 A 自己写入的 KV；假设朴素布局每轮的 diff 单占一行，并按轮次轮转到这些通道。

**怎么做：** 保持 H 的 master 分布不变，把 A 在 H 下的 D0–D{r - 1} 紧凑追加到 ch{channels - 1} 的一个 diff 行中；ch{channels - 1} 仍保存 master，其他 KV head 的 diff 各自处理。

{layout}

**预计收益：** 在这些轮次结束后的一次 K 扫描中，diff 共 `{r} × {k} = {r * k}` token，小于行容量 {chunk} token，所需行激活由 **{r} 次降为 {packed} 次**；连同 {r} 个 master 行，合计由 **{2 * r} 次降为 {r + packed} 次**，减少 **{1 - (r + packed) / (2 * r):.2%}**，V 扫描同理；这里仅统计该 head 的这些 master 和 diff 行，未把轮间自写 KV 计入这个子扫描，ACT 减少比例不等于延迟减少比例。""")

first, last = p['co_read_write_indices']
assert first % channels == last % channels and channels > 1
slot = first % channels
other = (slot + 1) % channels
put('placement', f"""**情况设定：** H 仍占 {channels} 个 channel，另取按 M{first} 到 M{last} 顺序写入的共享块，每块占一行；一次扫描只读 M{first} 和 M{last}，写入 M{last} 时已知这两个块会一起读，且有其他通道可选。

**怎么做：** 写入序轮转会把 M{first} 和 M{last} 都放到 ch{slot}；软件表根据共同读取关系，让 M{first} 留在 ch{slot}，在写入 M{last} 时把它放到 ch{other}，记录两块各自的 channel 和 row，后续扫描按表访问。

**预计收益：** 若每块扫描耗时为 `t`，没有其他竞争且通道可并行，两块由同通道串行的 **2t** 变为跨通道并行的 **t**，这两个块的局部扫描预计加速 **2×**，总读取量不变。""")

queries = buffer // q_slice
put('mq', f"""**情况设定：** H 占 {channels} 个 channel，历史 KV 已驻留 PIM，由 {c} 个 {chunk}-token chunk 组成，共 {length:,} token；agent A 的一次短 prefill 只计算 {queries} 个新增 token，产生 {queries} 个 query，均需读取这些历史行；head 维度为 {d}、每元素 {b} B，每个 bank 的 query 缓冲为 {buffer} B、每个 query 切片占 {q_slice} B，恰好驻留这批 query。

**怎么做：** GPU 生成新增 token 的 Q/K/V，将新 K/V 写入 PIM，并传入对齐后的 query；prefill attention 在 PIM 执行，扫描到 H 的一个共享行时，一次 MQ 列读取将同一份 K 数据交给这 {queries} 个 query 使用，新增 token 间的注意力按因果关系处理，最后返回 context。

**预计收益：** PIM prefill 避免本次将 `{length:,} × 2 × {d} × {b} B = {history_bytes / mib:g} MiB` 历史 KV 回读 GPU，链路仍需传新 KV、query 并返回 context；MQ 相对逐 query 重复列读取，将同一共享列服务整批的读取次数由 **{queries} 次降为 1 次**，减少 **{1 - 1 / queries:.1%}**，但仍需完成各 query 的 MAC 和 softmax，不能把列读取减少倍数直接当作 A4e → A5 的注意力加速比。""")

requests = p['prefills']
gpu = sum(x['gpu_us'] for x in requests)
pim = sum(x['pim_us'] for x in requests)
chosen = sum(min(x['gpu_us'], x['pim_us']) for x in requests)
choices = table(['本次计算量', 'GPU 估价（假设）', 'PIM 估价（假设）', '选择'], [
    [f"{x['queries']} query", f"{x['gpu_us']} µs", f"{x['pim_us']} µs",
     'PIM' if x['pim_us'] <= x['gpu_us'] else 'GPU'] for x in requests])
put('adaptive', f"""**情况设定：** H 占 {channels} 个 channel，两个请求都扫描 {length:,} token 的历史上下文，先后就绪且不重叠，但计算的 query 数不同；假设两侧估价已包含各自传输与注意力成本，共同的线性层成本省略。

**怎么做：** 对每个请求选择估价较小的一侧。

{choices}

**预计收益：** 按 A5 的固定全 PIM 策略，本例注意力总耗时为 **{pim} µs**；按 A6 动态选边，短请求留在 PIM、长请求转到 GPU，总耗时为 **{chosen} µs**，相对全 PIM 预计加速 **{pim / chosen:.2f}×**；全 GPU 则为 {gpu} µs，这些是假设耗时下的局部比较，实际 A6 的结果取决于估价和资源竞争，不能推成全局最优保证。""")
path.write_text(text)
```

</details>
