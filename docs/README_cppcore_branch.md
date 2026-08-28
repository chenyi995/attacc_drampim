# chenyi-822-cppcore-exp:核心 C++ 化实验分支(chenyi9 裁决 2026-08-28)

**一句话**:基于冻结的 `chenyi-822-dirty`(1a231b6)新建的实验分支,把
DAG 引擎的**调度状态机**迁入原生 C++ 核(`src/cppcore/eventcore.cpp`,
ctypes 桥 `src/cpp_eventcore.py`),Python 保留全部领域逻辑与
SplitEvent 记录;每个事件在创建时镜像进核((device, duration, deps)),
增量准入排序与全图调度都在原生侧完成。

## v1 范围与动机

- 消灭两处 Python 侧结构性成本:①增量调度器**每轮整拷 finish/
  availability 字典**(隐性二次方,suite 规模的构图大头之一);
  ②全图调度的逐事件 dict/max 解释开销。
- 浮点逐位一致:C++ 侧 max 折叠顺序与 Python `max([avail]+deps)` 相同,
  IEEE double 同序同运算 → 结果不是"近似",是**同一**。
- **纯 Python 路完整保留**(`KVPIM_CPPCORE=0` 或缺库时自动回退);
  overlap 契约校验器仍用 Python 重放核对原生调度——**每次运行都在做
  交叉验证**。

## 验证(2026-08-28)

- 41/41 单测(核默认开启,DAG 测试内嵌 Python 重放核对全过);
- wl_tiny @16HBM/PC 暖模式 A4/A6:`KVPIM_CPPCORE=1` vs `0` 报告
  **逐值相同**(排除缓存统计元数据块);本分支 Python 路 vs 822 冻结
  基线亦逐值相同(同仿真器二进制 + 同签名缓存快照);
- wl_tiny 墙钟:A4 48.2→38.1 s、A6 51.0→40.0 s(−21%;二次方项在
  star 规模的收益远大于此,见 experiment-1 复跑)。

## 构建与运行

```sh
make -C src/cppcore          # gcc-toolset-11,产出 libeventcore.so
KVPIM_CPPCORE=1 python3 main.py ... --engine dag ...   # 默认即开
```

## 路线图(后续)

1. experiment-1 复跑(= 822-dirty 在跑的 star @16HBM/PC 三点链)对数
   与计时——排队在 822 编排器收官之后(560 GB 纪律,两边不并跑);
2. v2:构图主体(TLB/布局/事件发射循环)下沉 C++——当前 v1 只搬了
   调度,构图的解释器常数仍在 Python;
3. v3:warm 台账/重定价与聚合下沉。
