# SNAP 数据集说明

本文档描述 `data/` 下数据的实际目录结构、文件语义、生成流程，以及当前实验代码读取这些文件的方式。后续分析或修改数据相关代码时，应优先参考本文档与根目录的 `README.md`。

## 1. 当前正式实验范围

当前论文主实验使用三类合成网络和七个真实网络：

- 合成网络：`BA`、`WS`、`ER`；
- 水分配网络：`WDN_Net1`、`WDN_Net2`、`WDN_Net3`；
- 电力网络：`IEEE_14Bus`、`IEEE_30Bus`、`IEEE_57Bus`、`IEEE_118Bus`。

当前数据规模如下。

| 数据划分 | 图类型 | 节点数 | 每组拓扑数 | 每张拓扑的延迟实现数 |
|---|---|---|---:|---:|
| 训练集 | BA、WS、ER | 20、50、100 | 1000 | 3 |
| 测试集 | BA、WS、ER | 20、50、100、200、300、500 | 100 | 12 |
| 真实网络测试 | 3个WDN、4个IEEE | 11、36、97、14、30、57、118 | 1 | 30 |

真实网络拓扑同时存在于 `train_data` 和 `test_data`，但当前PPO训练只使用 BA、WS、ER 的20/50/100节点合成图；真实网络只用于测试。

## 2. 目录结构

```text
data/
├── data.md
├── epanet/
│   ├── Net1.inp
│   ├── Net2.inp
│   └── Net3.inp
├── train_data/
│   ├── BA/{20,50,100}_nodes/
│   ├── WS/{20,50,100}_nodes/
│   ├── ER/{20,50,100}_nodes/
│   └── WDN_*/、IEEE_*/                 # 非当前训练输入
└── test_data/
    ├── BA/{20,50,100,200,300,500}_nodes/
    ├── WS/{20,50,100,200,300,500}_nodes/
    ├── ER/{20,50,100,200,300,500}_nodes/
    └── WDN_*/、IEEE_*/
```

标准目录命名为：

```text
data/{train_data|test_data}/{graph_type}/{n_nodes}_nodes/
```

图编号从0开始，正式合成训练集通常为 `0..999`，合成测试集为 `0..99`，真实网络只有编号 `0`。

## 3. 全局节点索引契约

项目中的图数据统一使用以下契约：

```text
矩阵索引 i == NetworkX 节点 i == PPO动作 i == baseline/SGBS传感器 i == 传播源 i
```

因此，同一张图的邻接矩阵 `A`、跳数矩阵 `D_hop`、延迟矩阵
`D_delay` 的第 `i` 行/列必须始终指向同一个节点。生成代码不再依赖
`G.nodes()` 的插入顺序，而是显式使用 `nodelist=range(N)`；非连续或非整数
标签在矩阵化前会被确定性地重标为 `0..N-1`。

`graph_indexing.py` 集中实现这一契约。`data_generator.py` 在写文件前校验
`D_hop` 是否等于从 `A` 重算的最短路矩阵；`generate_sd_realizations.py`
拒绝为错位拓扑生成延迟；`utils.data_loader` 在训练和测试加载时再次校验。
PPO环境、五个baseline、one-shot、SGBS和最终排名评价还会检查动作、传感器
和传播源索引是否越界或重复。

只读审计命令：

```powershell
python validate_graph_data.py
```

只检查拓扑和跳数矩阵（数据量很大时更快）：

```powershell
python validate_graph_data.py --skip-delays
```

只检查真实网络：

```powershell
python validate_graph_data.py --graph-types WDN_Net1,WDN_Net2,WDN_Net3,IEEE_14Bus,IEEE_30Bus,IEEE_57Bus,IEEE_118Bus
```

## 4. 单张拓扑的文件

当前多延迟格式中，一张编号为 `graph_num` 的图包含：

```text
{graph_num}.npy
{graph_num}_GDist.npy
{graph_num}_GDistDelay_{delay_num}.npy
{graph_num}_sd_{delay_num}.npy
```

### 4.1 `{graph_num}.npy`

- 形状：`N × N`；
- 内容：无向图邻接矩阵；
- 对称、非负，通常对角线为0；
- 只表示拓扑，不包含延迟权重。

### 4.2 `{graph_num}_GDist.npy`

- 形状：`N × N`；
- 内容：无权最短路距离矩阵，即跳数距离；
- 由拓扑唯一确定；
- `no_sd=True` 时，训练和评价均使用该矩阵；
- 图编码中的二跳比例、偏心率和直径也基于该矩阵。

### 4.3 `{graph_num}_GDistDelay_{delay_num}.npy`

- 形状：`N × N`；
- 内容：给每条边采样延迟后得到的全源对加权最短路距离；
- 它不是边延迟邻接矩阵，而是延迟加权后的最短路距离矩阵；
- 对称、非负、对角线为0；
- 同一拓扑对应多个不同的 `delay_num`。

每条边的延迟独立采样自截断正态分布：

```text
delay ~ TruncatedNormal(mu=1.0, sd, [mu-2*sd, mu+2*sd])
```

随后使用 Dijkstra 算法计算全源对加权最短路。

### 4.4 `{graph_num}_sd_{delay_num}.npy`

该文件包含三个数：

```text
[delay_sd, graph_param, n_nodes]
```

- `delay_sd`：当前延迟实现使用的边延迟标准差；
- `graph_param`：生成拓扑时的主要参数；
- `n_nodes`：节点数，用于完整性校验。

`graph_param` 的含义：

| 图类型 | `graph_param` |
|---|---|
| BA | 每个新节点连接的旧节点数 `m` |
| ER | 连边概率 `p` |
| WS | 重连概率 `p_rewire` |
| WDN | 当前写入0 |
| IEEE | IEEE bus编号，如14、30、57、118 |

## 5. 一个容易忽略的格式事实

虽然文件扩展名是 `.npy`，当前数据实际由 `numpy.savetxt` 写成文本格式，不是NumPy二进制格式。因此应使用：

```python
matrix = np.loadtxt(path)
```

不要使用 `np.load(path)`。修改存储格式会破坏当前 `utils.data_loader` 的兼容性，除非同时迁移全部数据和加载代码。

## 6. 延迟实现编号与SD区间

所有区间都是左开右闭：

```text
low:    (0.0, 0.1]
medium: (0.1, 0.3]
high:   (0.3, 0.5]
```

### 训练集合成图：3个实现

| `delay_num` | SD区间 |
|---:|---|
| 0 | `(0.0, 0.1]` |
| 1 | `(0.1, 0.3]` |
| 2 | `(0.3, 0.5]` |

### 测试集合成图：12个实现

| `delay_num` | SD区间 |
|---:|---|
| 0–3 | `(0.0, 0.1]` |
| 4–7 | `(0.1, 0.3]` |
| 8–11 | `(0.3, 0.5]` |

### 真实网络：30个实现

| `delay_num` | SD区间 |
|---:|---|
| 0–9 | `(0.0, 0.1]` |
| 10–19 | `(0.1, 0.3]` |
| 20–29 | `(0.3, 0.5]` |

### 在线搜索场景不是数据集文件

`mc-SGBS` 和 `pmc-SGBS` 在部署阶段通过 `sd_scenarios.py` 在线生成独立模拟场景；这些场景不写入上述编号数据文件，也不读取最终测试实现的延迟矩阵或实际 `sd` 值：

- `mc-SGBS`：默认 `M=12`，按三档等权分层采样，每档4个场景；
- `pmc-SGBS --known-sd-level low|medium|high`：只在已知的互斥档位内生成 `M=12` 个场景；
- `pmc-SGBS --known-sd-level all`：为 low、medium、high 分别生成12个场景，共36个场景和三套部署方案，最终测试时按档位匹配方案。

这里的档位信息可以作为条件先验，但最终测试实现的精确 `sd` 数值、延迟矩阵、传播源和观测时间均不能进入部署过程。`sd` 表示截断前正态分布的尺度参数；截断后样本的实际标准差通常不等于该值。

## 7. 旧式单延迟格式

历史数据使用：

```text
{graph_num}_GDistDelay.npy
{graph_num}_sd.npy
```

当前 `utils.data_loader` 仍能回退读取这种格式，但：

- SD-PPO训练要求每张拓扑至少有两个编号延迟实现，因此旧格式不能用于当前SD-PPO训练；
- 新版SD测试的分组统计需要编号实现；
- 正式BA、WS、ER和真实网络已经使用编号格式，旧文件在转换成功后被删除。

## 8. 拓扑生成

拓扑由 `data_generator.py` 生成。该脚本首先写入邻接矩阵、跳数矩阵，以及一个临时的旧式单延迟文件；多延迟实现随后由独立脚本生成。

单组生成示例：

```powershell
python data_generator.py --type BA --nodes 100 --num 1000 --base-dir data/train_data --seed 42
```

当前正式合成训练集可用PowerShell循环生成：

```powershell
foreach ($type in @("BA","WS","ER")) {
    foreach ($size in @(20,50,100)) {
        python data_generator.py --type $type --nodes $size --num 1000 --base-dir data/train_data --seed 42
    }
}
```

当前正式合成测试集：

```powershell
foreach ($type in @("BA","WS","ER")) {
    foreach ($size in @(20,50,100,200,300,500)) {
        python data_generator.py --type $type --nodes $size --num 100 --base-dir data/test_data --seed 2026
    }
}
```

生成真实网络拓扑：

```powershell
python data_generator.py --real --base-dir data/train_data --seed 42
python data_generator.py --real --base-dir data/test_data
```

`data/epanet/*.inp` 是WDN拓扑来源；IEEE拓扑的支路表内置于 `data_generator.py`。

注意：

- 相同编号文件会被覆盖；
- 如果新生成数量小于目录中原有数量，高编号旧文件不会自动删除；
- `--snap-test-data` 的内置生成范围不等于当前最终实验矩阵；
- 因此正式数据建议使用上面的显式图类型、规模循环。

## 9. 多延迟实现生成

多延迟生成由 `generate_sd_realizations.py` 独立完成，不修改：

```text
{graph_num}.npy
{graph_num}_GDist.npy
```

脚本默认只预览，不写文件：

```powershell
python generate_sd_realizations.py --graph-types BA,WS,ER,WDN_Net1,WDN_Net2,WDN_Net3,IEEE_14Bus,IEEE_30Bus,IEEE_57Bus,IEEE_118Bus
```

确认预览后执行：

```powershell
python generate_sd_realizations.py --graph-types BA,WS,ER,WDN_Net1,WDN_Net2,WDN_Net3,IEEE_14Bus,IEEE_30Bus,IEEE_57Bus,IEEE_118Bus --execute --seed 42
```

仅重新生成指定规模或图类型时，可以增加：

```text
--sizes 20,50,100
--overwrite
```

`--overwrite` 会确定性重建已有编号实现，应谨慎使用。脚本只有在一张图的全部新实现验证通过后，才删除其旧式无编号延迟文件。

随机数由基础seed、数据根目录绝对路径、图类型、节点数、图编号和延迟编号共同派生。因此在相同绝对路径下可重复生成；移动数据根目录后重新生成，随机实现可能不同。

## 10. 代码如何使用数据

### `no_sd=True`

- 只使用 `{graph_num}_GDist.npy`；
- `G_distance_delay` 在内存中直接设为跳数矩阵副本；
- 不读取编号延迟文件；
- 保持历史no-SD随机数路径不变。

### `no_sd=False`训练

- 当前PPO训练加载一张拓扑的全部3个编号延迟实现；
- 每个episode开始时均匀随机选择其中一个；
- 一个episode内延迟矩阵固定；
- 图特征始终只来自拓扑和跳数，不含SD或具体延迟矩阵。

### `no_sd=False`测试

- 延迟文件按拓扑懒加载，避免一次将全部矩阵放入内存；
- `one_shot`、五种baseline和 `hop-SGBS` 先基于拓扑/跳数选择一套传感器；
- `mc-SGBS` 根据三档等权先验生成一个共享场景池并选择一套传感器；
- `pmc-SGBS` 单档模式根据已知档位选择一套传感器；`all` 模式在最终测试矩阵物化前分别生成三套传感器方案，再按最终实现所属的 low/medium/high 档位匹配方案；
- 选定的集合再在该拓扑对应的全部12或30个最终延迟实现上评价；
- 测试端选择过程不知道具体延迟实现；
- 每张拓扑先对延迟实现求平均，再跨拓扑求 `ave_rank`，保证每张拓扑权重相同。

## 11. 数据完整性要求

正式数据应满足：

1. 邻接矩阵和距离矩阵均为 `N × N`；
2. `D_hop` 必须与从邻接矩阵按相同索引重算的最短路矩阵完全一致；
3. 邻接矩阵与延迟距离矩阵对称；
4. 图连通；
5. 距离矩阵对角线为0且不存在NaN/Inf；
6. `D_delay` 必须满足当前SD截断范围与 `D_hop` 推导出的路径上下界；
7. 编号延迟实现必须从0连续编号；
8. 每个延迟矩阵必须有同编号SD元数据；
9. 元数据中的节点数必须与目录和矩阵一致；
10. 同一训练组内每张拓扑的延迟实现数量必须相同。

不要直接删除某个单独的 `_sd_{delay_num}.npy` 或 `_GDistDelay_{delay_num}.npy`，否则加载器会将数据判为不完整。

## 12. 当前数据审计状态（2026-10-10）

当前目录实测数量为：

- 训练集：BA、WS、ER 的20/50/100节点每组1000张拓扑，每张3个编号延迟实现；
- 测试集：BA、WS、ER 的20/50/100/200/300/500节点每组100张拓扑，每张12个编号延迟实现；
- 七个真实网络：训练目录和测试目录各1张拓扑，每张30个编号延迟实现；真实网络不参与当前PPO训练。

2026-10-10 对测试目录七个真实网络执行了完整索引审计：

```powershell
& python validate_graph_data.py --roots data/test_data --graph-types WDN_Net1,WDN_Net2,WDN_Net3,IEEE_14Bus,IEEE_30Bus,IEEE_57Bus,IEEE_118Bus
```

结果为：检查7张拓扑和210个延迟实现，全部通过。现存正式真实网络数据已经按统一节点顺序重建；旧索引错位数据及由其产生的旧结果不得再用于论文统计。

