# SNAP 实验框架

> 本目录是用于公开发布的独立副本。它包含核心代码、回归测试、可运行的小型示例数据、两组最终 PPO 权重、完整的解压结果和绘图产物。完整数据集及历史模型权重由于体积较大，将通过外部数据仓库发布；参见 [数据说明](data/README.md)、[模型说明](model_save/README.md) 和 [发布清单](REPOSITORY_CONTENTS.md)。

## 快速开始

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest test_feature_reward.py test_graph_indexing.py test_sd_scenarios.py test_sd_sgbs.py -v
```

仓库内的 BA 20 节点样例包含一张训练拓扑和一张测试拓扑，可用于检查数据读取流程。完整规模实验需要先按 `data/README.md` 获取完整数据集。

SNAP 用于研究图上的传感器布设与传播源定位问题。给定网络拓扑和传感器预算，算法选择一组传感器；传播源产生观测后，再根据传感器观测对所有候选源节点排序。项目同时支持只考虑拓扑跳数的 `no_sd` 场景，以及考虑随机连边延迟分布的 `sd` 场景。

当前正式实验以 `PPOAgent` 为神经模型，区分策略直接推理与三种SGBS评分模式：

- `one_shot`：PPO 策略逐个选择传感器，不进行额外搜索；
- `hop-SGBS`：使用跳数代理评分完整候选方案；
- `mc-SGBS`：使用三档等权的独立模拟时延场景评分完整方案；
- `pmc-SGBS`：部署前已知 low/medium/high 离散档位，在对应条件先验内模拟场景并评分。三种SGBS的候选生成均由同一个PPO策略引导，搜索宽度由 `beta` 和 `gamma` 控制。

五种非神经网络基线为 `Random`、`BC`、`HCR`、`CB` 和 `HV_OBS`。当前正式模型使用 `ppo_mix_v3_no_sd` 和 `ppo_mix_v3_sd`；旧模型和旧结果仅用于复现或对照。

## 1. 权威代码与项目结构

根目录中的代码是当前实验框架的权威版本。工作目录中的 `20260708/`、`runs/` 等历史代码、运行记录和备份未纳入本公开副本。

| 文件 | 作用 |
|---|---|
| `run_experiment.py` | 训练与测试的统一命令行入口 |
| `experiment_config.py` | 图类型、规模、路径和模型默认参数 |
| `sensor_budget_config.py` | 节点规模与固定传感器预算的映射 |
| `data_generator.py` | 拓扑、跳数距离和旧式单延迟数据生成 |
| `generate_sd_realizations.py` | 为现有拓扑生成多组延迟实现 |
| `graph_indexing.py` | 统一矩阵、动作、传感器和传播源的节点索引契约 |
| `validate_graph_data.py` | 只读审计邻接、跳数和延迟矩阵是否采用同一索引 |
| `utils.py` | 数据加载、图特征、回放数据及通用工具 |
| `OP_Env.py` | 传感器选择强化学习环境与奖励计算 |
| `OP_Trainer.py` | PPO 训练流程和 checkpoint 保存 |
| `OP_Model.py` | GCN、actor 和 critic 网络定义 |
| `sourceLocalizationClass.py` | 定位评价、基线方法和 SGBS 推理 |
| `sd_scenarios.py` | 按预先声明的噪声先验生成独立、可复现的搜索时延场景 |
| `summarize_baselines.py` | 汇总非神经基线 JSON 结果为 CSV |
| `sweep_sgbs_tradeoff.py` | 扫描 `beta`、`gamma` 的质量—时间权衡 |
| `test_feature_reward.py` | 图特征和奖励逻辑的回归测试 |
| `test_graph_indexing.py` | 图生成、baseline、PPO、SGBS和评价索引的回归测试 |
| `test_sd_scenarios.py` | 独立时延场景的分层采样、随机流和矩阵合法性测试 |
| `test_sd_sgbs.py` | hop/mc/pMC-SGBS评分、缓存、隔离边界及结果命名测试 |
| `data/data.md` | 数据目录、文件格式和生成流程的详细说明 |

## 2. 当前实验矩阵

### 2.1 合成网络

当前正式实验使用 `BA`、`WS`、`ER` 三类网络。

| 阶段 | 图规模 | 每种类型、每个规模的拓扑数 | 每张拓扑的延迟实现数 |
|---|---|---:|---:|
| 训练 | 20、50、100 | 1000 | 3 |
| 测试 | 20、50、100、200、300、500 | 100 | 12 |

训练命令中的 `--num-episodes 5000` 表示每个“图类型 × 图规模”组训练 5000 个 episode。使用三类网络和三个规模时，共执行 9 组，即总计 45000 个 episode。

### 2.2 真实网络

真实网络只用于测试，每张网络有 30 组延迟实现。

| 网络 | 节点数 | 传感器预算 |
|---|---:|---:|
| WDN_Net1 | 11 | 3 |
| WDN_Net2 | 36 | 5 |
| WDN_Net3 | 97 | 10 |
| IEEE_14Bus | 14 | 3 |
| IEEE_30Bus | 30 | 5 |
| IEEE_57Bus | 57 | 6 |
| IEEE_118Bus | 118 | 12 |

## 3. 固定传感器预算

预算由 `sensor_budget_config.py` 统一管理，不再按节点数的固定百分比计算。

| 节点数 | 11 | 14 | 20 | 30 | 36 | 50 | 57 | 97 | 100 | 118 | 200 | 300 | 500 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 传感器数 | 3 | 3 | 3 | 5 | 5 | 5 | 6 | 10 | 10 | 12 | 15 | 20 | 30 |

新增图规模或真实网络时，应先在该配置文件中增加预算，避免训练、测试和 baseline 使用不同设置。

## 4. 图编码特征与消融设置

节点静态特征按 `--feature-set` 分为四组；每组还会拼接两个动态特征：“节点是否已选”和“剩余预算比例”。

| 特征集 | 静态节点特征 | 静态维度 | PPO 总输入维度 |
|---|---|---:|---:|
| `f0` | degree、betweenness、closeness、bias | 4 | 6 |
| `f1` | `f0` + k-core | 5 | 7 |
| `f2` | `f1` + clustering、two-hop ratio、eccentricity | 8 | 10 |
| `f3` | `f2` + graph density、diameter ratio | 10 | 12 |

所有中心性和结构量都经过归一化。`graph density` 和 `diameter ratio` 是复制到每个节点的全局图特征。`f3` 是正式模型的默认特征集；`f0`、`f1`、`f2` 用于特征消融。旧 checkpoint 的输入维度可能不同，不能跨特征集直接加载。

## 5. `no_sd` 与 `sd` 场景

### `--no-sd True`

- 训练和评价均使用无权最短路跳数矩阵 `{graph_num}_GDist.npy`；
- 不读取随机延迟实现；
- 结果文件名不带 `_delay` 后缀。

### `--no-sd False`

- PPO 训练时，每张拓扑有 3 组延迟实现，每个 episode 均匀抽取一组，并在该 episode 内固定；
- `one_shot` 始终只使用拓扑、f3特征、已选标记和剩余预算比例；
- `SGBS --sd-score-mode hop` 使用跳数矩阵评价完整候选方案，保持历史行为；
- `SGBS --sd-score-mode mc` 使用按测试前规定的先验独立生成的模拟时延场景评价完整方案；
- `SGBS --sd-score-mode pmc` 表示部署前已知离散SD档位，分别使用 `(0,0.1]`、`(0.1,0.3]`、`(0.3,0.5]` 三个互斥条件先验评价完整方案；
- 未指定 `--results-dir` 时，以上三种SD-SGBS结果分别默认写入 `result/hop_sgbs`、`result/mc_sgbs` 和 `result/pmc_sgbs`；显式 `--results-dir` 仍可覆盖默认目录；
- 选出的同一传感器集合在该拓扑的全部延迟实现上评价；
- 合成测试集每张拓扑评价 12 组延迟，真实网络评价 30 组延迟；
- 结果先在同一拓扑的延迟实现间平均，再在不同拓扑间平均；
- 当前代码的 SD 结果文件仍使用历史后缀 `_delay`。

#### 独立搜索场景与最终测试场景

二者用途不同，严禁混用：

- **独立搜索场景**：只在 `mc-SGBS` 部署时使用，由 `sd_scenarios.py` 根据显式 `--sd-prior`、`--search-scenarios` 和 `--search-seed` 在线生成。默认先验为三档等权分层：原始正态尺度参数分别落在 `(0,0.1]`、`(0.1,0.3]`、`(0.3,0.5]`，边时延从 `[mu-2sd,mu+2sd]` 截断正态采样。这里的 `sd` 是截断前正态尺度参数，不是截断后样本标准差。
- **最终测试场景**：数据目录中编号的 `GDistDelay` 实现，仅在传感器集合确定后用于最终rank评价和low/medium/high分组。

搜索上下文不包含最终测试延迟矩阵、实际 `sd` metadata、最终传播源或观测时间。每张拓扑的一次搜索只生成一个场景池，所有候选共享；缓存仅在同一拓扑、场景池和评分配置内有效。保存的 `A` 与 `D_hop` 不一致时会直接报错，不能静默继续。

因此，原mc-SGBS测试的是“只知道拓扑和预先规定的等权时延先验时”的稳健部署。pmc-SGBS额外允许部署前知道 `low`、`medium` 或 `high` 离散档位，但仍禁止使用最终实现的实际SD数值或延迟矩阵。当前任务复用已有 `ppo_mix_v3_sd`，不重新训练PPO；`mc`和`pmc`只改变SGBS完整方案评分。

## 6. 评价指标与奖励

核心质量指标是传播源的平均排名 `rank`，数值越小越好。每次传播以真实源节点为目标，根据传感器观测对候选源排序，再对源节点、图实例和必要时的延迟实现取平均。

PPO 每一步选择传感器后的奖励为：

\[
r_t=\frac{2(\operatorname{rank}_{t-1}-\operatorname{rank}_t)}{N-1}.
\]

其中 `N` 为节点数。当前训练保持 `gamma=1.0`，使一轮累计奖励只由初始排名和最终排名的差决定，不受传感器选择顺序影响。网络直径作为图特征使用，不作为奖励分母。

结果还记录推理时间。对时间比较时，应区分：策略前向时间、搜索时间、定位评价时间以及总时间；不同脚本中字段名称以生成的 JSON 为准。

## 7. 方法说明

### 非神经网络基线

- `Random`：随机选择传感器，默认重复 100 次并报告均值和标准差；
- `BC`：基于 betweenness centrality 的选择；
- `HCR`：基于层次覆盖/可达性的启发式；
- `CB`：组合式结构评分方法；
- `HV_OBS`：基于有限观测层数的启发式，当前默认层数为 4。

后四种方法是确定性的，通常无需像 `Random` 一样重复多次。

### 神经推理方法

- `one_shot`：按 PPO 输出逐步贪婪选择，速度快，主要用于衡量策略本身质量；
- `hop-SGBS`：维护 `beta` 个候选分支，每次从策略的 `gamma` 个高概率候选中扩展，以跳数代理目标筛选完整方案；
- `mc-SGBS`：候选生成与hop-SGBS完全相同，但以独立模拟场景上的全部源平均rank筛选完整方案。场景数增加会近似线性增加场景生成和候选评分成本，不保证质量单调改善。
- `pmc-SGBS`：候选生成仍不变；部署前已知互斥SD档位，并只在该档位的独立模拟场景上评价候选。`--known-sd-level all` 会为同一拓扑预先生成三套条件部署并将最终测试实现按档位匹配；现实中的一次已知档位部署只使用其中一套。

## 8. PPO 默认训练配置

当前正式配置包括：

- 隐藏层：`[8, 16, 32, 64, 32, 16, 8]`；
- learning rate：`1e-4`；
- PPO discount `gamma=1.0`；
- GAE lambda：`0.95`；
- clip epsilon：`0.2`；
- 每次更新 epoch：`4`；
- 初始 entropy coefficient：`0.01`；
- entropy decay：`0.999`，下限 `0.001`；
- dropout：`0`；
- update interval：`512`；
- gradient norm clip：`0.5`；
- 单图组 on-policy rollout，默认每组 5000 episode。

训练日志会记录特征集、输入维度和奖励归一化方式。不要用旧模型覆盖当前正式模型，建议为每次结构或特征修改使用新的 `--model-id`。

## 9. 环境准备

在 PowerShell 中进入项目目录并激活实际存在的 Python 环境：

```powershell
cd SNAP
conda info --envs
conda activate <你的环境名>
pip install -r requirements.txt
```

如果不使用 Conda，也可直接调用已经安装依赖的解释器，例如：

```powershell
& python run_experiment.py --help
```

OpenMP 运行库冲突应优先通过统一 NumPy、PyTorch、MKL 等依赖来源解决，不建议把 `KMP_DUPLICATE_LIB_OK=TRUE` 作为正式实验设置。

## 10. 数据生成

数据结构、文件字段、随机延迟分层、生成命令和完整性检查见 [data/data.md](data/data.md)。旧式批量选项的生成范围与当前实验矩阵不同，正式数据应按该文档中的显式图类型和规模生成。

所有模块遵循同一个节点索引约定：矩阵行列 `i`、NetworkX节点 `i`、PPO动作
`i`、baseline/SGBS传感器 `i` 和传播源 `i` 表示同一个节点。生成和加载阶段
都会验证 `D_hop` 是否能由邻接矩阵按相同索引精确重算。数据重建后应先执行：

```powershell
python validate_graph_data.py
```

只检查真实网络及其全部延迟实现时：

```powershell
python validate_graph_data.py --graph-types WDN_Net1,WDN_Net2,WDN_Net3,IEEE_14Bus,IEEE_30Bus,IEEE_57Bus,IEEE_118Bus
```

## 11. 训练命令

### 11.1 无 SD 正式模型

```powershell
python run_experiment.py train --model-id ppo_mix_v3_no_sd --model-method PPOAgent --graph-types BA,WS,ER --train-sizes 20,50,100 --num-graphs 1000 --num-episodes 5000 --no-sd True --seed 42 --feature-set f3
```

### 11.2 SD 正式模型

```powershell
python run_experiment.py train --model-id ppo_mix_v3_sd --model-method PPOAgent --graph-types BA,WS,ER --train-sizes 20,50,100 --num-graphs 1000 --num-episodes 5000 --no-sd False --seed 42 --feature-set f3
```

### 11.3 特征消融

分别将模型 ID 与特征集改为 `f0`、`f1`、`f2`，例如：

```powershell
python run_experiment.py train --model-id ppo_mix_v3_no_sd_f0 --model-method PPOAgent --graph-types BA,WS,ER --train-sizes 20,50,100 --num-graphs 1000 --num-episodes 5000 --no-sd True --seed 42 --feature-set f0
```

`f1`、`f2` 同理。消融模型必须分别训练，不要加载 `f3` checkpoint。

## 12. 合成网络推理测试

以下命令都复用 `model_save/ppo_mix_v3_sd/ppo_mix_v3_sd.pt`，不会训练模型。one-shot、hop-SGBS、mc-SGBS和pMC-SGBS分别写入独立文件。正式对照时，同一规模的SGBS模式必须使用相同的 `beta/gamma`。

### 12.1 小规模功能验证

每类取2张20节点图，`k=3`，仅用于检查流程：

```powershell
& python run_experiment.py test --model-id ppo_mix_v3_sd --model-method PPOAgent --graph-types BA,WS,ER --test-sizes 20 --num-graphs 2 --fixed-k 3 --beta 4 --gamma 4 --no-sd False --no-baselines --neural-methods one_shot --seed 42 --feature-set f3 --sd-score-mode hop --results-dir result/ppo_mix_v3_sd --result-tag one_shot

& python run_experiment.py test --model-id ppo_mix_v3_sd --model-method PPOAgent --graph-types BA,WS,ER --test-sizes 20 --num-graphs 2 --fixed-k 3 --beta 4 --gamma 4 --no-sd False --no-baselines --neural-methods SGBS --seed 42 --feature-set f3 --sd-score-mode hop --results-dir result/hop_sgbs --result-tag hop_beta4_gamma4

& python run_experiment.py test --model-id ppo_mix_v3_sd --model-method PPOAgent --graph-types BA,WS,ER --test-sizes 20 --num-graphs 2 --fixed-k 3 --beta 4 --gamma 4 --no-sd False --no-baselines --neural-methods SGBS --seed 42 --feature-set f3 --sd-score-mode mc --search-scenarios 12 --search-seed 42001 --sd-prior equal_lmh --delay-mu 1.0 --results-dir result/mc_sgbs --result-tag mc_beta4_gamma4_M12
```

旧no-SD路径仍可使用同一checkpoint做兼容性检查；`mc`只允许SD模式：

```powershell
& python run_experiment.py test --model-id ppo_mix_v3_sd --model-method PPOAgent --graph-types BA --test-sizes 20 --num-graphs 1 --fixed-k 3 --beta 4 --gamma 4 --no-sd True --no-baselines --neural-methods one_shot,SGBS --seed 42 --feature-set f3 --sd-score-mode hop --results-dir result/ppo_mix_v3_no_sd --result-tag one_shot_and_hop
```

### 12.2 BA/WS/ER：20、50、100节点

```powershell
& python run_experiment.py test --model-id ppo_mix_v3_sd --model-method PPOAgent --graph-types BA,WS,ER --test-sizes 20,50,100 --num-graphs 100 --beta 4 --gamma 4 --no-sd False --no-baselines --neural-methods one_shot --seed 42 --feature-set f3 --sd-score-mode hop --results-dir result/ppo_mix_v3_sd --result-tag one_shot

& python run_experiment.py test --model-id ppo_mix_v3_sd --model-method PPOAgent --graph-types BA,WS,ER --test-sizes 20,50,100 --num-graphs 100 --beta 4 --gamma 4 --no-sd False --no-baselines --neural-methods SGBS --seed 42 --feature-set f3 --sd-score-mode hop --results-dir result/hop_sgbs --result-tag hop_beta4_gamma4

& python run_experiment.py test --model-id ppo_mix_v3_sd --model-method PPOAgent --graph-types BA,WS,ER --test-sizes 20,50,100 --num-graphs 100 --beta 4 --gamma 4 --no-sd False --no-baselines --neural-methods SGBS --seed 42 --feature-set f3 --sd-score-mode mc --search-scenarios 12 --search-seed 42001 --sd-prior equal_lmh --delay-mu 1.0 --results-dir result/mc_sgbs --result-tag mc_beta4_gamma4_M12
```

### 12.3 BA/WS/ER：200、300、500节点

当前正式候选组合为200节点 `(2,16)`、300节点 `(2,24)`、500节点 `(2,32)`。下列循环保证同一规模的hop与mc参数一致：

```powershell
$cases = @(@(200,16), @(300,24), @(500,32))
$cases | ForEach-Object {
    $size = $_[0]
    $searchGamma = $_[1]

    & python run_experiment.py test --model-id ppo_mix_v3_sd --model-method PPOAgent --graph-types BA,WS,ER --test-sizes $size --num-graphs 100 --beta 2 --gamma $searchGamma --no-sd False --no-baselines --neural-methods one_shot --seed 42 --feature-set f3 --sd-score-mode hop --results-dir result/ppo_mix_v3_sd --result-tag "one_shot_${size}"

    & python run_experiment.py test --model-id ppo_mix_v3_sd --model-method PPOAgent --graph-types BA,WS,ER --test-sizes $size --num-graphs 100 --beta 2 --gamma $searchGamma --no-sd False --no-baselines --neural-methods SGBS --seed 42 --feature-set f3 --sd-score-mode hop --results-dir result/hop_sgbs --result-tag "hop_${size}_beta2_gamma${searchGamma}"

    & python run_experiment.py test --model-id ppo_mix_v3_sd --model-method PPOAgent --graph-types BA,WS,ER --test-sizes $size --num-graphs 100 --beta 2 --gamma $searchGamma --no-sd False --no-baselines --neural-methods SGBS --seed 42 --feature-set f3 --sd-score-mode mc --search-scenarios 12 --search-seed 42001 --sd-prior equal_lmh --delay-mu 1.0 --results-dir result/mc_sgbs --result-tag "mc_${size}_beta2_gamma${searchGamma}_M12"
}
```

大图的SGBS候选数和每个候选的全源rank评价均更昂贵；mc又会对每个未缓存完整方案评价M个场景。运行成本会随 `M`、`beta`、`gamma`、节点数和预算增加，不应在未测量前假定线性总耗时。

### 12.4 pMC-SGBS条件先验评价

下面的命令复用同一个 `ppo_mix_v3_sd` checkpoint。每张拓扑分别按三个互斥条件先验生成三套部署方案，每档使用12个独立模拟场景；最终测试实现只与相同档位的方案匹配。结果单独保存在 `result/pmc_sgbs`，不会覆盖hop或mc结果：

```powershell
& python run_experiment.py test --model-id ppo_mix_v3_sd --model-method PPOAgent --graph-types BA,WS,ER --test-sizes 20,50,100 --num-graphs 100 --beta 4 --gamma 4 --no-sd False --no-baselines --neural-methods SGBS --seed 42 --feature-set f3 --sd-score-mode pmc --known-sd-level all --search-scenarios 12 --search-seed 42001 --delay-mu 1.0 --results-dir result/pmc_sgbs --result-tag pmc_all_beta4_gamma4_M12
```

如只验证一个已知档位，可将 `all` 改为 `low`、`medium` 或 `high`。此时结果仅包含该档位最终实现的评价。JSON中的通用 `ave_deployment_time` 表示一次已知档位部署的平均时间；`ave_conditional_total_deployment_time` 表示为了完成 `all` 三档论文实验而依次执行三套部署的总时间。

### 12.5 五种非神经网络基线

```powershell
python run_experiment.py test --model-id baselines --graph-types BA,WS,ER --test-sizes 20,50,100,200,300,500 --num-graphs 100 --no-sd True --only-baselines --seed 42

python run_experiment.py test --model-id baselines --graph-types BA,WS,ER --test-sizes 20,50,100,200,300,500 --num-graphs 100 --no-sd False --only-baselines --seed 42
```

若命令行参数或输出目录有疑问，以 `python run_experiment.py test --help` 和当前代码为准。运行后可执行：

```powershell
python summarize_baselines.py
```

## 13. 真实网络推理测试

真实网络必须先通过节点/矩阵一致性检查：

```powershell
& python validate_graph_data.py --graph-types WDN_Net1,WDN_Net2,WDN_Net3,IEEE_14Bus,IEEE_30Bus,IEEE_57Bus,IEEE_118Bus
```

只有检查成功后才运行下列评价。每个case内hop与mc严格复用同一 `beta/gamma`。下列命令给出推荐的质量—时间配置：IEEE-118使用 `(2,16)`，其余网络使用 `(4,4)`。需要注意，`result/hop_sgbs`、`result/mc_sgbs` 和 `result/pmc_sgbs` 中现存的真实网络三方对照实际全部使用 `(4,4)`；该批结果不能标记为IEEE-118 `(2,16)`：

```powershell
$cases = @(
    @('WDN_Net1',11,4,4), @('WDN_Net2',36,4,4),
    @('WDN_Net3',97,4,4), @('IEEE_14Bus',14,4,4),
    @('IEEE_30Bus',30,4,4), @('IEEE_57Bus',57,4,4),
    @('IEEE_118Bus',118,2,16)
)
$cases | ForEach-Object {
    $graphType = $_[0]
    $size = $_[1]
    $searchBeta = $_[2]
    $searchGamma = $_[3]

    & python run_experiment.py test --model-id ppo_mix_v3_sd --model-method PPOAgent --graph-types $graphType --test-sizes $size --num-graphs 1 --beta $searchBeta --gamma $searchGamma --no-sd False --no-baselines --neural-methods one_shot --seed 42 --feature-set f3 --sd-score-mode hop --results-dir result/ppo_mix_v3_sd --result-tag "one_shot_${graphType}"

    & python run_experiment.py test --model-id ppo_mix_v3_sd --model-method PPOAgent --graph-types $graphType --test-sizes $size --num-graphs 1 --beta $searchBeta --gamma $searchGamma --no-sd False --no-baselines --neural-methods SGBS --seed 42 --feature-set f3 --sd-score-mode hop --results-dir result/hop_sgbs --result-tag "hop_${graphType}_beta${searchBeta}_gamma${searchGamma}"

    & python run_experiment.py test --model-id ppo_mix_v3_sd --model-method PPOAgent --graph-types $graphType --test-sizes $size --num-graphs 1 --beta $searchBeta --gamma $searchGamma --no-sd False --no-baselines --neural-methods SGBS --seed 42 --feature-set f3 --sd-score-mode mc --search-scenarios 12 --search-seed 42001 --sd-prior equal_lmh --delay-mu 1.0 --results-dir result/mc_sgbs --result-tag "mc_${graphType}_beta${searchBeta}_gamma${searchGamma}_M12"
}
```

真实网络每张拓扑有30个最终测试实现；这些实现只在部署完成后评价。基线测试仍使用 `--model-id baselines --only-baselines`，不需要 `beta/gamma`。

## 14. `beta`、`gamma` 质量—时间权衡

- `beta`：每一层保留的搜索分支数，增加后通常同时提高搜索质量和内存、时间开销；
- `gamma`：每个分支尝试的高概率候选数，增加后扩展数和定位评价次数近似线性增加；
- 单层候选规模大致与 `beta × gamma` 成正比，因此两者同时增加会显著延长推理时间。

参数扫描使用独立脚本，具体参数以帮助信息为准：

```powershell
python sweep_sgbs_tradeoff.py --help
```

正式报告应同时给出平均 `rank`、平均总时间以及相对最佳非神经 baseline 的改善率，不应只选择质量最优但耗时不可接受的参数。

## 15. 结果目录和主要字段

- `model_save/<model_id>/`：checkpoint、训练 return、loss 和配置记录；
- `result/ppo_mix_v3_no_sd/`、`result/ppo_mix_v3_sd/`：PPO v3直接推理结果；
- `result/baselines/`：非神经 baseline 的 JSON 与 `baselines_summary.csv`；
- `result/ppo_ablation_f0_no_sd/`、`result/ppo_ablation_f1_no_sd/`、`result/ppo_ablation_f2_no_sd/`：特征消融结果；
- `result/hop_sgbs/`、`result/mc_sgbs/`、`result/pmc_sgbs/`：hop、mc与pMC正式对照结果，覆盖20至500节点合成网络及七个真实网络；
- 参数扫描结果由 `sweep_sgbs_tradeoff.py` 的输出参数决定。

常见结果字段包括图类型、节点数、传感器预算、方法名、平均 rank、rank 标准差、平均时间、时间标准差、是否使用 SD、延迟实现数量、`beta` 和 `gamma`。其中：

- `rank_std`：重复图实例、随机运行或延迟实现上的 rank 离散程度；
- `time_std_seconds`：对应运行样本的推理时间标准差；
- SD 与 no-SD 的结果必须按相同图、规模、预算和方法配对比较；
- `Random` 的均值与标准差来自多次随机选择，不能拿单次偶然结果作结论。

新的神经推理JSON还包含：

- `run_config.checkpoint`：checkpoint路径、文件大小和SHA-256；
- `run_config.data`：数据路径、图类型、规模、拓扑数及拓扑/metadata标识；
- `run_config` 中的推理方法、f3、主seed、预算、`beta/gamma`、评分模式、prior、M和search seed；
- `placements_by_topology`：每张拓扑实际选择的传感器索引；
- `search_pool_fingerprints`：mc模式每张拓扑的独立场景池标识；
- `timing_semantics`：每个时间字段的包含关系。

时间字段不能简单相加：`search_scoring_time` 是 `search_time` 的子项；二者以及 `search_scenario_generation_time` 都已包含在 `deployment_time` 中。最终 `evaluation_time` 不属于部署，`time = deployment_time + evaluation_time`。

## 16. 回归测试

图特征或奖励逻辑修改后至少运行：

```powershell
& python -m unittest test_feature_reward.py test_graph_indexing.py test_sd_scenarios.py test_sd_sgbs.py -v
```

测试覆盖特征形状、有限性、重标号置换等变性、特殊图除零、累计奖励，
以及插入顺序错位复现、真实图写入、多延迟生成、五个baseline、one-shot、
SGBS和最终排名的统一索引，以及独立场景复现、随机流隔离、hop/mc评分、缓存作用域、最终测试信息隔离和结果文件命名。

## 17. 实验解读注意事项

1. `one_shot` 是衡量 PPO 策略质量的主要指标，`SGBS` 是搜索增强后的系统性能。
2. 在 SD 场景中，模型和 SGBS 选择传感器时不知道具体延迟实现，不能把加权距离矩阵作为测试输入。
3. 不同规模使用不同固定预算，跨规模比较原始 rank 时应同时给出相对改善率或归一化指标。
4. 大图 SGBS 的搜索参数与小图不同，运行时间比较必须同时标注 `beta`、`gamma`。
5. 当前mc/pMC-SGBS复用原 `ppo_mix_v3_sd`，没有把模拟场景或sd加入PPO输入；它们只改变完整候选方案评分。

