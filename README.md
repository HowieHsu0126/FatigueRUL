# FatigueRUL

FatigueRUL 是一个**沥青疲劳剩余寿命预测**项目。它根据试验中采集的传感器时序数据，通过程序“学习”历史规律，预测**材料在失效前还能承受多少加载周期**（即剩余寿命，RUL）。本说明面向不熟悉深度学习的用户，力求用通俗语言说明项目在做什么、如何运行、以及结果在哪里查看。

---

## 一、项目在做什么（通俗说明）

- **输入**：沥青疲劳试验的原始数据（一般为 `Input/raw/data.mat`），里面包含传感器随时间记录的信号。
- **中间过程**：程序会先根据物理关系从信号算出**弹性模量**（可理解为材料“刚度”的指标，会随疲劳逐渐下降），再根据“刚度降到初始值的多少算失效”的规则，为每段数据标出**剩余寿命**（RUL）。这些带标签的数据被切成一段段时序，供后续“学习”使用。
- **输出**：程序会用多种**预测方法**（见下文“模型类型”）从一段时序预测 RUL；训练完成后会得到**预测模型**（保存为文件）和**评估指标**（如误差、R² 等），便于比较哪种方法更准。

您不需要理解“神经网络”或“深度学习”的具体原理，只需知道：**程序在利用历史数据学习“看到一段曲线 → 预测还剩多少寿命”的规律，并把学到的规律存成模型，用于对新数据做预测。**

---

## 二、使用前需要准备什么

1. **原始数据**：将试验得到的原始数据文件放在 `Input/raw/` 下，默认文件名为 `data.mat`。若使用其它路径或文件名，需在配置中修改（见“配置说明”）。
2. **运行环境**：本机需已安装 **Python 3**，并安装项目依赖（如 PyTorch、numpy、pandas、scipy、yaml 等）。推荐使用 conda 创建独立环境，避免与其它项目冲突。
3. **工作目录**：后续所有命令都请在**项目根目录**下执行（即包含 `Libs`、`Input`、`Output`、`run_perf_compare.py` 的那一层目录）。

---

## 三、环境与依赖

项目依赖 Python 3 及若干科学计算与深度学习库（如 PyTorch、numpy、pandas、scipy、PyYAML 等），具体以 `Libs` 下代码实际引用为准。建议使用 conda 环境并先激活再运行脚本。

**激活 conda 环境示例**（请按您本机实际路径调整）：

```bash
source /share/home/hwxu/miniconda3/bin/activate hw
```

若上述命令无效，可尝试：

```bash
source /home/xuhaowei/miniconda3/etc/profile.d/conda.sh && conda activate hw
```

激活后，在项目根目录执行下文中的 `python3 -m ...` 命令即可。

---

## 四、操作步骤（按顺序执行）

### 4.1 第一步：准备数据集

**作用**：从原始 `.mat` 中读取数据，用物理模型计算弹性模量，再根据“失效阈值”为每段数据生成剩余寿命标签，并做清洗、滑窗、归一化等处理，得到供训练使用的表格和序列。  
**命令**（在项目根目录执行）：

```bash
python3 -m Libs.scripts.prepare_dataset
```

**完成后会生成**：  
- 主要在 `Input/processed/` 下，例如 `training_dataset.csv` 以及其它中间结果（若配置了保存序列等）。  
若未指定其它路径，原始数据路径由 `Libs/config/dataset.yaml` 中的 `data_paths.raw_data` 决定。

---

### 4.2 第二步：准备图数据

**作用**：为其中一种预测方法（“图神经网络”，即利用应力传播等关系把样本组织成图结构）准备图数据。  
**命令**（在项目根目录执行）：

```bash
python3 -m Libs.scripts.prepare_graph_data
```

**完成后会生成**：  
- `Input/graph_processed/graph_data.pt`  
- `Input/graph_processed/graph_metadata.json`  

只有需要训练“图模型”或运行完整流程时，才必须执行本步。

---

### 4.3 第三步：训练模型

**作用**：用前面准备好的数据，让程序反复学习“时序 → 剩余寿命”的映射，并把学好的**模型**保存下来，同时在**测试集**上计算误差、R² 等指标。

您可以选择：

| 目的           | 命令 |
|----------------|------|
| 训练所有已启用的深度学习模型 | `python3 -m Libs.scripts.train_all_models` |
| 只训练图神经网络（GNN）     | `python3 -m Libs.scripts.train_gnn`        |
| 只训练某一种时序模型（如 LSTM） | `python3 -m Libs.scripts.train_dl_baseline`（具体模型名由脚本/配置决定） |

**训练完成后**：  
- **模型文件**：保存在 `Output/models/` 下（如 `best_xxx_model.pth`）。  
- **指标结果**：保存在 `Output/results/` 下（如 `all_dl_models_results.json`、各模型的 `*_metrics.json`）。  
- **运行日志**：在 `Output/logs/` 下（如 `experiment.log`，若通过 SLURM 提交则还有 `train_<jobid>.out` / `train_<jobid>.err`）。

若您在**集群上使用 SLURM** 提交作业，可使用项目提供的脚本（示例）：

```bash
bash Libs/scripts/train.sh
```

具体参数（如分区、时长、GPU）见脚本内注释或 `train.sh -h`。

---

### 4.4 第四步：性能对比（可选）

**作用**：对比两种策略下的预测效果——  
- **Baseline**：不启用“样本加权”、失效前样本不额外复制；  
- **Mitigation**：启用样本加权，并对失效前样本做 2 倍过采样。  

脚本会训练相应模型并输出测试集上的 RMSE、MAE、R²、R²(RUL>0) 等，便于判断“加权+过采样”是否提升表现。

**命令**（在项目根目录执行）：

```bash
python3 run_perf_compare.py
```

默认只对比**图神经网络（GNN）**。若希望同时对比 **LSTM**，可先设置环境变量再执行：

```bash
RUN_LSTM=1 python3 run_perf_compare.py
```

结果会直接打印在终端，同时训练得到的模型与指标仍会写入 `Output/models/` 与 `Output/results/`。

---

## 五、结果文件在哪里看

| 内容         | 位置 |
|--------------|------|
| 训练好的模型   | `Output/models/`（如 `best_xxx_model.pth`） |
| 各模型评估指标 | `Output/results/`（如 `all_dl_models_results.json`、`*_metrics.json`） |
| 程序运行日志   | `Output/logs/`（如 `experiment.log`，SLURM 任务为 `train_<jobid>.out` / `.err`） |

指标文件为 JSON 格式，可用文本编辑器或脚本打开；其中常见名称含义见下文“名词解释”。

---

## 六、配置说明（不用改代码，只改配置）

项目里**所有重要参数和路径都通过 YAML 配置文件**设置，无需改 Python 代码。

| 文件 | 主要作用 |
|------|----------|
| `Libs/config/dataset.yaml` | 原始数据路径、预处理方式（清洗、滑窗长度、步长）、归一化方法、疲劳失效定义（如刚度降到初值的多少算失效）、图数据路径等。 |
| `Libs/config/exp.yaml`     | 实验名称与说明、是否使用 GPU、训练轮数、批大小、损失函数、是否启用样本加权、验证/测试划分、评估指标等。 |
| `Libs/config/model.yaml`   | 各“模型类型”的结构参数（如层数、隐藏维度等）；若需调整网络深度、宽度等，可在此修改。 |

修改配置后，重新执行对应的“准备数据”“准备图数据”或“训练”步骤即可生效。

---

## 七、目录结构一览

便于您快速知道“东西放在哪”：

```
项目根目录/
├── Input/                     # 输入与中间数据
│   ├── raw/                   # 原始数据（如 data.mat）
│   ├── processed/             # 第一步生成：处理后表格、序列等
│   └── graph_processed/       # 第二步生成：图数据与元信息
├── Libs/                      # 项目代码与配置
│   ├── config/                # 配置文件（dataset / exp / model 的 YAML）
│   ├── data/                  # 数据加载、图构建、标签生成、物理计算等
│   ├── exps/                  # 训练与评估逻辑
│   ├── models/                # 各类预测模型（网络、层、基线）
│   ├── scripts/               # 可执行脚本：数据准备、训练入口、全流程
│   └── utils/                 # 工具（如可解释性分析）
├── Output/                    # 所有输出
│   ├── logs/                  # 日志
│   ├── models/                # 保存的模型文件
│   └── results/               # 评估指标（JSON）
├── run_perf_compare.py        # 性能对比脚本（baseline vs mitigation）
└── README.md                  # 本说明
```

各目录下 `.keep` 文件仅用于在版本控制中保留空目录结构，可忽略。

### Input / Output 各文件夹应包含的文件

| 目录 | 应包含的文件 | 说明 |
|------|----------------|------|
| **Input/raw/** | `data.mat`（或配置中指定的其它 .mat 文件名） | 试验原始数据：传感器时序（力、位移、时间等），由 `Libs/config/dataset.yaml` 的 `data_paths.raw_data` 指定路径。 |
| **Input/processed/** | `training_dataset.csv`、`physics_results.pkl`、`sequences.npz`、`metadata.pkl`、`processed_metadata.json` 等 | 执行「准备数据集」后生成：带 RUL 标签的训练表、弹性模量等物理结果、滑窗序列与元数据；具体文件名以 `dataset.yaml` 的 `data_paths` 为准。 |
| **Input/graph_processed/** | `graph_data.pt`、`graph_metadata.json` | 执行「准备图数据」后生成：图结构样本与元信息，供 GNN 训练使用。 |
| **Output/logs/** | `experiment.log`；若用 SLURM 提交则有 `train_<jobid>.out`、`train_<jobid>.err` | 程序运行日志与任务标准输出/错误输出。 |
| **Output/models/** | `best_xxx_model.pth`（如 `best_lstm_model.pth`、`best_gnn_model.pth`）、`best_model.pth` | 训练完成后保存的模型权重；具体名称由 `model.yaml` 的 `model_paths` 及训练脚本决定。 |
| **Output/results/** | `all_dl_models_results.json`、`lstm_metrics.json`、`gru_metrics.json` 等 `*_metrics.json` | 各模型在验证/测试集上的评估指标（RMSE、MAE、R² 等），JSON 格式。 |

---

## 八、名词解释（方便阅读文档与结果）

- **剩余寿命（RUL）**：在当前状态下，材料在失效前还能承受的加载周期数（或等效时间）。预测目标就是 RUL。
- **弹性模量**：反映材料刚度；随疲劳发展会下降。程序中用物理模型从传感器信号反算弹性模量，再据此判断“何时算失效”并生成 RUL 标签。
- **训练**：程序根据历史数据（带 RUL 标签）反复调整内部参数，使预测值尽量接近真实 RUL；这一过程叫“训练”。
- **模型**：训练后得到的“预测规则”，保存成文件（如 `.pth`），可用来对新数据做 RUL 预测。
- **基线（baseline）**：一种简单或默认的预测方法，用作对比基准。
- **过采样（oversample）**：对“失效前”等少数样本多复制几份参与训练，以减轻数据不平衡。
- **样本加权（sample weighting）**：在损失计算时给某类样本（如 RUL>0）更大权重，让模型更关注这类样本。
- **RMSE / MAE / R²**：常用误差与拟合指标——RMSE、MAE 越小越好，R² 越接近 1 越好；结果文件里会出现这些缩写。

---

## 九、模型类型（供进一步了解）

若您想大致知道“程序里有哪些预测方法”，可参考下表（无需掌握实现细节）：

| 类型说明     | 简要说明 |
|--------------|----------|
| LSTM / GRU   | 基于时序的循环网络，适合处理连续时间序列。 |
| CNN-LSTM     | 先用卷积提取局部特征，再用 LSTM 建模时序。 |
| TCN          | 时间卷积网络，用因果卷积处理时序。 |
| Transformer  | 基于注意力机制的序列模型。 |
| Attention-LSTM | 带注意力机制的 LSTM。 |
| 时空 GNN     | 图神经网络，利用应力传播等关系把样本组织成图进行预测。 |
| 物理基线     | 基于物理公式的简单预测，用作对比。 |

具体启用哪些模型、训练多少轮等，均由 `Libs/config/exp.yaml` 与 `Libs/config/model.yaml` 控制。

---

## 十、小结

1. 把原始数据放到 `Input/raw/`（默认 `data.mat`），激活 conda 环境，在项目根目录执行：**准备数据 → 准备图数据 → 训练**；可选执行 **性能对比**。  
2. 所有可调参数在 `Libs/config/` 的 YAML 中修改，无需改代码。  
3. 训练好的模型在 `Output/models/`，指标在 `Output/results/`，日志在 `Output/logs/`。  
4. 若遇到报错，可先查看 `Output/logs/` 中的日志；若在集群上跑，可查看 SLURM 的 `.out` / `.err` 文件。

若您希望增加“常见报错与处理”或“如何用保存的模型对新数据预测”等小节，可以在此基础上再补充。
