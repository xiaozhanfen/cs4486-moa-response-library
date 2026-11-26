# Topic 2 · MoA Response Library · 最终项目

本目录是本组最终保留的项目。已评分版本为 **MoA Response Library / Version 1**，提交号 **57940904**。模型从随机权重训练，预测 206 个药物作用机制概率。历史项目归入同级 `../archive/`。

| 官方结果，log loss 越低越好 | 本项目 | 小组给出的历史第 5 名门槛 | 比较 |
|---|---:|---:|---|
| Public | 0.01814 | 0.01805 | 还高 0.00009 |
| Private | 0.01590 | 0.01602 | 低 0.00012，优于门槛 |

可以陈述“Private 优于历史第五名分数，Public 接近”，不能陈述“两榜均超过 Top 5”或已获得官方第五名排名。正式凭证在 [competition_score.json](evidence/kaggle_v1/competition_score.json)。课程文件内写的是 CS4486，目录沿用所在课程工作区的名称。

**先读 [架构讲解 ARCHITECTURE_CN.md](ARCHITECTURE_CN.md)**。它按任务、输入、CNN、MLP、药物库、概率融合、训练验证和运行顺序讲解，附流程图、尺寸、小例子和答辩练习。

## 先跑一次预测

本机已经配置好 `cs4486` 环境，无需重新安装。进入本目录后执行：

```powershell
conda activate cs4486
python predict.py
```

输出为 `outputs/submission.csv`，公开测试输入有 **3,982 行 × 206 个概率**，另加 `sig_id` 列。默认使用 CPU、随项目提供的压缩权重，从真实输入重新计算；训练集、验证预测和评分凭证都不会被推理入口读取。

如 Conda 没有初始化，可在本机使用：

```powershell
& 'D:\anaconda\envs\cs4486\python.exe' predict.py
```

组员更换数据目录或使用显卡时：

```powershell
python predict.py --data-dir "D:\your-data" --output outputs/submission.csv --device cuda
```

输入目录需要 `test_features.csv` 和 `sample_submission.csv`。列名与官方格式相同，输入行数可以改变，`sig_id` 可以是新 ID。输出按模板 ID 排序；模板 ID 与本次输入不同则按本次输入排序，仍使用模板定义的 206 个标签列。

## 检查与重新训练

```powershell
python verify_project.py
python baseline.py
```

第一条核对已评分包的 SHA-256，用整理后的代码与原始 Kaggle Notebook 分别推理，并检查新 ID、缺少部分时间/剂量类别、模板乱序、旧模板、全对照和空输入。结果写入 `evidence/final_verification.json`。第二条计算同一分折上的训练折标签频率基线，结果写入 `outputs/baseline_frequency/`。

已经训练好的最终权重可以直接使用。需要从头重训时，执行以下分阶段命令，产物默认写入 **`reproduction/`**：

```powershell
python reproduce.py cv
python reproduce.py study
python reproduce.py audit
python reproduce.py full
python reproduce.py package
```

也可 `python reproduce.py all` 顺序执行全部阶段。`cv` 是三种配置各 5 种子、每种子 5 折，共 75 个网络。`study` 计算药物库 OOF、交叉拟合融合与消融；`audit` 检验增大训练集是否有收益；`full` 在全部处理样本上训练三种配置各 8 种子，共 24 个部署网络；`package` 压缩并验证新的发布包。精确配置在 `recipes.json`，训练使用 CUDA BF16，RTX 5080 可以运行。

可用 `--data-dir` 指向完整六个官方 CSV 的目录，用 `--work-dir` 选择新的训练产物目录。已有完整 CV 配置可用 `cv --resume` 跳过，未完成的配置不会被覆盖。阶段依赖上一阶段产物；完整复现应从 `cv` 开始。随机种子与依赖已固定，GPU 算子和不同设备仍可能带来数值差异，不能保证重训逐字节等于原权重。

## 文件怎么分工

| 文件或目录 | 用途 |
|---|---|
| `predict.py` | 本地预测入口，读取实际输入，写 CSV |
| `models.py` | ResponseCNN、ProfileMLP、训练损失 |
| `features.py` / `response_features.py` | MLP 的 1,032 维特征 / CNN 的 937 维特征 |
| `drug_library.py` | 建立训练药物响应原型，按相似度软投票 |
| `infer_engine.py` | 加载模型、恢复特征、计算概率、融合、按 ID 写出 |
| `trainer.py` / `folds.py` / `metrics.py` | 单配置五折训练 / 冻结分折核对 / log loss |
| `response_study.py` | 同折对照、融合参数交叉拟合、消融和药物 bootstrap |
| `refit_response.py` | 训练规模检验、按 CV 轮数全量重训 |
| `response_release.py` / `artifact_utils.py` | 新发布包构建 / 通用 INT8 储存等工具 |
| `reproduce.py` / `recipes.json` | 完整重训入口 / 三种训练配置 |
| `baseline.py` / `verify_project.py` | 频率基线 / 独立运行及预测一致性检查 |
| `kernel.py` / `notebooks/` | Kaggle 入口源码 / 便于阅读的 Notebook |
| `response_release/` | **已正式评分的 V1 原包，保留原始字节** |
| `data/raw/` / `data/folds.csv` | 本地官方原始数据 / 固定五折清单 |
| `models/` / `runs/` | 本地保留的相关训练权重、OOF、配置与实验记录 |
| `evidence/` | 评分、发布核对、原配置和本次整理的验证凭证 |
| `course_materials/` | 作业要求、报告和展示模板 |

顶层代码已移除旧三阶段堆叠、旧网络和跨历史目录导入。`response_release/` 内是原提交的冻结快照，其历史兼容分支为保留哈希而原样存放。日常读代码从顶层文件开始。

## 分享包和 Kaggle

`../Topic2_MoA_Final.zip` 是给组员的项目分享包，含训练和推理代码、讲解、已评分权重、固定分折、课程模板及关键实验和评分凭证。它不含原始 CSV、可重建的 FP32 训练权重或大批 OOF 数组。解压后补齐 `data/raw/` 即可预测和从头训练。当前本地目录另外保留了这些训练产物。

正式 Kaggle 输入只用 `response_release/dataset/`，总量 **113.96 MB**；Notebook 在 `response_release/notebook/`。无需上传整个项目。Notebook 关闭 Internet、使用 CPU，实际读取 Kaggle 当次提供的测试数据。现有 [Notebook V1](https://www.kaggle.com/code/xiaozhanfen/moa-response-library) 已评分，本次整理没有再次提交。

`evidence/kaggle_v1/submission.csv` 仅为已下载的云端输出凭证。隐藏集预测由 Kaggle 重跑入口产生，本地没有隐藏标签，也不能从这份公开输入 CSV 重算官方 Private 分数。

新电脑可使用 Python 3.11，安装 PyTorch 后执行 `python -m pip install -r requirements.txt`。本次验证环境为 Python 3.11.17、PyTorch 2.11.0+cu128、CUDA 12.8、RTX 5080 16 GB，其余版本固定在依赖文件。CPU 推理也支持对应 CPU 版 PyTorch；精确核验环境见 `evidence/`。项目包不含 Kaggle key。

## 报告与归属

方法出处和本项目改动见架构讲解最后部分。课程要求按模板写报告、解释验证与失败案例、披露 AI 参与、填写成员实际贡献及工时。这里提供的是项目代码与学习讲解，正式报告仍需组员根据实际工作完成。模型和实现过程中有 AI 参与，不能将其全部写成组员独立完成；也不能把阅读懂了代码等同于满足课程对独立完成的要求。
