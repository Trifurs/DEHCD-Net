# 实验协议与训练稳定性

本协议定义训练、验证、测试与多种子统计流程。运行前核实数据划分、标签含义和裁块规则；不同实验条件分别保存。

## 功能

| 功能 | 实现与边界 |
| --- | --- |
| 多随机种子、公平预算与精度 | `tools/run_multiseed.py` 为每个模型/种子启动独立进程；默认 42、1051、2060、3069、4078；同一数据集固定训练 epoch，对比配置关闭早停、统一 FP32（AMP=false），避免基线固定 FP32 而 DEHCD 使用 AMP；均值和样本标准差 ddof=1 |
| 对比模型 | 4 个数据集 × (S/M/L + 9 个对比模型) = 48 个主实验配置；明确标识为项目适配版本 |
| 模块消融 | 独立控制 HOG、DPM、flow、difference gate、modality gate、BiCSF、GCB、IRB；名称由配置产生 |
| 参数量控制 | `plain_matched` 用实际参与计算的两层卷积替代 DPM；自动选择隐藏宽度，记录逐层实际参数与误差；另有 HOG 同容量 intensity prior、BiCSF/GCB 近似容量匹配局部卷积、IRB 等参数前馈对照；不宣称 FLOPs/感受野相等 |
| 损失与采样对照 | 基础 CE+Dice、完整策略、仅换损失、去加权采样、去定向裁块、去类别权重 6 组配置；Haiti 的 binary/subclass 权重也独立去除 |
| 敏感性 | BRIGHT/Haiti 上 HOG bins 2/4/6/8/10，IRB steps 0/1/2/3/4 |
| baseline 适配影响 | 四个数据集提供保留原 BN 的 36 个对照；仍有输入适配与多类输出适配，不等于官方训练协议 |
| 结果溯源 | 源码、完整配置、种子、环境、数据清单、checkpoint SHA256、原始混淆矩阵；聚合拒绝 val、截断测试、配置/数据/权重/指标不一致 |
| 跨事件验证 | 显式 CSV 清单支持重组原始目录的样本；按事件划分，禁止 source group 跨 split；不从文件名猜测地理位置 |
| 测试一致性 | train 的验证不使用 TTA；test/evaluate/infer 共用 TTA 实现；主对比实验显式 `none`，历史配置中的 flips 仍生效 |
| 效率 | 保留参数计数，但旧 hook 统计不覆盖 attention/SSM，完整 MACs/FLOPs 字段改为 null，部分算子估计单独标注；另有独立设备同步的 latency/throughput/峰值显存 benchmark |

`configs/experiments/catalog.json` 共列出 174 个配置。全量 5 种子需要 870 次训练，**不要把全量命令当作冒烟测试**。当前采用 `controlled_comparison_v1`；BiCSF 完整消融同时关闭 GCBM/WASM，额外定位监督移至独立对照。主协议、对照参照关系和公平性检查见 [FAIR_COMPARISONS.md](FAIR_COMPARISONS.md)。不同协议的结果不能合并续跑。
`tools/build_experiment_configs.py` 可以重新生成这些配置；重生成会覆盖该工具生成的 XML，应先保存手工定制。

## 建议运行顺序

以下命令在仓库根目录执行。先用当前可用的 PyTorch 环境运行测试：

```bash
python -m unittest discover -s tests -v
python tools/run_multiseed.py --suite main --datasets haiti \
  --data-root haiti=/path/to/Haiti1 --output runs/experiments/haiti_main --dry-run
python tools/run_multiseed.py --suite main --datasets haiti \
  --data-root haiti=/path/to/Haiti1 --output runs/experiments/haiti_main --preflight-only
```

预检只建立/核对协议，不训练。确认后，用相同选项加 `--resume` 开始训练：

```bash
python tools/run_multiseed.py --suite main --datasets haiti \
  --data-root haiti=/path/to/Haiti1 --output runs/experiments/haiti_main --resume
```

可先只选择 `--experiments haiti_dehcd_s`，用独立 output 验证流程，再启动完整实验。
BRIGHT、CAU-Flood、xBD 的 root 参数键分别是 `bright`、`cau_flood`、`xbd`。
`--suite ablation recipe sensitivity adapter` 可选多个类别；`--seeds` 可显式指定种子列表。
所有对比配置显式关闭 AMP；原基线核心一直固定 FP32，因此同精度比较需 DEHCD 也使用 FP32。效率工具拒绝为固定 FP32 基线请求 `--amp`，跨模型效率表统一省略该选项。RTX 5090 使用 `--runtime-profile rtx5090`，详见 [运行参数](RUNTIME_PROFILE.md)。其他设备的物理 batch 与梯度累积须对整个比较组统一调整。
默认 epochs 为 BRIGHT/CAU/xBD 100、Haiti 1000；如需调整，用 `--epochs` 并为整个同数据集比较设置同一预算。

一个 campaign 绑定代码、解析后的配置、种子和数据；改变这些内容必须使用新的 output，不能把新旧条件的种子拼在一起。
默认数据指纹使用文件路径/大小/mtime（速度快，**不是内容级完整性证明**）；正式归档可用 `--fingerprint sha256`，会读取全部影像。
多种子改变初始化、采样和增强，**不重划分数据集**。
启用确定性选项、显式 worker/sampler RNG，并按 epoch 重置随机流；CUDA 的 flow/grid_sample 反向仍可能存在非确定性，因此不保证跨设备逐位一致。

输出包括：

- campaign：`protocol.json`、`progress.json`、`progress.txt`、`summary/per_seed.csv`、`summary/aggregate.json`；缺失种子标为 incomplete，n=1 的 SD 为 null。
- 每次训练：`config_snapshot.json`、`protocol.json`、`history.jsonl`、`training_summary.json`、`checkpoints/best.pth`、`checkpoints/last.pth`。
- 每次测试：完整结果 JSON、原始混淆矩阵、逐样本混淆矩阵；有事件元数据时输出逐事件指标。

单次训练中 `best.pth` 始终保存严格最大验证分数，早停 min_delta 只控制 patience；每轮保存 `last.pth`。
恢复必须在原 run 目录，使用相同的科学配置。`--resume` 未指定 `--run-dir` 时自动识别原目录：

```bash
python tools/train.py --config path/to/run/config_snapshot.json \
  --resume path/to/run/checkpoints/last.pth
python tools/validate_results.py path/to/test_result.json --verify-checkpoints
```

故障训练没有完成标记，不能进入正式统计。批量入口会在协议不变时从 `last.pth` 恢复；若初始化阶段失败且没有检查点和历史记录，可在保留控制台日志的原目录重试。每次启动重新校验源数据，代码或科学配置有变化仍需新的 output。目录、进度及共享准备流程见 [运行管理](RUN_WORKFLOW.md)。
验证集用于选权重/超参数；测试集只对预先确定的方案做最终评价。

## Haiti 训练退化

稳定性措施：

1. 所有分割损失（包括层次损失和辅助输出）在关闭 autocast 的 FP32 区间计算。避免半精度输入下 CE、概率、面积、先验项等大规模累加的数值风险；空有效掩膜返回保留计算图的零损失。
2. Haiti 显式设置 `model.group_norm_eps=1e-3`，对网络及基线适配器中的所有 GroupNorm 生效；原 BN 对照的 BN epsilon 保留上游值。真实样本 1313 的光学输入归一化后为常量，SAR 仍有效。同一 4 样本 batch 和固定初始化下，GN epsilon 从 1e-5 调到 1e-3，裁剪前梯度范数从约 84631 降到 61.8。这是对低方差梯度放大的控制实验，不证明所有历史失败均由此引起。GRN 空间归约显式 FP32。Haiti 默认关闭模型前向 AMP，对所有模型采用相同设置；如另测 AMP，应使用独立配置/目录。
3. 每个 batch 检查输入、logits 和 loss；每次优化前 unscale、记录裁剪前梯度范数并裁剪。非有限梯度绝不更新参数；AMP 下允许有限次数降 scale 后重试后续 batch，连续异常则终止。
4. `history.jsonl` 保存梯度范数、优化步数、AMP 跳步/scale、每类 IoU、预测/真实前景比例。梯度尖峰或非有限梯度还会在 `gradient_spikes.jsonl` 中记录样本及范数最大的 10 个参数层。异常写入 `failure.json`，不产生正常完成标记。
5. 默认在已学到有效前景后，连续 5 次验证出现接近全背景的严重退化时终止；保护 `best.pth`，避免继续消耗数百轮。它不把一般验证波动误判为崩溃，也不把“全背景”直接诊断成“梯度爆炸”。
6. Haiti 默认 `hier_binary_reduction=class_mean`、binary 权重 `[1,1]`：分别计算有效背景/前景的二分类 CE，再按两个类别平均，避免背景像素数量支配 CE。其余损失项保留。这是明确的训练策略变化，不应归为架构增益。`haiti_legacy_pixel_mean` 保留旧的像素均值与 `[1,2.8]`，`haiti_pixel_mean_equal_weights` 仅保留像素均值，供隔离 reduction/权重效应；两者仍使用修复后的数值保护。
7. 增加前景停滞保护：Haiti 从第 40 轮起，若连续 10 次验证的预测前景不足真实前景的 1%，即使此前从未学到高分也会报错停止。旧的崩溃保护继续针对“曾学会、后丢失”的情况；计数随 checkpoint 恢复。失败不会生成正常完成标记，也不会自动更改预测阈值。
8. 修正最后一个不足整组的梯度累积窗口被过度缩小的问题。

诊断顺序：

- loss/logits 非有限：查看失败 batch 的样本 id、输入范围及是否 AMP；先用默认 FP32 复测。
- `grad_norm_max_before_clip` 很大且跳步密集：检查学习率、输入异常和损失权重；不要仅看裁剪后梯度。
- loss/梯度有限，但 `pred_foreground_ratio` 接近 0、真实前景仍存在：这是背景退化；检查层次损失的定位/分类权重、采样与数据标签，再用 recipe 实验隔离。
- 类别权重、采样、loss、学习率的任何调整均需固定后对相应比较组重跑，不能仅挑选有利种子。

CPU 单元检查可复现旧损失接受 `24×4×128×128` FP16 logits 时产生非有限值，新实现的 loss/gradient 有限。
这不是对历史 CUDA AMP 崩溃根因的证明：PyTorch autocast 本身会将部分算子提升为 FP32，实际 dtype 路径及旧日志没有完整记录。
参考 [PyTorch AMP](https://docs.pytorch.org/docs/stable/amp.html) 与 [AMP 梯度裁剪示例](https://docs.pytorch.org/docs/stable/notes/amp_examples.html#gradient-clipping)。
在 RTX 5090 的额外受控短测中，修复 GN 后的旧 pixel-mean 损失仍会在梯度有限时变成全背景，新的 class-mean 设置避免了该短测中的停滞。设置和局限见 [验证记录](VALIDATION.md)。完整训练、多种子收敛与精度仍须验证；新旧策略必须使用不同 output。

## 数据准备与跨事件清单

BRIGHT/xBD 裁块工具现在默认保留所有 split 的背景块。`--drop-background-train` 只过滤 train，val/test 始终保留。
这一修改**不能找回已有数据目录中早先删除的背景块**；需要从原图生成新的数据版本，保留旧数据以便溯源。
不要将历史“筛选后测试集”的分数与官方完整测试集分数直接并列。

跨事件实验先准备 CSV，每个 tile 一行，同一原始 scene 的所有裁块使用相同 group：

```csv
id,source_split,split,group,event
event_a_scene01_r00_c00,train,train,event_a_scene01,event_a
event_b_scene03_r00_c00,val,val,event_b_scene03,event_b
event_c_scene07_r00_c00,test,test,event_c_scene07,event_c
```

`id` 必须与现有 loader 识别的 id 一致，`source_split` 指影像实际所在目录，`split` 是这次实验的归属。
事件和 group 必须来自已核实的元数据；普通的文件名去重不能证明地理独立。

```bash
python tools/make_event_split.py --input verified_metadata.csv \
  --test-events event_c --val-events event_b --output heldout_c.csv
python tools/run_multiseed.py --experiments bright_dehcd_l \
  --data-root bright=/path/to/BRIGHT --manifest bright=heldout_c.csv \
  --output runs/experiments/heldout_c --preflight-only
```

验证事件也与训练事件分离。训练、选权重、测试全程不得接触测试事件标签用于调参。

## 数据与评估边界

- 新增的 ChangeOS-R50、DamageFormer、ChangeMamba 已完成可训练接入；[上游版本与适配说明](../compare/UPSTREAM.md)记录结构、双头损失、预训练与 CUDA 算子。完整正式精度实验仍需运行，不能把功能检查当成官方成绩复现。
- 当前基线均有项目适配；GN/原 BN 对照只能量化其中一项影响。官方原生训练策略应另表汇报。
- Haiti C1/C2/C3 的物理语义、官方 BRIGHT 数据划分一致性、完整负样本恢复、地理/场景元数据必须由数据来源核实；代码不会臆造。
- 种子 SD 描述固定划分上的训练波动，不是跨事件不确定性；像素不能作为独立训练重复。

## 效率测量

```bash
python tools/benchmark.py --config configs/dehcd/bright_l.xml \
  --optical-channels 3 --sar-channels 1 --size 256 256 \
  --batch-size 1 --device cuda:0 --warmup 20 --iterations 100 \
  --output runs/benchmark/bright_l.json
```

所有模型使用同设备、输入尺寸、batch 和精度报告；该脚本统计 device-resident 模型前向，不含磁盘 I/O 和 TTA。训练显存/总耗时需另测，不能与推理指标混用。


## 新增基线及训练策略对照

主实验对所有模型使用同一主任务损失/数据/预算，定位辅助损失权重统一为 0；`auxiliary` 单独将三种双头基线在四个数据集上的定位权重改为 1。`adapter` 只改变 BN→GN 开关。`head_recipe` 的 6 组配置以各自 `original_bn` 为参照，只改变双头目标函数；`optimizer_recipe` 的 4 组配置以各自 `head_recipe` 为参照，只改变指定优化器配方，选模指标仍相同。它们仍使用项目数据、采样与 epoch 预算，不能标成完整官方 benchmark。

```bash
python tools/run_multiseed.py --experiments bright_changeos bright_damageformer bright_changemamba \
  --data-root bright=/path/to/BRIGHT1 --output runs/experiments/new_baselines --dry-run
```

显存不足时可统一对整个比较设置 `--batch-size 2 --gradient-accumulation-steps 4`，有效 batch=8；要同时记录物理 batch 与 accumulation。原 BN 对照仍会受物理 batch 影响，累积不是 BN 的等价大 batch。不要为单个模型静默修改 batch 或缩图。

## 统计、原始指标和事件泛化

```bash
python tools/compare_results.py --campaign runs/experiments/bright_main \
  --reference bright_dehcd_l --metrics mean_iou foreground_miou oa \
  --output runs/experiments/bright_statistics
```

只有计划内全部种子完整、原始混淆矩阵/测试集/权重校验通过才输出统计。导出原始百分比均值、样本 SD、以种子为块的 percentile bootstrap 95% CI、配对种子差值、双侧 sign-flip p 和全报告 Holm 校正。置信区间是未校正的边际区间。5 个种子的精确双侧 p 最小为 0.0625，不用小样本结果自动宣称“显著”。种子是训练重复单位，不是新的独立灾害事件；该 CI 不能替代跨地域泛化区间。`raw_metrics_percent.csv`/`raw_metrics.md` 不对模型间小幅差异做 min–max 归一化，可用于绘制原始指标图表。

`mean_iou` 包括背景，`foreground_miou` 不包括背景；分割指标均从全测试集原始混淆矩阵计算。双头附加的 conditional damage harmonic F1 只在真实前景上统计。必须以同一定义比较 BRIGHT 原论文和本工作，并同时核查官方 split、输入分辨率、预训练、负样本裁块和归一化，不能把差距简单归因于一个指标名称。

```bash
python tools/build_event_cv.py --metadata verified_metadata.csv --dataset bright \
  --configs configs/experiments/main/bright_dehcd_l.xml configs/experiments/main/bright_changemamba.xml \
  --output runs/experiments/bright_event_protocol
python tools/run_multiseed.py --catalog runs/experiments/bright_event_protocol/catalog.json \
  --suite cross_event --datasets bright --data-root bright=/path/to/BRIGHT1 \
  --output runs/experiments/bright_event_campaign --preflight-only
```

每个已核实事件恰好作为测试事件一次，验证事件按排序固定轮转，训练事件与二者分离，场景 group 不跨 split；不通过测试表现选择验证事件。没有真实事件元数据时工具明确失败，不能凭裁块 id 声称留一事件验证已完成。
