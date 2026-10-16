# 地震事件迁移实验

在已安装依赖的 Python 环境、项目根目录执行：

```bash
python -W ignore tools/run_event_transfer.py --seeds 42 1051 2060
```

默认顺序为 DEHCD-Net-s、DEHCD-Net-m、DEHCD-Net-l。每个尺寸完成指定随机种子的训练后，使用 Turkey 验证集选出的最佳检查点，分别测试 Haiti、Morocco、Noto。三个测试事件不参与训练、调参、早停或检查点选择。

所附尺寸配置针对 RTX 5090 32 GB：

| 尺寸 | 配置 | Batch | 初始目标学习率 | 精度 |
| --- | --- | ---: | ---: | --- |
| s | `configs/event_transfer/dehcd_s.xml` | 16 | 0.001 | BF16 |
| m | `configs/event_transfer/dehcd_m.xml` | 8 | 0.0006 | BF16 |
| l | `configs/event_transfer/dehcd_l.xml` | 8 | 0.0006 | BF16 |

公共设置位于 `configs/event_transfer_earthquake.xml`。最多训练 100 epoch，5 epoch 学习率预热、余弦衰减；每个 epoch 验证一次，以验证集 `foreground_miou` 选择最佳检查点。连续 20 次验证未达到 `0.0002` 的有效提升时早停；小于该阈值的严格最佳分数仍会保存为最佳检查点。损失、类别权重和稀有类别采样均在配置中记录。

加速设置包括 BF16 前向计算、FP32 损失计算、TF32、fused AdamW、批量梯度裁剪、8 个持久数据 worker、预取、按抽样次数播种的数据增强，以及三个尺寸共享的标准化张量缓存。保留 best/last 检查点并支持续跑，关闭额外 epoch 检查点归档和验证图片导出。使用 cuDNN 算法自动选择，随机种子仍生效，但 GPU 计算不承诺逐位一致。这些设置不修改主实验的配置。

## 结果位置

公开默认输出为 `runs/event_transfer_earthquake`，通过 `local/settings.json`、`DEHCD_EVENT_OUTPUT` 或 `--output` 设置。数据目录通过 `DEHCD_EVENT_DATA_ROOT` 或 `--data-root` 设置。每个尺寸独立保存在 `sizes/s`、`sizes/m`、`sizes/l`。

每次训练的目录格式为：

```text
sizes/<size>/runs/event_transfer/custom/dehcd_<size>/seed_<seed>/
  config_snapshot.json
  history.jsonl
  checkpoints/best.pth
  checkpoints/last.pth
  test/haiti-earthquake/result.json
  test/morocco-earthquake/result.json
  test/noto-earthquake/result.json
```

每个事件有独立的测试覆盖数量、原始混淆矩阵、样本指标、数据指纹及检查点哈希。重跑时验证数据、精度和检查点后复用已完成的测试；未完成的事件会重新完整评估。

`summary/event_metrics.json` 按尺寸、事件、随机种子汇总；`summary/event_metrics.csv` 是逐尺寸/逐事件/逐种子的表格。单随机种子 `n=1`，标准差为 `null`；多个种子使用样本标准差（ddof=1）。只汇总本次命令明确指定的尺寸、事件和种子。

## 常用选项

```bash
# 多随机种子
python tools/run_event_transfer.py --seeds 3402 42 1051

# 指定尺寸
python tools/run_event_transfer.py --seeds 3402 --sizes m l

# 仅检查数据与配置
python tools/run_event_transfer.py --seeds 3402 --preflight-only

# GPU 显存有限时，指定新实验目录和较小批量
python tools/run_event_transfer.py --seeds 3402 --sizes l --batch-size 4 --output /path/to/new_results
```

`--batch-size` 和 `--epochs` 会覆盖本次命令的所有尺寸。改变影响训练的超参数时应使用新的 `--output`，原检查点不能作为不同训练配置的续跑状态。BF16 需要支持它的 GPU；如需 FP32，在公共配置中将 `training.amp` 设置为 `false` 并选择新结果目录。

三个尺寸采用独立的性能设置，适用于补充事件迁移分析；若报告不同尺寸的差异，需要同时公开这些设置，不能将结果解释为严格控制超参数的尺寸消融。短时速度和数值检查不代表已经验证完整训练收敛或迁移性能。

场景隔离的留一事件及小样本从头训练/权重微调对比见[迁移研究协议](../configs/event_transfer/README.md)。
