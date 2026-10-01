# 验证与故障诊断

## 回归检查

```bash
python -m unittest discover -s tests -v
```

测试覆盖模型接口、损失数值稳定性、消融模块、容量对照梯度、指标与测试结果完整性、实验控制变量，以及常驻数据加载进程在 epoch 边界恢复时的增强随机流。
CUDA 算子检查需要可用的 GPU；多进程测试需要本地进程通信权限。

## 数据与配置预检

```bash
python tools/run_multiseed.py --suite main --datasets haiti \
  --seeds 42 1051 2060 --runtime-profile rtx5090 \
  --data-root haiti=/path/to/Haiti1 --output runs/haiti_3seeds --preflight-only
```

预检核对配置参照关系、数据样本和 split 独立性，保存解析后的协议。只有预检通过才可启动训练。代码、参数、种子或数据变更后需使用新的输出目录。

## 数值与运行状态

- `history.jsonl`：逐轮损失、IoU、梯度范数、有效优化步数与前景比例。
- `gradient_spikes.jsonl`：异常梯度对应的样本 ID 和参数层。
- `failure.json`：训练故障原因；故障训练不能进入完整结果统计。
- `training_summary.json`：完成状态、最优验证轮数和训练时长。
- `protocol.json`：源码、数据、执行精度、优化器、环境与实现信息。

Haiti 的输入质量掩膜、FP32 损失、GN epsilon、梯度裁剪与前景停滞检查共同用于数值保护。有限梯度短测并不证明长期精度或收敛。

## 结果核验

```bash
python tools/validate_results.py path/to/test_result.json --verify-checkpoints
```

最终统计要求完整测试集、计划内全部种子、匹配的数据与 checkpoint 指纹。性能测速应记录设备、精度、输入尺寸、物理 batch 与梯度累积；模型前向延迟不能代替整轮训练耗时。
