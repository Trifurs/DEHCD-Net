# 验证与故障诊断

```bash
python -m unittest discover -s tests -v
python -W ignore tools/run_all.py --audit-only
python -W ignore tools/run_all.py --preflight-only
```

单元检查覆盖配置唯一性、图表扫描范围、控制变量、模型前后向、评价/监督标记、有限性与checkpoint恢复。CUDA检查需要可用设备及匹配extension，缺少设备的跳过项必须明确记录。短程冒烟使用隔离临时目录，不进入正式统计，不能证明长期收敛。

只读审计不改结果。预检建立当前计划、核验源数据/裁块/采样、审计已有训练并在证据兼容时登记导入。文档或元数据变化不应触发全量重训；科学设置、训练/选模行为变化必须核对，不能关闭一致性检查。运行中的文件不迁移、不删除。

状态证据：history.jsonl记录每轮损失/指标/梯度/前景，gradient_spikes.jsonl记录异常样本与层，failure.json记录故障，training_summary.json记录预算完成和最优轮次，protocol.json记录数据/源码/设置/环境，provenance保留导入来源。

Haiti保留FP32损失、GN epsilon=1e-3、质量掩膜、裁剪、非有限及前景退化保护。普通early stopping在正式配置关闭，控制台的无提升轮数只作监控；崩溃/停滞保护与正常早停分别展示。有限梯度不能排除背景退化，全背景也不能直接等同梯度爆炸。

```bash
python tools/validate_results.py path/to/test/result.json --verify-checkpoints
```

结果必须来自完整测试集及匹配的配置/数据/权重。汇总保留每seed原始值、n、expected_n=3及complete状态。参数匹配需同时核对实际计数/误差和梯度参与；匹配参数不等于匹配感受野或FLOPs。效率统计记录同设备、形状、batch、精度/backend；模型前向延迟不代表整轮训练耗时。CPU状态恢复检查可要求逐位一致；CUDA flow/grid_sample反向不保证逐位复现。
