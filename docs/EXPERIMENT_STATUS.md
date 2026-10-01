# 实验能力与适用边界

正式目标为80个唯一配置、三种子、240个任务。实际完成/剩余数量以结果目录的reuse_manifest、audit_report和summary为准；配置数量不代表完成训练。

| 功能 | 当前范围 | 边界 |
| --- | --- | --- |
| 主比较 | 四数据集S/M/L及原六基线，BRIGHT/Haiti另含三基线，共42项 | 统一策略适配比较，不是官方最优复现 |
| 组件/组合 | BRIGHT/Haiti L各8行，full复用主模型 | BiCSF包括GCBM和WASM |
| 容量/DPM/训练策略 | BRIGHT L分别4/2/3个额外控制 | 不宣称四数据集均隔离这些贡献 |
| 敏感性 | Haiti L五个bins；BRIGHT M九个steps、四个levels | 默认点共用主模型，其余独立训练 |
| 复用 | 完成、重评、续训、新训、不兼容、待核实六类 | 不改写来源，不能只凭同名/可加载判断 |
| 三种子统计 | 原始指标、mean、sample SD、n、expected_n、complete、同seed差值 | n=3精确双侧sign-flip最小p=0.25 |
| 稳定性 | FP32 loss、Haiti GN epsilon、有限性/裁剪、前景保护 | 短程检查不证明长期收敛 |
| 后处理 | 定性、混淆、特征、参数和测速复用checkpoint | 预固定seed/样本，部分算子计数不是完整FLOPs |

操作见[协议](EXPERIMENT_PROTOCOL.md)、[运行流程](RUN_WORKFLOW.md)、[控制因素](FAIR_COMPARISONS.md)。
