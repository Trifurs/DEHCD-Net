# 实验协议

正式集合由 `configs/experiments/` 的 80 个唯一科学配置组成，每项固定使用 `[42,1051,2060]`，共 240 个目标任务。已有合规成果属于这 240 项，不是额外任务。正式配置的 `canonical_id`、`roles`、`uses`、`paper_refs` 和 `comparison` 是用途及参照的来源；catalog 与说明从元数据生成。多个组选取先取 canonical ID 并集，再展开种子。

## 定义与范围

| 类别 | 额外配置数 | 型号及参照 |
| --- | ---: | --- |
| 主比较 | 42 | 四数据集 S/M/L 和原六基线；BRIGHT/Haiti 另有 ChangeOS、DamageFormer、ChangeMamba |
| 组件与组合 | 14 | BRIGHT/Haiti 各7项，另引用各自L组成8行 |
| 参数匹配 | 4 | BRIGHT-L：HOG intensity、DPM conv、完整BiCSF conv、IRB feedforward |
| DPM分项 | 2 | BRIGHT-L：关闭flow或difference gate |
| 训练策略 | 3 | BRIGHT-L：CE+Dice损失方案、关闭weighted sampler、关闭class weights |
| HOG bins | 4 | Haiti-L：2/4/8/10，默认6引用主模型 |
| IRB次数 | 8 | BRIGHT-M：0/1/2/4/5/6/7/8，默认3引用主模型 |
| HOG引导层级 | 3 | BRIGHT-M：1/3/4，默认2引用主模型 |

BRIGHT/Haiti/xBD/CAU-Flood 分别39/23/9/9项、117/69/27/27个种子任务。主比较、scaling、默认扫描点和后处理共用训练资产。BRIGHT-M T=0与BRIGHT-L no_irb不能合并。每个非默认点独立完成训练，不在默认checkpoint上临时改变循环代替训练。共享base、dataset、dehcd模板不独立入队，也不能直接执行模板后混入正式统计。

## 公平协议

`controlled_comparison_v1` 固定同数据集的划分、波段、标签/ignore、质量掩膜、归一化、增强、采样、裁块、物理batch、累积、预算、优化器、调度、种子、精度和验证/测试规则，只允许声明的因素变化。同effective batch不代表相同microbatch/累积。

全部主比较从头训练、主任务监督；localization、deep-supervision和feature-pair auxiliary权重为零。普通early stopping关闭，完整验证集的严格最大 `foreground_miou` 选best；完整测试集主logits argmax、无TTA。数值异常和前景崩溃/停滞保护保留，触发属于失败，不是正常提前完成。DEHCD S/M/L外部dropout统一0.15，仅预声明结构规格变化。

这是统一策略的适配架构比较，不声称复现基线官方最优成绩。输入stem、BN转GN、输出适配和HRSICD 64×64内核限制见[基线说明](../compare/README.md)。RTX5090使用FP32张量/loss并开启TF32矩阵及卷积运算，不是严格IEEE FP32乘法，详见[运行参数](RUNTIME_PROFILE.md)。

## 实际损失与采样

BRIGHT完整损失为 CE + Dice + 0.5 Focal + 0.2 Lovasz + 0.3 Tversky；CE smoothing=0.01，类别权重 `[0.5,2,18,14]`，Focal gamma=2，Tversky alpha/beta=0.3/0.7。背景及缺失类别处理以配置和 `utils/losses.py` 为准。CE+Dice对照同时移除其他损失项和smoothing、保留采样，因此是损失方案对照，不是单项损失的孤立贡献。

Haiti使用hierarchical_change：二分类CE/Dice/Focal为1/0.7/0.15，CE对有效背景/前景分别取均值再以 `[1,1]` 平均；子类CE/Dice/Focal为0.8/0.2/0.10，子类权重 `[1,1.8,1.3]`、smoothing=0.02；前景先验权重0.15，前景logit offset=-0.90，两个Focal gamma均为2。所有Haiti模型相同。本集合只在BRIGHT隔离三个训练策略因素，不能写成四数据集都完成该控制。

WeightedRandomSampler根据训练类别像素统计计算权重，每epoch有放回抽取与训练集等长的序列。BRIGHT目标类别 `[2,3]`、上限8；Haiti `[1,2,3]`、上限4。上限不是固定复制次数。预检记录实际权重范围及非均匀性。定向裁块须在对齐后至少一维大于patch，且存在相应有效前景/稀有类别；预检记录自由度及实际适用比例，不能只凭配置概率声称全部样本都发生定向裁块。

## 稳定性、复用与恢复

保留FP32 loss、Haiti GN epsilon=1e-3、输入/输出/梯度/参数有限性、梯度裁剪、末尾不足整组的累积归一化、前景保护及best/last。短程有限梯度不能证明长期收敛，全背景也不能直接诊断成梯度爆炸。

续训恢复model、optimizer、scheduler、scaler、epoch、RNG/采样流、best tracker和保护计数；纯推理权重不能完整续训。可信本地checkpoint与纯权重采用明确加载策略，无全局不安全补丁。

目标任务分为reuse_complete、reevaluate_only、resume_training、train_new、retrain_required、blocked_review。复用核对解析CLI/runtime后的结构、训练设置、三个split、checkpoint和原始记录。展示元数据/文档不改变科学配置，模型/loss/dataset/RNG/训练循环/验证选模变化须逐项核对。保留原来源及导入记录，不改写旧checkpoint哈希。原路径逐文件stat核对只能标记stat_verified；本次SHA256不能证明历史内容。活跃文件不迁移、不删除。

## 三种子统计与后处理

每seed完整测试指标独立计算，再输出mean、sample SD(ddof=1)、n、expected_n=3、complete和来源。少于3种子为incomplete，不补零、不复制、不选最好种子。差值按同seed与正确参照对齐：图17各自L，IRB/层级BRIGHT-M，bins Haiti-L；相对扫描起点和默认点分别标注。同一full在多图出现仍只有3个独立训练重复。

默认mean±SD及逐seed差值。n=3的精确双侧sign-flip最小p=0.25，不能达到p<0.05；bootstrap也受小样本限制。种子是固定划分上的训练重复，不是像素/tile/独立灾害事件。

指标来自完整测试原始混淆计数。mean_iou包括背景，foreground_miou仅前景；均值只纳入有有效union的类别。混淆分析同时给原始计数、每类指标和分母。行归一化对角线是该类recall，不是总体OA。双头未监督定位头标记localization_supervised=false，正式localization_f1=null，诊断输出不参与定位排名。conditional damage F1、foreground mIoU和全图mIoU不是同一指标。

定性图、特征图、混淆矩阵、参数与测速复用正式checkpoint；展示seed预固定42、样本预登记，不按测试成绩挑选。hook部分算子估计不能称完整FLOPs；实测延迟/吞吐需相同设备、输入、batch、精度/backend。未开展跨事件训练，不能宣称证明unseen-event generalization。

入口与结果管理见[运行流程](RUN_WORKFLOW.md)，控制因素见[公平比较](FAIR_COMPARISONS.md)，验证见[测试说明](VALIDATION.md)。
