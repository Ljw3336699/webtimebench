# 数据构建报告

版本：timeagent_revision_v2.1。固定随机种子：20261103。原始输入文件未修改。

| 划分 | clean | temporal |
|---|---:|---:|
| train | 17557 | 9087 |
| dev | 923 | 471 |
| test | 2894 | 1448 |

隔离记录：583；去重/跨集合排除记录：1043。
全部 23,000 条原始记录的去向均在 data/id_mapping.json 中；理由可重叠，理由计数不能直接求和当作记录数。

## 隔离原因

- conflicting_gold_for_body: 455
- conflicting_regime_for_body: 130
- invalid_gold: 28

## 排除原因

- duplicate_body_same_labels: 497
- train_body_in_original_test: 546

## 验证边界

已验证严格公历日期、单一金标、gold/reward 一致、ID 完整追踪、精确归一正文去重和跨集合无重叠、原测试集不进入训练/开发集、时间子集约束。
不能从缺少 URL、日期来源证据和事件簇的本地导出确认 source-disjoint、event-disjoint、语义近重复消除或 gold 真实性。所有冲突记录仅隔离，未猜测或修补日期。
新版划分必须重新训练；在旧 checkpoint 上运行新版测试集不能宣称已经消除历史训练污染。Parquet 和 JSON 等价检查状态见 dataset_build.json。
