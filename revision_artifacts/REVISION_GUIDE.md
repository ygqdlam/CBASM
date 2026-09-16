# Major Revision Code Guide

本指南按审稿意见逐条说明：哪些代码覆盖该意见、如何运行、输出文件在哪里。所有新实验默认写入 `code_revised/results_revised/`，原 `code/result` 或 `code/results` 不会被覆盖。

## 通用运行约定

建议从 `code_revised/` 目录运行：

```bash
cd code_revised
```

3D 主训练入口：

```bash
python train_post_3d_aut_addema.py --cfg config_3d_pan_aut.yml --exp major_revision --labeled_num 4 --enable_profile --enable_diversity_metrics
```

2D 主训练入口：

```bash
python train_post_2d_aut_addema.py --cfg config_2d_pro_aut.yml --exp major_revision --labeled_num 7 --enable_profile --enable_diversity_metrics
```

如果服务器上的配置文件路径不在默认候选位置，可以直接传绝对路径：

```bash
python train_post_3d_aut_addema.py --cfg /path/to/config_3d_pan_aut.yml
```

## R1-2 / R2-4：训练时间、iter/s、GPU-hours、FLOPs、显存

相关代码：

- `train_post_3d_aut_addema.py`
- `train_post_2d_aut_addema.py`
- `revision_utils/profiling.py`

运行时加：

```bash
--enable_profile --profile_interval 10
```

输出：

- `results_revised/<dataset>/<labeled>_labeled_<exp>/<model>/profiling/iteration_profile.csv`
- `results_revised/<dataset>/<labeled>_labeled_<exp>/<model>/profiling/model_complexity.csv`
- TensorBoard: `profile/iter_time_sec`, `profile/iter_per_sec`, `profile/cumulative_gpu_hours`, `profile/cuda_max_memory_allocated_mb`

可用于回复：

- FLOPs/Params 来自 `model_complexity.csv`
- iter/s、GPU-hours、memory curve 来自 `iteration_profile.csv` 或 TensorBoard

## R1-5 / R2-3 / R3-5：mean ± std、paired significance test、HD 显著性

相关代码：

- `test_performance_3d.py`
- `test_performance_2d.py`
- `val_3D.py`
- `revision_utils/metrics_export.py`
- `tools/run_significance_tests.py`

3D 测试：

```bash
python test_performance_3d.py \
  --dataset LA \
  --res_path ./results_revised/LA \
  --exp major_revision \
  --model vnet \
  --model_type ema \
  --labeled_num 4 \
  --checkpoint_path ./results_revised/LA/4_labeled_major_revision/vnet/vnet_best_ema_model.pth
```

2D 测试：

```bash
python test_performance_2d.py \
  --root_path ../data_split/Prostate \
  --data_path /path/to/Prostate/h5file \
  --res_path ./results_revised/Prostate \
  --exp major_revision \
  --model unet \
  --model_type ema \
  --labeled_num 7 \
  --checkpoint_path ./results_revised/Prostate/7_labeled_major_revision/unet/unet_best_ema_model.pth
```

输出：

- `*_predictions_<model_type>/case_metrics_<model_type>.csv`
- `*_predictions_<model_type>/summary_metrics_<model_type>.csv`

显著性检验：

```bash
python tools/run_significance_tests.py \
  --baseline_csv /path/to/baseline/case_metrics_ema.csv \
  --candidate_csv /path/to/ours/case_metrics_ema.csv \
  --output_csv ./results_revised/significance/ours_vs_baseline.csv \
  --metrics dice hd95
```

可用于回复：

- Tables I-III 的 `mean ± std` 来自 `summary_metrics_*.csv`
- paired t-test p-value 来自 `tools/run_significance_tests.py` 的输出
- PROMISE12 4 labeled 的 HD 显著性同样用 `--metrics hd95`

## R2-2 / R3-4 / R3-8：diversity、similarity、KL/JS/disagreement

相关代码：

- `train_post_3d_aut_addema.py`
- `train_post_2d_aut_addema.py`
- `revision_utils/diversity.py`

运行时加：

```bash
--enable_diversity_metrics --diversity_interval 50
```

输出：

- 训练 CSV 字段：`diversity_ema_active_teacher_kl_pq`, `kl_qp`, `js`, `disagreement`, `entropy_p`, `entropy_q`
- TensorBoard：`diversity/ema_active_teacher/*`
- 原有 cosine similarity 仍在：`ratio/cosine_sim_t12`, `ratio/cosine_sim_st1`, `ratio/cosine_sim_st2`, `ratio/cosine_sim_sema`

可用于回复：

- Figure 4 similarity metric 定义：feature cosine similarity，即两模型对同一 unlabeled weak input 的 feature tensor 展平后做 cosine similarity。
- diversity 定量：预测概率分布 KL/JS divergence 与 hard prediction disagreement。

## R3-1：DRSCM frozen teacher 伪标签质量随时间变化

相关代码：

- `train_post_3d_aut_addema.py`
- `train_post_2d_aut_addema.py`

已有输出：

- `log/seg_*_pseudo_dice.txt`
- 训练 CSV：`mask_ratio`, `mask_ratio_t`, `error_ratio`, `error_ratio_t`, `conflict_ratio`, `high_ratio`, `high_ratio_t`
- TensorBoard：`ratio/mask_ratio`, `ratio/error_ratio`, `info/conflict_ratio`, `ratio/avg_dice_s`, `ratio/avg_dice_t`

可用于回复：

- 用 iteration/epoch 曲线展示 pseudo-label Dice、error ratio、mask ratio、conflict ratio 随训练变化。
- 如果要和同步 co-training/2CPS 比较，用相同测试脚本导出 per-case CSV 后再跑显著性检验。

## R3-2 / R3-7：Margin Loss 理论解释、hinge/no-ML/alpha 消融、loss scale 曲线

相关代码：

- `train_post_3d_aut_addema.py`
- `train_post_2d_aut_addema.py`

新增参数：

```bash
--margin_type l1      # 原始默认设置
--margin_type none    # no Margin Loss
--margin_type hinge   # hinge margin 消融
--alpha 0.5           # margin 权重消融
--margin_m 0.8        # hinge 目标 margin
```

建议消融命令：

```bash
python train_post_3d_aut_addema.py --exp ablate_no_ml --margin_type none
python train_post_3d_aut_addema.py --exp ablate_l1_alpha05 --margin_type l1 --alpha 0.5
python train_post_3d_aut_addema.py --exp ablate_hinge --margin_type hinge --margin_m 0.8
```

输出：

- 训练 CSV：`loss_ce`, `loss_dice`, `margin_loss_raw`, `margin_loss_weighted`, `margin_type`
- TensorBoard：`loss/loss_ce`, `loss/loss_dice`, `loss/margin_loss_raw`, `loss/margin_loss`

可用于回复：

- loss scale 曲线直接来自 TensorBoard 或训练 CSV。
- 消融结果用测试脚本导出 `summary_metrics_*.csv` 后汇总。

## R3-6：multi-class general applicability

相关代码：

- `test_performance_3d.py`
- `test_performance_2d.py`
- `val_3D.py`
- `train_post_3d_aut_addema.py`
- `train_post_2d_aut_addema.py`

新增支持：

- 训练时 `my_DICE` 跟随 `--num_classes` 初始化。
- 3D 推理 `num_classes=2` 保持前景阈值，`num_classes>2` 改用 softmax argmax。
- 2D/3D 测试脚本均支持 `--num_classes`。

运行模板：

```bash
python test_performance_3d.py --num_classes 4 --checkpoint_path /path/to/multiclass_model.pth
```

说明：

- 这覆盖了 multi-class evaluation/training interface。
- 若论文不补完整 Synapse/BTCV 实验，应在 limitation 中明确当前实证主要是二分类医学分割任务。

## 汇总工具

训练/测试结束后可扫描所有修订结果：

```bash
python tools/summarize_revision_results.py
```

输出：

- `results_revised/revision_summary/revision_result_summary.md`
- `training_last_rows.csv`
- `validation_last_rows.csv`
- `profiling_last_rows.csv`
- `metric_summary_rows.csv`
