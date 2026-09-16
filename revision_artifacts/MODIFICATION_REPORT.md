# Code Modification Report

本报告只记录 `code_revised/` 相对原始 `code/` 的大修相关改动。原始 `code/` 未作为修改目标；新实验默认保存到 `code_revised/results_revised/`。

## 修改过的代码文件

### `train_post_3d_aut_addema.py`

删除/替换的关键代码：

```python
import shutil
...
shutil.copy('../code/train_post_3d_aut_addema.py', snapshot_path)
shutil.copy('../code/networks/vnet.py', snapshot_path)
shutil.copy('../code/networks/net_factory.py', snapshot_path)
...
parser.add_argument('--res_path', type=str,
                    default='./results/LA', help='Path to save resutls')
...
margin_loss = args['alpha'] * cal_confidence_loss(pred_lb, target_lb)
loss_lb = (ce_loss(pred_lb, target_lb.long()) +
           dice_loss(torch.softmax(pred_lb, dim=1),
                     target_lb.unsqueeze(1).float(),
                     ignore=torch.zeros_like(target_lb).float())
           + margin_loss
           ) / 3.0
return loss_lb, margin_loss
```

新增的关键代码：

```python
from revision_utils.diversity import diversity_from_torch_probs
from revision_utils.profiling import IterationProfiler, append_profile_csv, profile_model_complexity, write_complexity_csv
from revision_utils.run_io import copy_sources, resolve_config_path, write_run_metadata
...
profile_csv = os.path.join(snapshot_path, "profiling", "iteration_profile.csv")
profiler = IterationProfiler(enabled=args.get("enable_profile", False), num_gpus=len(args.get("gpu_id", [0])))
...
diversity_metrics = maybe_log_diversity(
    writer, iter_num + 1, "diversity/ema_active_teacher",
    ema_outputs_soft_1, ema_outputs_soft_2, args,
)
...
loss_lb, margin_loss, loss_ce, loss_dice, margin_loss_raw = cal_sup_loss(
    ce_loss, dice_loss, pred_lb, target_lb, args)
...
parser.add_argument('--margin_type', choices=['none', 'l1', 'hinge'], default='l1')
parser.add_argument('--enable_profile', action='store_true')
parser.add_argument('--enable_diversity_metrics', action='store_true')
```

修改目的：

- 覆盖 R1-2/R2-4：记录 FLOPs、Params、iter/s、GPU-hours、CUDA 显存峰值。
- 覆盖 R2-2/R3-4/R3-8：记录 KL/JS divergence、disagreement、entropy，并保留原有 feature cosine similarity。
- 覆盖 R3-2/R3-7：新增 `none/l1/hinge` margin 消融和 CE/Dice/Margin loss scale。
- 覆盖 R3-1：保留并强化伪标签质量相关 CSV/TensorBoard 字段。
- 防止覆盖旧结果：默认输出转到脚本目录下的 `results_revised/`。

### `train_post_2d_aut_addema.py`

删除/替换的关键代码：

```python
import shutil
...
shutil.copy('./train_post_2d_aut_addema.py', snapshot_path)
...
parser.add_argument('--res_path', type=str,
                    default='./results/ACDC', help='Path to save resutls')
...
margin_loss = args['alpha']*cal_confidence_loss(pred_lb, 0, 0, target_lb,args)
return loss_lb,margin_loss
```

新增的关键代码：

```python
from revision_utils.diversity import diversity_from_torch_probs
from revision_utils.profiling import IterationProfiler, append_profile_csv, profile_model_complexity, write_complexity_csv
from revision_utils.run_io import copy_sources, resolve_config_path, write_run_metadata
...
global my_DICE
my_DICE = losses.my_DiceLoss(nclass=num_classes)
...
loss_lb, mar_loss, loss_ce, loss_dice, margin_loss_raw = cal_sup_loss(
    ce_loss, dice_loss, pred_lb, target_lb, args)
...
def cal_margin_loss(pred_lb, target_lb, args):
    if args.get("margin_type", "l1") == "none":
        return pred_lb.new_tensor(0.0)
    if args.get("margin_type", "l1") == "l1":
        return cal_confidence_loss(pred_lb, 0, 0, target_lb, args)
    ...
```

修改目的：

- 与 3D 主脚本对齐，补齐效率、diversity、loss scale 与 margin 消融证据。
- 保留 2D 原始 `right/err` L1 margin 口径，默认行为仍对应原始方法。
- 支持 `--num_classes` 下的伪标签 Dice 日志。

### `val_3D.py`

删除/替换的关键代码：

```python
y = y.cpu().data.numpy()
y = y[0,1,:,:,:]
...
label_map = (score_map[0]>0.5).astype(np.int32)
...
if np.sum(prediction)==0:
    single_metric = (0,0,0,0)
else:
    single_metric = calculate_metric_percase(prediction, label[:])
...
return avg_metric
```

新增的关键代码：

```python
from revision_utils.metrics_export import rows_from_metric_array, summarize_case_metrics, write_case_metrics_csv, write_summary_csv
...
y = y.cpu().data.numpy()[0]
...
if num_classes == 2:
    label_map = (score_map[1] > 0.5).astype(np.int32)
else:
    label_map = np.argmax(score_map, axis=0).astype(np.int32)
...
case_metrics = calculate_metric_perclass(prediction, label[:], num_classes)
rows = rows_from_metric_array(case_metric_arrays, case_ids=case_ids, extra=extra_fields)
write_case_metrics_csv(case_metric_csv, rows, extra_fields=list((extra_fields or {}).keys()))
write_summary_csv(summary_csv, summarize_case_metrics(rows))
```

修改目的：

- 覆盖 R1-5/R2-3/R3-5：3D 测试输出 per-case CSV 和 mean/std summary。
- 覆盖 R3-6：`num_classes>2` 时支持 multi-class argmax 推理。
- 保持二分类兼容：`num_classes=2` 仍用前景概率阈值。

### `test_performance_3d.py`

删除/替换的关键代码：

```python
parser.add_argument('--res_path', type=str, default='./results/LA', help='Path to results')
...
num_classes = 2
...
save_model_path = os.path.join(snapshot_path, '{}_best_{}_model.pth'.format(FLAGS.model,FLAGS.model_type))
```

新增的关键代码：

```python
DEFAULT_RESULTS_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_revised")
parser.add_argument('--res_path', type=str, default=os.path.join(DEFAULT_RESULTS_ROOT, 'LA'))
parser.add_argument('--num_classes', type=int, default=2)
parser.add_argument('--checkpoint_path', type=str, default='')
parser.add_argument('--case_metrics_csv', type=str, default='')
parser.add_argument('--summary_csv', type=str, default='')
...
case_metrics_csv = FLAGS.case_metrics_csv or os.path.join(test_save_path, "case_metrics_{}.csv".format(FLAGS.model_type))
summary_csv = FLAGS.summary_csv or os.path.join(test_save_path, "summary_metrics_{}.csv".format(FLAGS.model_type))
```

修改目的：

- 让 3D 测试脚本直接生成统计分析需要的 CSV。
- 支持显式指定服务器上的 `.pth`，不用依赖本地已有 checkpoint。
- 支持 multi-class evaluation 参数。

### `test_performance_2d.py`

删除/替换的关键代码：

```python
parser.add_argument('--res_path', type=str, default='./results/Prostate', help='Path to results')
...
if os.path.exists(test_save_path):
    shutil.rmtree(test_save_path)
os.makedirs(test_save_path)
save_model_path = os.path.join(snapshot_path, 'student/ep_416_dice_0.8576.pth'.format(...))
```

新增的关键代码：

```python
from revision_utils.metrics_export import rows_from_metric_array, summarize_case_metrics, write_case_metrics_csv, write_summary_csv
...
os.makedirs(test_save_path, exist_ok=True)
save_model_path = FLAGS.checkpoint_path or os.path.join(
    snapshot_path, '{}_best_{}_model.pth'.format(FLAGS.model, FLAGS.model_type))
...
rows = rows_from_metric_array(metric_list, case_ids=image_list, extra=extra_fields)
write_case_metrics_csv(case_metrics_csv, rows, extra_fields=list(extra_fields.keys()))
write_summary_csv(summary_csv, summarize_case_metrics(rows))
```

修改目的：

- 让 2D 测试脚本输出 per-case CSV 和 mean/std summary。
- 避免测试时删除整个预测目录。
- 支持显式 checkpoint 路径，便于调用服务器已有 `.pth`。

## 新增代码文件

### `revision_utils/metrics_export.py`

用途：统一 per-case metric CSV、summary mean/std CSV 的读写和数组转行逻辑。

为什么新增：R1-5/R2-3/R3-5 都需要 case-level 结果和 mean ± std，训练脚本和测试脚本不应各自手写 CSV 逻辑。

### `revision_utils/diversity.py`

用途：计算 prediction KL、reverse KL、JS divergence、hard prediction disagreement、entropy。

为什么新增：覆盖 R2-2/R3-4/R3-8 对 diversity/similarity 定义和定量分析的要求。

### `revision_utils/profiling.py`

用途：记录 iteration time、iter/s、GPU-hours、CUDA memory，并在可用时用 `thop` 计算 FLOPs/Params。

为什么新增：覆盖 R1-2/R2-4 的效率和计算开销审稿意见。

### `revision_utils/run_io.py`

用途：解析 cfg 路径、复制源码快照、写 `run_meta.json`。

为什么新增：服务器与本地启动目录不同，原脚本固定 `../cfgs` 和 `../code` 容易失效；同时 run metadata 可支撑可复现性说明。

### `revision_utils/stats.py`

用途：读取两个 per-case CSV，按 `case_id + class_id` 对齐，做 paired significance test。

为什么新增：覆盖 R2-3/R3-5 的统计显著性要求。

### `tools/run_significance_tests.py`

用途：命令行运行 paired significance test，输出 p-value CSV。

为什么新增：让审稿回复中的显著性检验可复现，而不是只在论文里写结果。

### `tools/summarize_revision_results.py`

用途：扫描 `results_revised/` 下训练、验证、profiling、测试 summary CSV，生成汇总 markdown 和 CSV。

为什么新增：大修实验数量多，需要一个统一汇总入口。

### `tests/test_revision_utils.py`

用途：测试 metrics summary、diversity、profiling、cfg path resolution、paired significance。

为什么新增：这些工具不依赖 torch，可以在本地快速确认核心逻辑没有坏。
