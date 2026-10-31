# CIFAR-10 CNN Project

## 环境

```bash
conda activate dl_env     # 或直接用绝对路径
~/miniconda3/envs/dl_env/bin/python xxx.py
```

## 文件

| 文件 | 作用 |
|---|---|
| `dataset.py` | 读取 CIFAR-10，切分 train/val/test，返回 DataLoader |
| `model.py` | `CNN`（三层卷积）和 `MLP`（全连接基线） |
| `metrics.py` | accuracy / precision / recall / F1 / 混淆矩阵 |
| `train.py` | 训练循环，TensorBoard 记录，checkpoint 保存 |
| `evaluate.py` | 用某个 checkpoint 在 val 或 test 上评估 |
| `experiments.py` | 批量跑实验组（调参 / 容量 / 正则化 / 长训练 / 衰减） |
| `analyze.py` | 读实验结果，出表格和图 |
| `ANALYSIS.md` | **实验分析与结论** |

## 常用命令

```bash
# 训练（默认 CNN + Adam，20 epochs）
python train.py

# 换优化器
python train.py --optimizer sgd --lr 0.01
python train.py --optimizer adamw --lr 1e-3 --epochs 30

# 换模型 / 结构选项
python train.py --model mlp
python train.py --width 2 --norm bn --dropout 0.3 --init kaiming

# 学习率衰减
python train.py --optimizer sgd --lr 0.05 --scheduler cosine --epochs 30

# 批量实验
python experiments.py --group tuning --epochs 12
python analyze.py

# 评估
python evaluate.py --ckpt checkpoints/schedule_sgd_cosine_best.pt --split test
```

当前最好的单模型是 `schedule_sgd_cosine`（SGD 0.05 + cosine，30 epoch），
test accuracy **0.8385**。完整实验分析见 `ANALYSIS.md`。

## 输出

```
runs/
  <run_name>/              TensorBoard 事件文件
  <run_name>_history.json  每轮指标 + 超参数 + 耗时
  experiments.json         五组实验的汇总指标
  exp_<group>.png          每组的主图
  exp_<group>_curves.png   验证集曲线
checkpoints/
  <run_name>_best.pt       val accuracy 最高的那一轮
  <run_name>_last.pt       最后一轮
  eval_<ckpt>_<split>.json  评估结果 + 混淆矩阵
```

查看 TensorBoard：

```bash
~/miniconda3/envs/dl_env/bin/tensorboard --logdir runs --port 6006
```

然后在浏览器打开 `http://localhost:6006`。

## 数据

```
data/cifar10/cifar-10-batches-py/
```

训练集 50000 张按 9:1 切成 train(45000) / val(5000)，test(10000) 留到最后。
