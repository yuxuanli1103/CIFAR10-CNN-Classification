# -------------------------------------------------------------------
# References:
# [1] Scikit-learn: confusion_matrix.
#     https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
# [2] CIFAR-10 Official Dataset (class order).
#     https://www.cs.toronto.edu/~kriz/cifar.html
# [3] torch.load: map_location 与 weights_only.
#     https://docs.pytorch.org/docs/stable/generated/torch.load.html
# [4] torch.nn.Module: train() / eval().
#     https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html
# [5] Ian Goodfellow, Yoshua Bengio, Aaron Courville, "Deep Learning",
#     MIT Press 2016, 第 7 章：early stopping（选验证集最好的那一轮）.
#     https://www.deeplearningbook.org/contents/regularization.html
# -------------------------------------------------------------------


import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn

from dataset import get_loaders
from metrics import compute_confusion, compute_metrics
from model import build_model

# 依据 [2]，索引顺序必须与 pickle 里的 label 一致，否则每类准确率会错位
CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--split", default="val", choices=["val", "test"])
    p.add_argument("--batch-size", type=int, default=256)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, val_loader, test_loader, _ = get_loaders(batch_size=args.batch_size)
    # 依据 [5]，模型是按 val 选出来的，所以默认也在 val 上报告；test 只在最后确认时用
    loader = val_loader if args.split == "val" else test_loader

    # 依据 [3]，map_location 允许在 CPU 上读 GPU 存的权重；
    # weights_only=False 是因为文件里还有 args 字典等非张量对象
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    saved = ckpt["args"]
    model = build_model(saved["model"], width=saved.get("width", 1),
                        norm=saved.get("norm", "bn"),
                        dropout=saved.get("dropout", 0.3),
                        init=saved.get("init", "default")).to(device)
    model.load_state_dict(ckpt["model"])
    # 依据 [4]，评估前必须切到 eval：关掉 Dropout，BN 改用训练期累积的滑动统计量
    model.eval()

    criterion = nn.CrossEntropyLoss()
    total_loss, y_true, y_pred = 0.0, [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            total_loss += criterion(logits, labels).item() * len(labels)
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(logits.argmax(1).cpu().tolist())

    result = compute_metrics(y_true, y_pred)
    result["loss"] = total_loss / len(y_true)
    result["split"] = args.split
    result["checkpoint_epoch"] = ckpt["epoch"]

    print(f"checkpoint: {args.ckpt} (epoch {ckpt['epoch']})")
    print(f"split: {args.split}  n={len(y_true)}")
    for k in ("loss", "accuracy", "precision", "recall", "f1"):
        print(f"  {k:10} {result[k]:.4f}")

    # 依据 [1]，按行归一化得到每一类的召回率
    cm = compute_confusion(y_true, y_pred)
    print("\nper-class accuracy:")
    for i, name in enumerate(CLASSES):
        print(f"  {name:12} {cm[i, i] / cm[i].sum():.4f}")

    out = Path(args.ckpt).parent / f"eval_{Path(args.ckpt).stem}_{args.split}.json"
    out.write_text(json.dumps({**result, "confusion": cm.tolist()}, indent=2))
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
