# -------------------------------------------------------------------
# References:
# [1] Adam: Diederik P. Kingma, Jimmy Ba, ICLR 2015.
#     https://arxiv.org/abs/1412.6980
# [2] AdamW: Ilya Loshchilov, Frank Hutter, ICLR 2019.
#     https://arxiv.org/abs/1711.05101
# [3] SGD with momentum: Sutskever et al., ICML 2013.
#     https://proceedings.mlr.press/v28/sutskever13.html
# [4] torch.optim: SGD / Adam / AdamW.
#     https://docs.pytorch.org/docs/stable/optim.html
# [5] CosineAnnealingLR.
#     https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html
# [6] StepLR.
#     https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.StepLR.html
# [7] torch.nn.CrossEntropyLoss.
#     https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
# [8] PyTorch: Autograd mechanics (set_grad_enabled).
#     https://docs.pytorch.org/docs/stable/autograd.html
# [9] torch.utils.tensorboard: Scalars.
#     https://docs.pytorch.org/docs/stable/tensorboard.html
# [10] torch.nn.Module: train() / eval().
#     https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html
# [11] PyTorch: Reproducibility (torch.manual_seed).
#     https://pytorch.org/docs/stable/notes/randomness.html
# [12] Ian Goodfellow, Yoshua Bengio, Aaron Courville, "Deep Learning",
#     MIT Press 2016, 第 7 章：early stopping.
#     https://www.deeplearningbook.org/contents/regularization.html
# -------------------------------------------------------------------


import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import get_loaders
from metrics import compute_metrics
from model import build_model, count_params

RUNS_DIR = Path(__file__).resolve().parent / "runs"
CKPT_DIR = Path(__file__).resolve().parent / "checkpoints"


def build_optimizer(name, params, lr):
    # 依据 [4]，三种优化器的接口一致，只是默认超参与更新规则不同
    if name == "sgd":
        # 依据 [3]，动量 0.9 是常用的默认值
        return torch.optim.SGD(params, lr=lr, momentum=0.9)
    if name == "adam":
        # 依据 [1]
        return torch.optim.Adam(params, lr=lr)
    if name == "adamw":
        # 依据 [2]，weight decay 5e-4 只作用在权重上，不与自适应步长混在一起
        return torch.optim.AdamW(params, lr=lr, weight_decay=5e-4)
    raise ValueError(f"unknown optimizer: {name}")


def build_scheduler(name, optimizer, epochs):
    if name == "none":
        return None
    if name == "cosine":
        # 依据 [5]，T_max 取总轮数，学习率按余弦从 lr 降到 0
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    if name == "step":
        # 依据 [6]，每 1/3 轮把学习率乘 0.1
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, epochs // 3),
                                               gamma=0.1)
    raise ValueError(f"unknown scheduler: {name}")


def run_epoch(model, loader, criterion, optimizer, device, train, progress=True):
    # 依据 [10]，训练模式与评估模式的差别只在 Dropout 和 BN 的统计量，用同一个布尔量一起切换
    model.train(train)
    total_loss, y_true, y_pred = 0.0, [], []

    for images, labels in tqdm(loader, leave=False, disable=not progress):
        images, labels = images.to(device), labels.to(device)

        # 依据 [8]，评估时不建计算图，省显存也更快
        with torch.set_grad_enabled(train):
            logits = model(images)
            loss = criterion(logits, labels)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # 按样本数加权累加，避免最后一个不满的 batch 被过度加权
        total_loss += loss.item() * len(labels)
        y_true.extend(labels.cpu().tolist())
        y_pred.extend(logits.argmax(1).cpu().tolist())

    result = compute_metrics(y_true, y_pred)
    result["loss"] = total_loss / len(y_true)
    return result


def save_checkpoint(path, model, epoch, best_acc, args):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "epoch": epoch,
        "best_acc": best_acc,
        "args": vars(args),
    }, path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="cnn", choices=["cnn", "mlp"])
    p.add_argument("--optimizer", default="adam", choices=["sgd", "adam", "adamw"])
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--run-name", default=None)
    p.add_argument("--width", type=int, default=1)
    p.add_argument("--norm", default="bn", choices=["bn", "none"])
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--init", default="default",
                   choices=["default", "kaiming", "xavier", "zero"])
    p.add_argument("--scheduler", default="none", choices=["none", "cosine", "step"])
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    # 依据 [11]，固定种子后权重初始化与 DataLoader 的 shuffle 每次都相同
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    run_name = args.run_name or f"{args.model}_{args.optimizer}_lr{args.lr}"
    writer = SummaryWriter(RUNS_DIR / run_name)
    print(f"run: {run_name}  device: {device}")

    train_loader, val_loader, _, train_eval_loader = get_loaders(
        batch_size=args.batch_size,
        seed=args.seed,
        num_workers=args.num_workers,
    )

    model = build_model(args.model, width=args.width, norm=args.norm,
                        dropout=args.dropout, init=args.init).to(device)
    # 依据 [7]，交叉熵内部含 softmax，模型直接输出 logits
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(args.optimizer, model.parameters(), args.lr)
    scheduler = build_scheduler(args.scheduler, optimizer, args.epochs)
    print(f"params: {count_params(model):,}")

    history = []
    best_acc = 0.0
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        train_m = run_epoch(model, train_loader, criterion, optimizer, device,
                            True, progress=not args.quiet)
        val_m = run_epoch(model, val_loader, criterion, optimizer, device,
                          False, progress=not args.quiet)
        # 依据 [12]，过拟合看的是 train 与 val 的落差，所以训练集还要在
        # 不做增强的条件下再测一遍（增强后的 acc 与 val 不可比）
        clean_m = run_epoch(model, train_eval_loader, criterion, optimizer,
                            device, False, progress=False)

        # 依据 [9]，三组标量分别打到 train/、val/、train_clean/ 下
        for k, v in train_m.items():
            writer.add_scalar(f"train/{k}", v, epoch)
        for k, v in val_m.items():
            writer.add_scalar(f"val/{k}", v, epoch)
        for k, v in clean_m.items():
            writer.add_scalar(f"train_clean/{k}", v, epoch)

        history.append({"epoch": epoch, "train": train_m, "val": val_m,
                        "train_clean": clean_m,
                        "lr": optimizer.param_groups[0]["lr"]})
        if scheduler is not None:
            scheduler.step()
        print(f"epoch {epoch:02d} | train loss {train_m['loss']:.4f} "
              f"acc {train_m['accuracy']:.4f} f1 {train_m['f1']:.4f} "
              f"| clean acc {clean_m['accuracy']:.4f} "
              f"| val loss {val_m['loss']:.4f} acc {val_m['accuracy']:.4f} "
              f"f1 {val_m['f1']:.4f}")

        save_checkpoint(CKPT_DIR / f"{run_name}_last.pt", model, epoch,
                        best_acc, args)
        # 依据 [12]，保存验证集上最好的一轮，而不是最后一轮
        if val_m["accuracy"] > best_acc:
            best_acc = val_m["accuracy"]
            save_checkpoint(CKPT_DIR / f"{run_name}_best.pt", model, epoch,
                            best_acc, args)
            print(f"  -> new best val acc {best_acc:.4f}, checkpoint saved")

    elapsed = time.time() - start
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    (RUNS_DIR / f"{run_name}_history.json").write_text(
        json.dumps({"args": vars(args), "best_acc": best_acc,
                    "elapsed_sec": round(elapsed, 1), "history": history},
                   indent=2)
    )
    writer.close()
    print(f"done in {elapsed / 60:.1f} min | best val acc {best_acc:.4f}")


if __name__ == "__main__":
    main()
