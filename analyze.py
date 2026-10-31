import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RUNS_DIR = Path(__file__).resolve().parent / "runs"
EXP_FILE = RUNS_DIR / "experiments.json"


def load_experiments():
    if not EXP_FILE.exists():
        raise SystemExit(f"no experiment results; run experiments.py first ({EXP_FILE})")
    return json.loads(EXP_FILE.read_text())


def read_history(name):
    return json.loads((RUNS_DIR / f"{name}_history.json").read_text())["history"]


def save(fig, filename):
    fig.tight_layout()
    fig.savefig(RUNS_DIR / filename, dpi=150)


def plot_tuning(results):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for r in results:
        name = r["name"].replace("tuning_", "")
        ax.bar(name, r["best_val_acc"])
        ax.text(name, r["best_val_acc"], f"{r['best_val_acc']:.3f}",
                ha="center", va="bottom", fontsize=8)
    ax.set(title="Hyperparameter tuning: best val accuracy",
           ylabel="accuracy", ylim=(0, 1))
    ax.tick_params(axis="x", rotation=35, labelsize=8)
    ax.grid(axis="y", alpha=0.3)
    save(fig, "exp_tuning.png")


def plot_capacity(results):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    labels, train_acc, val_acc = [], [], []
    for r in results:
        c = r["config"]
        labels.append(f"{c['model']} w{c.get('width', 1)}")
        train_acc.append(r["final_train_acc"])
        val_acc.append(r["final_val_acc"])

    x = np.arange(len(labels))
    axes[0].bar(x - 0.2, train_acc, 0.4, label="train (clean)")
    axes[0].bar(x + 0.2, val_acc, 0.4, label="val")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, fontsize=9)
    axes[0].set(title="MLP vs CNN capacity", ylabel="accuracy", ylim=(0, 1))
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="y", alpha=0.3)

    for r in results:
        c = r["config"]
        hist = read_history(r["name"])
        axes[1].plot([h["epoch"] for h in hist],
                     [h["train_clean"]["accuracy"] for h in hist],
                     marker="o", ms=3, label=f"{c['model']} w{c.get('width', 1)}")
    axes[1].set(title="Train accuracy over epochs", xlabel="epoch",
                ylabel="accuracy")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=7)
    save(fig, "exp_capacity.png")


def plot_regularization(results):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    labels = [r["config"].get("label") or r["name"].replace("regularization_", "")
              for r in results]
    x = np.arange(len(labels))

    axes[0].bar(x - 0.2, [r["final_train_acc"] for r in results], 0.4,
                label="train (clean)")
    axes[0].bar(x + 0.2, [r["final_val_acc"] for r in results], 0.4, label="val")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=40, ha="right", fontsize=7)
    axes[0].set(title="Normalization / dropout / init", ylabel="accuracy",
                ylim=(0, 1))
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(x, [r["gap"] for r in results], 0.6)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=40, ha="right", fontsize=7)
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].set(title="Generalization gap (train - val)", ylabel="gap")
    axes[1].grid(axis="y", alpha=0.3)
    save(fig, "exp_regularization.png")


def plot_curves(group, results):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for r in results:
        hist = read_history(r["name"])
        epochs = [h["epoch"] for h in hist]
        label = r["config"].get("label") or r["name"].replace(f"{group}_", "")
        axes[0].plot(epochs, [h["val"]["accuracy"] for h in hist],
                     marker="o", ms=3, label=label)
        axes[1].plot(epochs, [h["val"]["loss"] for h in hist],
                     marker="o", ms=3, label=label)
    for ax, ylabel in zip(axes, ("accuracy", "loss")):
        ax.set(xlabel="epoch", ylabel=ylabel)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    axes[0].set_title(f"{group}: val accuracy")
    axes[1].set_title(f"{group}: val loss")
    save(fig, f"exp_{group}_curves.png")


def plot_overfit(results):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for r in results:
        hist = read_history(r["name"])
        epochs = [h["epoch"] for h in hist]
        label = r["config"].get("label", r["name"])
        axes[0].plot(epochs, [h["train_clean"]["accuracy"] for h in hist],
                     lw=2, label=f"{label} train")
        axes[0].plot(epochs, [h["val"]["accuracy"] for h in hist],
                     lw=1.5, ls="--", label=f"{label} val")

        gap = [h["train_clean"]["accuracy"] - h["val"]["accuracy"] for h in hist]
        axes[1].plot(epochs, gap, lw=1.8, label=label)

    axes[0].set(title="Overfitting: train vs val accuracy",
                xlabel="epoch", ylabel="accuracy")
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].set(title="Generalization gap over training",
                xlabel="epoch", ylabel="train - val")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    save(fig, "exp_overfit.png")


def plot_schedule(results):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for r in results:
        hist = read_history(r["name"])
        epochs = [h["epoch"] for h in hist]
        label = r["config"].get("label", r["name"])
        axes[0].plot(epochs, [h["val"]["accuracy"] for h in hist],
                     marker="o", ms=3, lw=1.8, label=label)
        axes[1].plot(epochs, [h["lr"] for h in hist], lw=1.8, label=label)
    axes[0].set(title="Scheduler: val accuracy", xlabel="epoch",
                ylabel="accuracy")
    axes[1].set(title="Learning rate schedule", xlabel="epoch", ylabel="lr")
    axes[1].set_yscale("log")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    save(fig, "exp_schedule.png")


def main():
    exps = load_experiments()
    for group, results in exps.items():
        print(f"\n=== {group} ===")
        print(f"{'run':42} {'best_val':>9} {'train*':>7} {'val':>7} {'gap':>7}")
        for r in sorted(results, key=lambda x: -x["best_val_acc"]):
            print(f"{r['name']:42} {r['best_val_acc']:>9.4f} "
                  f"{r['final_train_acc']:>7.4f} {r['final_val_acc']:>7.4f} "
                  f"{r['gap']:>+7.4f}")

        if group == "tuning":
            plot_tuning(results)
        elif group == "capacity":
            plot_capacity(results)
        elif group == "regularization":
            plot_regularization(results)
        elif group == "overfit":
            plot_overfit(results)
        elif group == "schedule":
            plot_schedule(results)
        plot_curves(group, results)
        print(f"  -> runs/exp_{group}.png, runs/exp_{group}_curves.png")


if __name__ == "__main__":
    main()
