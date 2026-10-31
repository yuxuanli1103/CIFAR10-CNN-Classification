import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "runs" / "experiments.json"

TUNING = [
    {"optimizer": "adam", "lr": 1e-4},
    {"optimizer": "adam", "lr": 1e-3},
    {"optimizer": "adam", "lr": 1e-2},
    {"optimizer": "adamw", "lr": 3e-4},
    {"optimizer": "sgd", "lr": 1e-2},
    {"optimizer": "sgd", "lr": 5e-2},
]

CAPACITY = [
    {"model": "mlp", "width": 1},
    {"model": "mlp", "width": 2},
    {"model": "cnn", "width": 1},
    {"model": "cnn", "width": 2},
]

REGULARIZATION = [
    {"norm": "none", "dropout": 0.0, "init": "default", "label": "no_norm_no_drop"},
    {"norm": "bn", "dropout": 0.0, "init": "default", "label": "bn_only"},
    {"norm": "none", "dropout": 0.5, "init": "default", "label": "drop_only"},
    {"norm": "bn", "dropout": 0.3, "init": "default", "label": "bn_drop"},
    {"norm": "bn", "dropout": 0.3, "init": "xavier", "label": "bn_drop_xavier"},
    {"norm": "bn", "dropout": 0.3, "init": "kaiming", "label": "bn_drop_kaiming"},
    {"norm": "bn", "dropout": 0.3, "init": "zero", "label": "bn_drop_zero_init"},
]


OVERFIT = [
    {"norm": "none", "dropout": 0.0, "init": "default", "label": "long_no_reg"},
    {"norm": "bn", "dropout": 0.0, "init": "default", "label": "long_bn"},
    {"norm": "bn", "dropout": 0.5, "init": "default", "label": "long_bn_drop"},
]


SCHEDULE = [
    {"optimizer": "sgd", "lr": 0.05, "scheduler": "cosine", "label": "sgd_cosine"},
    {"optimizer": "sgd", "lr": 0.05, "scheduler": "step", "label": "sgd_step"},
    {"optimizer": "adam", "lr": 1e-3, "scheduler": "cosine", "label": "adam_cosine"},
    {"optimizer": "sgd", "lr": 0.05, "scheduler": "none", "label": "ctrl_sgd_fixed"},
    {"optimizer": "adam", "lr": 1e-3, "scheduler": "none", "label": "ctrl_adam_fixed"},
]


def run_one(tag, config, epochs, seed):
    cfg = dict(config)
    label = cfg.pop("label", None)
    name = f"{tag}_{label or '_'.join(f'{k}{v}' for k, v in cfg.items())}"
    cmd = [sys.executable, str(HERE / "train.py"), "--epochs", str(epochs),
           "--seed", str(seed), "--run-name", name, "--quiet"]
    for k, v in cfg.items():
        cmd += [f"--{k.replace('_', '-')}", str(v)]

    print(f"\n>>> {name}")
    subprocess.run(cmd, cwd=HERE, check=True)

    hist = json.loads((HERE / "runs" / f"{name}_history.json").read_text())
    last = hist["history"][-1]
    return {
        "name": name,
        "config": {**cfg, "label": label or name},
        "epochs": epochs,
        "best_val_acc": hist["best_acc"],
        "final_train_acc": last["train_clean"]["accuracy"],
        "final_val_acc": last["val"]["accuracy"],
        "gap": last["train_clean"]["accuracy"] - last["val"]["accuracy"],
        "elapsed_sec": hist["elapsed_sec"],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--group", required=True,
                   choices=["tuning", "capacity", "regularization", "overfit",
                            "schedule"])
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    groups = {"tuning": TUNING, "capacity": CAPACITY,
              "regularization": REGULARIZATION, "overfit": OVERFIT,
              "schedule": SCHEDULE}
    results = [run_one(args.group, cfg, args.epochs, args.seed)
               for cfg in groups[args.group]]

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    existing = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    existing[args.group] = results
    RESULTS.write_text(json.dumps(existing, indent=2))

    print(f"\n{'run':38} {'best_val':>9} {'train*':>7} {'val':>7} {'gap':>7}")
    print("-" * 72)
    for r in sorted(results, key=lambda x: -x["best_val_acc"]):
        print(f"{r['name']:38} {r['best_val_acc']:>9.4f} "
              f"{r['final_train_acc']:>7.4f} {r['final_val_acc']:>7.4f} "
              f"{r['gap']:>+7.4f}")
    print("* train = 干净数据（无增强）上的训练集准确率")


if __name__ == "__main__":
    main()
