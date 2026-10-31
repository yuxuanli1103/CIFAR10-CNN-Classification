# -------------------------------------------------------------------
# References:
# [1] CS231n: Data Preprocessing. https://cs231n.github.io/neural-networks-2/
# [2] PyTorch PR #2405: CIFAR-10 normalization stats.
#     https://github.com/pytorch/tutorials/pull/2405
# [3] Python 3 Documentation: pickle.
#     https://docs.python.org/3/library/pickle.html
# [4] CIFAR-10 Official Dataset.
#     https://www.cs.toronto.edu/~kriz/cifar.html
# [5] Alex Krizhevsky, Ilya Sutskever, Geoffrey Hinton, "ImageNet Classification
#     with Deep Convolutional Neural Networks", NIPS 2012.
#     https://papers.nips.cc/paper_files/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html
# [6] Ian Goodfellow, Yoshua Bengio, Aaron Courville, "Deep Learning",
#     MIT Press 2016, 第 5 章：训练集 / 验证集 / 测试集的划分.
#     https://www.deeplearningbook.org/contents/ml.html
# [7] CS231n: Neural Networks Part 3 (训练动态).
#     https://cs231n.github.io/neural-networks-3/
# -------------------------------------------------------------------


import pickle
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

DATA_ROOT = Path(__file__).resolve().parent / "data" / "cifar10" / "cifar-10-batches-py"
# 依据 [2] 的官方统计结果
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)


def load_batch(name):
    with open(DATA_ROOT / name, "rb") as f:
        # 依据 [3]，处理 Python 2 二进制流的兼容性问题
        d = pickle.load(f, encoding="latin1")
    # 依据 [4]，官方格式为展平向量，此处依据 PyTorch NCHW 协议重塑为 3D 图像
    images = torch.from_numpy(d["data"]).reshape(-1, 3, 32, 32)
    return images, torch.tensor(d["labels"])


class CIFAR10(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx]


def train_transform():
    # 依据 [5]，训练时随机平移 + 水平翻转，让模型见到位置和左右翻转后的同一物体
    return v2.Compose([
        v2.RandomCrop(32, padding=4),
        v2.RandomHorizontalFlip(),
        # 依据 [1]，先缩放到 [0, 1] 再按通道均值方差标准化
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(MEAN, STD),
    ])


def eval_transform():
    return v2.Compose([
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(MEAN, STD),
    ])


def get_loaders(batch_size=128, val_ratio=0.1, seed=0, num_workers=0):
    images, labels = zip(*(load_batch(f"data_batch_{i}") for i in range(1, 6)))
    images, labels = torch.cat(images), torch.cat(labels)

    # 依据 [6]，验证集用于选模型，测试集只在最后做一次最终评估
    # 依据 [7]，单折验证即可，不必做交叉验证
    perm = torch.randperm(len(labels), generator=torch.Generator().manual_seed(seed))
    n_val = int(len(labels) * val_ratio)
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    train_set = CIFAR10(images[train_idx], labels[train_idx], train_transform())
    val_set = CIFAR10(images[val_idx], labels[val_idx], eval_transform())

    # 同一批训练图像再过一遍干净变换，用来测“没有增强的训练准确率”；
    # 依据 [6]，比较过拟合时必须让 train 与 val 的预处理一致，否则差距里混进了增强的扰动
    train_eval_set = CIFAR10(images[train_idx], labels[train_idx], eval_transform())

    test_images, test_labels = load_batch("test_batch")
    test_set = CIFAR10(test_images, test_labels, eval_transform())

    return (
        DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        DataLoader(val_set, batch_size=batch_size, num_workers=num_workers),
        DataLoader(test_set, batch_size=batch_size, num_workers=num_workers),
        DataLoader(train_eval_set, batch_size=batch_size, num_workers=num_workers),
    )


if __name__ == "__main__":
    train_loader, val_loader, test_loader, train_eval_loader = get_loaders()
    print("train/val/test:", len(train_loader.dataset), len(val_loader.dataset),
          len(test_loader.dataset))
    images, labels = next(iter(train_loader))
    print("batch:", images.shape, images.dtype, "| labels:", labels.shape)
    print("cuda:", torch.cuda.is_available())
