# -------------------------------------------------------------------
# References:
# [1] CS231n: Convolutional Networks.
#     https://cs231n.github.io/convolutional-networks/
# [2] Sergey Ioffe, Christian Szegedy, "Batch Normalization", ICML 2015.
#     https://arxiv.org/abs/1502.03167
# [3] Nitish Srivastava et al., "Dropout", JMLR 2014.
#     https://jmlr.org/papers/v15/srivastava14a.html
# [4] Kaiming He et al., "Delving Deep into Rectifiers", ICCV 2015.
#     https://arxiv.org/abs/1502.01852
# [5] Xavier Glorot, Yoshua Bengio, AISTATS 2010.
#     https://proceedings.mlr.press/v9/glorot10a.html
# [6] CS231n: Neural Networks Part 2 (权重初始化与对称性).
#     https://cs231n.github.io/neural-networks-2/
# -------------------------------------------------------------------


import torch.nn as nn


def make_norm(norm, channels):
    if norm == "bn":
        # 依据 [2]，在卷积输出上按通道计算 batch 统计量
        return nn.BatchNorm2d(channels)
    if norm == "none":
        return nn.Identity()
    raise ValueError(f"unknown norm: {norm}")


def conv_block(c_in, c_out, norm):
    # 依据 [1]，3x3 卷积 + padding=1 保持尺寸，再池化把边长减半
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1),
        make_norm(norm, c_out),
        nn.ReLU(),
        nn.MaxPool2d(2),
    )


class CNN(nn.Module):
    def __init__(self, num_classes=10, width=1, norm="bn", dropout=0.3):
        super().__init__()
        c1, c2, c3 = 32 * width, 64 * width, 128 * width
        # 依据 [1]，三次 2x2 池化把 32x32 降到 4x4
        self.features = nn.Sequential(
            conv_block(3, c1, norm),
            conv_block(c1, c2, norm),
            conv_block(c2, c3, norm),
        )
        # 依据 [3]，只在全连接层之间随机置零，卷积部分保留
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c3 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class MLP(nn.Module):
    def __init__(self, num_classes=10, width=1, norm="none", dropout=0.3):
        super().__init__()
        h1, h2 = 512 * width, 256 * width
        # 依据 [1]，不用卷积，直接把 3x32x32 展平成一维
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 32 * 32, h1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def init_weights(model, scheme="default"):
    if scheme == "default":
        return model
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            if scheme == "kaiming":
                # 依据 [4]，fan_out 模式配 ReLU 增益
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif scheme == "xavier":
                # 依据 [5]，兼顾 fan_in 与 fan_out
                nn.init.xavier_uniform_(m.weight)
            elif scheme == "zero":
                # 依据 [4]，全零会让同一层神经元梯度相同，对称性无法打破
                nn.init.zeros_(m.weight)
            else:
                raise ValueError(f"unknown init: {scheme}")
            if m.bias is not None:
                # 依据 [6]，偏置置零即可，打破对称性靠的是权重的随机性
                nn.init.zeros_(m.bias)
    return model


def build_model(name="cnn", num_classes=10, width=1, norm="bn",
                dropout=0.3, init="default"):
    if name == "cnn":
        model = CNN(num_classes, width, norm, dropout)
    elif name == "mlp":
        model = MLP(num_classes, width, norm, dropout)
    else:
        raise ValueError(f"unknown model: {name}")
    return init_weights(model, init)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
