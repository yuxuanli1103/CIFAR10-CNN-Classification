# -------------------------------------------------------------------
# References:
# [1] Scikit-learn: accuracy_score.
#     https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html
# [2] Scikit-learn: precision_recall_fscore_support.
#     https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html
# [3] Scikit-learn: confusion_matrix.
#     https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
# [4] Marina Sokolova, Guy Lapalme, "A systematic analysis of performance
#     measures for classification tasks", 2009.
#     https://doi.org/10.1016/j.ipm.2009.03.002
# -------------------------------------------------------------------


import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_recall_fscore_support)


def compute_metrics(y_true, y_pred, num_classes=10):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # 依据 [2]，四项指标一次算出
    # 依据 [4]，宏平均对每一类同等加权，某一类全错时掉 1/num_classes，
    # 比加权平均更能暴露类别不均衡的失败
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=range(num_classes),
        average="macro", zero_division=0,
    )

    # 依据 [1]
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def compute_confusion(y_true, y_pred, num_classes=10):
    # 依据 [3]，固定 labels 保证输出恒为 10x10，不会被缺失的类别压缩
    return confusion_matrix(y_true, y_pred, labels=range(num_classes))
