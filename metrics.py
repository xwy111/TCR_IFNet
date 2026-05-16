import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)


def cal_acc_precision_recall_f1(real_labels, pred_labels, pred_probs=None, average="binary", is_test=False):
    acc = accuracy_score(real_labels, pred_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(
        real_labels, pred_labels, average=average, zero_division=0
    )
    cm = confusion_matrix(real_labels, pred_labels)

    out = {
        "acc": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": cm,
    }

    if pred_probs is not None:
        pos_probs = np.array(pred_probs)[:, 1]
        try:
            out["auc"] = float(roc_auc_score(real_labels, pos_probs))
        except Exception:
            out["auc"] = float("nan")
        try:
            out["ap"] = float(average_precision_score(real_labels, pos_probs))
        except Exception:
            out["ap"] = float("nan")

    return out
