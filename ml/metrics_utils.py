"""Metric computation (numpy + scikit-learn only, no torch)."""
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve


def compute_metrics(y_true, probs, class_names):
    y_true = np.asarray(y_true)
    probs = np.asarray(probs)
    n_classes = len(class_names)
    y_pred = probs.argmax(axis=1)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    total = cm.sum()

    per_class, roc_curves = {}, {}
    aucs = []
    for i, name in enumerate(class_names):
        tp = int(cm[i, i])
        fn = int(cm[i, :].sum() - tp)
        fp = int(cm[:, i].sum() - tp)
        tn = int(total - tp - fn - fp)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        specificity = tn / (tn + fp) if (tn + fp) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        binary_true = (y_true == i).astype(int)
        auc = None
        if 0 < binary_true.sum() < len(binary_true):
            auc = float(roc_auc_score(binary_true, probs[:, i]))
            aucs.append(auc)
            fpr, tpr, _ = roc_curve(binary_true, probs[:, i])
            roc_curves[name] = {
                "fpr": [round(float(v), 4) for v in fpr],
                "tpr": [round(float(v), 4) for v in tpr],
                "auc": round(auc, 4),
            }
        per_class[name] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),  # a.k.a. sensitivity
            "specificity": round(specificity, 4),
            "f1": round(f1, 4),
            "roc_auc": None if auc is None else round(auc, 4),
            "support": int(cm[i, :].sum()),
        }

    def macro(key):
        return round(float(np.mean([per_class[c][key] for c in class_names])), 4)

    supports = np.array([per_class[c]["support"] for c in class_names], dtype=float)
    weighted_f1 = float(np.sum([per_class[c]["f1"] * s for c, s in zip(class_names, supports)]) / supports.sum())

    return {
        "n_samples": int(total),
        "accuracy": round(float(np.trace(cm) / total), 4),
        "macro": {
            "precision": macro("precision"),
            "recall": macro("recall"),
            "specificity": macro("specificity"),
            "f1": macro("f1"),
            "roc_auc": round(float(np.mean(aucs)), 4) if aucs else None,
        },
        "weighted_f1": round(weighted_f1, 4),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "class_names": list(class_names),
        "roc_curves": roc_curves,
    }
