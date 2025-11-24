import matplotlib
matplotlib.use("Agg")  # Use a non-interactive backend for saving plots

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

def evaluate_and_plot(y_test, preds, model_name: str, csv: str, save_dir: str = "plots"):
    """
    Compute evaluation metrics, print them, and save:
      - confusion matrix
      - bar plot of Accuracy / Precision / Recall / F1

    Returns a dict with the metric values (useful if you want
    to compare models in a table later).
    """
    os.makedirs(save_dir, exist_ok=True)

    # ===== Metrics =====
    acc  = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    rec  = recall_score(y_test, preds, zero_division=0)
    f1   = f1_score(y_test, preds, zero_division=0)

    print(f"\n🔍 {model_name} METRICS")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")

    # ===== Confusion matrix plot =====
    cm = confusion_matrix(y_test, preds)
    fig_cm, ax_cm = plt.subplots()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    disp.plot(ax=ax_cm, colorbar=False)
    ax_cm.set_title(f"{model_name} – Confusion Matrix")
    fig_cm.tight_layout()
    fig_cm.savefig(os.path.join(save_dir, f"{csv}_{model_name}_confusion_matrix.png"))
    plt.close(fig_cm)

    # ===== Bar plot of metrics =====
    metric_names  = ["Accuracy", "Precision", "Recall", "F1-score"]
    metric_values = [acc, prec, rec, f1]

    fig_bar, ax_bar = plt.subplots()
    ax_bar.bar(metric_names, metric_values)
    ax_bar.set_ylim(0, 1)
    ax_bar.set_ylabel("Score")
    ax_bar.set_title(f"{model_name} – Evaluation Metrics")

    for i, v in enumerate(metric_values):
        ax_bar.text(i, v + 0.01, f"{v:.2f}", ha="center")

    fig_bar.tight_layout()
    fig_bar.savefig(os.path.join(save_dir, f"{csv}_{model_name}_metrics_bar.png"))
    plt.close(fig_bar)

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
    }
