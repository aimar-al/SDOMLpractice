"""Evaluation and diagnostic visualization functions.

Provides functions to compute, plot, and save model performance figures including
confusion matrix, calibration curve, ROC curve, precision-recall curve, and
epoch-wise loss/accuracy progression.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn import calibration

def confusion_matrix(y_true, y_pred, path):
    """Plots and saves the confusion matrix for binary classification.

    Args:
        y_true (array-like): Ground truth binary labels.
        y_pred (array-like): Predicted binary class labels.
        path (str): Destination file path to save the generated plot image.

    Example:
        >>> confusion_matrix(y_true, y_pred, "reports/figures/cm.png")
    """
    cm = metrics.confusion_matrix(y_true, y_pred)
    disp = metrics.ConfusionMatrixDisplay(cm,display_labels = ['Human', 'AI'])
    disp.plot()
    plt.savefig(path, dpi=150)
    plt.show()

def calibration_curve(y_true, y_prob, path):
    """Plots and saves the probability calibration curve.

    Args:
        y_true (array-like): Ground truth binary labels.
        y_prob (array-like): Predicted class probabilities of shape `(n_samples, 2)`.
        path (str): Destination file path to save the generated plot image.

    Example:
        >>> calibration_curve(y_true, y_prob, "reports/figures/calibration.png")
    """
    prob_true, prob_pred = calibration.calibration_curve(y_true, y_prob[:,1])
    disp = calibration.CalibrationDisplay(prob_true, prob_pred, y_prob)
    disp.plot()
    plt.savefig(path, dpi=150)
    plt.show()

def roc_curve(y_true, y_prob, path):
    """Plots and saves the Receiver Operating Characteristic (ROC) curve.

    Args:
        y_true (array-like): Ground truth binary labels.
        y_prob (array-like): Predicted class probabilities of shape `(n_samples, 2)`.
        path (str): Destination file path to save the generated plot image.

    Example:
        >>> roc_curve(y_true, y_prob, "reports/figures/roc.png")
    """
    fpr, tpr, _ = metrics.roc_curve(y_true,y_prob[:,1])
    disp = metrics.RocCurveDisplay(fpr=fpr,tpr=tpr,roc_auc=metrics.auc(fpr,tpr))
    disp.plot()
    plt.savefig(path, dpi=150)
    plt.show()

def precision_recall_curve(y_true, y_prob, path):
    """Plots and saves the Precision-Recall curve.

    Args:
        y_true (array-like): Ground truth binary labels.
        y_prob (array-like): Predicted class probabilities of shape `(n_samples, 2)`.
        path (str): Destination file path to save the generated plot image.

    Example:
        >>> precision_recall_curve(y_true, y_prob, "reports/figures/pr_curve.png")
    """
    precision, recall, _ = metrics.precision_recall_curve(y_true,y_prob[:,1])
    disp = metrics.PrecisionRecallDisplay(precision,recall)
    disp.plot()
    plt.savefig(path, dpi=150)
    plt.show()

def loss_accuracy_curves(image_path, log_path):
    """Plots and saves epoch-level training loss and accuracy curves from a CSV log.

    Args:
        image_path (str): Destination file path to save the generated figure.
        log_path (str): File path to the CSV metrics log file generated during training.

    Example:
        >>> loss_accuracy_curves("reports/figures/curves.png", "logs/train_log/version_0/metrics.csv")
    """
    metrics = pd.read_csv(log_path)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(metrics['epoch_train_loss'].dropna())
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Loss curve', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(metrics['epoch_train_acc'].dropna())
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title('Accuracy curve', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    plt.savefig(image_path, dpi=150)
    plt.show()