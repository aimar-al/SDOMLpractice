import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn import calibration

def confusion_matrix(y_true, y_pred, path):
    cm = metrics.confusion_matrix(y_true, y_pred)
    disp = metrics.ConfusionMatrixDisplay(cm,display_labels = ['Human', 'AI'])
    disp.plot()
    plt.savefig(path, dpi=150)
    plt.show()

def calibration_curve(y_true, y_prob, path):
    prob_true, prob_pred = calibration.calibration_curve(y_true, y_prob[:,1])
    disp = calibration.CalibrationDisplay(prob_true, prob_pred, y_prob)
    disp.plot()
    plt.savefig(path, dpi=150)
    plt.show()

def roc_curve(y_true, y_prob, path):
    fpr, tpr, _ = metrics.roc_curve(y_true,y_prob[:,1])
    disp = metrics.RocCurveDisplay(fpr=fpr,tpr=tpr,roc_auc=metrics.auc(fpr,tpr))
    disp.plot()
    plt.savefig(path, dpi=150)
    plt.show()

def precision_recall_curve(y_true, y_prob, path):
    precision, recall, _ = metrics.precision_recall_curve(y_true,y_prob[:,1])
    disp = metrics.PrecisionRecallDisplay(precision,recall)
    disp.plot()
    plt.savefig(path, dpi=150)
    plt.show()

def loss_accuracy_curves(image_path, log_path):
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