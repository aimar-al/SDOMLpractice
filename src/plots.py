"""Evaluation and diagnostic visualization functions.

Provides functions to compute, plot, and save model performance figures including
confusion matrix, calibration curve, ROC curve, precision-recall curve, and
epoch-wise loss/accuracy progression, as well as dataset exploration plots.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics
from sklearn import calibration

def confusion_matrix(y_true, y_pred, path=None):
    """Plots the confusion matrix for binary classification."""
    cm = metrics.confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = metrics.ConfusionMatrixDisplay(cm, display_labels=['Human', 'AI'])
    disp.plot(ax=ax, cmap='Blues')
    if path:
        plt.savefig(path, dpi=150)
    return fig

def calibration_curve(y_true, y_prob, path=None):
    """Plots the probability calibration curve."""
    prob_true, prob_pred = calibration.calibration_curve(y_true, y_prob[:, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = calibration.CalibrationDisplay(prob_true, prob_pred, y_prob[:, 1])
    disp.plot(ax=ax)
    if path:
        plt.savefig(path, dpi=150)
    return fig

def roc_curve(y_true, y_prob, path=None):
    """Plots the Receiver Operating Characteristic (ROC) curve."""
    fpr, tpr, _ = metrics.roc_curve(y_true, y_prob[:, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=metrics.auc(fpr, tpr))
    disp.plot(ax=ax)
    if path:
        plt.savefig(path, dpi=150)
    return fig

def precision_recall_curve(y_true, y_prob, path=None):
    """Plots the Precision-Recall curve."""
    precision, recall, _ = metrics.precision_recall_curve(y_true, y_prob[:, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = metrics.PrecisionRecallDisplay(precision=precision, recall=recall)
    disp.plot(ax=ax)
    if path:
        plt.savefig(path, dpi=150)
    return fig

def loss_accuracy_curves(image_path=None, log_path=None, metrics_df=None):
    """Plots epoch-level training loss and accuracy curves."""
    if metrics_df is None and log_path is not None:
        metrics_df = pd.read_csv(log_path)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    if metrics_df is not None:
        if 'epoch_train_loss' in metrics_df.columns:
            axes[0].plot(metrics_df['epoch_train_loss'].dropna())
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Loss', fontsize=12)
        axes[0].set_title('Loss curve', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        if 'epoch_train_acc' in metrics_df.columns:
            axes[1].plot(metrics_df['epoch_train_acc'].dropna())
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Accuracy', fontsize=12)
        axes[1].set_title('Accuracy curve', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    if image_path:
        plt.savefig(image_path, dpi=150)
    return fig

def plot_class_distribution(df):
    """Plots human vs AI class distribution."""
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(data=df, x='is_ai_generated', hue='is_ai_generated', palette='colorblind', legend=False, ax=ax)
    ax.set_title('Text Origin Distribution: Human vs AI')
    ax.set_xlabel('Text Origin')
    ax.set_ylabel('Frequency')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Human', 'AI'])
    fig.tight_layout()
    return fig

def plot_domain_distribution(df):
    """Plots text domain class distribution."""
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.countplot(data=df, x='domain_context', hue=df['is_ai_generated'].map({0: 'Human', 1: 'AI'}), palette='colorblind', ax=ax)
    ax.set_title('Class Distribution by Text Domain')
    ax.set_xlabel('Text Domain')
    ax.set_ylabel('Sample Count')
    ax.tick_params(axis='x', rotation=30)
    ax.legend(title='Text Origin')
    fig.tight_layout()
    return fig

def plot_perplexity_distribution(df):
    """Plots perplexity distribution by origin class."""
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(data=df, x='is_ai_generated', y='simulated_perplexity', hue='is_ai_generated', palette='colorblind', legend=False, ax=ax)
    ax.set_title('Perplexity Distribution by Class')
    ax.set_xlabel('Text Origin')
    ax.set_ylabel('Simulated Perplexity')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Human', 'AI'])
    fig.tight_layout()
    return fig

def plot_readability_distribution(df):
    """Plots readability score histogram by origin."""
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(data=df, x='readability_score', hue=df['is_ai_generated'].map({0: 'Human', 1: 'AI'}), kde=True, palette='colorblind', alpha=0.5, ax=ax)
    ax.set_title('Readability Score Distribution by Text Origin')
    ax.set_xlabel('Readability score')
    ax.set_ylabel('Count')
    fig.tight_layout()
    return fig

def plot_length_scatterplot(df):
    """Plots word count vs sentence length."""
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.scatterplot(data=df, x='word_count', y='avg_sentence_length', hue=df['is_ai_generated'].map({0: 'Human', 1: 'AI'}), alpha=0.6, palette='colorblind', ax=ax)
    ax.set_title('Document Length vs Sentence Length')
    ax.set_xlabel('Total Word Count')
    ax.set_ylabel('Average Sentence Length')
    ax.legend(title='Text Origin')
    fig.tight_layout()
    return fig

def plot_correlation_matrix(df):
    """Plots Pearson correlation heatmap of numeric features."""
    fig, ax = plt.subplots(figsize=(6, 5))
    numeric_columns = ['is_ai_generated', 'word_count', 'readability_score', 'avg_sentence_length', 'simulated_perplexity']
    corr_matrix = df[numeric_columns].corr()
    corr_matrix.rename(index={'is_ai_generated': 'is_ai (Target)'}, columns={'is_ai_generated': 'is_ai (Target)'}, inplace=True)
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
    ax.set_title('Pearson Correlation Matrix')
    fig.tight_layout()
    return fig

def plot_source_model_distribution(df):
    """Plots perplexity boxplot by specific source model."""
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.boxplot(data=df, x='source_model', y='simulated_perplexity', hue='source_model', palette='colorblind', legend=False, ax=ax)
    ax.set_title('Perplexity Distribution by Source Model')
    ax.set_xlabel('Source Model')
    ax.set_ylabel('Simulated Perplexity')
    ax.tick_params(axis='x', rotation=25)
    fig.tight_layout()
    return fig