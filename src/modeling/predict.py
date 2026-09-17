"""Prediction and inference utilities for the AI text classifier.

Provides functions to load trained model checkpoints and generate
predictions and class probabilities over a dataset.
"""

import torch
from modeling.train import Net

def predict(trainer,data_module,path):
    """Loads a saved checkpoint and runs batch inference on a DataModule.

    Args:
        trainer (pl.Trainer): PyTorch Lightning Trainer used to execute prediction.
        data_module (pl.LightningDataModule): DataModule containing the inference dataset.
        path (str): Path to the saved model checkpoint file (`.ckpt`).

    Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - y_pred (torch.Tensor): 1D tensor of predicted class labels (0 or 1).
            - y_prob (torch.Tensor): 2D tensor of predicted class probabilities.

    Example:
        >>> y_pred, y_prob = predict(trainer, data_module, "models/best.ckpt")
    """
    net = Net.load_from_checkpoint(path)
    predictions = trainer.predict(net, data_module)
    y_pred = torch.cat([pred['predictions'] for pred in predictions])
    y_prob = torch.cat([prob['probabilities'] for prob in predictions])
    return y_pred, y_prob