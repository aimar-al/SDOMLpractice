"""Prediction and inference utilities for the AI text classifier.

Provides functions to load trained model checkpoints and generate
predictions and class probabilities over a dataset.
"""

import torch
try:
    from modeling.train import Net
except ImportError:
    from .train import Net

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

def predict_single(net, x_features):
    """Predicts class and probabilities for a single sample.

    Args:
        net (Net): Trained Net model instance.
        x_features (torch.Tensor or numpy.ndarray): Feature vector of shape (10,) or (1, 10).

    Returns:
        tuple[int, list[float]]: Predicted class (0=Human, 1=AI) and probabilities [prob_human, prob_ai].
    """
    net.eval()
    if not isinstance(x_features, torch.Tensor):
        x_tensor = torch.tensor(x_features, dtype=torch.float32)
    else:
        x_tensor = x_features.clone().detach().float()
    if x_tensor.ndim == 1:
        x_tensor = x_tensor.unsqueeze(0)
    with torch.no_grad():
        logits = net(x_tensor)
        probs = torch.softmax(logits, dim=1)[0].tolist()
        pred = int(torch.argmax(logits, dim=1).item())
    return pred, probs