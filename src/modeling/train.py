"""Neural network model definition and training loop for AI text detection.

This module provides the `Net` PyTorch Lightning module representing a multi-layer
perceptron classifier, as well as the `train` routine with checkpointing and logging.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint

class Net(pl.LightningModule):
    """Multi-Layer Perceptron (MLP) binary classifier for AI vs human text detection.

    Consists of an input linear layer (10 inputs -> 64 hidden units), ReLU activation,
    and a final linear layer producing binary classification logits (2 outputs).

    Attributes:
        l1 (nn.Sequential): Sequential feed-forward layers.
        loss_fn (nn.CrossEntropyLoss): Cross-entropy loss function.
        train_losses (list[float]): History of epoch average training losses.
        train_accs (list[float]): History of epoch average training accuracies.
        training_step_outputs (list[dict]): Temporary metric container for step-level outputs.
    """

    def __init__(self):
        """Initializes the network layers, loss function, and tracking metrics."""
        super().__init__()
        self.l1 = nn.Sequential(nn.Linear(10, 64), nn.ReLU(), nn.Linear(64, 2))
        self.loss_fn = nn.CrossEntropyLoss()
        self.train_losses = []
        self.train_accs = []
        self.training_step_outputs = []

    def forward(self, x):
        """Performs a forward pass through the network.

        Args:
            x (torch.Tensor): Input feature tensor of shape `(batch_size, 10)`.

        Returns:
            torch.Tensor: Unnormalized output logits of shape `(batch_size, 2)`.
        """
        return self.l1(x)

    def training_step(self, batch, batch_idx):
        """Executes a single training step.

        Computes Cross-Entropy loss and accuracy, logs step and epoch values,
        and records batch metrics.

        Args:
            batch (tuple[torch.Tensor, torch.Tensor]): Tuple containing batch features and labels.
            batch_idx (int): Index of the current batch.

        Returns:
            torch.Tensor: Computed training loss for backpropagation.
        """
        xb, yb = batch
        out = self(xb)
        loss = self.loss_fn(out, yb)
        preds = torch.argmax(out, dim=1)
        acc = (preds == yb).float().mean()
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log('train_acc', acc, prog_bar=True, on_step=True, on_epoch=True)
        self.training_step_outputs.append({
            'loss': loss.item(),
            'acc': acc.item()
        })
        return loss

    def on_train_epoch_end(self):
        """Aggregates and logs epoch-level average loss and accuracy."""
        if self.training_step_outputs:
            avg_loss = sum([x['loss'] for x in self.training_step_outputs]) / len(self.training_step_outputs)
            avg_acc = sum([x['acc'] for x in self.training_step_outputs]) / len(self.training_step_outputs)
            self.train_losses.append(avg_loss)
            self.train_accs.append(avg_acc)
            self.log('epoch_train_loss', avg_loss)
            self.log('epoch_train_acc', avg_acc)
            self.training_step_outputs.clear()
    
    def configure_optimizers(self):
        """Configures the optimizer for model training.

        Returns:
            torch.optim.Optimizer: Adam optimizer initialized with model parameters.
        """
        return optim.Adam(self.parameters())
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """Runs prediction on a batch during trainer inference.

        Args:
            batch (tuple[torch.Tensor, torch.Tensor]): Tuple of features and ground-truth labels.
            batch_idx (int): Index of the current batch.
            dataloader_idx (int, optional): Index of the dataloader. Defaults to 0.

        Returns:
            dict[str, torch.Tensor]: Dictionary containing 'predictions', 'probabilities',
                'labels', and 'features'.
        """
        x, y = batch
        logits = self(x)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)
        return {
            'predictions': preds,
            'probabilities': probs,
            'labels': y,
            'features': x
        }
def train(data_module,model_path,logger_path):
    """Initializes and trains the Net classifier using PyTorch Lightning.

    Configures a CSVLogger, best-model ModelCheckpoint callback, and CPU Trainer,
    then fits the model for 20 epochs on the provided DataModule.

    Args:
        data_module (pl.LightningDataModule): Initialized DataModule providing training data.
        model_path (str): Directory where model checkpoints will be stored.
        logger_path (str): Directory where CSV training logs will be saved.

    Returns:
        pl.Trainer: Trained Lightning Trainer instance.

    Example:
        >>> dm = CSVDataModule("data/benchmark.csv")
        >>> trainer = train(dm, model_path="models/", logger_path="logs/")
    """
    net = Net()
    logger = CSVLogger(save_dir=logger_path, name="train_log")
    checkpoint_callback = ModelCheckpoint(
        dirpath=model_path,
        filename='best-{epoch:02d}-{train_loss:.2f}',
        monitor='train_loss',
        mode='min',
        save_top_k=1,
        save_last=True
    )
    trainer = pl.Trainer(
        max_epochs=20, 
        callbacks=[checkpoint_callback],
        default_root_dir=model_path, 
        accelerator="cpu", devices=1, 
        enable_progress_bar=True,
        logger=logger
    )
    trainer.fit(net, datamodule=data_module)
    return trainer