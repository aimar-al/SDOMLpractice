import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint

class Net(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Sequential(nn.Linear(10, 64), nn.ReLU(), nn.Linear(64, 2))
        self.loss_fn = nn.CrossEntropyLoss()
        self.train_losses = []
        self.train_accs = []
        self.training_step_outputs = []

    def forward(self, x):
        return self.l1(x)

    def training_step(self, batch, batch_idx):
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
        if self.training_step_outputs:
            avg_loss = sum([x['loss'] for x in self.training_step_outputs]) / len(self.training_step_outputs)
            avg_acc = sum([x['acc'] for x in self.training_step_outputs]) / len(self.training_step_outputs)
            self.train_losses.append(avg_loss)
            self.train_accs.append(avg_acc)
            self.log('epoch_train_loss', avg_loss)
            self.log('epoch_train_acc', avg_acc)
            self.training_step_outputs.clear()
    
    def configure_optimizers(self):
        return optim.Adam(self.parameters())
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
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