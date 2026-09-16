import torch
from modeling.train import Net

def predict(trainer,data_module,path):
    net = Net.load_from_checkpoint(path)
    predictions = trainer.predict(net, data_module)
    y_pred = torch.cat([pred['predictions'] for pred in predictions])
    y_prob = torch.cat([prob['probabilities'] for prob in predictions])
    return y_pred, y_prob