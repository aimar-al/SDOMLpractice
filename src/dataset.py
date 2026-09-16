import torch
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

class CSVDataset(Dataset):
    def __init__(self, csv_path="../data/benchmark_ai_detection_multimodel_2026.csv", transform=None):
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        self.feature_columns = self.df.columns.drop(['id', 'timestamp', 'is_ai_generated','text_content', 'source_model'])
        self.label_column = 'is_ai_generated'
        self.X = pd.get_dummies(self.df[self.feature_columns]).values.astype('float32')
        self.y = self.df[self.label_column].values.astype('int64')
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        X = torch.from_numpy(self.X[idx])
        y = torch.tensor(self.y[idx], dtype=torch.long)
        return X, y

class CSVDataModule(pl.LightningDataModule):
    def __init__(self, data_dir="../data/benchmark_ai_detection_multimodel_2026.csv", batch_size=64, num_workers=8):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transforms.ToTensor()
        
    def setup(self, stage=None):
        self.train_ds = CSVDataset(self.data_dir)
    
    def train_dataloader(self, shuffle=True):
        return DataLoader(
            self.train_ds, 
            batch_size=self.batch_size, 
            num_workers=0, 
            shuffle=shuffle
        )   
    
    def predict_dataloader(self):
        return self.train_dataloader(shuffle=False)
        