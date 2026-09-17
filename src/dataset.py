"""Dataset and DataModule definitions for AI text detection.

This module handles loading tabular text metrics from CSV files,
converting features into tensors, and supplying PyTorch DataLoaders
via a LightningDataModule.
"""

import torch
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

class CSVDataset(Dataset):
    """PyTorch Dataset for tabular AI text detection data.

    Extracts statistical text features (e.g. word count, perplexity,
    readability) and binary AI-generation labels from a CSV file.

    Attributes:
        df (pd.DataFrame): Raw dataframe loaded from the CSV file.
        transform (callable, optional): Optional transform applied to samples.
        feature_columns (pd.Index): Selected feature column names.
        label_column (str): Name of the label column ('is_ai_generated').
        X (np.ndarray): Preprocessed feature array (float32).
        y (np.ndarray): Target labels array (int64).
    """

    def __init__(self, csv_path="../data/benchmark_ai_detection_multimodel_2026.csv", transform=None):
        """Initializes the dataset from a CSV file.

        Args:
            csv_path (str): Path to the CSV dataset file.
            transform (callable, optional): Optional data transformation. Defaults to None.
        """
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        self.feature_columns = self.df.columns.drop(['id', 'timestamp', 'is_ai_generated','text_content', 'source_model'])
        self.label_column = 'is_ai_generated'
        self.X = pd.get_dummies(self.df[self.feature_columns]).values.astype('float32')
        self.y = self.df[self.label_column].values.astype('int64')
        
    def __len__(self):
        """Returns the total number of samples in the dataset.

        Returns:
            int: Number of samples in the dataset.
        """
        return len(self.df)
    
    def __getitem__(self, idx):
        """Retrieves the feature tensor and label for a sample at the given index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Tuple containing the feature tensor (X)
                and the target label tensor (y).
        """
        X = torch.from_numpy(self.X[idx])
        y = torch.tensor(self.y[idx], dtype=torch.long)
        return X, y

class CSVDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for managing training and prediction DataLoaders.

    Encapsulates dataset preparation and DataLoader instantiation for model training
    and inference pipelines.

    Example:
        >>> data_module = CSVDataModule(data_dir="data/dataset.csv", batch_size=32)
        >>> data_module.setup()
        >>> loader = data_module.train_dataloader()
    """

    def __init__(self, data_dir="../data/benchmark_ai_detection_multimodel_2026.csv", batch_size=64, num_workers=8):
        """Initializes the DataModule.

        Args:
            data_dir (str): Path to the CSV dataset file.
            batch_size (int, optional): Batch size for dataloaders. Defaults to 64.
            num_workers (int, optional): Number of subprocesses for data loading. Defaults to 8.
        """
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transforms.ToTensor()
        
    def setup(self, stage=None):
        """Initializes the dataset instance for the requested pipeline stage.

        Args:
            stage (str, optional): Stage identifier ('fit', 'validate', 'test', or 'predict').
                Defaults to None.
        """
        self.train_ds = CSVDataset(self.data_dir)
    
    def train_dataloader(self, shuffle=True):
        """Builds the training DataLoader.

        Args:
            shuffle (bool, optional): Whether to shuffle samples each epoch. Defaults to True.

        Returns:
            DataLoader: PyTorch DataLoader configured for training.
        """
        return DataLoader(
            self.train_ds, 
            batch_size=self.batch_size, 
            num_workers=0, 
            shuffle=shuffle
        )   
    
    def predict_dataloader(self):
        """Builds the prediction DataLoader with shuffling disabled.

        Returns:
            DataLoader: PyTorch DataLoader configured for inference.
        """
        return self.train_dataloader(shuffle=False)
        