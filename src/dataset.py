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
    """

    def __init__(self, csv_path="../data/benchmark_ai_detection_multimodel_2026.csv", transform=None):
        """Initializes the dataset from a CSV file.

        Args:
            csv_path (str): Path to the CSV dataset file.
            transform (callable, optional): Optional data transformation. Defaults to None.
        """
        self._df = pd.read_csv(csv_path)
        self._transform = transform
        self._feature_columns = self._df.columns.drop(['id', 'timestamp', 'is_ai_generated','text_content', 'source_model'])
        self._label_column = 'is_ai_generated'
        self._X = pd.get_dummies(self._df[self._feature_columns]).values.astype('float32')
        self._y = self._df[self._label_column].values.astype('int64')
        
    def __len__(self):
        """Returns the total number of samples in the dataset.

        Returns:
            int: Number of samples in the dataset.
        """
        return len(self._df)
    
    def __getitem__(self, idx):
        """Retrieves the feature tensor and label for a sample at the given index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Tuple containing the feature tensor (X)
                and the target label tensor (y).
        """
        X = torch.from_numpy(self._X[idx])
        y = torch.tensor(self._y[idx], dtype=torch.long)
        return X, y

    @property
    def y(self):
        """np.ndarray: Target labels array (int64)."""
        return self._y

    @property
    def X(self):
        """np.ndarray: Preprocessed feature array (float32)."""
        return self._X

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
        self._data_dir = data_dir
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._transform = transforms.ToTensor()
        self._train_ds = None
        
    def setup(self, stage=None):
        """Initializes the dataset instance for the requested pipeline stage.

        Args:
            stage (str, optional): Stage identifier ('fit', 'validate', 'test', or 'predict').
                Defaults to None.
        """
        self._train_ds = CSVDataset(self._data_dir)
    
    def train_dataloader(self, shuffle=True):
        """Builds the training DataLoader.

        Args:
            shuffle (bool, optional): Whether to shuffle samples each epoch. Defaults to True.

        Returns:
            DataLoader: PyTorch DataLoader configured for training.
        """
        return DataLoader(
            self._train_ds, 
            batch_size=self._batch_size, 
            num_workers=0, 
            shuffle=shuffle
        )   
    
    def predict_dataloader(self):
        """Builds the prediction DataLoader with shuffling disabled.

        Returns:
            DataLoader: PyTorch DataLoader configured for inference.
        """
        return self.train_dataloader(shuffle=False)

    @property
    def train_ds(self):
        """CSVDataset: Underlying training dataset instance."""
        return self._train_ds
        