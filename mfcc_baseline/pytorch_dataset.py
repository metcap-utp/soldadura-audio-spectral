import torch
from torch.utils.data import Dataset
import numpy as np


class AudioDataset(Dataset):
    def __init__(self, X: np.ndarray, y: dict = None, task: str = None):
        self.X = torch.FloatTensor(X)
        self.y = y
        self.task = task
        
        if y is not None and task is not None:
            self.labels = torch.LongTensor(y[task])
        else:
            self.labels = None
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        if self.labels is not None:
            return self.X[idx], self.labels[idx]
        return self.X[idx]
    
    def get_feature_dim(self):
        return self.X.shape[1]
    
    def get_num_classes(self):
        if self.labels is not None:
            return len(torch.unique(self.labels))
        return None


class MultiTaskDataset(Dataset):
    def __init__(self, X: np.ndarray, y: dict):
        self.X = torch.FloatTensor(X)
        self.y = {k: torch.LongTensor(v) for k, v in y.items()}
        self.tasks = list(y.keys())
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        labels = {task: self.y[task][idx] for task in self.tasks}
        return self.X[idx], labels
    
    def get_feature_dim(self):
        return self.X.shape[1]
    
    def get_num_classes_per_task(self):
        return {task: len(torch.unique(labels)) for task, labels in self.y.items()}
