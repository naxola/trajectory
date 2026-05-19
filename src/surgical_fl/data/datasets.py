"""
Este es un wrapper de pytorch dataset para el generador de datos de trayectorias quirúrgicas.
"""
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from .generators.base import SurgicalDataGenerator

class TrajectoryDataset(Dataset):
    """
    Este será el dataset de trayectorias quirúrgicas para el entrenamiento con pytorch.
    Dado el punto t, predice el punto t+1
     - input (x_t, y_t) -> shape(T-1,2)
     - output (x_{t+1}, y_{t+1}) -> shape(T-1,2)
    para comenzar sencillo, se puede usar ocmo autoencoder input = outut = trayectoria completa.
    Args:
     trajectories: array (N, T, 2 ) con las trayectorias.
    """
    def __init__(self, trajectories: np.ndarray):
        # (N, T, 2) → inputs: (N, T-1, 2), targets: (N, T-1, 2)
        self.inputs = torch.tensor(trajectories[:,:-1,:],dtype =torch.float32)
        self.outputs = torch.tensor(trajectories[:,1:,:],dtype =torch.float32)
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx)->tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[idx], self.outputs[idx]

def build_dataloader(generator: SurgicalDataGenerator, n_samples: int, batch_size: int = 32, shuffle: bool = True) -> DataLoader:
    """
    Genera trayectorias y construye un DataLoader.
    Args:
        generator: Generador de trayectorias
        n_samples: Número de trayectorias a generar
        batch_size: Tamaño del batch
        shuffle: Si se debe barajar el dataset
    Returns:
        DataLoader con las trayectorias
    """
    trajectories = generator.generate(n_samples)
    dataset = TrajectoryDataset(trajectories)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
     