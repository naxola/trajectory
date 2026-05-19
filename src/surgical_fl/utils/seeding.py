"""
Control de determinismo para reproducibilidad experimental
Se fija la semilla para una sola llamada
Llamar set_global_seed() al inicio de cualquier script
"""
import os
import random
import numpy as np
import torch

def set_global_seed(seed: int = 42, deterministic: bool = True) -> None:
    """
    Se fijan todas las fuentes de aleaotriedad relevantes.
        Args:
        seed:           semilla numérica
        deterministic:  si True, fuerza determinismo en operaciones CUDA
                        (más lento pero exactamente reproducible)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["PYTHONHASHSEED"] = str(seed)