"""
Lógica de entrenamiento local.

Esta función NO conoce Flower, NO conoce federación.
Es PyTorch puro: train loop + evaluate.

Flower la llamará desde el ClientApp, pero también
puede llamarse desde scripts/train_centralized.py para validar
que el modelo aprende antes de federar.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """
    Entrena el modelo durante una epoch completa.

    Returns:
        loss media de la epoch
    """
    model.train()
    criterion = nn.MSELoss()
    total_loss = 0.0

    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        predictions = model(inputs)
        loss = criterion(predictions, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[float, dict[str, float]]:
    """
    Evalúa el modelo en un dataloader.

    Returns:
        (loss, metrics_dict)
        metrics incluye 'mse' y 'rmse'
    """
    model.eval()
    criterion = nn.MSELoss()
    total_loss = 0.0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            predictions = model(inputs)
            loss = criterion(predictions, targets)
            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    metrics = {
        "mse": avg_loss,
        "rmse": avg_loss ** 0.5,
    }
    return avg_loss, metrics


def train_local(
    model: nn.Module,
    dataloader: DataLoader,
    epochs: int,
    learning_rate: float,
    device: torch.device,
) -> tuple[nn.Module, list[float]]:
    """
    Entrenamiento completo de N epochs. Usado por el ClientApp de Flower.

    Returns:
        (modelo entrenado, lista de losses por epoch)
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    losses = []

    for epoch in range(epochs):
        loss = train_one_epoch(model, dataloader, optimizer, device)
        losses.append(loss)

    return model, losses