"""
Persistencia de resultados experimentales.

Cada ejecución del entrenamiento crea un directorio único bajo outputs/runs/
con la siguiente estructura:

    outputs/runs/<experiment>_<timestamp>/
    ├── config.json         ← config serializada (qué se ejecutó)
    ├── metrics.json        ← resultados (loss por epoch, métricas finales)
    ├── checkpoints/        ← modelo guardado (mejor + final)
    └── figures/            ← curva de aprendizaje, predicciones, etc.
De esta forma podré comparar runs a posteriori sin parsear logs.
"""
import json
import os
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path

import torch


class RunDirectory:
    """
    Gestiona el directorio de salida de una ejecución.

    Uso:
        run = RunDirectory.create("cutting_baseline")
        run.save_config(config)
        run.save_metrics({"final_loss": 0.001})
        run.save_checkpoint(model, name="final")
        path = run.figure_path("learning_curve.png")
    """

    def __init__(self, root: Path):
        self.root = Path(root)
        self.checkpoints = self.root / "checkpoints"
        self.figures = self.root / "figures"
        self.checkpoints.mkdir(parents=True, exist_ok=True)
        self.figures.mkdir(parents=True, exist_ok=True)

    @classmethod
    def create(
        cls,
        experiment_name: str,
        base_dir: str = "outputs/runs",
    ) -> "RunDirectory":
        """Crea un directorio nuevo con timestamp único."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = Path(base_dir) / f"{experiment_name}_{timestamp}"
        path.mkdir(parents=True, exist_ok=True)
        return cls(path)

    def save_config(self, config) -> None:
        """Serializa la config (dataclass) como JSON."""
        if is_dataclass(config):
            data = asdict(config)
        elif isinstance(config, dict):
            data = config
        else:
            raise TypeError(f"Config debe ser dataclass o dict, no {type(config)}")

        with open(self.root / "config.json", "w") as f:
            json.dump(data, f, indent=2)

    def save_metrics(self, metrics: dict) -> None:
        """Guarda métricas finales y/o por epoch."""
        with open(self.root / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

    def save_checkpoint(self, model: torch.nn.Module, name: str = "final") -> Path:
        """Guarda los pesos del modelo. name = 'final', 'best', 'epoch_10', etc."""
        path = self.checkpoints / f"{name}.pt"
        torch.save(model.state_dict(), path)
        return path

    def figure_path(self, filename: str) -> str:
        return str(self.figures / filename)