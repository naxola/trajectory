"""
Configuración de experimentos.

Cada experimento es un fichero TOML versionable en experiments/.
La dataclass ExperimentConfig sirve como contrato único de hiperparámetros
para los scripts centralizado y federado.

Estructura del TOML esperado:

    name = "cutting_baseline"
    skill = "cutting"
    profiles = ["hospital_a", "hospital_b"]

    [model]
    name = "mlp"
    hidden_dim = 64
    dropout = 0.1

    [data]
    num_samples = 400
    trajectory_length = 50
    val_split = 0.2

    [training]
    epochs = 20
    batch_size = 32
    learning_rate = 0.001
    seed = 42
"""
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ModelConfig:
    name: str = "mlp"
    hidden_dim: int = 64
    dropout: float = 0.1


@dataclass
class DataConfig:
    num_samples: int = 400
    trajectory_length: int = 50
    val_split: float = 0.2


@dataclass
class TrainingConfig:
    epochs: int = 20
    batch_size: int = 32
    learning_rate: float = 0.001
    seed: int = 42


@dataclass
class ExperimentConfig:
    name: str = "default"
    skill: str = "cutting"
    profiles: list[str] = field(default_factory=lambda: ["hospital_a", "hospital_b"])
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    @classmethod
    def from_toml(cls, path: str | Path) -> "ExperimentConfig":
        """Carga configuración desde fichero TOML."""
        path = Path(path)
        with open(path, "rb") as f:
            raw = tomllib.load(f)

        return cls(
            name=raw.get("name", path.stem),
            skill=raw.get("skill", "cutting"),
            profiles=raw.get("profiles", ["hospital_a", "hospital_b"]),
            model=ModelConfig(**raw.get("model", {})),
            data=DataConfig(**raw.get("data", {})),
            training=TrainingConfig(**raw.get("training", {})),
        )