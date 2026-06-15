"""
Utilidades para construir datasets a partir de generadores y perfiles.

Funciones reutilizables entre el script centralizado y el federado:

  build_dataset_for_profile    → trayectorias de UN solo perfil
                                 (uso típico: cliente federado o ablations)
  build_dataset_from_profiles  → mezcla trayectorias de varios perfiles
                                 (uso típico: baseline centralizado)
  build_centralized_split      → train mezclado + val SEPARADO por perfil
                                 + curva de referencia por perfil
                                 (uso típico: baseline que compara generalización
                                  del modelo conjunto a cada hospital)

Cada perfil recibe una SEMILLA DERIVADA distinta para que sus datos sean
estadísticamente independientes (simula hospitales que no comparten pacientes).
"""
import zlib
from dataclasses import dataclass

import numpy as np

from .datasets import TrajectoryDataset
from .generators.factory import build_generator


def profile_seed(base_seed: int, profile_name: str) -> int:
    """Semilla determinista y distinta por perfil derivada de la semilla base.

    Usa crc32 (estable entre procesos, a diferencia de hash() de Python) para
    que dos perfiles nunca compartan el mismo flujo pseudoaleatorio.
    """
    return (base_seed + zlib.crc32(profile_name.encode())) % (2 ** 32)


def build_dataset_for_profile(
    skill: str,
    profile_name: str,
    n_samples: int,
    trajectory_length: int = 50,
    seed: int = 42,
) -> tuple[TrajectoryDataset, np.ndarray]:
    """
    Construye dataset con datos de UN perfil.

    Útil para:
      - Cliente federado (cada cliente ve solo un perfil)
      - Ablations (entrenar solo con Hospital A para comparar)

    Returns:
        (dataset, raw_trajectories)  — el array crudo se devuelve por si
        scripts/análisis posteriores lo necesitan sin reconstruirlo.
    """
    generator = build_generator(
        skill=skill,
        profile_name=profile_name,
        trajectory_length=trajectory_length,
        seed=seed,
    )
    trajectories = generator.generate(n_samples)
    return TrajectoryDataset(trajectories), trajectories


def build_dataset_from_profiles(
    skill: str,
    profile_names: list[str],
    total_samples: int,
    trajectory_length: int = 50,
    seed: int = 42,
) -> tuple[TrajectoryDataset, dict[str, np.ndarray]]:
    """
    Construye un dataset centralizado mezclando datos de varios perfiles.

    Reparte total_samples equitativamente entre los perfiles. Cada perfil usa
    una semilla derivada distinta → datos independientes entre hospitales.

    Returns:
        (dataset, trajectories_per_profile)
        El dict permite calcular métricas por perfil sin regenerar.
    """
    samples_per_profile = total_samples // len(profile_names)
    per_profile = {}
    all_trajs = []

    for profile_name in profile_names:
        _, trajs = build_dataset_for_profile(
            skill=skill,
            profile_name=profile_name,
            n_samples=samples_per_profile,
            trajectory_length=trajectory_length,
            seed=profile_seed(seed, profile_name),
        )
        per_profile[profile_name] = trajs
        all_trajs.append(trajs)

    combined = np.concatenate(all_trajs, axis=0)
    return TrajectoryDataset(combined), per_profile


@dataclass
class CentralizedSplit:
    """Resultado de build_centralized_split.

    train:            dataset de entrenamiento con datos de TODOS los perfiles.
    val_combined:     dataset de validación con todos los perfiles (selección
                      del mejor checkpoint / métrica global).
    val_per_profile:  dataset de validación SEPARADO por perfil. Permite ver si
                      el modelo conjunto generaliza a cada hospital o solo al
                      mayoritario — la pregunta central del experimento.
    references:       curva ideal (R, 2) por perfil, para medir desviación.
    train_per_profile: trayectorias crudas de train por perfil (calidad/plots).
    """
    train: TrajectoryDataset
    val_combined: TrajectoryDataset
    val_per_profile: dict[str, TrajectoryDataset]
    references: dict[str, np.ndarray]
    train_per_profile: dict[str, np.ndarray]


def build_centralized_split(
    skill: str,
    profile_names: list[str],
    total_samples: int,
    val_split: float = 0.2,
    trajectory_length: int = 50,
    seed: int = 42,
) -> CentralizedSplit:
    """
    Construye el split centralizado con validación SEPARADA por perfil.

    A diferencia de build_dataset_from_profiles + random_split (que mezcla los
    perfiles antes de partir), aquí cada perfil se parte en train/val por
    separado. Así la val de cada hospital contiene solo su estilo y se puede
    medir la generalización del modelo conjunto hospital a hospital.
    """
    samples_per_profile = total_samples // len(profile_names)
    train_trajs, val_trajs = [], []
    val_per_profile: dict[str, TrajectoryDataset] = {}
    references: dict[str, np.ndarray] = {}
    train_per_profile: dict[str, np.ndarray] = {}

    for profile_name in profile_names:
        generator = build_generator(
            skill=skill,
            profile_name=profile_name,
            trajectory_length=trajectory_length,
            seed=profile_seed(seed, profile_name),
        )
        trajs = generator.generate(samples_per_profile)

        n_val = int(len(trajs) * val_split)
        prof_val, prof_train = trajs[:n_val], trajs[n_val:]

        train_trajs.append(prof_train)
        val_trajs.append(prof_val)
        train_per_profile[profile_name] = prof_train
        val_per_profile[profile_name] = TrajectoryDataset(prof_val)
        references[profile_name] = generator.reference_curve

    return CentralizedSplit(
        train=TrajectoryDataset(np.concatenate(train_trajs, axis=0)),
        val_combined=TrajectoryDataset(np.concatenate(val_trajs, axis=0)),
        val_per_profile=val_per_profile,
        references=references,
        train_per_profile=train_per_profile,
    )
