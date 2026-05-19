"""
Utilidades para construir datasets a partir de generadores y perfiles.

Funciones reutilizables entre el script centralizado y el federado:

  build_dataset_from_profiles  → mezcla trayectorias de varios perfiles
                                 (uso típico: baseline centralizado)
  build_dataset_for_profile    → trayectorias de UN solo perfil
                                 (uso típico: cliente federado o ablations)
"""
import numpy as np
from .datasets import TrajectoryDataset
from .generators.factory import build_generator


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

    Reparte total_samples equitativamente entre los perfiles.

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
            seed=seed,
        )
        per_profile[profile_name] = trajs
        all_trajs.append(trajs)

    combined = np.concatenate(all_trajs, axis=0)