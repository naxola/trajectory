"""
Visualización de trayectorias quirúrgicas.

Funciones reutilizables entre script centralizado y federado.
"""
import os
import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from surgical_fl.data.generators.factory import build_generator


PROFILE_COLORS = ["#2563EB", "#DC2626", "#16A34A", "#D97706", "#7C3AED", "#0891B2"]


def plot_learning_curve(
    losses: list[float],
    output_path: str,
    title: str = "Curva de aprendizaje",
) -> str:
    """Guarda PNG con la evolución de la loss por epoch/ronda."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(losses, color="#2563EB", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (MSE)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_predictions_per_profile(
    model: torch.nn.Module,
    skill: str,
    profiles: list[str],
    output_path: str,
    trajectory_length: int = 50,
    device: torch.device | None = None,
    title: str | None = None,
) -> str:
    """
    Una trayectoria real vs predicha por cada perfil registrado.

    Args:
        model:             modelo ya entrenado
        skill:             nombre de la habilidad
        profiles:          lista de identificadores de perfil
        output_path:       ruta del PNG resultante
        trajectory_length: longitud de cada trayectoria a visualizar
        device:            torch.device (default: cpu)
        title:             título del gráfico (opcional)
    """
    device = device or torch.device("cpu")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    model.eval()

    n = len(profiles)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, profile_name, color in zip(axes, profiles, PROFILE_COLORS):
        gen = build_generator(
            skill=skill,
            profile_name=profile_name,
            trajectory_length=trajectory_length,
            seed=0,
        )
        traj   = gen.generate_one()
        inputs = torch.tensor(traj[:-1], dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            predicted = model(inputs).squeeze(0).cpu().numpy()

        ax.plot(traj[:, 0], traj[:, 1], "o-", color=color,
                alpha=0.6, markersize=3, label="Real", linewidth=1.5)
        ax.plot(predicted[:, 0], predicted[:, 1], "s--", color="gray",
                alpha=0.8, markersize=3, label="Predicha", linewidth=1.5)
        ax.set_title(profile_name.replace("_", " ").title())
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal")

    if title:
        plt.suptitle(title, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path