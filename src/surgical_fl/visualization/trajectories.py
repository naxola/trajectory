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


def _autoregressive_rollout(
    model: torch.nn.Module,
    start_point: np.ndarray,
    steps: int,
    device: torch.device,
) -> np.ndarray:
    """Despliega el modelo realimentando su propia predicción.

    Parte del primer punto real y predice toda la trayectoria sin volver a ver
    datos reales: p_{t+1} = model(p_t). Es la prueba HONESTA de si el modelo
    capturó la trayectoria — a diferencia del teacher forcing (predecir un paso
    desde cada punto real), que se ve engañosamente bien.
    """
    cur = torch.tensor(start_point, dtype=torch.float32).view(1, 1, -1).to(device)
    points = [np.asarray(start_point, dtype=np.float32)]
    with torch.no_grad():
        for _ in range(steps - 1):
            cur = model(cur)
            points.append(cur.squeeze().cpu().numpy())
    return np.stack(points, axis=0)


def plot_predictions_per_profile(
    model: torch.nn.Module,
    skill: str,
    profiles: list[str],
    output_path: str,
    trajectory_length: int = 50,
    device: torch.device | None = None,
    references: dict[str, np.ndarray] | None = None,
    title: str | None = None,
) -> str:
    """
    Trayectoria real vs ROLLOUT autorregresivo del modelo, por cada perfil.

    Args:
        model:             modelo ya entrenado
        skill:             nombre de la habilidad
        profiles:          lista de identificadores de perfil
        output_path:       ruta del PNG resultante
        trajectory_length: longitud de cada trayectoria a visualizar
        device:            torch.device (default: cpu)
        references:        curva ideal por perfil para superponerla (opcional)
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
        traj = gen.generate_one()
        rollout = _autoregressive_rollout(model, traj[0], len(traj), device)

        if references and profile_name in references:
            ref = references[profile_name]
            ax.plot(ref[:, 0], ref[:, 1], "-", color="black",
                    alpha=0.25, linewidth=3, label="Referencia ideal")

        ax.plot(traj[:, 0], traj[:, 1], "o-", color=color,
                alpha=0.6, markersize=3, label="Real", linewidth=1.5)
        ax.plot(rollout[:, 0], rollout[:, 1], "s--", color="gray",
                alpha=0.85, markersize=3, label="Rollout", linewidth=1.5)
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


def _vertical_deviations(trajectories: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Desviación media (vertical) de cada trayectoria respecto a la curva ideal.

    Como las incisiones son funciones de x, se compara y(traj) con y(referencia)
    interpolada en las mismas x. Es rápido y coherente con la banda vertical que
    se rellena en el plot.
    """
    rx, ry = reference[:, 0], reference[:, 1]
    return np.array([
        float(np.mean(np.abs(t[:, 1] - np.interp(t[:, 0], rx, ry))))
        for t in trajectories
    ])


def plot_training_dataset(
    trajectories: np.ndarray,
    reference: np.ndarray,
    output_path: str,
    title: str | None = None,
    max_background: int = 200,
) -> str:
    """
    Visualiza un dataset de cortes (spline, curvo o recto) frente a su ideal.

    - Nube de fondo: el resto del dataset (submuestreado a max_background) en gris.
    - Las DOS curvas con mayor desviación respecto a la curva ideal, resaltadas.
    - La curva ideal (referencia) ploteada por encima de todo.

    Args:
        trajectories: (N, T, 2) trayectorias del dataset.
        reference:    (R, 2) curva ideal contra la que se mide la desviación.
        output_path:  ruta del PNG resultante.
        title:        título del gráfico (opcional).
        max_background: nº máximo de trayectorias dibujadas como nube de fondo.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    trajectories = np.asarray(trajectories)

    # Desviación de cada corte respecto a la incisión ideal → los dos extremos.
    deviations = _vertical_deviations(trajectories, reference)
    order = np.argsort(deviations)
    hi, second = int(order[-1]), int(order[-2])  # las dos más desviadas

    fig, ax = plt.subplots(figsize=(9, 5))

    # ── Nube de fondo (resto del dataset) ─────────────────────────────────────
    n = len(trajectories)
    if n > max_background:
        bg_idx = np.linspace(0, n - 1, max_background).astype(int)
    else:
        bg_idx = np.arange(n)
    for i in bg_idx:
        if i in (hi, second):
            continue
        t = trajectories[i]
        ax.plot(t[:, 0], t[:, 1], color="#94A3B8", alpha=0.15, linewidth=0.7)

    # ── Curvas extremas resaltadas ───────────────────────────────────────────
    for idx, lab in ((hi, "Extrema 1"), (second, "Extrema 2")):
        c = trajectories[idx]
        ax.plot(c[:, 0], c[:, 1], color="#1D4ED8", linewidth=0.9,
                label=f"{lab} (desv={deviations[idx]:.3f})")

    # ── Curva ideal por encima de todo ───────────────────────────────────────
    ax.plot(reference[:, 0], reference[:, 1], color="#16A34A", linewidth=1.5,
            label="Curva ideal")

    ax.set_title(title or "Dataset de entrenamiento")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path