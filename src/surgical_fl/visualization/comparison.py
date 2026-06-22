"""
Visualización comparativa: Aprendizaje Federado vs Centralizado.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_comparison_bars(
    centralized_metrics: dict[str, dict],
    federated_metrics: dict[str, dict],
    output_path: str,
    title: str = "RMSE por hospital — Centralizado vs Federado",
) -> str:
    """Barras agrupadas de RMSE por hospital para ambos enfoques."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    profiles = list(centralized_metrics.keys())
    cent_rmse = [centralized_metrics[p]["rmse"] for p in profiles]
    fed_rmse = [federated_metrics[p]["rmse"] for p in profiles]

    x = np.arange(len(profiles))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars_c = ax.bar(x - width / 2, cent_rmse, width, label="Centralizado",
                    color="#2563EB", alpha=0.85)
    bars_f = ax.bar(x + width / 2, fed_rmse, width, label="Federado",
                    color="#DC2626", alpha=0.85)

    ax.set_ylabel("RMSE")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels([p.replace("_", " ").title() for p in profiles])
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    for bars in (bars_c, bars_f):
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.5f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 4), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_fl_rounds(
    round_losses: list[float],
    output_path: str,
    title: str = "Federado — Loss por ronda",
) -> str:
    """Curva de loss global del servidor a lo largo de las rondas FL."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    rounds = list(range(1, len(round_losses) + 1))
    ax.plot(rounds, round_losses, "o-", color="#DC2626", linewidth=2, markersize=5)
    ax.set_xlabel("Ronda")
    ax.set_ylabel("Loss (MSE)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_comparison_predictions(
    model_cent,
    model_fed,
    skill: str,
    profiles: list[str],
    output_path: str,
    trajectory_length: int = 50,
    device=None,
    references: dict | None = None,
    title: str | None = None,
) -> str:
    """Predicciones side-by-side: centralizado (fila 1) vs federado (fila 2)."""
    import torch
    from surgical_fl.data.generators.factory import build_generator
    from surgical_fl.visualization.trajectories import _autoregressive_rollout

    device = device or torch.device("cpu")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    n = len(profiles)
    fig, axes = plt.subplots(2, n, figsize=(6 * n, 8))
    if n == 1:
        axes = axes.reshape(2, 1)

    row_labels = ["Centralizado", "Federado"]
    models = [model_cent, model_fed]
    colors = ["#2563EB", "#DC2626"]

    for row, (model, label, color) in enumerate(zip(models, row_labels, colors)):
        model.eval()
        for col, profile_name in enumerate(profiles):
            ax = axes[row, col]
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
                        alpha=0.25, linewidth=3, label="Referencia")

            ax.plot(traj[:, 0], traj[:, 1], "o-", color=color,
                    alpha=0.6, markersize=3, label="Real", linewidth=1.5)
            ax.plot(rollout[:, 0], rollout[:, 1], "s--", color="gray",
                    alpha=0.85, markersize=3, label="Rollout", linewidth=1.5)

            if row == 0:
                ax.set_title(profile_name.replace("_", " ").title())
            ax.set_xlabel("X")
            if col == 0:
                ax.set_ylabel(f"{label}\nY")
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)
            ax.set_aspect("equal")

    if title:
        plt.suptitle(title, y=1.02, fontsize=13)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path
