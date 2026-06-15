"""
Entrenamiento centralizado — BASELINE DE VALIDACIÓN.

Lee la configuración desde un fichero TOML versionable y persiste
toda la ejecución (config + métricas + checkpoints + figuras) en
outputs/runs/<experiment>_<timestamp>/.

Uso:
    python scripts/train_centralized.py
    python scripts/train_centralized.py --config experiments/cutting_baseline/config.toml
    python scripts/train_centralized.py --config <ruta> --epochs 50
"""
import argparse
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import torch
from torch.utils.data import DataLoader

from surgical_fl.utils.config import ExperimentConfig
from surgical_fl.utils.seeding import set_global_seed
from surgical_fl.utils.run_io import RunDirectory

from surgical_fl.domain.skills.registry import get_skill
from surgical_fl.data.builders import build_centralized_split
from surgical_fl.models.registry import get_model
from surgical_fl.training.trainer import train_one_epoch, evaluate

from surgical_fl.visualization.trajectories import (
    plot_learning_curve,
    plot_predictions_per_profile,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entrenamiento centralizado.")
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/cutting_baseline/config.toml",
        help="Ruta al fichero TOML de configuración",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Sobrescribe epochs")
    parser.add_argument("--seed",   type=int, default=None, help="Sobrescribe seed")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = ExperimentConfig.from_toml(args.config)

    # Sobrescrituras CLI (útiles para sweep rápido sin editar TOML)
    if args.epochs is not None:
        cfg.training.epochs = args.epochs
    if args.seed is not None:
        cfg.training.seed = args.seed

    set_global_seed(cfg.training.seed)
    device = torch.device("cpu")
    run    = RunDirectory.create(cfg.name)

    print("=" * 60)
    print(f"   BASELINE CENTRALIZADO — '{cfg.name}'")
    print(f"   Run: {run.root}")
    print("=" * 60)
    print(f"  Skill:      {cfg.skill}")
    print(f"  Profiles:   {cfg.profiles}")
    print(f"  Model:      {cfg.model.name} (hidden_dim={cfg.model.hidden_dim})")
    print(f"  Samples:    {cfg.data.num_samples} (val_split={cfg.data.val_split})")
    print(f"  Epochs:     {cfg.training.epochs}  |  seed={cfg.training.seed}")
    print()

    # Persistir la config exacta usada (incluye sobrescrituras CLI)
    run.save_config(cfg)

    # ── Dataset: train mezclado + val SEPARADO por perfil ─────────────────────
    # Cada perfil usa una semilla derivada distinta (datos independientes) y se
    # parte en train/val por separado, para poder medir la generalización del
    # modelo conjunto hospital a hospital.
    skill = get_skill(cfg.skill)
    split = build_centralized_split(
        skill=cfg.skill,
        profile_names=cfg.profiles,
        total_samples=cfg.data.num_samples,
        val_split=cfg.data.val_split,
        trajectory_length=cfg.data.trajectory_length,
        seed=cfg.training.seed,
    )

    # Calidad de cada perfil: desviación respecto a SU curva de referencia ideal.
    for profile_name, trajs in split.train_per_profile.items():
        ref = split.references[profile_name]
        scores = [
            skill.evaluate_trajectory(t, reference=ref)["task_score"]
            for t in trajs[:15]
        ]
        print(
            f"  {profile_name}: calidad {np.mean(scores):.3f} ± {np.std(scores):.3f}"
        )

    print(
        f"\nDataset total: train={len(split.train)}, "
        f"val={len(split.val_combined)} ({len(cfg.profiles)} perfiles)\n"
    )

    train_loader = DataLoader(split.train, batch_size=cfg.training.batch_size, shuffle=True)
    val_loader   = DataLoader(split.val_combined, batch_size=cfg.training.batch_size, shuffle=False)
    val_loaders_per_profile = {
        name: DataLoader(ds, batch_size=cfg.training.batch_size, shuffle=False)
        for name, ds in split.val_per_profile.items()
    }

    # ── Modelo ────────────────────────────────────────────────────────────────
    model = get_model(
        cfg.model.name,
        hidden_dim=cfg.model.hidden_dim,
        dropout=cfg.model.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.learning_rate)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Modelo: {n_params} parámetros\n")

    # ── Loop de entrenamiento con validación y mejor checkpoint ──────────────
    train_losses, val_losses = [], []
    best_val_loss = float("inf")

    for epoch in range(1, cfg.training.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, _ = evaluate(model, val_loader, device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            run.save_checkpoint(model, name="best")

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"  Epoch {epoch:3d}/{cfg.training.epochs}  |  "
                f"train={train_loss:.5f}  val={val_loss:.5f}"
            )

    # ── Evaluación final con el mejor modelo ─────────────────────────────────
    run.save_checkpoint(model, name="final")
    model.load_state_dict(torch.load(run.checkpoints / "best.pt"))
    final_val_loss, final_metrics = evaluate(model, val_loader, device)

    # Generalización hospital a hospital: ¿predice igual de bien cada estilo?
    val_per_profile = {}
    for name, loader in val_loaders_per_profile.items():
        loss, metrics = evaluate(model, loader, device)
        val_per_profile[name] = {"mse": loss, "rmse": metrics["rmse"]}

    print(f"\n{'─'*50}")
    print(f"Mejor val loss (global):  {final_val_loss:.5f}")
    print(f"RMSE val final (global):  {final_metrics['rmse']:.5f}")
    print("Val por hospital:")
    for name, m in val_per_profile.items():
        print(f"  {name}: mse={m['mse']:.5f}  rmse={m['rmse']:.5f}")
    print(f"{'─'*50}\n")

    # ── Persistir resultados ──────────────────────────────────────────────────
    run.save_metrics({
        "train_losses":        train_losses,
        "val_losses":          val_losses,
        "best_val_loss":       best_val_loss,
        "final_val_rmse":      final_metrics["rmse"],
        "val_per_profile":     val_per_profile,
        "n_params":            n_params,
    })

    plot_learning_curve(
        train_losses,
        run.figure_path("learning_curve.png"),
        title=f"{cfg.name} — train loss",
    )
    plot_predictions_per_profile(
        model=model,
        skill=cfg.skill,
        profiles=cfg.profiles,
        output_path=run.figure_path("predictions.png"),
        trajectory_length=cfg.data.trajectory_length,
        device=device,
        references=split.references,
        title=f"{cfg.name} — Real vs Rollout",
    )

    print(f"✓ Resultados completos en: {run.root}")
    print(f"  └ config.json, metrics.json, checkpoints/, figures/\n")


if __name__ == "__main__":
    main()