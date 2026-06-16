"""
Re-visualización de un run YA ENTRENADO — sin reentrenar.

Carga la configuración exacta, el checkpoint y las métricas que el run guardó
en outputs/runs/<experimento>_<timestamp>/ y regenera las figuras:

  - learning_curve.png  → la curva de aprendizaje (loss por epoch)
  - predictions.png     → real vs rollout autorregresivo + incisión de referencia

Útil para iterar sobre la visualización sin volver a pagar el entrenamiento.

Uso:
    python scripts/visualize.py --run outputs/runs/cutting_c_20260615_120000
    python scripts/visualize.py --run <ruta> --checkpoint final
    python scripts/visualize.py --run <ruta> --out-suffix _v2   # no sobrescribe
"""
import argparse
import json
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch

from surgical_fl.utils.config import ExperimentConfig
from surgical_fl.models.registry import get_model
from surgical_fl.data.generators.factory import build_generator
from surgical_fl.data.builders import build_centralized_split
from surgical_fl.visualization.trajectories import (
    plot_learning_curve,
    plot_predictions_per_profile,
    plot_training_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenera las figuras de un run ya entrenado sin reentrenar."
    )
    parser.add_argument(
        "--run",
        type=str,
        required=True,
        help="Directorio del run (outputs/runs/<experimento>_<timestamp>)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="best",
        help="Checkpoint a cargar: 'best' (def.), 'final', 'epoch_10', ...",
    )
    parser.add_argument(
        "--out-suffix",
        type=str,
        default="",
        help="Sufijo para los PNG (p. ej. '_v2') para no sobrescribir los originales",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run = Path(args.run)
    figures = run / "figures"

    config_path = run / "config.json"
    checkpoint_path = run / "checkpoints" / f"{args.checkpoint}.pt"
    if not config_path.exists():
        raise FileNotFoundError(f"No encuentro {config_path}. ¿Es un directorio de run válido?")
    if not checkpoint_path.exists():
        available = sorted(p.stem for p in (run / "checkpoints").glob("*.pt"))
        raise FileNotFoundError(
            f"No existe el checkpoint '{args.checkpoint}'. Disponibles: {available}"
        )

    cfg = ExperimentConfig.from_json(config_path)
    device = torch.device("cpu")

    print(f"Run:        {run}")
    print(f"Experimento:{cfg.name}  |  skill={cfg.skill}  |  profiles={cfg.profiles}")
    print(f"Checkpoint: {checkpoint_path.name}")

    # ── Reconstruir el modelo y cargar pesos ─────────────────────────────────
    model = get_model(
        cfg.model.name,
        hidden_dim=cfg.model.hidden_dim,
        dropout=cfg.model.dropout,
    ).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    # ── Curvas de referencia por perfil (mismas que en entrenamiento) ────────
    references = {
        profile: build_generator(
            skill=cfg.skill,
            profile_name=profile,
            trajectory_length=cfg.data.trajectory_length,
            seed=0,
        ).reference_curve
        for profile in cfg.profiles
    }

    suffix = args.out_suffix

    # ── Curva de aprendizaje (desde las métricas guardadas) ──────────────────
    metrics_path = run / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)
        train_losses = metrics.get("train_losses")
        if train_losses:
            out = plot_learning_curve(
                train_losses,
                str(figures / f"learning_curve{suffix}.png"),
                title=f"{cfg.name} — train loss",
            )
            print(f"✓ {out}")
        else:
            print("  (metrics.json sin 'train_losses' → no regenero learning_curve)")
    else:
        print("  (sin metrics.json → no regenero learning_curve)")

    # ── Predicciones: real vs rollout autorregresivo + referencia ────────────
    out = plot_predictions_per_profile(
        model=model,
        skill=cfg.skill,
        profiles=cfg.profiles,
        output_path=str(figures / f"predictions{suffix}.png"),
        trajectory_length=cfg.data.trajectory_length,
        device=device,
        references=references,
        title=f"{cfg.name} — Real vs Rollout",
    )
    print(f"✓ {out}")

    # ── Dataset de entrenamiento vs curva ideal (regenerado, determinista) ───
    # Se reconstruye con la misma semilla/config, así coincide con el del run.
    split = build_centralized_split(
        skill=cfg.skill,
        profile_names=cfg.profiles,
        total_samples=cfg.data.num_samples,
        val_split=cfg.data.val_split,
        trajectory_length=cfg.data.trajectory_length,
        seed=cfg.training.seed,
    )
    for profile in cfg.profiles:
        out = plot_training_dataset(
            trajectories=split.train_per_profile[profile],
            reference=split.references[profile],
            output_path=str(figures / f"dataset_{profile}{suffix}.png"),
            title=f"{cfg.name} — {profile}: dataset vs curva ideal",
        )
        print(f"✓ {out}")


if __name__ == "__main__":
    main()
