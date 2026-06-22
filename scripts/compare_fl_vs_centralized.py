"""
Comparación: Aprendizaje Federado (Flower) vs Centralizado.

Entrena (o carga desde caché) un modelo centralizado y ejecuta aprendizaje
federado con Flower, luego evalúa ambos sobre los mismos datos de validación
para una comparación justa.

Uso:
    python scripts/compare_fl_vs_centralized.py
    python scripts/compare_fl_vs_centralized.py --config experiments/federated_vs_centralized/config.toml
"""
import argparse
import copy
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import torch
from torch.utils.data import DataLoader

from surgical_fl.utils.config import ExperimentConfig
from surgical_fl.utils.seeding import set_global_seed
from surgical_fl.utils.run_io import RunDirectory

from surgical_fl.data.builders import build_centralized_split
from surgical_fl.data.datasets import TrajectoryDataset
from surgical_fl.models.registry import get_model
from surgical_fl.training.trainer import train_one_epoch, evaluate

from surgical_fl.federation.client import FlowerClient

from surgical_fl.visualization.trajectories import plot_learning_curve
from surgical_fl.visualization.comparison import (
    plot_comparison_bars,
    plot_comparison_predictions,
    plot_fl_rounds,
)


CACHE_DIR = "outputs/models/centralized"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Comparación FL vs Centralizado.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/federated_vs_centralized/config.toml",
    )
    parser.add_argument("--force-retrain", action="store_true",
                        help="Ignora la caché y re-entrena el modelo centralizado")
    return parser.parse_args()


# ── Centralizado (con caché) ────────────────────────────────────────────────

def _cache_path(cfg: ExperimentConfig) -> str:
    return os.path.join(CACHE_DIR, f"{cfg.name}.pt")


def _cache_config_path(cfg: ExperimentConfig) -> str:
    return os.path.join(CACHE_DIR, f"{cfg.name}_config.json")


def train_centralized(
    cfg: ExperimentConfig,
    split,
    device: torch.device,
    initial_state_dict: dict,
    force: bool = False,
) -> tuple[torch.nn.Module, list[float], dict]:
    """Entrena centralizado o carga de caché. Devuelve (modelo, losses, val_per_profile)."""
    cache = _cache_path(cfg)

    if not force and os.path.exists(cache):
        print(f"  [caché] Cargando modelo centralizado desde {cache}")
        model = get_model(
            cfg.model.name,
            hidden_dim=cfg.model.hidden_dim,
            dropout=cfg.model.dropout,
        ).to(device)
        model.load_state_dict(torch.load(cache, weights_only=True))

        metrics_path = _cache_config_path(cfg).replace("_config.json", "_metrics.json")
        cached_metrics = {}
        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                cached_metrics = json.load(f)

        return (
            model,
            cached_metrics.get("train_losses", []),
            cached_metrics.get("val_per_profile", {}),
        )

    print("  Entrenando modelo centralizado...")
    model = get_model(
        cfg.model.name,
        hidden_dim=cfg.model.hidden_dim,
        dropout=cfg.model.dropout,
    ).to(device)
    model.load_state_dict(initial_state_dict)

    train_loader = DataLoader(
        split.train, batch_size=cfg.training.batch_size, shuffle=True,
    )
    val_loader = DataLoader(
        split.val_combined, batch_size=cfg.training.batch_size, shuffle=False,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.learning_rate)

    train_losses = []
    best_val_loss = float("inf")
    best_state = None

    for epoch in range(1, cfg.training.epochs + 1):
        loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, _ = evaluate(model, val_loader, device)
        train_losses.append(loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())

        if epoch % 5 == 0 or epoch == 1:
            print(f"    Epoch {epoch:3d}/{cfg.training.epochs}  "
                  f"train={loss:.6f}  val={val_loss:.6f}")

    model.load_state_dict(best_state)

    val_per_profile = {}
    for name, ds in split.val_per_profile.items():
        loader = DataLoader(ds, batch_size=cfg.training.batch_size, shuffle=False)
        loss, metrics = evaluate(model, loader, device)
        val_per_profile[name] = {"mse": loss, "rmse": metrics["rmse"]}

    os.makedirs(CACHE_DIR, exist_ok=True)
    torch.save(model.state_dict(), cache)
    metrics_path = cache.replace(".pt", "_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({
            "train_losses": train_losses,
            "val_per_profile": val_per_profile,
        }, f, indent=2)
    print(f"  [caché] Modelo centralizado guardado en {cache}")

    return model, train_losses, val_per_profile


# ── Federado (Flower) ───────────────────────────────────────────────────────

def _fedavg(client_params: list[tuple[list[np.ndarray], int]]) -> list[np.ndarray]:
    """FedAvg: media ponderada por número de muestras."""
    total = sum(n for _, n in client_params)
    averaged = [
        np.zeros_like(client_params[0][0][i])
        for i in range(len(client_params[0][0]))
    ]
    for params, n in client_params:
        weight = n / total
        for i, p in enumerate(params):
            averaged[i] += p * weight
    return averaged


def train_federated(
    cfg: ExperimentConfig,
    split,
    device: torch.device,
    initial_state_dict: dict,
) -> tuple[torch.nn.Module, list[float], dict]:
    """Ejecuta FL con Flower NumPyClient y FedAvg manual. Devuelve (modelo, round_losses, val_per_profile)."""
    print("  Entrenando modelo federado...")

    clients: dict[str, FlowerClient] = {}
    for profile_name in cfg.profiles:
        train_trajs = split.train_per_profile[profile_name]
        train_ds = TrajectoryDataset(train_trajs)
        train_loader = DataLoader(
            train_ds, batch_size=cfg.training.batch_size, shuffle=True,
        )

        val_ds = split.val_per_profile[profile_name]
        val_loader = DataLoader(
            val_ds, batch_size=cfg.training.batch_size, shuffle=False,
        )

        client_model = get_model(
            cfg.model.name,
            hidden_dim=cfg.model.hidden_dim,
            dropout=cfg.model.dropout,
        ).to(device)
        client_model.load_state_dict(initial_state_dict)

        clients[profile_name] = FlowerClient(
            model=client_model,
            train_loader=train_loader,
            val_loader=val_loader,
            local_epochs=cfg.federation.local_epochs,
            learning_rate=cfg.training.learning_rate,
            device=device,
        )

    global_model = get_model(
        cfg.model.name,
        hidden_dim=cfg.model.hidden_dim,
        dropout=cfg.model.dropout,
    ).to(device)
    global_model.load_state_dict(initial_state_dict)
    global_params = global_model.get_parameters()

    val_combined_loader = DataLoader(
        split.val_combined, batch_size=cfg.training.batch_size, shuffle=False,
    )

    round_losses = []

    for rnd in range(1, cfg.federation.num_rounds + 1):
        client_results = []
        for name, client in clients.items():
            new_params, n_samples, fit_metrics = client.fit(global_params, {})
            client_results.append((new_params, n_samples))

        global_params = _fedavg(client_results)

        global_model.set_parameters(global_params)
        val_loss, val_metrics = evaluate(global_model, val_combined_loader, device)
        round_losses.append(val_loss)

        if rnd % 2 == 0 or rnd == 1:
            print(f"    Ronda {rnd:3d}/{cfg.federation.num_rounds}  "
                  f"val_loss={val_loss:.6f}  rmse={val_metrics['rmse']:.6f}")

    val_per_profile = {}
    for name, ds in split.val_per_profile.items():
        loader = DataLoader(ds, batch_size=cfg.training.batch_size, shuffle=False)
        loss, metrics = evaluate(global_model, loader, device)
        val_per_profile[name] = {"mse": loss, "rmse": metrics["rmse"]}

    return global_model, round_losses, val_per_profile


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    cfg = ExperimentConfig.from_toml(args.config)

    set_global_seed(cfg.training.seed)
    device = torch.device("cpu")
    run = RunDirectory.create(f"{cfg.name}_comparison")

    print("=" * 65)
    print(f"   COMPARACIÓN: FL vs CENTRALIZADO — '{cfg.name}'")
    print(f"   Run: {run.root}")
    print("=" * 65)
    print(f"  Skill:        {cfg.skill}")
    print(f"  Profiles:     {cfg.profiles}")
    print(f"  Model:        {cfg.model.name} (hidden_dim={cfg.model.hidden_dim})")
    print(f"  Samples:      {cfg.data.num_samples} (val_split={cfg.data.val_split})")
    print(f"  Centralizado: {cfg.training.epochs} epochs")
    print(f"  Federado:     {cfg.federation.num_rounds} rondas × "
          f"{cfg.federation.local_epochs} epochs locales")
    print()

    run.save_config(cfg)

    # ── Datos compartidos ────────────────────────────────────────────────────
    split = build_centralized_split(
        skill=cfg.skill,
        profile_names=cfg.profiles,
        total_samples=cfg.data.num_samples,
        val_split=cfg.data.val_split,
        trajectory_length=cfg.data.trajectory_length,
        seed=cfg.training.seed,
    )
    print(f"  Dataset: train={len(split.train)}, val={len(split.val_combined)}\n")

    # ── Pesos iniciales compartidos ──────────────────────────────────────────
    init_model = get_model(
        cfg.model.name,
        hidden_dim=cfg.model.hidden_dim,
        dropout=cfg.model.dropout,
    ).to(device)
    initial_state_dict = copy.deepcopy(init_model.state_dict())

    # ── Centralizado ─────────────────────────────────────────────────────────
    print("━" * 65)
    print("  CENTRALIZADO")
    print("━" * 65)
    model_cent, cent_losses, cent_val = train_centralized(
        cfg, split, device, initial_state_dict, force=args.force_retrain,
    )
    print()
    for name, m in cent_val.items():
        print(f"    {name}: rmse={m['rmse']:.6f}")

    # ── Federado ─────────────────────────────────────────────────────────────
    print()
    print("━" * 65)
    print("  FEDERADO (Flower)")
    print("━" * 65)
    model_fed, fed_round_losses, fed_val = train_federated(
        cfg, split, device, initial_state_dict,
    )
    print()
    for name, m in fed_val.items():
        print(f"    {name}: rmse={m['rmse']:.6f}")

    # ── Comparación ──────────────────────────────────────────────────────────
    print()
    print("━" * 65)
    print("  RESUMEN COMPARATIVO")
    print("━" * 65)
    print(f"  {'Hospital':<20} {'Cent. RMSE':>12} {'Fed. RMSE':>12} {'Diferencia':>12}")
    print(f"  {'─'*56}")

    for name in cfg.profiles:
        c_rmse = cent_val[name]["rmse"]
        f_rmse = fed_val[name]["rmse"]
        diff = f_rmse - c_rmse
        sign = "+" if diff > 0 else ""
        print(f"  {name:<20} {c_rmse:>12.6f} {f_rmse:>12.6f} {sign}{diff:>11.6f}")

    c_avg = np.mean([m["rmse"] for m in cent_val.values()])
    f_avg = np.mean([m["rmse"] for m in fed_val.values()])
    diff = f_avg - c_avg
    sign = "+" if diff > 0 else ""
    print(f"  {'─'*56}")
    print(f"  {'PROMEDIO':<20} {c_avg:>12.6f} {f_avg:>12.6f} {sign}{diff:>11.6f}")
    print()

    # ── Persistir resultados ─────────────────────────────────────────────────
    run.save_checkpoint(model_cent, name="centralized_best")
    run.save_checkpoint(model_fed, name="federated_final")

    run.save_metrics({
        "centralized": {
            "train_losses": cent_losses,
            "val_per_profile": cent_val,
            "avg_rmse": c_avg,
        },
        "federated": {
            "round_losses": fed_round_losses,
            "val_per_profile": fed_val,
            "avg_rmse": f_avg,
        },
    })

    # ── Figuras ──────────────────────────────────────────────────────────────
    if cent_losses:
        plot_learning_curve(
            cent_losses,
            run.figure_path("centralized_learning_curve.png"),
            title=f"{cfg.name} — Centralizado: loss por epoch",
        )

    plot_fl_rounds(
        fed_round_losses,
        run.figure_path("fl_round_losses.png"),
        title=f"{cfg.name} — Federado: loss por ronda",
    )

    plot_comparison_bars(
        cent_val,
        fed_val,
        run.figure_path("comparison_rmse.png"),
        title=f"{cfg.name} — RMSE: Centralizado vs Federado",
    )

    plot_comparison_predictions(
        model_cent=model_cent,
        model_fed=model_fed,
        skill=cfg.skill,
        profiles=cfg.profiles,
        output_path=run.figure_path("comparison_predictions.png"),
        trajectory_length=cfg.data.trajectory_length,
        device=device,
        references=split.references,
        title=f"{cfg.name} — Predicciones: Centralizado vs Federado",
    )

    print(f"Resultados en: {run.root}")
    print(f"  config.json, metrics.json, checkpoints/, figures/")


if __name__ == "__main__":
    main()
