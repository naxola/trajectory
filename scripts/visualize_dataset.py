"""
Visualización de un dataset generado por un generador de cortes.

Crea tres figuras:
  - dataset.png       → nube de trayectorias simuladas + curva de referencia ideal
  - nodes.png         → (solo SplineCutGenerator) nodos de referencia y un ejemplo
                        de nodos perturbados superpuestos a la curva de referencia
  - reference.png     → curva de referencia aislada

Uso básico:
    python scripts/visualize_dataset.py --generator spline
    python scripts/visualize_dataset.py --generator linear --num-samples 300
    python scripts/visualize_dataset.py --generator curved --curvature 0.2

Parámetros por generador
─────────────────────────
  linear:
    --noise-std          desviación lateral en Y           (def: 0.02)
    --speed-variance     varianza en la velocidad          (def: 0.01)

  curved:
    --curvature          amplitud de la parábola           (def: 0.1)
    --noise-std          ruido gaussiano en Y              (def: 0.02)
    --speed-variance     varianza en la velocidad          (def: 0.01)

  spline:
    --radius             radio del disco de perturbación   (def: 0.05)
    --noise-std          jitter isótropo final             (def: 0.0)
    --n-nodes            nodos de control del corte        (def: igual que ref.)
    --ref-nodes          nodos de referencia como JSON     (def: curva en S)
                         Ejemplo: "[[0,0],[0.5,0.1],[1,0]]"

Parámetros comunes:
    --num-samples        nº de trayectorias                (def: 200)
    --trajectory-length  puntos por trayectoria            (def: 50)
    --max-background     máx. trayectorias en fondo        (def: 200)
    --seed               semilla aleatoria                 (def: 0)
    --out-dir            directorio de salida              (def: outputs/datasets)
    --title              título del plot                   (def: automático)
    --show               mostrar en pantalla además de guardar
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from surgical_fl.data.generators.trajectories.cutting import (
    LinearCutGenerator,
    CurvedCutGenerator,
    SplineCutGenerator,
    DEFAULT_REFERENCE_NODES,
    catmull_rom,
    resample_by_arclength,
)
from surgical_fl.visualization.trajectories import plot_training_dataset


# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Visualiza un dataset generado por un generador de cortes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Generador
    p.add_argument(
        "--generator", "-g",
        choices=["linear", "curved", "spline"],
        default="spline",
        help="Tipo de generador (def: spline)",
    )

    # Parámetros comunes
    p.add_argument("--num-samples",       type=int,   default=200,  help="Nº de trayectorias a generar")
    p.add_argument("--trajectory-length", type=int,   default=50,   help="Puntos por trayectoria")
    p.add_argument("--seed",              type=int,   default=0,    help="Semilla aleatoria")
    p.add_argument("--max-background",   type=int,   default=200,  help="Máx. trayectorias en el fondo del plot")
    p.add_argument("--out-dir",          type=str,   default="outputs/datasets", help="Directorio de salida")
    p.add_argument("--title",            type=str,   default=None, help="Título del gráfico principal")
    p.add_argument("--show",             action="store_true",      help="Abrir figura en pantalla")

    # Parámetros específicos de linear / curved
    p.add_argument("--noise-std",        type=float, default=None, help="Ruido gaussiano (linear, curved, spline)")
    p.add_argument("--speed-variance",   type=float, default=None, help="Varianza de velocidad (linear, curved)")
    p.add_argument("--curvature",        type=float, default=None, help="Amplitud parabólica (curved)")

    # Parámetros específicos de spline
    p.add_argument("--radius",    type=float, default=None, help="Radio de perturbación por nodo (spline)")
    p.add_argument("--n-nodes",   type=int,   default=None, help="Nodos de control del corte (spline)")
    p.add_argument(
        "--ref-nodes",
        type=str, default=None,
        help='Nodos de referencia como JSON, ej: "[[0,0],[0.5,0.1],[1,0]]" (spline)',
    )

    return p.parse_args()


# ─── Construcción del generador ───────────────────────────────────────────────

def build_generator(args: argparse.Namespace):
    tl  = args.trajectory_length
    sd  = args.seed

    if args.generator == "linear":
        return LinearCutGenerator(
            trajectory_length=tl,
            noise_std=args.noise_std if args.noise_std is not None else 0.02,
            speed_variance=args.speed_variance if args.speed_variance is not None else 0.01,
            seed=sd,
        )

    if args.generator == "curved":
        return CurvedCutGenerator(
            trajectory_length=tl,
            curvature=args.curvature if args.curvature is not None else 0.1,
            noise_std=args.noise_std if args.noise_std is not None else 0.02,
            speed_variance=args.speed_variance if args.speed_variance is not None else 0.01,
            seed=sd,
        )

    # spline
    ref_nodes = DEFAULT_REFERENCE_NODES
    if args.ref_nodes is not None:
        ref_nodes = np.array(json.loads(args.ref_nodes), dtype=np.float64)

    return SplineCutGenerator(
        trajectory_length=tl,
        reference_nodes=ref_nodes,
        n_nodes=args.n_nodes,
        radius=args.radius if args.radius is not None else 0.05,
        noise_std=args.noise_std if args.noise_std is not None else 0.0,
        seed=sd,
    )


# ─── Plot extra: nodos de control (solo spline) ───────────────────────────────

def plot_nodes(gen: SplineCutGenerator, out_path: str, title: str | None, show: bool):
    """Muestra los nodos de referencia, la curva ideal y un ejemplo de corte perturbado."""
    ref      = gen.reference_nodes
    ref_curve = gen.reference_curve
    base_nodes = gen._control_nodes()

    # Generar UN ejemplo de nodos perturbados
    rng = np.random.default_rng(0)
    angle = rng.uniform(0.0, 2 * np.pi, gen.n_nodes)
    rad   = gen.radius * np.sqrt(rng.uniform(0.0, 1.0, gen.n_nodes))
    offsets = np.stack([rad * np.cos(angle), rad * np.sin(angle)], axis=1)
    perturbed = base_nodes + offsets

    fig, ax = plt.subplots(figsize=(9, 5))

    # Curva de referencia (Catmull-Rom densa)
    ax.plot(ref_curve[:, 0], ref_curve[:, 1], color="black", linewidth=2.5,
            label="Curva referencia (Catmull-Rom)", zorder=5)

    # Nodos de referencia
    ax.scatter(ref[:, 0], ref[:, 1], color="black", s=80, zorder=10,
               label=f"Nodos referencia ({len(ref)})", marker="D")

    # Nodos base (interpolados si n_nodes > n_ref)
    if gen.n_nodes != len(ref):
        ax.scatter(base_nodes[:, 0], base_nodes[:, 1], color="#7C3AED", s=55,
                   zorder=9, label=f"Nodos base interpolados ({gen.n_nodes})", marker="^")

    # Nodos perturbados + trayectoria resultante (Catmull-Rom, igual que generate())
    traj_dense      = catmull_rom(perturbed)
    traj_perturbed  = resample_by_arclength(traj_dense, gen.trajectory_length)
    ax.plot(traj_perturbed[:, 0], traj_perturbed[:, 1], color="#2563EB",
            linewidth=1.5, alpha=0.8, label="Corte perturbado (ejemplo)", zorder=4)
    ax.scatter(perturbed[:, 0], perturbed[:, 1], color="#2563EB", s=55, zorder=8,
               label=f"Nodos perturbados (r={gen.radius})", marker="o")

    # Discos de radio r alrededor de cada nodo base
    for node in base_nodes:
        circle = plt.Circle(node, gen.radius, color="#2563EB", fill=False,
                             alpha=0.2, linewidth=0.8, linestyle="--")
        ax.add_patch(circle)

    ax.set_title(title or f"Nodos de control — radio={gen.radius}, n_nodes={gen.n_nodes}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"✓ {out_path}")
    if show:
        plt.show()
    plt.close(fig)


# ─── Plot de la curva de referencia aislada ───────────────────────────────────

def plot_reference(gen, out_path: str, title: str | None, show: bool):
    ref_curve = gen.reference_curve

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(ref_curve[:, 0], ref_curve[:, 1], color="black", linewidth=2.5,
            label="Curva de referencia ideal")

    # Nodos de referencia si es spline
    if isinstance(gen, SplineCutGenerator):
        ref = gen.reference_nodes
        ax.scatter(ref[:, 0], ref[:, 1], color="black", s=80, zorder=5,
                   label=f"Nodos ({len(ref)})", marker="D")

    ax.set_title(title or "Curva de referencia ideal")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"✓ {out_path}")
    if show:
        matplotlib.use("TkAgg")
        plt.show()
    plt.close(fig)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    gen  = build_generator(args)

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    label = args.generator
    title = args.title or f"{label} — {args.num_samples} muestras, T={args.trajectory_length}"

    print(f"\n{'='*60}")
    print(f"  Generador : {label}")
    print(f"  Muestras  : {args.num_samples}")
    print(f"  Longitud  : {args.trajectory_length}")
    if isinstance(gen, SplineCutGenerator):
        print(f"  Radio     : {gen.radius}")
        print(f"  n_nodes   : {gen.n_nodes}  (ref: {len(gen.reference_nodes)})")
        print(f"  noise_std : {gen.noise_std}")
    print(f"  Salida    : {out_dir}/")
    print(f"{'='*60}\n")

    # ── Generar dataset ───────────────────────────────────────────────────────
    print("Generando trayectorias…")
    trajectories = gen.generate(args.num_samples)   # (N, T, 2)
    ref_curve    = gen.reference_curve              # (M, 2)
    print(f"  Shape: {trajectories.shape}\n")

    # ── Figura 1: dataset vs curva ideal ─────────────────────────────────────
    dataset_path = os.path.join(out_dir, f"{label}_dataset.png")
    plot_training_dataset(
        trajectories=trajectories,
        reference=ref_curve,
        output_path=dataset_path,
        title=title,
        max_background=args.max_background,
    )
    print(f"✓ {dataset_path}")

    # ── Figura 2 (solo spline): nodos de control ──────────────────────────────
    if isinstance(gen, SplineCutGenerator):
        nodes_path = os.path.join(out_dir, f"{label}_nodes.png")
        plot_nodes(
            gen=gen,
            out_path=nodes_path,
            title=args.title,
            show=args.show,
        )

    # ── Figura 3: curva de referencia aislada ─────────────────────────────────
    ref_path = os.path.join(out_dir, f"{label}_reference.png")
    plot_reference(
        gen=gen,
        out_path=ref_path,
        title=args.title,
        show=args.show,
    )

    print(f"\n✓ Figuras guardadas en: {out_dir}/")


if __name__ == "__main__":
    main()
