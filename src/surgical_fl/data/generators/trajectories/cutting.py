"""
Aqui definimos la geometria de corte (no los perfiles de los hospitales)
La variabilidad entre cada hospital (ruido, velocidad, estilo) se inyecta desde profiles y se combina en la factory

Aqui generamos tres tipos de cortes:
- LinearCutGenerator -> Corte lineal a lo largo del eje X
- CurvedCutGenerator -> Corte con parábola suave (puede ser un tejido con resistencia al corte variable)
- SplineCutGenerator -> Corte simulado alrededor de una curva de referencia compartida.

Todos los generadores exponen `reference_curve`: la trayectoria ideal (sin ruido)
contra la que se mide la DESVIACIÓN de un corte real. Así dos hospitales que
comparten la misma referencia son comparables: lo que cambia es la calidad de
ejecución, no la forma objetivo.
"""

import numpy as np
from .base import TrajectoryGenerator


# ─── Geometría compartida ────────────────────────────────────────────────────

# Curva de referencia por defecto del spline: incisión en S suave de (0,0) a (1,0).
# Cuatro nodos de control. Los cortes simulados pueden tener MÁS nodos que estos.
DEFAULT_REFERENCE_NODES = np.array(
    [[0.0, 0.0], [0.33, 0.07], [0.66, -0.07], [1.0, 0.0]],
    dtype=np.float64,
)


def catmull_rom(points: np.ndarray, samples_per_segment: int = 24) -> np.ndarray:
    """Spline de Catmull-Rom que INTERPOLA los puntos de control dados.

    A diferencia de np.interp (lineal a trozos), produce una curva suave que
    pasa exactamente por cada nodo de control. Los extremos se clampean
    duplicándolos para que la curva empiece y acabe en el primer y último nodo.

    Args:
        points: (n, 2) nodos de control, n >= 2.
        samples_per_segment: resolución de cada tramo entre nodos.

    Returns:
        (m, 2) curva densa.
    """
    P = np.asarray(points, dtype=np.float64)
    n = len(P)
    if n < 2:
        return P.copy()

    ext = np.vstack([P[0:1], P, P[-1:]])  # clamp de extremos → longitud n+2
    t = np.linspace(0.0, 1.0, samples_per_segment, endpoint=False)
    t2, t3 = t * t, t * t * t

    segments = []
    for k in range(n - 1):  # tramo entre P[k] y P[k+1]
        p0, p1, p2, p3 = ext[k], ext[k + 1], ext[k + 2], ext[k + 3]
        seg = 0.5 * (
            (2.0 * p1)
            + (-p0 + p2) * t[:, None]
            + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2[:, None]
            + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3[:, None]
        )
        segments.append(seg)
    segments.append(P[-1:])  # cierra exactamente en el último nodo
    return np.vstack(segments)


def resample_by_arclength(curve: np.ndarray, n_out: int) -> np.ndarray:
    """Re-muestrea una curva a n_out puntos equiespaciados por longitud de arco."""
    curve = np.asarray(curve, dtype=np.float64)
    deltas = np.diff(curve, axis=0)
    seg_len = np.sqrt((deltas ** 2).sum(axis=1))
    s = np.concatenate([[0.0], np.cumsum(seg_len)])
    if s[-1] == 0.0:
        return np.repeat(curve[:1], n_out, axis=0)
    s /= s[-1]
    u = np.linspace(0.0, 1.0, n_out)
    x = np.interp(u, s, curve[:, 0])
    y = np.interp(u, s, curve[:, 1])
    return np.stack([x, y], axis=1)


class LinearCutGenerator(TrajectoryGenerator):
    """
    Respresenta el movimiento perfecto recto de un bisturí en linea recta.
    El ruido y la velocidad son parámetros que dependen de los perfiles.

    Args:
    noise_std: desviación lateral en Y (ruido del hospital)
    speed_variance: varianza en la velocidad (estilo del hospital)

    """
    def __init__(self, trajectory_length: int = 50, noise_std: float = 0.02, speed_variance: float = 0.01, seed: int | None = None):
        super().__init__("cutting", trajectory_length = trajectory_length, seed = seed)
        self.noise_std = noise_std
        self.speed_variance = speed_variance

    @property
    def reference_curve(self) -> np.ndarray:
        """Corte ideal: recta y=0 a lo largo de x∈[0,1]."""
        m = max(200, self.trajectory_length)
        x = np.linspace(0.0, 1.0, m)
        return np.stack([x, np.zeros_like(x)], axis=1).astype(np.float32)

    def generate(self, num_samples: int) -> np.ndarray:
        """ Genera un lote de trayectorias lineales """
        trajectories = np.zeros((num_samples, self.trajectory_length, 2))

        for i in range(num_samples):
           speeds = (1/self.trajectory_length) + self.rng.normal(0, self.speed_variance, self.trajectory_length)
           speeds = np.clip(speeds, 1e-4, None)
           x = np.cumsum(speeds)
           x = x / x[-1]
           y = self.rng.normal(0, self.noise_std, self.trajectory_length)
           trajectories[i, :, 0] = x
           trajectories[i, :, 1] = y
        return trajectories.astype(np.float32)


class CurvedCutGenerator(TrajectoryGenerator):
    """
    Corte con curvatura parabólica en Y.
    Modela tejidos con resistencia variable que desvían la trayectoria.
    """
    def __init__(self, trajectory_length: int = 50, curvature: float = 0.1,
                 noise_std: float = 0.02, speed_variance: float = 0.01,
                 seed: int | None = None):
        super().__init__("cutting", trajectory_length=trajectory_length, seed=seed)
        self.curvature = curvature
        self.noise_std = noise_std
        self.speed_variance = speed_variance

    @property
    def reference_curve(self) -> np.ndarray:
        """Corte ideal: la parábola determinista sin ruido."""
        m = max(200, self.trajectory_length)
        x = np.linspace(0.0, 1.0, m)
        y = self.curvature * (x - 0.5) ** 2
        return np.stack([x, y], axis=1).astype(np.float32)

    def generate(self, num_samples: int) -> np.ndarray:
        trajectories = np.zeros((num_samples, self.trajectory_length, 2))
        for i in range(num_samples):
            speeds = (1 / self.trajectory_length) + self.rng.normal(0, self.speed_variance, self.trajectory_length)
            speeds = np.clip(speeds, 1e-4, None)
            x = np.cumsum(speeds)
            x = x / x[-1]
            # U-shape: edges (x≈0, x≈1) are higher than center (x≈0.5)
            y_curve = self.curvature * (x - 0.5) ** 2
            y_noise = self.rng.normal(0, self.noise_std, self.trajectory_length)
            trajectories[i, :, 0] = x
            trajectories[i, :, 1] = y_curve + y_noise
        return trajectories.astype(np.float32)


class SplineCutGenerator(TrajectoryGenerator):
    """
    Corte simulado alrededor de una curva de referencia compartida.

    Modelo de la maniobra real:
      1. Existe una CURVA DE REFERENCIA (la incisión ideal), definida por unos
         nodos de control e interpolada con Catmull-Rom.
      2. Un corte real se genera tomando `n_nodes` anclas a lo largo de esa
         referencia y desplazando cada una dentro de un DISCO de radio `radius`
         (la "mano" del cirujano). Un hospital experto tiene `radius` pequeño.
      3. Se interpola un nuevo spline a través de los nodos perturbados.

    Nótese que `n_nodes` puede ser MAYOR que el nº de nodos de la referencia:
    la calidad no se mide comparando nodos, sino la DESVIACIÓN de la curva
    generada respecto a `reference_curve` (ver CuttingSkill).

    Args:
        reference_nodes: (n_ref, 2) nodos de la incisión ideal. Compartirlos
                         entre hospitales es lo que los hace comparables.
        n_nodes:         nº de nodos de control del corte simulado (>= 2).
                         Por defecto, el mismo nº que la referencia.
        radius:          radio del disco de perturbación por nodo (skill).
        noise_std:       jitter isótropo de instrumentación, añadido al final.
    """
    def __init__(self, trajectory_length: int = 50,
                 reference_nodes: np.ndarray | None = None,
                 n_nodes: int | None = None,
                 radius: float = 0.05,
                 noise_std: float = 0.0,
                 seed: int | None = None):
        super().__init__("cutting", trajectory_length=trajectory_length, seed=seed)
        self.reference_nodes = (
            DEFAULT_REFERENCE_NODES if reference_nodes is None
            else np.asarray(reference_nodes, dtype=np.float64)
        )
        self.n_nodes = n_nodes if n_nodes is not None else len(self.reference_nodes)
        self.radius = radius
        self.noise_std = noise_std

    @property
    def reference_curve(self) -> np.ndarray:
        """Incisión ideal: Catmull-Rom denso sobre los nodos de referencia."""
        m = max(200, self.trajectory_length)
        dense = catmull_rom(self.reference_nodes)
        return resample_by_arclength(dense, m).astype(np.float32)

    def _disk_offsets(self, count: int) -> np.ndarray:
        """`count` desplazamientos uniformes dentro de un disco de radio `radius`."""
        angle = self.rng.uniform(0.0, 2.0 * np.pi, count)
        # sqrt para distribución uniforme en área (no concentrada en el centro)
        rad = self.radius * np.sqrt(self.rng.uniform(0.0, 1.0, count))
        return np.stack([rad * np.cos(angle), rad * np.sin(angle)], axis=1)

    def generate(self, num_samples: int) -> np.ndarray:
        # Anclas equiespaciadas (por longitud de arco) sobre la referencia ideal.
        ref_dense = catmull_rom(self.reference_nodes)
        anchors = resample_by_arclength(ref_dense, self.n_nodes)

        trajectories = np.zeros((num_samples, self.trajectory_length, 2))
        for i in range(num_samples):
            nodes = anchors + self._disk_offsets(self.n_nodes)
            curve = catmull_rom(nodes)
            traj = resample_by_arclength(curve, self.trajectory_length)
            if self.noise_std > 0.0:
                traj = traj + self.rng.normal(0.0, self.noise_std, traj.shape)
            trajectories[i] = traj
        return trajectories.astype(np.float32)
