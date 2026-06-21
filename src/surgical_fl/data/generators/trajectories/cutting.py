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
    """Re-muestrea una curva a n_out puntos equiespaciados por longitud de arco.

    AVISO: internamente usa np.interp sobre X e Y por separado, lo que requiere
    que la longitud de arco sea una función biyéctiva de X (curvas sin retrocesos
    laterales pronunciados). Solo debe usarse con curvas de referencia, donde X
    es monótonamente creciente por construcción.
    Para curvas perturbadas usa resample_by_parameter.
    """
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


def resample_by_parameter(curve: np.ndarray, n_out: int) -> np.ndarray:
    """Re-muestrea una curva densa a n_out puntos equiespaciados por parámetro.

    A diferencia de resample_by_arclength, NO interpola X e Y de forma
    independiente: toma índices uniformes en la curva densa (parametrización
    uniforme en t). Preserva la suavidad original de la curva incluso cuando
    X no es monótonamente creciente (p. ej. splines perturbados con curvatura).

    Args:
        curve:  (m, 2) curva densa ya generada (p. ej. por catmull_rom).
        n_out:  número de puntos de salida.

    Returns:
        (n_out, 2) puntos muestreados uniformemente por parámetro.
    """
    curve = np.asarray(curve, dtype=np.float64)
    m = len(curve)
    if m <= n_out:
        return curve.copy()
    idx = np.linspace(0, m - 1, n_out)
    idx_lo = np.floor(idx).astype(int)
    idx_hi = np.minimum(idx_lo + 1, m - 1)
    frac = (idx - idx_lo)[:, None]
    return (curve[idx_lo] * (1.0 - frac) + curve[idx_hi] * frac).astype(np.float64)


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
    Corte simulado perturbando los NODOS DE CONTROL de una incisión de referencia.

    Modelo de la maniobra real:
      1. Existe una CURVA DE REFERENCIA (la incisión ideal), definida por un
         conjunto de nodos de control e interpolada con Catmull-Rom.
      2. Un corte real se genera PERTURBANDO CADA NODO DE CONTROL interior dentro
         de un disco de radio `radius` (la "mano" del cirujano) y volviendo a
         interpolar con Catmull-Rom. La trayectoria resultante sigue siendo una
         spline suave que pasa por los nodos desplazados, remuestreada a
         `trajectory_length` puntos.
      3. Si `n_nodes` > nº de nodos de referencia, se añaden nodos intermedios
         por interpolación lineal según longitud de arco sobre los nodos de
         referencia (NO sobre la curva densa), de modo que todos los nodos
         generados tienen correspondencia directa con puntos de control reales.

    Un hospital experto tiene `radius` pequeño (nodos perturbados cerca de los
    de referencia). La calidad se mide como la DESVIACIÓN de la curva generada
    respecto a `reference_curve` (ver CuttingSkill).

    Args:
        reference_nodes: (n_ref, 2) nodos de la incisión ideal. Compartirlos
                         entre hospitales es lo que los hace comparables.
        n_nodes:         nº de nodos de control del corte simulado (>= 2).
                         Por defecto, el mismo nº que la referencia.
                         Puede ser MAYOR: se interpolan nodos intermedios.
        radius:          radio del disco de perturbación por nodo (skill).
        noise_std:       jitter de instrumentación. Se inyecta en los NODOS de
                         control antes de interpolar (no en los puntos finales),
                         de modo que la salida es suave y monótona en X para
                         cualquier valor.
    """

    # Gap mínimo en X entre nodos consecutivos: garantiza que la spline avance
    # siempre de izquierda a derecha (incisión sin retrocesos laterales).
    _EPS = 1e-3
    # Resolución mínima por tramo de la spline densa antes de remuestrear.
    _MIN_SAMPLES_PER_SEGMENT = 24

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

    def _control_nodes(self) -> np.ndarray:
        """Devuelve los nodos de control base para perturbar.

        Si ``n_nodes == len(reference_nodes)`` devuelve directamente los nodos
        de referencia. Si ``n_nodes > len(reference_nodes)`` interpola nodos
        intermedios mediante longitud de arco sobre los propios nodos de
        referencia (interpolación lineal a trozos entre nodos, NO sobre la
        curva densa de Catmull-Rom).
        """
        ref = self.reference_nodes
        if self.n_nodes <= len(ref):
            return ref.copy()
        # Longitud de arco acumulada y normalizada sobre los nodos de referencia
        seg_len = np.sqrt((np.diff(ref, axis=0) ** 2).sum(axis=1))
        s = np.concatenate([[0.0], np.cumsum(seg_len)])
        s /= s[-1]
        u = np.linspace(0.0, 1.0, self.n_nodes)
        x = np.interp(u, s, ref[:, 0])
        y = np.interp(u, s, ref[:, 1])
        return np.stack([x, y], axis=1)

    def _disk_offsets(self, n: int) -> np.ndarray:
        """(n, 2) desplazamientos uniformes dentro de un disco de radio `radius`."""
        angle = self.rng.uniform(0.0, 2.0 * np.pi, n)
        rad = self.radius * np.sqrt(self.rng.uniform(0.0, 1.0, n))
        return np.stack([rad * np.cos(angle), rad * np.sin(angle)], axis=1)

    def _node_jitter(self, n: int) -> np.ndarray:
        """(n, 2) jitter gaussiano de instrumentación por nodo (cero si no aplica)."""
        if self.noise_std <= 0.0:
            return np.zeros((n, 2))
        jx = self.rng.normal(0.0, self.noise_std, n)
        jy = self.rng.normal(0.0, self.noise_std, n)
        return np.stack([jx, jy], axis=1)

    def _perturb_nodes(self, base_nodes: np.ndarray) -> np.ndarray:
        """Perturba los nodos de control interiores (radio + jitter de nodo).

        Restricciones para que la spline no zigzaguee:
          - El primer y último nodo NO se perturban (anclan los extremos del corte).
          - La X de cada nodo interior se clampea entre la del nodo anterior (ya
            perturbado) y la del siguiente nodo BASE, con un gap mínimo ``_EPS``,
            para mantener el orden estrictamente creciente en X.
          - El desplazamiento en Y es libre.
        """
        n = len(base_nodes)
        offsets = self._disk_offsets(n) + self._node_jitter(n)  # radio + ruido
        nodes = base_nodes.copy()

        for i in range(1, n - 1):  # extremos fijos
            x_lo = nodes[i - 1, 0] + self._EPS          # nodo anterior ya perturbado
            x_hi = base_nodes[i + 1, 0] - self._EPS     # nodo siguiente base
            nodes[i, 0] = np.clip(base_nodes[i, 0] + offsets[i, 0], x_lo, x_hi)
            nodes[i, 1] = base_nodes[i, 1] + offsets[i, 1]

        return nodes

    def _trajectory_from_nodes(self, nodes: np.ndarray) -> np.ndarray:
        """Interpola los nodos con Catmull-Rom y los remuestrea a la salida.

        La densidad de la spline se elige para que SIEMPRE supere
        ``trajectory_length`` (``resample_by_parameter`` solo submuestrea), de
        modo que pocas nodos + trayectorias largas no produzcan menos puntos de
        los pedidos. Se remuestrea por parámetro uniforme y se ordena por X para
        garantizar monotonía: Catmull-Rom puede sobrepasar localmente en X aunque
        los nodos estén ordenados, y el retroceso es un artefacto numérico sin
        significado físico (la incisión siempre avanza de izquierda a derecha).
        """
        n_seg = len(nodes) - 1
        sps = max(self._MIN_SAMPLES_PER_SEGMENT,
                  int(np.ceil((self.trajectory_length + 1) / n_seg)) + 1)
        dense = catmull_rom(nodes, samples_per_segment=sps)
        traj = resample_by_parameter(dense, self.trajectory_length)
        return traj[np.argsort(traj[:, 0], kind="stable")]

    def generate(self, num_samples: int) -> np.ndarray:
        """Genera `num_samples` trayectorias perturbando los nodos de control.

        Los nodos base (referencia o interpolados) son fijos para todas las
        muestras; cada trayectoria desplaza esos nodos (``_perturb_nodes``) y los
        convierte en una spline suave (``_trajectory_from_nodes``).
        """
        base_nodes = self._control_nodes()  # fijos para todas las muestras
        trajectories = np.empty((num_samples, self.trajectory_length, 2))
        for i in range(num_samples):
            trajectories[i] = self._trajectory_from_nodes(self._perturb_nodes(base_nodes))
        return trajectories.astype(np.float32)
