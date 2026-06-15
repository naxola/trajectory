"""
Aqui definimos los hositales simulados
"""
from .base import HospitalProfile

HOSPITAL_A = HospitalProfile(
    #Este hospital solo genera trayectorias lineales
    name= "Hospital A - EXPERTO",
    noise_std= 0.015,
    speed_variance= 0.004,
    curvature_bias= 0.0,
    local_epochs_factor= 1.0,
    lr_multiplier= 1.0,
    skill_overrides= {
        
    }
)

HOSPITAL_B = HospitalProfile(
    #Este hospital genera trayectorias curvas con mas ruido y variabilidad en la velocidad
    name= "Hospital B - INTERMEDIO",
    noise_std=0.055,
    speed_variance=0.015,
    curvature_bias=0.08,            # tiende a curvar ligeramente
    local_epochs_factor=1.0,
    lr_multiplier=1.0,
    skill_overrides={
        "cutting": {"curvature": 0.08},
    },
)
HOSPITAL_C = HospitalProfile(
    # Spline EXPERTO: corta muy cerca de la incisión de referencia (radio pequeño).
    # Comparte la curva de referencia por defecto con Hospital D → comparables.
    name= "Hospital C - SPLINE EXPERTO",
    noise_std=0.010,               # jitter de instrumentación bajo
    speed_variance=0.015,          # (no usado por el spline, sí por otros estilos)
    curvature_bias=0.0,
    local_epochs_factor=1.0,
    lr_multiplier=1.0,
    skill_overrides={
        "cutting": {"cut_style": "spline", "radius": 0.030, "n_nodes": 6},
    },
)
HOSPITAL_D = HospitalProfile(
    # Spline INTERMEDIO: misma incisión de referencia, mano menos precisa
    # (radio mayor → mayor desviación respecto a la curva ideal).
    name= "Hospital D - SPLINE INTERMEDIO",
    noise_std=0.020,
    speed_variance=0.015,
    curvature_bias=0.0,
    local_epochs_factor=1.0,
    lr_multiplier=1.0,
    skill_overrides={
        "cutting": {"cut_style": "spline", "radius": 0.080, "n_nodes": 6},
    },
)