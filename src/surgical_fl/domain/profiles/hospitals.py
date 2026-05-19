"""
Aqui definimos los hositales simulados
"""
from .base import HospitalProfile

HOSPITAL_A = HospitalProfile(
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