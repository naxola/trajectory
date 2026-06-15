"""
Aqui se registran los hospitales

API simétrica a skills/registry.py:
  get_profile("hospital_a")  → HospitalProfile
  list_profiles()            → ["hospital_a", "hospital_b", ...]

  skills y profiles son los dos ejes de variación del experimento federado, para que se manejen con el mismo patrón mental.
  """

from .base import HospitalProfile
from .hospitals import HOSPITAL_A, HOSPITAL_B, HOSPITAL_C, HOSPITAL_D

_REGISTRY: dict[str, HospitalProfile] = {
    "hospital_a": HOSPITAL_A,
    "hospital_b": HOSPITAL_B,
    "hospital_c": HOSPITAL_C,
    "hospital_d": HOSPITAL_D,
}

def get_profile(name: str) -> HospitalProfile:
    """ Devuelve el perfil del hospital
    Args:
        name: identificador del hospital que debe ser el key del registro
    Raises:
        ValueError: si el perfil no está registrado
    Returns:
        HospitalProfile: perfil del hospital
    """
    if name not in _REGISTRY:
        available = list(_REGISTRY.keys())
        raise ValueError(f"Perfil {name} no encontrado. Perfiles disponibles: {available}")
    return _REGISTRY[name]

def list_profiles() -> list[str]:
    """ Devuelve la lista de perfiles """
    return list(_REGISTRY.keys())

