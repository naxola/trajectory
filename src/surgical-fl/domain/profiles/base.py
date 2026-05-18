"""
Aqui simulamos los hospitales

El pefil base de cada hospital define las condiciones que diferencia a un clientre de otro en la federacion (no solo las que afectan a la generación de datos)

"""

from dataclasses import dataclass, field

@dataclass
class HospitalProfile:
    """
    Parámetros que caracterizan a un hgospital (cliente federado)

    Estos parámetros se modifican en el generador de datos sintéticos, el optimizador local y cualquier otra componente que necesite saber "qué hospital es este"

    Aqui se pueden ir añadiendo los nuevos campos (coste computacional, politica de privacidad, frecuencia de comunicación... etc)
    """
    name: str #Identificador del hospital
    nose_std: float #Ruidos o imprecisiones por instrumentación
    speed_variance: float #Varianza en la velocidad de ejecución de la maniobra
    curvature_bias: float #Sesgo en la curvatura de la trayectoria
    local_epochs_factor: float = 1.0 #Factor multiplicador de epochs locales
    lr_multiplier: float = 1.0 #Factor multiplicador del learning rate local
    skill_overrides: dict = field(default_factory=dict) #Sobreescritura de parámetros por skill. por ejemplo {"skill_name": {"noise_std": 0.1}}

    def get_skill_param(self, skill: str, param: str, default):
        """ Devuelve el parámetro específico de una skill, si no existe devuelve el parámetro base"""
        return self.skill_overrides.get(skill, {}).get(param, default)