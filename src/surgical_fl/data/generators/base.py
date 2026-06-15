# Generador de datos abstracto. Puede ser de cualquier tipo de dato que necesitemos. 

"""
Esta clase la vamos a usar como interfaz para generar datos sintéticos de cualquier tipo de cirugia.

"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np

@dataclass
class SurgicalGeneratorMetadata:
    """
    Metadatos de un generador de datos sintéticos.
    Lo utilizaré para describir los datos que produce el generador antes de indagar sobre ellos.
    Es como una base de datos utili para logign, validacton, etc.
    """
    skill: str #Puede ser "cutting", "suturing", "dissection", etc.
    output_type: str #trayetory, force tissue_state. multimodal... 
    output_shape: tuple # shape de un solo sample por ejemplo (T,2) en trayectorias 2d (que es en lo que estamos ahora)
    units: str #unidades de los datos (metros, newtons, ....)
    description: str =""
    version: str = "0.0.1"

@dataclass
class SurgicalDataGenerator(ABC):
    """
    Contrato que debe cumplir cualquier generador de datos sintéticos.
    
    Tener en cuenta que un generador produce lotes de samples homogeneos que representan alguna habilidad.

    No se habla nada de pytorch ni flowre (son solo datos)
    """
    def __init__(self, seed: int | None = None):
        self.rng = np.random.default_rng(seed)
    
    @property
    @abstractmethod
    def metadata(self) -> SurgicalGeneratorMetadata:
        """ Se describen la forma y tipo de datos que se van a producir"""
        ...
    
    @abstractmethod
    def generate(self, num_samples: int) -> np.ndarray | dict[str, np.ndarray]:
        """ Genera un lote de datos sintéticos
        Devuelve un array de numpy o un diccionario de arrays de numpy (para datos multimodales)
        """
        ...

    def generate_one(self) -> np.ndarray | dict[str, np.ndarray]:
        """ Genera un solo sample de datos sintéticos"""
        result = self.generate(1)
        if isinstance(result, dict):
            return {k: v[0] for k, v in result.items()}
        return result[0]
    
    def validate_output(self, output:np.ndarray | dict[str, np.ndarray]) -> bool:
        """ Valida que el output sea correcto, es decir que tenga la forma esperada sgún el metadata.
        Esto es útlil para cuando vaya a hacer test de los generadores.
        """
        if isinstance(output, dict):
            return True  # validación multimodal delegada a subclase
        expected = self.metadata.output_shape
        actual = output.shape[1:]  # ignorar dimensión de batch
        if actual != expected:
            raise ValueError(
                f"[{self.__class__.__name__}] Shape inesperado: "
                f"esperado {expected}, obtenido {actual}"
            )
        return True
