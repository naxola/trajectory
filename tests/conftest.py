"""
Configuración de pytest para el proyecto.

Añade src/ al path para que los imports funcionen
sin necesidad de pip install -e . durante desarrollo.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))