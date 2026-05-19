# surgical-fl

Framework de investigación para el aprendizaje federado de habilidades quirúrgicas mediante trayectorias sintéticas.

El objetivo es estudiar si varios hospitales pueden entrenar conjuntamente un modelo de predicción de maniobras quirúrgicas sin compartir datos de pacientes, usando [Flower](https://flower.ai/) como infraestructura federada.

---

## Qué hace el sistema

1. **Genera trayectorias sintéticas** que simulan maniobras quirúrgicas (actualmente: corte con bisturí). Cada hospital tiene un perfil propio —nivel de ruido, velocidad, curvatura— que modela la variabilidad real entre centros.

2. **Entrena un modelo** (MLP baseline) que aprende a predecir el siguiente punto de una trayectoria dado el punto actual, capturando así el estilo de ejecución de cada habilidad.

3. **Evalúa la calidad** de cada trayectoria con métricas de dominio: error de camino, suavidad, puntuación compuesta (0–1).

4. **Soporta dos modos de entrenamiento:**
   - **Centralizado** (`scripts/train_centralized.py`) — mezcla datos de todos los hospitales. Sirve como baseline para comparar contra el federado.
   - **Federado** (via Flower) — cada hospital entrena localmente y solo comparte pesos del modelo, no datos.

---

## Estructura del proyecto

```
src/surgical_fl/
├── domain/
│   ├── skills/          # Qué significa ejecutar bien una maniobra
│   │   ├── base.py      # Clase abstracta SurgicalSkill
│   │   ├── cutting.py   # Evaluación de la habilidad de corte
│   │   └── registry.py  # get_skill("cutting")
│   └── profiles/        # Características de cada hospital
│       ├── base.py      # HospitalProfile (ruido, velocidad, curvatura…)
│       ├── hospitals.py # Hospital A (experto) y Hospital B (intermedio)
│       └── registry.py  # get_profile("hospital_a")
│
├── data/
│   ├── generators/
│   │   ├── trajectories/
│   │   │   └── cutting.py   # LinearCutGenerator, CurvedCutGenerator, SplineCutGenerator
│   │   ├── factory.py       # build_generator(skill, profile) — cruza los dos ejes
│   │   └── registry.py      # get_generator_classes("cutting")
│   ├── datasets.py          # TrajectoryDataset (PyTorch), build_dataloader()
│   └── builders.py          # build_dataset_from_profiles(), build_dataset_for_profile()
│
├── models/
│   ├── base.py              # SurgicalModel (nn.Module + serialización Flower)
│   ├── registry.py          # get_model("mlp")
│   └── trajectory/
│       ├── base.py          # TrajectoryModel
│       └── mlp.py           # TrajectoryMLP — baseline de 3 capas
│
├── training/
│   └── trainer.py           # train_one_epoch(), evaluate(), train_local()
│
└── utils/
    ├── config.py            # ExperimentConfig (cargada desde TOML)
    ├── seeding.py           # Reproducibilidad
    └── run_io.py            # Persistencia de runs (config, métricas, checkpoints)
```

---

## Ejes de variación del experimento

El sistema tiene dos ejes ortogonales que se combinan en `data/generators/factory.py`:

| Eje | Ejemplos actuales | Dónde se define |
|---|---|---|
| **Skill** (qué maniobra) | `cutting` | `domain/skills/` |
| **Perfil** (qué hospital) | `hospital_a`, `hospital_b` | `domain/profiles/` |

Añadir un nuevo hospital o una nueva habilidad (sutura, disección) solo requiere registrar un nuevo perfil o skill; el resto del sistema los descubre automáticamente.

---

## Instalación

```bash
# Python 3.10+
pip install -e .
```

---

## Uso

### Entrenamiento centralizado (baseline)

```bash
python scripts/train_centralized.py
# Con configuración personalizada
python scripts/train_centralized.py --config experiments/cutting_baseline/config.toml
# Sobrescribir parámetros desde CLI
python scripts/train_centralized.py --epochs 50 --seed 0
```

Los resultados se guardan en `outputs/runs/<experimento>_<timestamp>/` con la configuración exacta, métricas, checkpoints y figuras.

### Configuración de experimentos

Cada experimento se define en un fichero TOML versionable:

```toml
name    = "cutting_baseline"
skill   = "cutting"
profiles = ["hospital_a", "hospital_b"]

[model]
name       = "mlp"
hidden_dim = 64
dropout    = 0.1

[data]
num_samples       = 400
trajectory_length = 50
val_split         = 0.2

[training]
epochs        = 20
batch_size    = 32
learning_rate = 0.001
seed          = 42
```

### Tests

```bash
pytest tests/ -v
```

---

## Hospitales simulados

| Hospital | Perfil | Ruido (`noise_std`) | Curvatura |
|---|---|---|---|
| Hospital A | Experto | 0.015 | 0.0 (lineal) |
| Hospital B | Intermedio | 0.055 | 0.08 (curvo) |

La diferencia de perfiles hace que el modelo federado deba aprender a generalizar entre estilos quirúrgicos distintos, replicando la heterogeneidad real entre centros.

---

## Licencia

Apache 2.0
