# surgical-fl

Framework de investigación para el aprendizaje federado de habilidades quirúrgicas mediante trayectorias sintéticas.

El objetivo es estudiar si varios hospitales pueden entrenar conjuntamente un modelo de predicción de maniobras quirúrgicas sin compartir datos de pacientes, usando [Flower](https://flower.ai/) como infraestructura federada.

---

## Qué hace el sistema

1. **Genera trayectorias sintéticas** que simulan maniobras quirúrgicas (actualmente: corte con bisturí). Cada hospital tiene un perfil propio —ruido, velocidad, curvatura, pericia— que modela la variabilidad real entre centros. Hay tres geometrías de corte: **recta**, **curva** y **spline alrededor de una curva de referencia** (ver más abajo).

2. **Entrena un modelo** (MLP baseline) que aprende a predecir el siguiente punto de una trayectoria dado el punto actual, capturando así el estilo de ejecución de cada habilidad.

3. **Evalúa la calidad** de cada trayectoria con métricas de dominio: **desviación respecto a la incisión ideal**, suavidad, longitud y puntuación compuesta (0–1).

4. **Visualiza** el aprendizaje y los datos: curva de loss, *rollout* autorregresivo del modelo frente a la trayectoria real, y el dataset de entrenamiento frente a su curva ideal.

5. **Soporta dos modos de entrenamiento:**
   - **Centralizado** (`scripts/train_centralized.py`) — mezcla datos de todos los hospitales. Sirve como baseline para comparar contra el federado.
   - **Federado** (via Flower) — cada hospital entrena localmente y solo comparte pesos del modelo, no datos. *(Pendiente: aún no implementado.)*

---

## El modelo de corte tipo spline (clave del experimento)

Cortes de **distinto estilo no son comparables** (una métrica de desviación contra una recta penaliza por diseño a los cortes curvos). Por eso la comparación se hace entre cortes del **mismo tipo** que comparten una **curva de referencia** (la incisión ideal):

1. Existe una **curva de referencia** común, definida por unos nodos de control e interpolada con Catmull-Rom (`DEFAULT_REFERENCE_NODES` en `data/generators/trajectories/cutting.py`).
2. Un **corte real** se genera tomando `n_nodes` anclas sobre esa referencia y desplazando cada una dentro de un **disco de radio `r`** (la "mano" del cirujano). Un hospital experto tiene `r` pequeño. El corte puede tener **más nodos** que la referencia.
3. La **calidad** no compara nodos: mide la **desviación geométrica de la curva generada respecto a la referencia** (`CuttingSkill.evaluate_trajectory(..., reference=...)`).

Dos hospitales que comparten la misma referencia y difieren solo en `r` son directamente comparables: lo que cambia es la pericia, no la forma objetivo.

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
│       ├── base.py      # HospitalProfile (ruido, velocidad, curvatura, pericia…)
│       ├── hospitals.py # A (recto), B (curvo), C y D (spline experto/intermedio)
│       └── registry.py  # get_profile("hospital_a")
│
├── data/
│   ├── generators/
│   │   ├── trajectories/
│   │   │   └── cutting.py   # Linear/Curved/SplineCutGenerator + Catmull-Rom y
│   │   │   │                #   reference_curve (la incisión ideal)
│   │   ├── factory.py       # build_generator(skill, profile) — cruza los dos ejes
│   │   └── registry.py      # get_generator_classes("cutting")
│   ├── datasets.py          # TrajectoryDataset (PyTorch), build_dataloader()
│   └── builders.py          # build_centralized_split() (train mezclado + val
│                            #   y referencia por hospital), seed por perfil
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
├── visualization/
│   └── trajectories.py      # curva de loss, rollout vs real, dataset vs ideal
│
└── utils/
    ├── config.py            # ExperimentConfig (desde TOML o config.json del run)
    ├── seeding.py           # Reproducibilidad
    └── run_io.py            # Persistencia de runs (config, métricas, checkpoints)

scripts/
├── train_centralized.py     # entrena un experimento y genera todas las figuras
└── visualize.py             # regenera las figuras de un run SIN reentrenar

experiments/                 # un TOML por experimento (versionable)
├── cutting_baseline/        # A + B (estilos distintos)
├── cutting_spline/          # C vs D (mismo spline, distinta pericia)
└── cutting_c/               # solo hospital C
```

---

## Ejes de variación del experimento

El sistema tiene dos ejes ortogonales que se combinan en `data/generators/factory.py`:

| Eje | Ejemplos actuales | Dónde se define |
|---|---|---|
| **Skill** (qué maniobra) | `cutting` | `domain/skills/` |
| **Perfil** (qué hospital) | `hospital_a`…`hospital_d` | `domain/profiles/` |

Añadir un nuevo hospital o una nueva habilidad (sutura, disección) solo requiere registrar un nuevo perfil o skill; el resto del sistema los descubre automáticamente.

---

## Instalación

```bash
# Python 3.10+
pip install -e .
```

---

## Uso

> En sistemas donde `python` no existe, usa `python3`. No hace falta `pip install -e .`: los scripts añaden `src/` al path.

### Entrenar un experimento

```bash
# Baseline A + B
python3 scripts/train_centralized.py --config experiments/cutting_baseline/config.toml

# Comparar dos splines (experto vs intermedio, misma incisión ideal)
python3 scripts/train_centralized.py --config experiments/cutting_spline/config.toml

# Un solo hospital (C)
python3 scripts/train_centralized.py --config experiments/cutting_c/config.toml

# Sobrescribir parámetros desde CLI
python3 scripts/train_centralized.py --config experiments/cutting_c/config.toml --epochs 50 --seed 0
```

Cada ejecución crea `outputs/runs/<experimento>_<timestamp>/` con la config exacta (`config.json`), métricas (`metrics.json`, incluye **val loss por hospital**), checkpoints (`best.pt`, `final.pt`) y figuras:

| Figura | Qué muestra |
|---|---|
| `learning_curve.png` | loss de entrenamiento por epoch |
| `predictions.png` | trayectoria real vs **rollout autorregresivo** del modelo + curva ideal |
| `dataset_<hospital>.png` | nube del dataset, **banda azul** entre los dos cortes más desviados y la **curva ideal** encima |

### Re-visualizar un run sin reentrenar

```bash
python3 scripts/visualize.py --run outputs/runs/cutting_c_<timestamp>
python3 scripts/visualize.py --run <ruta> --checkpoint final     # usa final.pt
python3 scripts/visualize.py --run <ruta> --out-suffix _v2       # no sobrescribe
```

Reconstruye el modelo desde `config.json`, carga el checkpoint y regenera todas las figuras (el dataset se regenera de forma determinista con la misma semilla).

### Ajustar la geometría del corte spline

Dos perillas, en distintos sitios:

- **Nodos del corte** (`n_nodes`) y **pericia** (`radius`) → en el perfil del hospital, `domain/profiles/hospitals.py`. Más `n_nodes` o más `radius` ⇒ mayor desviación respecto a la curva ideal.
- **Forma de la incisión ideal** (`DEFAULT_REFERENCE_NODES`) → en `data/generators/trajectories/cutting.py`. Afecta a todos los hospitales spline (los mantiene comparables).

### Configuración de experimentos

Cada experimento se define en un fichero TOML versionable:

```toml
name     = "cutting_spline"
skill    = "cutting"
profiles = ["hospital_c", "hospital_d"]   # mismo spline, distinta pericia

[model]
name       = "mlp"
hidden_dim = 80
dropout    = 0.2

[data]
num_samples       = 4000
trajectory_length = 100
val_split         = 0.2

[training]
epochs        = 30
batch_size    = 64
learning_rate = 0.0005
seed          = 10
```

> La geometría del spline (`n_nodes`, `radius`) vive hoy en el perfil del hospital, no en el TOML.

### Tests

```bash
pytest tests/ -v
```

---

## Hospitales simulados

| Hospital | Estilo de corte | Ruido (`noise_std`) | Geometría |
|---|---|---|---|
| Hospital A | Recto, experto | 0.015 | línea recta |
| Hospital B | Curvo, intermedio | 0.055 | parábola (`curvature` 0.08) |
| Hospital C | Spline experto | 0.010 | spline, `radius` 0.030 |
| Hospital D | Spline intermedio | 0.020 | spline, `radius` 0.080 |

A y B usan estilos distintos (no comparables entre sí por desviación). **C y D comparten la misma incisión de referencia** y solo difieren en la pericia (`radius`): ese es el par diseñado para comparar de forma justa.

La diferencia de perfiles hace que un futuro modelo federado deba generalizar entre estilos quirúrgicos distintos, replicando la heterogeneidad real entre centros. Cada hospital recibe además una **semilla derivada distinta**, de modo que sus datos son estadísticamente independientes.

---

## Licencia

Apache 2.0
