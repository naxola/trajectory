# surgical-fl

Framework de investigación para el aprendizaje federado de habilidades quirúrgicas mediante trayectorias sintéticas.

El objetivo es estudiar si varios hospitales pueden entrenar conjuntamente un modelo de predicción de maniobras quirúrgicas sin compartir datos de pacientes, usando [Flower](https://flower.ai/) como infraestructura federada.

---

## Qué hace el sistema

1. **Genera trayectorias sintéticas** que simulan maniobras quirúrgicas (actualmente: corte con bisturí). Cada hospital tiene un perfil propio —ruido, velocidad, curvatura, pericia— que modela la variabilidad real entre centros. Hay tres geometrías de corte: **recta**, **curva** y **spline alrededor de una curva de referencia** (ver más abajo).

2. **Entrena un modelo** (MLP baseline o RNN/GRU) que aprende a predecir el siguiente punto de una trayectoria dado el punto actual, capturando así el estilo de ejecución de cada habilidad.

3. **Evalúa la calidad** de cada trayectoria con métricas de dominio: **desviación respecto a la incisión ideal**, suavidad, longitud y puntuación compuesta (0–1).

4. **Visualiza** el aprendizaje y los datos: curva de loss, *rollout* autorregresivo del modelo frente a la trayectoria real, y el dataset de entrenamiento frente a su curva ideal.

5. **Soporta tres modos de entrenamiento:**
   - **Centralizado** (`scripts/train_centralized.py`) — mezcla datos de todos los hospitales. Sirve como baseline para comparar contra el federado.
   - **Federado** (via Flower) — cada hospital entrena localmente y solo comparte pesos del modelo, no datos. Usa `FlowerClient` (`NumPyClient`) y FedAvg como estrategia de agregación.
   - **Comparación FL vs Centralizado** (`scripts/compare_fl_vs_centralized.py`) — entrena ambos modelos con los mismos datos iniciales y evalúa sobre los mismos conjuntos de validación para una comparación justa.

---

## El modelo de corte tipo spline (clave del experimento)

Cortes de **distinto estilo no son comparables** (una métrica de desviación contra una recta penaliza por diseño a los cortes curvos). Por eso la comparación se hace entre cortes del **mismo tipo** que comparten una **curva de referencia** (la incisión ideal):

1. Existe una **curva de referencia** común, definida por unos nodos de control e interpolada con Catmull-Rom (`DEFAULT_REFERENCE_NODES` en `data/generators/trajectories/cutting.py`).
2. Un **corte real** se genera perturbando cada **nodo de control** dentro de un **disco de radio `r`** (la "mano" del cirujano) y reconstruyendo el spline con Catmull-Rom sobre los nodos perturbados. Un hospital experto tiene `r` pequeño. El corte puede tener **más nodos** que la referencia (interpolados linealmente por longitud de arco sobre los nodos de referencia). El **jitter de instrumentación** (`noise_std`) también se inyecta en los nodos de control, *antes* de interpolar: así la trayectoria es siempre una spline **suave** y **monótona en X**, sin picos ni retrocesos, para cualquier nivel de ruido.
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
│   ├── registry.py          # get_model("mlp"), get_model("rnn")
│   └── trajectory/
│       ├── base.py          # TrajectoryModel
│       ├── mlp.py           # TrajectoryMLP — baseline de 3 capas
│       └── rnn.py           # TrajectoryRNN — GRU con estado oculto
│
├── federation/
│   ├── __init__.py
│   ├── client.py            # FlowerClient (NumPyClient): fit/evaluate con Flower
│   ├── client_app.py        # ClientApp para flwr run / run_simulation
│   └── server_app.py        # ServerApp con FedAvg para flwr run / run_simulation
│
├── training/
│   └── trainer.py           # train_one_epoch(), evaluate(), train_local()
│
├── visualization/
│   ├── trajectories.py      # curva de loss, rollout vs real, dataset vs ideal
│   └── comparison.py        # barras RMSE, predicciones side-by-side, loss FL
│
└── utils/
    ├── config.py            # ExperimentConfig + FederationConfig (desde TOML)
    ├── seeding.py           # Reproducibilidad
    └── run_io.py            # Persistencia de runs (config, métricas, checkpoints)

scripts/
├── train_centralized.py          # entrena un experimento centralizado
├── compare_fl_vs_centralized.py  # compara FL vs centralizado con mismos datos
├── visualize.py                  # regenera figuras de un run SIN reentrenar
└── visualize_dataset.py          # inspecciona un generador en crudo

experiments/                      # un TOML por experimento (versionable)
├── cutting_baseline/             # A + B (estilos distintos)
├── cutting_spline/               # C vs D (mismo spline, distinta pericia)
├── cutting_spline_rnn/           # C vs D con modelo RNN
├── cutting_c/                    # solo hospital C
└── federated_vs_centralized/     # comparación FL vs centralizado (C + D, RNN)

tests/                            # pytest
├── test_datasets.py
├── test_factory.py
├── test_federation.py            # FlowerClient, FedAvg, ronda FL, config
├── test_generators.py
├── test_models.py
├── test_registries.py
└── test_skills.py
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

Esto instala `torch`, `numpy`, `matplotlib` y `flwr[simulation]`.

---

## Uso

> En sistemas donde `python` no existe, usa `python3`. No hace falta `pip install -e .`: los scripts añaden `src/` al path.

### Comparar Federado vs Centralizado (experimento principal)

```bash
python3 scripts/compare_fl_vs_centralized.py
python3 scripts/compare_fl_vs_centralized.py --config experiments/federated_vs_centralized/config.toml
```

El script:
1. **Entrena (o carga de caché)** un modelo centralizado con datos mezclados de los hospitales C y D.
2. **Entrena un modelo federado** con Flower: cada hospital es un cliente FL que entrena localmente y comparte pesos via FedAvg.
3. **Evalúa ambos** sobre los mismos conjuntos de validación por hospital.
4. **Genera figuras comparativas** y guarda métricas.

La comparación es justa:
- **Mismos datos**: mismo split train/val derivado de la misma semilla.
- **Misma inicialización**: ambos modelos parten de los mismos pesos aleatorios.
- **Mismo cómputo total**: `epochs` centralizado = `num_rounds × local_epochs` federado.
- **Misma evaluación**: RMSE sobre los mismos val sets por hospital.

El modelo centralizado se **cachea** en `outputs/models/centralized/` y no se reentrena en ejecuciones posteriores (usa `--force-retrain` para forzarlo).

Salida en `outputs/runs/<nombre>_comparison_<timestamp>/`:

| Archivo | Qué contiene |
|---|---|
| `config.json` | Configuración exacta del experimento |
| `metrics.json` | RMSE por hospital para ambos enfoques |
| `checkpoints/centralized_best.pt` | Mejor modelo centralizado |
| `checkpoints/federated_final.pt` | Modelo federado tras la última ronda |
| `figures/comparison_rmse.png` | Barras RMSE: centralizado vs federado por hospital |
| `figures/comparison_predictions.png` | Rollout autorregresivo side-by-side |
| `figures/centralized_learning_curve.png` | Loss por epoch del centralizado |
| `figures/fl_round_losses.png` | Loss global por ronda del federado |

### Entrenar solo el modelo centralizado

```bash
# Baseline A + B
python3 scripts/train_centralized.py --config experiments/cutting_baseline/config.toml

# Comparar dos splines (experto vs intermedio, misma incisión ideal)
python3 scripts/train_centralized.py --config experiments/cutting_spline/config.toml

# RNN sobre splines
python3 scripts/train_centralized.py --config experiments/cutting_spline_rnn/config.toml

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
| `dataset_<hospital>.png` | nube del dataset, los **dos cortes más desviados** resaltados y la **curva ideal** (verde) encima |

### Re-visualizar un run sin reentrenar

```bash
python3 scripts/visualize.py --run outputs/runs/cutting_c_<timestamp>
python3 scripts/visualize.py --run <ruta> --checkpoint final     # usa final.pt
python3 scripts/visualize.py --run <ruta> --out-suffix _v2       # no sobrescribe
```

Reconstruye el modelo desde `config.json`, carga el checkpoint y regenera todas las figuras (el dataset se regenera de forma determinista con la misma semilla).

### Visualizar un dataset en crudo (sin entrenar)

`scripts/visualize_dataset.py` genera figuras directamente desde el generador, sin necesitar ningún run ni modelo entrenado. Útil para ajustar la geometría antes de lanzar un experimento.

```bash
# SplineCutGenerator — configuración básica
python3 scripts/visualize_dataset.py --generator spline

# Spline con 6 nodos de control y radio mayor
python3 scripts/visualize_dataset.py --generator spline --n-nodes 6 --radius 0.08 --num-samples 300

# Spline con nodos de referencia personalizados (JSON)
python3 scripts/visualize_dataset.py --generator spline \
  --ref-nodes "[[0,0],[0.2,0.05],[0.4,0.1],[0.6,-0.05],[0.8,-0.1],[1,0]]"

# Corte lineal
python3 scripts/visualize_dataset.py --generator linear --noise-std 0.05

# Corte parabólico
python3 scripts/visualize_dataset.py --generator curved --curvature 0.2 --noise-std 0.03

# Directorio de salida personalizado
python3 scripts/visualize_dataset.py --generator spline --out-dir outputs/mi_prueba
```

Por cada ejecución se generan hasta **tres figuras** en `--out-dir` (por defecto `outputs/datasets/`):

| Figura | Descripción |
|---|---|
| `<gen>_dataset.png` | Nube de trayectorias simuladas, los dos cortes más desviados resaltados y curva de referencia ideal (verde) |
| `<gen>_nodes.png` | *(solo spline)* Nodos de referencia, discos de radio `r`, y un corte perturbado de ejemplo con los nodos perturbados |
| `<gen>_reference.png` | Curva de referencia ideal aislada (con nodos marcados en spline) |

#### Parámetros por generador

| Parámetro | Generadores | Descripción | Defecto |
|---|---|---|---|
| `--num-samples` | todos | Nº de trayectorias a generar | `200` |
| `--trajectory-length` | todos | Puntos por trayectoria | `50` |
| `--seed` | todos | Semilla aleatoria | `0` |
| `--max-background` | todos | Máx. trayectorias dibujadas en el fondo | `200` |
| `--out-dir` | todos | Directorio de salida | `outputs/datasets` |
| `--noise-std` | linear, curved, spline | Ruido gaussiano. En spline es el jitter de instrumentación inyectado en los nodos de control (mantiene la curva suave); en linear/curved se añade en Y | `0.02` / `0.0` |
| `--speed-variance` | linear, curved | Varianza en la velocidad | `0.01` |
| `--curvature` | curved | Amplitud de la parábola | `0.1` |
| `--radius` | spline | Radio del disco de perturbación por nodo | `0.05` |
| `--n-nodes` | spline | Nº de nodos de control del corte (≥ nodos de ref.) | igual que ref. |
| `--ref-nodes` | spline | Nodos de referencia como JSON `"[[x,y],…]"` | curva en S de 4 nodos |

### Ajustar la geometría del corte spline

Dos perillas, en distintos sitios:

- **Nodos del corte** (`n_nodes`) y **pericia** (`radius`) → en el perfil del hospital, `domain/profiles/hospitals.py`. Más `n_nodes` o más `radius` ⇒ mayor desviación respecto a la curva ideal.
- **Forma de la incisión ideal** (`DEFAULT_REFERENCE_NODES`) → en `data/generators/trajectories/cutting.py`. Afecta a todos los hospitales spline (los mantiene comparables).

---

## Configuración de experimentos

Cada experimento se define en un fichero TOML versionable:

```toml
name     = "federated_vs_centralized"
skill    = "cutting"
profiles = ["hospital_c", "hospital_d"]

[model]
name       = "rnn"
hidden_dim = 80
dropout    = 0.2

[data]
num_samples       = 4000
trajectory_length = 100
val_split         = 0.2

[training]
epochs        = 30          # epochs para el entrenamiento centralizado
batch_size    = 64
learning_rate = 0.0005
seed          = 10

[federation]
num_rounds   = 10           # rondas de comunicación FL
local_epochs = 3            # epochs locales por ronda (num_rounds × local_epochs = epochs)
```

> La geometría del spline (`n_nodes`, `radius`) vive hoy en el perfil del hospital, no en el TOML.

---

## Tests

```bash
# Ejecutar todos los tests
python3 -m pytest tests/ -v

# Solo tests de federación
python3 -m pytest tests/test_federation.py -v

# Solo un test específico
python3 -m pytest tests/test_federation.py::TestFederatedRound::test_one_round_reduces_loss -v
```

### Estructura de los tests

| Fichero | Qué verifica |
|---|---|
| `test_datasets.py` | `TrajectoryDataset`: shapes, dtypes, desplazamiento input/target |
| `test_factory.py` | Fábrica de generadores: combinaciones válidas, heterogeneidad entre hospitales |
| `test_federation.py` | `FlowerClient` (fit/evaluate), FedAvg, ronda FL completa, `FederationConfig` |
| `test_generators.py` | Linear/Curved/SplineCutGenerator: shapes, monotonía, reproducibilidad |
| `test_models.py` | MLP/RNN: forward shapes, serialización Flower, registry |
| `test_registries.py` | Registros de skills, profiles y generadores |
| `test_skills.py` | `CuttingSkill`: métricas, score, validación de trayectorias |

### Cómo escribir un test nuevo

1. **Crear o editar** un fichero `tests/test_<modulo>.py`. Pytest los descubre automáticamente.
2. **Importar** desde `surgical_fl.*` (el `conftest.py` ya añade `src/` al path).
3. **Agrupar** tests relacionados en una clase `TestNombreDescriptivo`.
4. **Usar fixtures** para datos compartidos (ver `@pytest.fixture(scope="module")` en `test_federation.py` para evitar regenerar datos en cada test).

Ejemplo mínimo:

```python
"""Tests para mi_modulo."""
import torch
from surgical_fl.models.registry import get_model


class TestMiModelo:

    def test_forward_shape(self):
        model = get_model("rnn", hidden_dim=16)
        x = torch.randn(4, 10, 2)  # batch=4, T=10, dims=2
        y = model(x)
        assert y.shape == (4, 10, 2)

    def test_loss_decreases_after_training(self):
        """Verifica que el modelo aprende algo en unas pocas epochs."""
        from surgical_fl.data.builders import build_dataset_for_profile
        from surgical_fl.training.trainer import train_local, evaluate
        from torch.utils.data import DataLoader

        ds, _ = build_dataset_for_profile("cutting", "hospital_c", 40, 20, seed=0)
        loader = DataLoader(ds, batch_size=16)

        model = get_model("rnn", hidden_dim=16)
        loss_before, _ = evaluate(model, loader, torch.device("cpu"))
        model, _ = train_local(model, loader, epochs=5, learning_rate=0.001,
                               device=torch.device("cpu"))
        loss_after, _ = evaluate(model, loader, torch.device("cpu"))
        assert loss_after < loss_before
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

La diferencia de perfiles hace que el modelo federado deba generalizar entre estilos quirúrgicos distintos, replicando la heterogeneidad real entre centros. Cada hospital recibe además una **semilla derivada distinta**, de modo que sus datos son estadísticamente independientes.

---

## Licencia

Apache 2.0
