# Connect-4 First-Visit Monte Carlo Agent

Agente para Connect-4 basado en First-Visit Monte Carlo (FVMC) con self-play y aproximación tabular de la función de valor Q.

## Descripción

El agente aprende una función Q tabular mediante episodios de Monte Carlo:

El entrenamiento combina partidas contra un jugador aleatorio y episodios de self-play, donde el agente se enfrenta a sí mismo para aprender estrategias más sofisticadas.

## Características del agente

- **First-Visit MC**: actualiza Q solo en la primera visita a cada par (estado, acción) por episodio
- **Self-play configurable**: proporción de episodios donde el oponente también usa los Q-values aprendidos

## Requisitos

```bash
pip install numpy matplotlib
```

> `zipfile`, `os`, `sys`, `importlib` y `collections` son parte de la librería estándar de Python, no requieren instalación.

## Archivos necesarios

Antes de ejecutar el notebook, asegúrate de tener estos 4 archivos en el directorio de trabajo (en Colab, súbelos manualmente):

| Archivo          | Descripción                                                                      |
|------------------|----------------------------------------------------------------------------------|
| `tournament.zip` | Paquete del torneo; contiene `connect4/` con el entorno y la clase base `Policy` |
| `aha.py`         | Clase `Aha` — este agente (FVMC mejorado con self-play)                          |
| `policy (1).py`  | Clase `AhaMCTS_Greedy` — agente MCTS con el que se compara                       |
| `policy_adp.py`  | Clase `MyPolicy` — agente ADP con el que se compara                              |

## Uso

```python
from aha import Aha

agente = Aha(n_episodes=20000, selfplay_ratio=0.6)
agente.mount()
accion = agente.act(tablero)
```

## Parámetros principales

| Parámetro        | Descripción                                              | Default |
|------------------|----------------------------------------------------------|---------|
| `n_episodes`     | Número de episodios de entrenamiento                     | 20000   |
| `selfplay_ratio` | Proporción de episodios jugados contra sí mismo          | 0.6     |
| `gamma`          | Factor de descuento                                      | 0.95    |
| `epsilon`        | Probabilidad de exploración (ε-greedy)                   | 0.15    |

## Estructura del tablero

El tablero es un `np.ndarray` de forma `(6, 7)` con valores:

- `1` — ficha del jugador actual
- `-1` — ficha del oponente
- `0` — celda vacía
