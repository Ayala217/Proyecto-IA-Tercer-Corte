# Connect-4 ADP Agent

Agente para Connect-4 basado en Approximate Dynamic Programming (ADP) utilizando aproximación lineal de la función de valor y aprendizaje TD(0).

## Descripción

El agente aprende una función de valor aproximada:

:contentReference[oaicite:0]{index=0}

donde:

- `φ(s)` representa un vector de características del tablero.
- `θ` corresponde a los pesos aprendidos mediante Temporal Difference Learning TD(0).

El sistema combina:
- aprendizaje por refuerzo,
- evaluación heurística,
- y reglas tácticas de seguridad.

---

# Características del agente

El agente utiliza las siguientes features:

| Feature | Descripción |
|---|---|
| bias | término constante |
| own1-own4 | ventanas propias de 1 a 4 fichas |
| opp1-opp4 | ventanas rivales de 1 a 4 fichas |
| center | control de la columna central |

Además incorpora heurísticas:
- victoria inmediata,
- bloqueo inmediato,
- evitar movimientos suicidas,
- preferencia por el centro.

---

# Requisitos

## Dependencias

Instalar:

```bash
pip install numpy
