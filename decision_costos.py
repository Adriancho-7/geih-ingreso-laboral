"""Regla de decisión por matriz de costos para los cuatro tramos ordenados C1–C4.

Sustituye el umbral por costo de la versión binaria (falso negativo = 3,
falso positivo = 1). Filas = clase real, columnas = clase asignada.

- C1 (ingreso <= P10 de su ciudad y año) es el grupo prioritario de la
  aplicación: asignar un tramo superior a una persona de C1 la deja fuera de
  la orientación que le corresponde y cuesta PRIORIDAD = 3 por cada tramo de
  distancia, como el falso negativo binario.
- Los demás errores cuestan la distancia ordinal |real - asignada|: confundir
  tramos vecinos es menos grave que confundir extremos, en ambos sentidos.

La decisión elige, para cada persona, la clase con menor costo esperado
sum_j p_j * C[j, k]. Con esta matriz se asigna C1 si y solo si
P(C1 | x) >= TAU = 1 / (1 + PRIORIDAD) = 0,25: exactamente el umbral binario
p* = 1 / (1 + 3); entre C2, C3 y C4 decide el costo ordinal esperado.
Las unidades son hipotéticas (no COP); la matriz se fija por la aplicación
antes de mirar validación o prueba y no se ajusta con los datos.
"""
import numpy as np
import pandas as pd

PRIORIDAD = 3.0
TAU = 1 / (1 + PRIORIDAD)
PRIORIDADES = [1.0, 2.0, 3.0, 5.0]      # solo para la sensibilidad en validación
PROBABILISTICOS = ['Logistica', 'Arbol', 'Random Forest', 'KNN', 'Red MLP']
NOMBRES = ['C1', 'C2', 'C3', 'C4']
EDADES = ['18–29', '30–44', '45–59', '60 o más']


def matriz(prioridad=PRIORIDAD):
    """Matriz 4x4 de costos: fila = clase real, columna = clase asignada."""
    real, asignada = np.meshgrid(np.arange(4), np.arange(4), indexing='ij')
    c = np.abs(real - asignada).astype(float)
    c[0, 1:] = prioridad * np.arange(1, 4)
    return c


COSTOS = matriz()


def decidir(prob, costos=COSTOS):
    """Clase de mínimo costo esperado; en empate se conserva el tramo más bajo."""
    prob = np.asarray(prob, dtype=float)
    assert prob.ndim == 2 and prob.shape[1] == 4, prob.shape
    assert np.all(prob >= -1e-9) and np.allclose(prob.sum(axis=1), 1, atol=1e-5), 'Se requieren probabilidades'
    return (prob @ np.asarray(costos, dtype=float)).argmin(axis=1).astype('int8')


def costo_medio(y, pred, costos=COSTOS):
    """Costo promedio por persona según la matriz (multiplicar por 1000 para costo por mil)."""
    return float(np.asarray(costos)[np.asarray(y, dtype=int), np.asarray(pred, dtype=int)].mean())


def _tasas(y, pred, c):
    y = np.asarray(y) == c
    p = np.asarray(pred) == c
    tp = int((y & p).sum()); fn = int((y & ~p).sum()); fp = int((~y & p).sum()); tn = int((~y & ~p).sum())
    div = lambda a, b: a / b if b > 0 else np.nan
    return dict(positivos=tp + fn, predichos=tp + fp, VP=tp, FN=fn, FP=fp, VN=tn,
                TPR=div(tp, tp + fn), FPR=div(fp, fp + tn), VPP=div(tp, tp + fp))


def tasas_ovr(y, pred):
    """TPR, FPR y VPP de cada clase frente a las demás (uno contra el resto)."""
    n = len(np.asarray(y))
    rows = []
    for c in range(4):
        t = _tasas(y, pred, c)
        rows.append(dict(clase=NOMBRES[c], n=n, prevalencia=t['positivos'] / n, **t))
    return pd.DataFrame(rows)


def tasas_por_grupo(frame, variable, real='real', pred='predicha'):
    """Tasas uno contra el resto por clase dentro de cada grupo de `variable`."""
    rows = []
    for grupo, g in frame.dropna(subset=[variable]).groupby(variable, sort=True, observed=True):
        for c in range(4):
            t = _tasas(g[real].to_numpy(), g[pred].to_numpy(), c)
            rows.append(dict(variable=variable, grupo=str(grupo), clase=NOMBRES[c], n=len(g),
                             prevalencia=t['positivos'] / len(g), **t))
    return pd.DataFrame(rows)


def brechas(tasas, minimo=30):
    """Diferencia máxima entre grupos (máx − mín) de cada tasa, por variable y clase.

    Solo se comparan grupos con al menos `minimo` casos en el denominador de cada tasa.
    """
    rows = []
    for (variable, clase), g in tasas.groupby(['variable', 'clase'], sort=False):
        r = dict(variable=variable, clase=clase)
        for tasa, den in [('TPR', 'positivos'), ('FPR', None), ('VPP', 'predichos')]:
            if den is None:
                h = g[(g.n - g.positivos) >= minimo]
            else:
                h = g[g[den] >= minimo]
            v = h[tasa].dropna()
            r[f'brecha_{tasa}'] = float(v.max() - v.min()) if len(v) > 1 else np.nan
            r[f'grupo_min_{tasa}'] = h.loc[v.idxmin(), 'grupo'] if len(v) > 1 else ''
            r[f'grupo_max_{tasa}'] = h.loc[v.idxmax(), 'grupo'] if len(v) > 1 else ''
        rows.append(r)
    return pd.DataFrame(rows)


def calibracion(y, prob, bins=10):
    """Frecuencia observada frente a probabilidad media por clase (diagrama de fiabilidad)."""
    y = np.asarray(y)
    prob = np.asarray(prob, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    rows = []
    for c in range(4):
        idx = np.clip(np.digitize(prob[:, c], edges[1:-1]), 0, bins - 1)
        for b in range(bins):
            m = idx == b
            if m.sum() == 0:
                continue
            rows.append(dict(clase=NOMBRES[c], intervalo=b, n=int(m.sum()),
                             prob_media=float(prob[m, c].mean()), frecuencia=float((y[m] == c).mean())))
    return pd.DataFrame(rows)
