"""Despliegue del modelo final: dominio nacional con la ciudad como predictor.

Es el único modelo que se despliega y el que describe la ficha (modelo_final/FICHA_MODELO.md).
Los modelos de las cinco áreas son solo de comparación y no se sirven aquí.

Uso:  python cargar_y_predecir.py --entrada perfiles.csv [--salida predicciones_nuevas.csv]
Columnas: edad (años), antiguedad (meses), horas (semanales), sexo (Hombre/Mujer), jefe (Si/No),
estudia (Si/No), estado_civil (1-6), educacion (Primaria o menos/Secundaria/Media/Superior) y
ciudad (Medellín AM, Barranquilla AM, Bogotá, Bucaramanga AM, Cali AM o Resto del país).

La clase se asigna con la matriz de costos (mínimo costo esperado), no con la probabilidad
máxima. No requiere ingreso observado. Los tramos son relativos a los cortes P10/P50/P90 de
entrenamiento de la ciudad y el año; no se estiman cortes de años futuros.
"""
from pathlib import Path
import argparse, json
import numpy as np
import pandas as pd
import joblib
from ml_final import FEATURES_FINAL, CIUDADES, RESTO, inputs, scores, NEIGHBOR_CACHE
import decision_costos as dc


def cargar_modelo(root=None):
    p = Path(root or Path(__file__).resolve().parent)/'modelo_final'
    config = json.loads((p/'configuracion.json').read_text(encoding='utf-8'))
    return config, joblib.load(p/'preprocesador.joblib'), joblib.load(p/'modelo.joblib')


def predecir(frame, root=None):
    config, pre, model = cargar_modelo(root)
    frame = frame.copy()
    missing = set(FEATURES_FINAL)-set(frame.columns)
    if missing: raise ValueError('Faltan variables: '+', '.join(sorted(missing)))
    for v in ['edad', 'antiguedad', 'horas']:
        frame[v] = pd.to_numeric(frame[v], errors='raise')
    if ((frame.edad.dropna() < 18) | (frame.edad.dropna() > 110)).any(): raise ValueError('Edad fuera del alcance adulto')
    if ((frame.horas.dropna() <= 0) | (frame.horas.dropna() > 100)).any(): raise ValueError('Horas fuera del alcance (0,100]')
    desconocidas = set(frame.ciudad.dropna().astype(str))-set(CIUDADES+[RESTO])
    if desconocidas: raise ValueError('Ciudad no reconocida: '+', '.join(sorted(desconocidas)))
    NEIGHBOR_CACHE.clear()
    s = scores(model, pre.transform(inputs(frame, FEATURES_FINAL)).astype('float32'))
    costos = np.asarray(config['matriz_costos'], dtype=float)
    decision = dc.decidir(s, costos); esperado = s @ costos
    out = pd.DataFrame({'modelo': config['familia'], 'clase_asignada': decision+1, 'clase_max_probabilidad': s.argmax(axis=1)+1})
    for c in range(4): out[f'prob_C{c+1}'] = s[:, c]
    for c in range(4): out[f'costo_esperado_C{c+1}'] = esperado[:, c]
    return out


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--entrada', required=True); p.add_argument('--salida', default='predicciones_nuevas.csv'); a = p.parse_args()
    result = predecir(pd.read_csv(a.entrada)); result.to_csv(a.salida, index=False); print(result.to_string(index=False))
