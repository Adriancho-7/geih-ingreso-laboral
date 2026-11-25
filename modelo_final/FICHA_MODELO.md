# Ficha del modelo final

| Campo | Contenido |
|:--|:--|
| Nombre y versión | Clasificador GEIH de tramos de ingreso laboral · modelo final · versión 2026-11-25 |
| Modelo desplegado | Random Forest {"max_features": 0.5}, entrenado con el 60 % nacional (n = 1 027 959); es el único que se despliega (carpeta modelo_final/ y cargar_y_predecir.py) |
| Pregunta | Asignar a un asalariado uno de cuatro tramos relativos de ingreso laboral: C1 ≤ P10; P10 < C2 ≤ P50; P50 < C3 ≤ P90; C4 > P90 |
| Cortes | P10, P50 y P90 por ciudad y año, calculados solo con entrenamiento (archivo modelo_final/cortes_entrenamiento.csv) |
| Datos | GEIH 2008–2025, base nacional armonizada, sin ponderar: 1 713 513 personas-mes; 60/20/20 por hogar-mes |
| Predictores | edad, antigüedad, horas, sexo, jefatura, estudia, estado civil, educación y ciudad (cinco áreas + Resto del país); el ingreso nunca es predictor |
| Uso previsto | Priorizar el contenido de una campaña informativa de orientación laboral y financiera cuando el ingreso no se declara; siempre revisable por la persona |
| Usos no previstos | Negar o asignar empleo, crédito, subsidios o beneficios; estimar el ingreso en COP; decisiones automáticas sin revisión humana |
| Regla de decisión | Mínimo costo esperado con la matriz de costos: asignar un tramo superior a una persona de C1 cuesta 3 por tramo; los demás errores cuestan la distancia entre tramos. Se asigna C1 si y solo si P(C1) ≥ 0.25, el umbral de la versión binaria |
| Selección | Familia probabilística de menor costo esperado en validación externa; hiperparámetros por CV anidada 3x2; prueba usada una sola vez |
| Desempeño en prueba | Costo 880 por mil personas (constante: 1271); F1 macro 0.381; exactitud 0.433; AUC macro 0.761 |
| Tasas por clase (prueba) | C1: TPR 0.69, FPR 0.33, VPP 0.38; C2: TPR 0.27, FPR 0.24, VPP 0.31; C3: TPR 0.47, FPR 0.21, VPP 0.59; C4: TPR 0.13, FPR 0.01, VPP 0.60 |
| Equidad | TPR, FPR y VPP por clase frente a las demás según sexo, edad, educación y ciudad; las brechas se informan, no se corrigen |
| Limitaciones | Muestra no ponderada de asalariados con contrato escrito; asociaciones, no causas; cortes de años pasados; sexo es predictor; calibración no garantizada fuera de la GEIH |
| Responsables | Autores del proyecto; revisión humana obligatoria antes de cualquier uso |

## Matriz de costos (fila = clase real, columna = clase asignada)

| Real \ Asignada | C1 | C2 | C3 | C4 |
|:--|--:|--:|--:|--:|
| C1 | 0 | 3 | 6 | 9 |
| C2 | 1 | 0 | 1 | 2 |
| C3 | 2 | 1 | 0 | 1 |
| C4 | 3 | 2 | 1 | 0 |
