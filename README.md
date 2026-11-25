# GEIH Colombia | Machine Learning y series de tiempo

**Distribución, clasificación y pronóstico del ingreso laboral en Colombia: GEIH 2007–2025 y proyección 2026.**

Proyecto académico de la Universidad Tecnológica de Bolívar (2026-II). Integrantes: Luis Florez Tapia, Carlos Arias Nobmann, Daniel Seluan, Duban Quintana Alcalá (Machine Learning) y Angel Vega (Series de tiempo).

## Pregunta y modelo final

**¿Qué tan bien clasifica un único modelo nacional, con la ciudad como variable, el ingreso laboral de un asalariado en cuatro tramos (C1 ≤ P10 < C2 ≤ P50 < C3 ≤ P90 < C4), con cortes P10, P50 y P90 calculados en entrenamiento por ciudad y año?**

- **Modelo final (el único que se despliega):** dominio nacional con la ciudad como predictor (Medellín AM, Barranquilla AM, Bogotá, Bucaramanga AM, Cali AM y Resto del país). Está empaquetado en [`modelo_final/`](modelo_final/) con su [ficha](modelo_final/FICHA_MODELO.md) y se usa con `cargar_y_predecir.py`.
- **Regla de decisión por matriz de costos** ([`decision_costos.py`](decision_costos.py)): dejar por fuera a una persona de C1 (asignarle un tramo superior) cuesta 3 por tramo; los demás errores cuestan la distancia entre tramos. Cada persona recibe la clase de mínimo costo esperado; con esta matriz se asigna C1 si y solo si P(C1) ≥ 0,25.
- **Tasas por grupo:** TPR, FPR y VPP de cada clase frente a las demás por sexo, edad, educación y ciudad.
- **Comparación:** las cinco áreas conservan modelos propios con el mismo protocolo; la partición alineada permite compararlos con el modelo final sobre las mismas personas de prueba.
- **Series de tiempo:** 25 procedimientos para pronosticar P10, P50, P90 y promedio mensual en 2026, con validación temporal y escenario sin pandemia.

## Informes

- [GEIH_MACHINE_LEARNING.pdf](GEIH_MACHINE_LEARNING.pdf) y [GEIH_SERIES_TIEMPO.pdf](GEIH_SERIES_TIEMPO.pdf): mismo proyecto, una portada por curso. En esta copia pública se omiten los códigos estudiantiles de la portada.
- Documentos Quarto: [GEIH_MACHINE_LEARNING.qmd](GEIH_MACHINE_LEARNING.qmd) y [GEIH_SERIES_TIEMPO.qmd](GEIH_SERIES_TIEMPO.qmd).

## Estructura

- `ejecutar_proyecto.py`: punto de entrada; calcula, verifica y genera el PDF de cada QMD.
- `ml_final.py`, `decision_costos.py`, `auditoria_clusters_final.py`: clasificación con CV anidada 3×2, regla de costos, tasas por grupo, ACP y K-means.
- `regresion_final.py`: regresión individual (511 subconjuntos, Ridge y Lasso).
- `series_temporales.py`, `series_final.py`, `parametros_series_final.py`: series, pronósticos, intervalos y parámetros.
- `informe_final.py`, `compilar_historico_local.py`, `verificar_final.py`: informe y verificación independiente.
- `cargar_y_predecir.py`: despliegue del modelo final.
- `datos_historicos/`, `datos_ml_nacional/`, `resultados_historicos/`: entradas analíticas y manifiestos.

## Reproducción

Requisitos: Python 3.14 de 64 bits, Quarto, TinyTeX/XeLaTeX y las fuentes TeX Gyre Pagella y Heros; se recomiendan 16 GB de RAM.

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements_final.txt
quarto install tinytex
.\.venv\Scripts\python.exe -X utf8 ejecutar_proyecto.py --recalcular
```

`GEIH_NJOBS` fija los núcleos de bosques, vecinos y álgebra lineal sin cambiar resultados. Consulte [LEEME_PROFESOR.md](LEEME_PROFESOR.md) para más detalles.

Para clasificar perfiles nuevos con el modelo final:

```powershell
.\.venv\Scripts\python.exe cargar_y_predecir.py --entrada perfiles.csv --salida predicciones.csv
```

## Datos y alcance

Fuente: [GEIH – DANE, catálogo ANDA](https://microdatos.dane.gov.co/index.php/catalog/853). Se incluyen bases analíticas procesadas a partir de microdatos anonimizados de acceso público, no los archivos brutos. Los ingresos son nominales y sin ponderar; los resultados no son cifras oficiales del DANE ni sirven para decisiones individuales automatizadas. Las series no alcanzan las 240 observaciones mensuales exigidas en la rúbrica.
