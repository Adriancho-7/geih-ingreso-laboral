# GEIH — paquete para revisar y regenerar el proyecto (Machine Learning y Series de tiempo)

Integrantes: Luis Florez Tapia, Carlos Arias Nobmann, Daniel Seluan, Duban Quintana Alcalá (Machine Learning) y Angel Vega (Series de tiempo).

Este paquete contiene el QMD, sus módulos Python, las entradas analíticas (Parquet y CSV), bibliografía, estilo, logo, los manifiestos de trazabilidad, el **modelo final empaquetado** (`modelo_final/`) y los PDF `GEIH_MACHINE_LEARNING.pdf` y `GEIH_SERIES_TIEMPO.pdf`. No incluye los resultados intermedios de todos los modelos ni los gráficos sueltos: se regeneran con el ejecutor. No hay que descargar otras bases.

## Qué cambió frente a la revisión

1. **Cuatro clases con regla de decisión por costos.** El umbral por costo de la versión binaria se reemplazó por una matriz de costos (`decision_costos.py`). Dejar por fuera a una persona de C1 (asignarle un tramo superior) cuesta 3 por tramo, como el falso negativo binario; los demás errores cuestan la distancia entre tramos. Cada persona recibe la clase de mínimo costo esperado; con esta matriz se asigna C1 si y solo si P(C1) ≥ 0,25, el mismo umbral binario.
2. **Tasas por grupo.** Para cada clase frente a las demás se reportan TPR, FPR y VPP por sexo, edad, educación y ciudad (sección «Tasas por grupo» y anexo).
3. **Un solo modelo final.** El modelo que se despliega y que describe la ficha es el del **dominio nacional con la ciudad como variable** (cinco áreas + Resto del país). Los cortes P10, P50 y P90 se calculan en entrenamiento por ciudad y año. Los modelos de las cinco áreas quedan como comparación, sobre las mismas personas de prueba gracias a una partición alineada.

## Ejecución en Windows (VS Code o PowerShell)

Instale Python 3.14 de 64 bits y Quarto. Descomprima el ZIP y abra la carpeta `GEIH_Proyecto_Final`. En una terminal PowerShell situada en esa carpeta:

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements_final.txt
quarto install tinytex
.\.venv\Scripts\python.exe -X utf8 ejecutar_proyecto.py --recalcular
```

En Linux/macOS use `.venv/bin/python`. XeLaTeX necesita las fuentes TeX Gyre Pagella y Heros. El entrenamiento completo requiere memoria y tiempo considerables (se recomiendan 16 GB de RAM); la variable `GEIH_NJOBS` fija cuántos núcleos usan bosques, vecinos y álgebra lineal sin cambiar los resultados (por ejemplo, `$env:GEIH_NJOBS='6'`). Si falta memoria en series temporales, establezca `$env:GEIH_SERIES_WORKERS='1'`.

## Usar el modelo final

```powershell
.\.venv\Scripts\python.exe cargar_y_predecir.py --entrada perfiles.csv --salida predicciones.csv
```

El CSV necesita las columnas `edad`, `antiguedad` (meses), `horas`, `sexo`, `jefe`, `estudia`, `estado_civil`, `educacion` y `ciudad` (Medellín AM, Barranquilla AM, Bogotá, Bucaramanga AM, Cali AM o Resto del país). La salida trae la clase asignada por la regla de costos, la clase de máxima probabilidad, las probabilidades y los costos esperados. La ficha está en `modelo_final/FICHA_MODELO.md`.

## Resultado

El ejecutor calcula los modelos, verifica los resultados (`verificar_final.py`) y genera el PDF de cada QMD presente. Después del primer cálculo puede usar `--solo-informe` para regenerar solo figuras y PDF. La ejecución reemplaza el PDF incluido.

## Datos y límites

Las entradas son las bases analíticas GEIH (Parquet de las cinco áreas y nacional), la partición original de las áreas y los indicadores mensuales y anuales. Las bases procesadas no equivalen a los microdatos crudos del DANE, que no se incluyen. La base nacional no conserva el código de las demás áreas metropolitanas, por eso la ciudad del modelo final tiene la categoría Resto del país. La serie no alcanza las 240 observaciones exigidas; sigue pendiente la excepción docente. Los resultados pueden variar ligeramente por versiones, plataforma y fuentes tipográficas.
